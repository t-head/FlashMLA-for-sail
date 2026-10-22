"""Native E4M3 cache dispatch, separate from packed-FP8/BF16 attention."""

from functools import cache

import torch

# Folding reuses KV once the grid fills the device. The larger H16 groups
# favor reuse after the active KV working set exceeds two cache footprints.
# Keep both thresholds device-based.
MAX_VERIFY_QUERIES = 32
VERIFY_TOPK = 2048


@cache
def _device_properties(device: torch.device):
    # Q provides an explicit device index; immutable hardware properties can
    # be reused without repeating CUDA initialization checks on every decode.
    return torch.cuda.get_device_properties(device)


def use_verify_union(
    batch: int, queries: int, heads: int, width: int, compute_units: int,
    cache_bytes: int, head_dim: int = 512, kv_rows: int = 0,
) -> bool:
    if not (2 <= queries <= MAX_VERIFY_QUERIES and width == VERIFY_TOPK
            and heads in (8, 16) and batch >= compute_units):
        return False
    if heads == 8:
        return True
    folded_heads = queries * heads
    # Complete narrow tiles can share resident KV without the padded wide
    # tail that the general working-set heuristic accounts for below.
    if (head_dim == 512 and 0 < kv_rows * head_dim <= cache_bytes
            and folded_heads > 64 and folded_heads % 64 == 32):
        return True
    # A narrow final tile repeats the whole KV scan. Keep folding for pools
    # that fit in cache, and let the C++ reuse hint handle larger streaming pools.
    if (head_dim == 512 and heads == 16 and folded_heads > 64
            and 0 < folded_heads % 64 <= heads
            and cache_bytes < kv_rows * head_dim <= 16 * cache_bytes):
        return False
    working_set = batch * width * head_dim
    folded_tiles = (queries * heads + 63) // 64
    # D512 compact queries favor a resident pool over a padded folded tile.
    # D576 retains its separate producer/consumer pipeline crossover.
    accessed_bytes = (min(working_set, kv_rows * head_dim)
                      if head_dim == 512 and kv_rows > 0 else working_set)
    if folded_tiles == 1 and folded_heads < 64 and accessed_bytes < 8 * cache_bytes:
        return False
    # H16 needs full tiles while KV is near cache capacity. For larger working
    # sets, three queries sharing each tile amortize the classifier and padding.
    # D576's streaming fallback uses the wider three-stage gather pipeline.
    # A large backing pool can need it even when the selected rows are smaller.
    streaming_pool = head_dim == 576 and kv_rows * head_dim > 24 * cache_bytes
    # D576 amortizes the folded path at a smaller active working set.
    reuse_cache_footprints = 3 if head_dim == 576 else 4
    minimum_reuse = 3 if working_set >= reuse_cache_footprints * cache_bytes or streaming_pool else 4
    return working_set >= 2 * cache_bytes and queries >= minimum_reuse * folded_tiles



def paged_attention(
    extension,
    decode,
    q,
    k_cache,
    head_dim_v,
    softmax_scale,
    causal,
    indices,
    attn_sink,
    extra_k_cache,
    extra_indices_in_kvcache,
    topk_length,
    extra_topk_length,
    out,
    num_queries_per_request=None,
):
    """Flatten paged inputs while retaining per-query masks and optional lengths."""
    if causal:
        raise ValueError(
            "native sparse attention requires caller-masked indices and causal=False"
        )
    q_shape, cache_shape, index_shape = q.shape, k_cache.shape, indices.shape
    if len(q_shape) != 4 or len(cache_shape) != 4 or cache_shape[2] != 1:
        raise ValueError(
            "native paged attention expects Q[B,S,H,D], cache[pages,page_size,1,D]"
        )
    batch, queries, heads, dim = q_shape
    if cache_shape[-1] != dim or head_dim_v != 512:
        raise ValueError(
            "native paged attention requires matching Q/KV dimensions and DV=512"
        )
    if len(index_shape) != 3 or index_shape[:2] != (batch, queries):
        raise ValueError("native indices must be [batch,queries,sparse_width]")
    width = index_shape[-1]
    # Preserve explicit grouping for flattened callers; divisibility alone
    # cannot identify request boundaries for variable draft lengths.
    request_queries = queries
    request_batch = batch
    if num_queries_per_request is not None:
        if num_queries_per_request < 1:
            raise ValueError("num_queries_per_request must be positive")
        if queries == 1:
            if batch % num_queries_per_request:
                raise ValueError("flattened Q must contain complete request groups")
            request_queries = num_queries_per_request
            request_batch = batch // request_queries
        elif queries != num_queries_per_request:
            raise ValueError("num_queries_per_request disagrees with Q's query axis")
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    flat_q = q.reshape(batch * queries, heads, dim).contiguous()
    flat_indices = indices.reshape(batch * queries, 1, width).contiguous()
    flat_out = None if out is None else out.view(batch * queries, heads, head_dim_v)

    def lengths(x):
        if x is None:
            return None
        if x.numel() == batch and queries > 1:
            return x.repeat_interleave(queries).contiguous()
        return x.reshape(batch * queries).contiguous()

    has_extra = extra_k_cache is not None or extra_indices_in_kvcache is not None
    if has_extra:
        if extra_k_cache is None or extra_indices_in_kvcache is None:
            raise ValueError("native extra cache and indices must be provided together")
        if (
            extra_k_cache.dtype != torch.float8_e4m3fn
            or extra_k_cache.ndim != 4
            or extra_k_cache.shape[2:] != (1, dim)
            or extra_indices_in_kvcache.shape[:2] != (batch, queries)
        ):
            raise ValueError("native extra cache must use the same flat E4M3 layout")
    # The classifier validates the full index lists on every graph replay.
    properties = _device_properties(q.device)
    compute_units = properties.multi_processor_count
    union_width = ((width + request_queries - 1 + 63) // 64) * 64
    if has_extra:
        union_width += extra_indices_in_kvcache.shape[-1]
    if (
        has_extra and dim == 512 and heads == 8 and 1 < request_queries <= MAX_VERIFY_QUERIES
        and q.dtype in (torch.float8_e4m3fn, torch.uint8)
        and width % 64 == 0 and union_width <= 2048
        and request_batch * ((request_queries * heads + 63) // 64) >= compute_units
        and request_queries * heads * 4 >= ((request_queries * heads + 63) // 64) * 64 * 3
    ):
        output, lse = extension.native_fp8_verify_fwd(
            flat_q, k_cache.reshape(-1, 1, dim), flat_indices, request_queries, scale,
            attn_sink, lengths(topk_length), flat_out, True,
            extra_k_cache.reshape(-1, 1, dim) if has_extra else None,
            extra_indices_in_kvcache.reshape(batch * queries, 1, extra_indices_in_kvcache.shape[-1]).contiguous() if has_extra else None,
            lengths(extra_topk_length), queries,
        )
        return output.view(batch, queries, heads, 512), lse.view(batch, heads, queries)
    if (
        not has_extra
        and dim in (512, 576)
        and q.dtype in (torch.float8_e4m3fn, torch.uint8)
        and use_verify_union(
            request_batch, request_queries, heads, width,
            compute_units, properties.L2_cache_size, dim,
            kv_rows=cache_shape[0] * cache_shape[1],
        )
    ):
        output, lse = extension.native_fp8_verify_fwd(
            flat_q,
            k_cache.reshape(-1, 1, dim),
            flat_indices,
            request_queries,
            scale,
            attn_sink,
            lengths(topk_length),
            flat_out,
            lse_queries=queries,
        )
        return output.view(
            batch, queries, heads, 512
        ), lse.view(batch, heads, queries)
    if (
        has_extra
        and dim == 512
        and head_dim_v == 512
        and width + extra_indices_in_kvcache.shape[-1] <= 2048
    ):
        output, lse = extension.native_fp8_extra_fwd(
            flat_q,
            k_cache.reshape(-1, 1, dim),
            flat_indices,
            extra_k_cache.reshape(-1, 1, dim),
            extra_indices_in_kvcache.reshape(batch * queries, 1, extra_indices_in_kvcache.shape[-1]).contiguous(),
            scale,
            attn_sink,
            lengths(topk_length),
            lengths(extra_topk_length),
            flat_out,
            queries,
        )
        return output.view(
            batch, queries, heads, 512
        ), lse.view(batch, heads, queries)
    if not has_extra:
        output, lse = extension.sparse_decode_fp8_fwd(
            flat_q, k_cache.reshape(-1, 1, dim), flat_indices,
            scale, 1.0, 1.0, head_dim_v, attn_sink, lengths(topk_length),
            flat_out, queries,
        )
        return output.view(batch, queries, heads, head_dim_v), lse.view(batch, heads, queries)
    output, lse = decode(
        flat_q,
        k_cache.reshape(-1, 1, dim),
        flat_indices,
        scale,
        d_v=head_dim_v,
        topk_length=lengths(topk_length),
    )
    other, other_lse = decode(
        flat_q,
        extra_k_cache.reshape(-1, 1, dim),
        extra_indices_in_kvcache.reshape(batch * queries, 1, extra_indices_in_kvcache.shape[-1]).contiguous(),
        scale,
        d_v=head_dim_v,
        topk_length=lengths(extra_topk_length),
    )
    output, lse = extension.native_fp8_merge(
        torch.stack((output, other), dim=1),
        torch.stack((lse, other_lse), dim=1),
        attn_sink,
        flat_out,
    )
    return output.view(
        batch, queries, heads, head_dim_v
    ), extension.native_fp8_lse_to_natural(lse, batch, queries)
