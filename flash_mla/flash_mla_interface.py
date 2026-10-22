from typing import Optional, Tuple
import dataclasses

import torch

import flash_mla_cuda
from . import _sparse_native_fp8

@dataclasses.dataclass
class FlashMLASchedMeta:
    """
    A class that stores the tile scheduler metadata of FlashMLA
    """

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int

        causal: bool
        is_fp8_kvcache: bool
        qkv_fp8: bool
        topk: Optional[int]

        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False

    config: Optional[Config] = None

    tile_scheduler_metadata: Optional[torch.Tensor] = None   # (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
    num_splits: Optional[torch.Tensor] = None                # (batch_size + 1), dtype torch.int32.

# def get_mla_metadata(
#     cache_seqlens: torch.Tensor,
#     num_heads_per_head_k: int,
#     num_heads_k: int,
#     num_heads_q: Optional[int] = None,
#     is_fp8_kvcache: bool = False,
#     qkv_fp8: bool = False
#     topk: Optional[int] = None
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Arguments:
#         cache_seqlens: (batch_size), dtype torch.int32.
#         num_heads_per_head_k: Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
#         num_heads_k: num_heads_k.
#         num_heads_q: The number of q heads. This argument is optional when sparse attention is not enabled
#         is_fp8_kvcache: Whether the k_cache and v_cache are in fp8 format.
#         qkv_fp8: Whether both q and k_cache are in fp8 format (requires the fp8 kernel path).
#         topk: If not None, sparse attention will be enabled, and only tokens in the `indices` array passed to `flash_mla_with_kvcache_sm90` will be attended to.

#     Returns:
#         tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
#         num_splits: (batch_size + 1), dtype torch.int32.
#     """
#     return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k, num_heads_q, is_fp8_kvcache, topk, qkv_fp8)

def get_mla_metadata(
    *args,
    **kwargs
) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty instance of FlashMLASchedMeta. The actual scheduling metadata will be generated during the first invocation of flash_mla_with_kvcache.

    Arguments:
        This function does not need any arguments, but we keep *args and **kwargs to be compatible with the old interface.

    Return:
        A tuple. Due to historical reasons, we return a tuple of (FlashMLASchedMeta, None) now. Only the first element is useful.
    """
    return FlashMLASchedMeta(), None


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    num_queries_per_request: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
                Different modes (including fp8/bf16, sparsity, and model version (i.e. V3.2 or MODEL1)) has different KV cache layouts. See comments below for details.
                The KV cache must be contiguously valid for sparse attention on sm100. Here "contiguously valid" means that every byte, from the very beginning of the KV cache, till the last byte in the KV cache, is valid memory address to visit (i.e. won't IMA). In other words, the KV cache could be a slice of a larger array, but cannot be a list of disjoint memory blocks.
                Besides, some kernels also have their own requirements on the layout of k cache, including:
                    - For sparse fp8 decoding kernel on F3, k_cache.stride(0) must be a multiple of 656B (for V32) or 576B (for MODEL1). Padding is needed sometimes.
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32. Can be None when sparse attention is used.
        cache_seqlens: (batch_size), torch.int32. Can be None when sparse attention is used.
        head_dim_v: Head_dim of v. Must be 512
        tile_scheduler_metadata: FlashMLASchedMeta, return by get_mla_metadata. You may reuse the same sched_meta across different invocations, but only when the tensor shapes and the values of cache_seqlens, topk_length, and extra_topk_length remain the same.
        num_queries_per_request: Optional explicit request grouping for native FP8
            Q flattened as [B*S,1,H,D]. Use S; grouped [B,S,H,D] infers it.
            This controls KV-reuse dispatch without assuming a fixed draft count.
        num_splits placeholder: must be "None" (to be compatible with the old interface).
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim_k).
        causal: bool. Whether to apply causal attention mask. Only valid for dense attention
        is_fp8_kvcache: bool.
        indices: (batch_size, seq_len_q, topk). KV indices when sparse attention is enabled.
                    Pay attention that indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * block_size + (the offset of token t among the page block),
                    where t is the k-th token of the j-th q-sequence in the i-th batch.
        attn_sink: Optional[torch.Tensor], (num_heads_q, ), torch.float32. If presented, the final output will be scaled by exp(lse) / (exp(lse) + exp(attn_sink)). Have no affect on the returned softmax_lse. +inf will cause the result to become 0.
        extra_k_cache and extra_indices_in_kvcache: If provided, will attend to these extra tokens in addition to those in k_cache and indices_in_kvcache. This is used to support MODEL1. Their format requirements are the same as k_cache and indices_in_kvcache respectively.
        topk_length/extra_topk_length: (batch_size, ), torch.int32. If provided, only the leftmost topk_length indices will be processed. Useful when the actual topk for different queries are different so that we can save some computation, compared to masking.
        out: Optional contiguous output with shape
            (batch_size, seq_len_q, num_heads_q, head_dim_v). Native FP8 requires
            BF16 output with a 16-byte-aligned base address; other paths use q's
            dtype. If provided, results are written into this buffer to avoid
            allocation. Dense attention supports only num_heads_k == 1 (MLA).

    For DeepSeek V3, DeepSeek V3.1, and DeepSeek V3.2:
        head_dim should be 576 while head_dim_v should be 512.
        In FP8+sparse mode, each token's KV cache is 656 Bytes, structured as:
            - The shape of the tensor `k_cache` is (num_blocks, page_block_size, num_heads_k, head_dim), and num_heads_k must be 1.
            - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
            - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
            - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.

    For DeepSeek MODEL1:
        head_dim should be 512 while head_dim_v is also 512.

        In FP8+sparse mode, every block can be divided into two parts. The first parts stores NoPE0, RoPE0, NoPE1, RoPE1, ... while the second part stores scale factors: 7xue8m0, 1Bpad, 7xue8m0, 1Bpad, ...

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    # Packed FP8 caches may also expose an E4M3 view, but include scale/rope
    # storage beyond D. Only a flat, dimension-matched cache uses native Q8/KV8.
    if (
        indices is not None
        and k_cache.dtype == torch.float8_e4m3fn
        and k_cache.shape[-1] == q.shape[-1]
    ):
        return _sparse_native_fp8.paged_attention(
            flash_mla_cuda,
            flash_mla_sparse_decode_fp8_fwd,
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
            num_queries_per_request,
        )

    sched_meta = tile_scheduler_metadata
    indices_in_kvcache = indices
    if isinstance(sched_meta, torch.Tensor):
        # Preserve caller-provided schedules.
        assert indices is not None and not causal, (
            "raw metadata requires sparse attention"
        )
        assert isinstance(num_splits, torch.Tensor), "raw metadata requires num_splits"
        scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        result = flash_mla_cuda.sparse_decode_fwd(
            q,
            k_cache,
            head_dim_v,
            indices,
            attn_sink,
            topk_length,
            sched_meta,
            num_splits,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            scale,
            out,
        )
        return result[0], result[1]
    assert isinstance(sched_meta, FlashMLASchedMeta), "tile_scheduler_metadata must be of type FlashMLASchedMeta"
    assert num_splits is None, "num_splits must be None"

    topk = indices_in_kvcache.shape[-1] if indices_in_kvcache is not None else None
    extra_k_page_block_size = extra_k_cache.shape[1] if extra_k_cache is not None else None
    extra_topk = extra_indices_in_kvcache.shape[-1] if extra_indices_in_kvcache is not None else None
    qkv_fp8 = (q.dtype == torch.float8_e4m3fn) and (k_cache.dtype == torch.float8_e4m3fn)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not sched_meta.have_initialized:
        # Sanity check. We only perform sanity check during the first invocation to save CPU time.
        if indices_in_kvcache is not None:
            assert not causal, "causal must be False when indices_in_kvcache is not None (i.e. sparse attention is enabled)"
        # Initialize the tile scheduler metadata during the first invocation.
        sched_meta.tile_scheduler_metadata = None
        sched_meta.num_splits = None
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],
            causal,
            is_fp8_kvcache,
            qkv_fp8,
            topk,
            extra_k_page_block_size,
            extra_topk,
        )
        # Metadata will be generated lazily in C++ on first call

    else:
        # Check whether the input arguments are consistent with sched_meta
        helper_msg = " Your input arguments are inconsistent with sched_meta. Please make sure the input arguments are consistent across different invocations of flash_mla_with_kvcache on the same sched_meta."
        assert sched_meta.config is not None
        assert sched_meta.config.b == q.shape[0], "sched_meta.config.b must be equal to batch_size." + helper_msg
        assert sched_meta.config.s_q == q.shape[1], "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
        assert sched_meta.config.h_q == q.shape[2], "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
        assert sched_meta.config.page_block_size == k_cache.shape[1], "sched_meta.config.page_block_size must be equal to page_block_size." + helper_msg
        assert sched_meta.config.h_k == k_cache.shape[2], "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
        assert sched_meta.config.causal == causal, "sched_meta.config.causal must be equal to causal." + helper_msg
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache." + helper_msg
        assert sched_meta.config.qkv_fp8 == qkv_fp8, "sched_meta.config.qkv_fp8 must be equal to (q.dtype == torch.float8_e4m3fn and k_cache.dtype == torch.float8_e4m3fn)." + helper_msg
        assert sched_meta.config.topk == topk, "sched_meta.config.topk must be equal to the last dim of indices_in_kvcache." + helper_msg
        assert sched_meta.config.extra_page_block_size == extra_k_page_block_size, "sched_meta.config.extra_page_block_size must be equal to the page_block_size of extra_k_cache." + helper_msg
        assert sched_meta.config.extra_topk == extra_topk, "sched_meta.config.extra_topk must be equal to the last dim of extra_indices_in_kvcache." + helper_msg

    # Dispatch to appropriate C++ function based on whether indices are provided
    if topk is not None:
        # Sparse decode path
        assert causal == False, "causal must be `false` if sparse attention is enabled."
        out, softmax_lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.sparse_decode_fwd(
            q,
            k_cache,
            head_dim_v,
            indices,
            attn_sink,
            topk_length,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            softmax_scale,
            out
        )
    else:
        # Dense decode path
        out, softmax_lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.dense_decode_fwd(
            q,
            k_cache,
            head_dim_v,
            cache_seqlens,
            block_table,
            softmax_scale,
            causal,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
            out
        )

    # Update metadata from C++ return values
    sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
    sched_meta.num_splits = new_num_splits

    return out, softmax_lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by exp(lse) / (exp(lse) + exp(attn_sink)).
            +-inf in attn_sink will be handled normally (i.e., -inf has no effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32. If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]], ignoring later k/v tokens (even if provided in indices).
            In extremely rare cases (topk_length provided, there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token containing NaN), operator output will contain NaN, so please avoid this situation.
        out: optional pre-allocated output tensor with shape [s_q, h_q, d_v], bfloat16, contiguous on the last dim. If provided, the result will be written into this buffer to avoid allocation.

    Returns:
        (output, max_logits, lse)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
    )
    return results


def flash_mla_sparse_fp8_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    compute_aux: bool = True,
    trusted_full_topk: bool = False,
    out_rope_freqs_cis: Optional[torch.Tensor] = None,
    out_rope_positions: Optional[torch.Tensor] = None,
    out_fp8: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    out_quant_heads_per_group: int = 0,
    out_token_offset: int = 0,
    out_static_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Native FP8 (q8 x kv8) sparse attention prefill kernel.

    Distinct from ``flash_mla_sparse_fwd``: that kernel takes bf16 q/kv (its fp8-KV
    option dequantises during the gather and still runs the bf16 MMA), whereas this
    one keeps both operands in e4m3 and uses the FP8 tensor-core path end to end.

    Args:
        q: [s_q, h_q, d_qk], bfloat16, float8_e4m3fn, or uint8 E4M3 bytes.
            h_q is a positive multiple of 8;
            d_qk must be 512 or 576. BF16 Q is converted to e4m3 while loading
            the kernel's shared-memory Q tile, avoiding a materialized FP8 tensor.
            BF16 Q may have strided token/head axes with dense, 16-byte-aligned rows.
            FP8 Q must be contiguous. Head tile selection is internal.
        kv: [s_kv, 1, d_qk], bfloat16, float8_e4m3fn, or uint8 E4M3 bytes.
            BF16 KV is converted once
            to a temporary E4M3 tensor using saturating round-to-nearest conversion,
            without changing the persistent cache. BF16 rows may be strided with
            dense, 4-byte-aligned channels. FP8 KV must be contiguous with a
            16-byte-aligned base address.
        indices: [s_q, h_kv, topk], int32. topk must be a positive multiple of 64 and
            <= 2176. Invalid indices (< 0 or >= s_kv) are masked exactly, so the tensor
            may be padded with -1; there is no need to append zero KV rows.
        sm_scale: float, the softmax scale.
        q_scale: float, per-tensor dequantisation scale for q.
        kv_scale: float, per-tensor dequantisation scale for kv. Both scales are folded
            into the softmax exponent and the epilogue, so dequantisation is free.
        d_v: The dimension of value vectors. Can only be 512.
        attn_sink: optional [h_q] float32 attention sink in natural-log space.
        topk_length: optional [s_q] int32 valid prefix length for each indices row.
            Entries after this length are masked even if their index value is valid.
            For BF16 KV, omitted lengths are derived from the last valid column,
            retaining valid entries after holes or an unpooled sparse tail.
        out: optional preallocated contiguous [s_q, h_q, 512] bfloat16 output,
            with a 16-byte-aligned base address. Must be None when out_fp8 is supplied.
        compute_aux: whether to materialize max_logits and lse. Set False when the
            caller only consumes output; returned aux tensors then have shape [0].
        trusted_full_topk: certifies that every index before ``topk_length`` is in
            ``[0, s_kv)`` and every later slot is padding. This removes validity
            scans and masks from complete active tiles; only the final partial tile
            is slot-masked. The default is False and preserves support for arbitrary
            invalid indices anywhere in the active prefix, including length-full
            rows.
        Fused inverse-RoPE/FP8 output writes directly to out_fp8 and out_scale.
            It requires contiguous Q, h_q divisible by 64, topk <= 2048,
            a 2-byte-aligned out_fp8 base address, out=None, and compute_aux=False.
            Callers must consume these supplied buffers; all three returned
            tensors are empty sentinels with shape [0].

    Returns:
        (output, max_logits, lse)
        - output: [s_q, h_q, d_v], bfloat16; shape [0] in fused output mode.
        - max_logits: [s_q, h_q], float. The ceil-quantised running max used internally,
            i.e. an upper bound on the true max rather than the max itself.
            Shape [0] when compute_aux=False.
        - lse: [s_q, h_q], float, base-2 log-sum-exp of quantized QK scores,
            computed in FP32 before attention-sink scaling. Shape [0] when compute_aux=False.
    """
    return flash_mla_cuda.sparse_prefill_fp8_fwd(
        q,
        kv,
        indices,
        sm_scale,
        q_scale,
        kv_scale,
        d_v,
        attn_sink,
        topk_length,
        out,
        compute_aux,
        trusted_full_topk,
        out_rope_freqs_cis,
        out_rope_positions,
        out_fp8,
        out_scale,
        out_quant_heads_per_group,
        out_token_offset,
        0.0 if out_static_scale is None else float(out_static_scale),
    )


def flash_mla_sparse_decode_fp8_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Native FP8 (q8 x kv8) sparse attention decode with automatic kernel selection.

    Same math as ``flash_mla_sparse_fp8_fwd``. FlashMLA selects head tiles,
    aligned key splits and pipeline stages from the input shape and device.
    Unsplit queries write final output directly; split queries merge their
    partial outputs internally. Q, K, P and V use native E4M3 tensor operations.

    Args:
        q: [n_tok, h_q, d_qk] BF16, float8_e4m3fn or uint8 E4M3 bytes.
            h_q is a positive multiple of 8; d_qk is 512 or 576.
            For decode n_tok is the batch size and the
            h_q rows are the q heads, which share one topk list.
        kv: [s_kv, 1, d_qk] float8_e4m3fn or uint8 E4M3 bytes,
            with a 16-byte-aligned base address.
            A paged cache flattens to this, and decode
            indices are already absolute slot ids.
        indices: [n_tok, 1, topk] int32; topk a positive multiple of 64 and <= 2048.
            Invalid entries may be -1, but they are *not* cheaper than valid ones, so
            padding is not a way to vary per-token work.
        attn_sink: optional [h_q] float32, natural-log space, excluded from LSE.
        topk_length: optional [n_tok] int32 valid prefix bounds.
        out: optional contiguous [n_tok,h_q,512] BF16 output with a
            16-byte-aligned base address.
            BF16 Q is quantized once in shared memory, with no global Q8 buffer.

    Returns:
        (output, lse):
        - output: [n_tok, h_q, d_v] bfloat16.
        - lse: [n_tok, h_q] float32, 2-based, and -inf when a row has no valid index.
    """
    return flash_mla_cuda.sparse_decode_fp8_fwd(
        q,
        kv,
        indices,
        sm_scale,
        q_scale,
        kv_scale,
        d_v,
        attn_sink,
        topk_length,
        out,
    )
