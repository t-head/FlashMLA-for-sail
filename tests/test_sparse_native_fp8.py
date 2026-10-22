"""Native Q8/KV8/P8 contracts: direct APIs, paged verify and graph replay.

Use the FP64 oracle for output error, finite values and empty-row semantics.
Numerical LSE proximity is deliberately not an acceptance criterion.
"""

import math

import pytest
import torch

import flash_mla

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="PPU1.5 native FP8 requires sm89 compatibility",
)


def native(kind, q, kv, indices, **kwargs):
    scale = kwargs.pop("sm_scale", q.shape[-1] ** -0.5)
    call = (
        flash_mla.flash_mla_sparse_decode_fp8_fwd
        if kind == "decode"
        else flash_mla.flash_mla_sparse_fp8_fwd
    )
    return call(q, kv, indices, scale, **kwargs)


def inputs(heads=64, dim=576, bf16=False):
    torch.manual_seed(771)
    dtype = torch.bfloat16 if bf16 else torch.float8_e4m3fn
    q = (torch.randn(5, heads, dim, device="cuda") * 0.3).to(dtype)
    kv = (torch.randn(193, 1, dim, device="cuda") * 0.3).to(dtype)
    ids = torch.randint(193, (5, 1, 128), device="cuda", dtype=torch.int32)
    ids[0] = -1
    ids[1, :, 3::7] = 193
    ids[2, :, 5::11] = torch.iinfo(torch.int32).min
    ids[3, :, 1::19] = torch.iinfo(torch.int32).max
    return q, kv, ids


def reference(q, kv, ids, qs=1.0, ks=1.0, lengths=None, sink=None, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    q = q.float().clamp(-448, 448).to(torch.float8_e4m3fn).double() * qs
    kv = (
        kv.reshape(-1, q.shape[-1])
        .float()
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .double()
        * ks
    )
    ids = ids.reshape(len(q), -1)
    outputs, lses = [], []
    # Bound the gathered KV allocation for large verify batches.
    for start in range(0, len(q), 4):
        ix = ids[start : start + 4]
        valid = (ix >= 0) & (ix < len(kv))
        if lengths is not None:
            valid &= (
                torch.arange(ids.shape[-1], device=ids.device)
                < lengths[start : start + 4, None]
            )
        selected = kv[ix.long().clamp(0, len(kv) - 1)]
        scores = torch.bmm(q[start : start + 4], selected.transpose(1, 2)) * scale
        scores.masked_fill_(~valid[:, None], -math.inf)
        lse = torch.logsumexp(scores, -1)
        denominator = lse if sink is None else torch.logaddexp(lse, sink.double())
        weights = torch.exp(scores - denominator[..., None]).nan_to_num()
        outputs.append(torch.bmm(weights, selected[:, :, :512]))
        lses.append(lse / math.log(2))
    return torch.cat(outputs), torch.cat(lses)


def check(result, expected, natural_lse=False):
    want, reference_lse = expected
    out, lse = result[0], result[-1]
    if natural_lse:
        out = out.flatten(0, 1)
        lse = lse.transpose(1, 2).flatten(0, 1)
    assert out.shape == want.shape and lse.shape == reference_lse.shape
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert torch.isfinite(out).all()
    error = (out.double() - want).norm() / want.norm().clamp_min(1e-12)
    assert error < 0.04, f"output relative L2: {error.item():.6f}"
    finite = torch.isfinite(reference_lse)
    assert torch.isfinite(lse[finite]).all()
    is_empty_lse = torch.isposinf if natural_lse else torch.isneginf
    assert is_empty_lse(lse[~finite]).all() and (out[~finite] == 0).all()


def capture(call):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    torch.cuda.empty_cache()
    return graph, result


@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("dim", [512, 576])
def test_decode_masks_scales_graph(heads, dim):
    q, kv, ids = inputs(heads, dim)

    def call():
        return native("decode", q, kv, ids, q_scale=0.25, kv_scale=2.0)

    check(call(), reference(q, kv, ids, 0.25, 2.0))
    graph, result = capture(call)
    graph.replay()
    check(result, reference(q, kv, ids, 0.25, 2.0))
    # Invalid zero-filled keys must not dominate strongly negative valid logits.
    q.copy_(-torch.ones_like(q.float()))
    kv.copy_(torch.ones_like(kv.float()))
    ids.fill_(-1)
    ids[1:, 0, 0] = 7
    graph.replay()
    check(result, reference(q, kv, ids, 0.25, 2.0))


@pytest.mark.parametrize(
    "heads,dim,bf16",
    [
        (16, 512, True),
        (32, 512, False),
        (64, 512, False),
        (64, 576, False),
        (128, 576, True),
    ],
)
def test_prefill_lengths_sink_preallocated(heads, dim, bf16):
    q, kv, ids = inputs(heads, dim, bf16)
    if bf16:
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
    lengths = torch.tensor([0, 1, 125, 64, 128], device="cuda", dtype=torch.int32)
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[0], sink[-1] = -math.inf, math.inf
    out = torch.empty((5, heads, 512), device="cuda", dtype=torch.bfloat16)
    result = native(
        "prefill",
        q,
        kv,
        ids,
        q_scale=0.25,
        kv_scale=2.0,
        attn_sink=sink,
        topk_length=lengths,
        out=out,
    )
    assert result[0].data_ptr() == out.data_ptr()
    check(result, reference(q, kv, ids, 0.25, 2.0, lengths, sink))


@pytest.mark.parametrize("kind", ["decode", "prefill"])
def test_uint8_empty_and_optional_statistics(kind):
    q, kv, ids = inputs()
    plain = native(kind, q, kv, ids)
    packed = native(kind, q.view(torch.uint8), kv.view(torch.uint8), ids)
    torch.testing.assert_close(plain[0], packed[0], atol=0, rtol=0)
    assert native(kind, q[:0], kv, ids[:0])[0].shape == (0, 64, 512)
    if kind == "prefill":
        noaux = native(kind, q, kv, ids, compute_aux=False)
        assert noaux[1].numel() == noaux[2].numel() == 0
        torch.testing.assert_close(noaux[0], plain[0], atol=0, rtol=0)


@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("path", ["prefill_half", "prefill_full", "decode"])
def test_ws_value_column_order(path, dim):
    """One valid key exposes V fragment permutations without softmax error."""
    cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    decode = path == "decode"
    queries = 2 * cu if decode else (cu if path == "prefill_full" else 1)
    heads, width = (16, 2048) if decode else (64, 128)
    q = torch.zeros(queries, heads, dim, device="cuda", dtype=torch.float8_e4m3fn)
    # Different values in adjacent columns and N16 fragments make a swapped
    # byte, fragment or packed output pair observable in the final output.
    kv = torch.linspace(-4, 4, dim, device="cuda").to(q.dtype).reshape(1, 1, dim)
    ids = torch.full((queries, 1, width), -1, device="cuda", dtype=torch.int32)
    ids[:, :, 0] = 0

    def call():
        return native("decode" if decode else "prefill", q, kv, ids)

    graph, result = capture(call)
    for sign in (1, -1):
        kv.copy_((sign * torch.linspace(-4, 4, dim, device="cuda")).to(q.dtype).reshape_as(kv))
        graph.replay()
        expected = kv[0, 0, :512].to(torch.bfloat16).expand_as(result[0])
        torch.testing.assert_close(result[0], expected, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["decode", "prefill"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("q_scale", 0.0),
        ("kv_scale", -1.0),
        ("q_scale", math.nan),
        ("sm_scale", math.inf),
    ],
)
def test_invalid_scale_rejected(kind, field, value):
    q, kv, ids = inputs()
    with pytest.raises(RuntimeError, match="finite and positive"):
        native(kind, q, kv, ids, **{field: value})


@pytest.mark.parametrize("kind", ["decode", "prefill"])
def test_invalid_inputs(kind):
    q, kv, ids = inputs()
    bad_q = q.transpose(0, 1).contiguous().transpose(0, 1)
    for query, cache in ((q[0], kv), (q, kv[:0]), (q, kv.cpu()), (bad_q, kv)):
        with pytest.raises(RuntimeError):
            native(kind, query, cache, ids)


@pytest.mark.parametrize("kind", ["prefill", "decode"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.uint8])
@pytest.mark.parametrize("offset", [1, 16])
def test_kv_storage_offset_alignment(kind, dtype, offset):
    q, kv, ids = inputs()
    kv = kv.view(dtype)
    backing = torch.empty(kv.numel() + offset, device=kv.device, dtype=dtype)
    shifted = backing[offset:].view_as(kv)
    shifted.copy_(kv)
    assert shifted.is_contiguous()
    if offset % 16:
        with pytest.raises(RuntimeError, match="16-byte aligned"):
            native(kind, q, shifted, ids)
        torch.cuda.synchronize()  # Rejection must not enqueue a faulting kernel.
    else:
        torch.testing.assert_close(
            native(kind, q, shifted, ids)[0], native(kind, q, kv, ids)[0], atol=0, rtol=0
        )


@pytest.mark.parametrize("kind", ["prefill", "decode", "extra", "verify", "window", "merge", "paged"])
@pytest.mark.parametrize("offset", [1, 2, 4, 8])
def test_output_storage_offset_alignment(kind, offset):
    import flash_mla_cuda

    q, kv, ids = inputs(heads=8, dim=512)
    q, ids = q[:4], ids[:4]
    # W2048 also covers uint4 output stores in the split-KV merge.
    if kind == "decode":
        ids = ids.repeat(1, 1, 16)
    shape = (4, 8, 512)
    backing = torch.full((math.prod(shape) + offset + 8,), -123., device="cuda", dtype=torch.bfloat16)
    out = backing[offset:offset + math.prod(shape)].view(shape)

    def call(output):
        if kind in ("prefill", "decode"):
            return native(kind, q, kv, ids, out=output)
        if kind == "extra":
            return flash_mla_cuda.native_fp8_extra_fwd(
                q, kv, ids, kv, ids, 512**-.5, None, None, None, output
            )
        if kind in ("verify", "window"):
            return flash_mla_cuda.native_fp8_verify_fwd(
                q, kv, ids, 2, 512**-.5, None, None, output, kind == "window"
            )
        if kind == "merge":
            return flash_mla_cuda.native_fp8_merge(
                q.to(torch.bfloat16)[:, None], torch.zeros(4, 1, 8, device="cuda"), None, output
            )
        return flash_mla.flash_mla_with_kvcache(
            q[:, None], kv[:, None], None, None, 512, flash_mla.FlashMLASchedMeta(),
            indices=ids, out=None if output is None else output[:, None],
        )

    assert out.is_contiguous()
    if out.data_ptr() % 16:
        with pytest.raises(RuntimeError, match="out must be 16-byte aligned"):
            call(out)
        torch.cuda.synchronize()
    else:
        actual = call(out)[0]
        assert actual.data_ptr() == out.data_ptr()
        torch.testing.assert_close(actual, call(None)[0], atol=0, rtol=0)
    assert (backing[:offset] == -123).all() and (backing[-8:] == -123).all()


def test_merge_rejects_unaligned_partial_output():
    import flash_mla_cuda

    backing = torch.empty(4 * 2 * 8 * 512 + 1, device="cuda", dtype=torch.bfloat16)
    part = backing[1:].view(4, 2, 8, 512)
    lse = torch.zeros(4, 2, 8, device="cuda")
    with pytest.raises(RuntimeError, match="part must be 16-byte aligned"):
        flash_mla_cuda.native_fp8_merge(part, lse, None, None)
    torch.cuda.synchronize()


# Exercise each automatic pipeline boundary and trusted-index contract without
# repeating their full Cartesian product. Grid sizes follow the device CU count.
@pytest.mark.parametrize(
    "grid,width,trusted",
    [
        ("one", 128, False),
        ("half", 512, False),
        ("half_next", 512, False),
        ("cu", 512, False),
        ("cu_next", 512, False),
        ("large", 2176, False),
        ("one", 128, True),
        ("half_next", 512, True),
        ("cu_next", 512, True),
        ("large", 2176, True),
    ],
)
@pytest.mark.parametrize("dim", [512, 576])
def test_pipeline_boundaries_graph_updates(grid, width, trusted, dim):
    cu = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    queries = {
        "one": 1,
        "half": max(1, cu // 2),
        "half_next": cu // 2 + 1,
        "cu": cu,
        "cu_next": cu + 1,
        "large": 2 * cu,
    }[grid]
    torch.manual_seed(4813)
    q = (torch.randn(queries, 64, dim, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    kv = (torch.randn(257, 1, dim, device="cuda") * 0.3).to(q.dtype)
    ids = torch.randint(257, (queries, 1, width), device="cuda", dtype=torch.int32)
    lengths = (
        torch.arange(queries, device="cuda", dtype=torch.int32)
        .mul(71)
        .remainder(width + 1)
    )
    lengths[-1] = width
    if trusted:
        ids.masked_fill_(
            torch.arange(width, device="cuda") >= lengths[:, None, None], -1
        )
    else:
        ids[:, :, 31::47] = torch.iinfo(torch.int32).min
        if queries > 1:
            ids[0] = -1
    sink = torch.linspace(-3, 3, 64, device="cuda")
    sink[0], sink[-1] = -math.inf, math.inf

    def call():
        return native(
            "prefill",
            q,
            kv,
            ids,
            q_scale=0.25,
            kv_scale=2.0,
            topk_length=lengths,
            attn_sink=sink,
            trusted_full_topk=trusted,
        )

    check(call(), reference(q, kv, ids, 0.25, 2.0, lengths, sink))
    graph, result = capture(call)
    # Refresh all captured operands; no Q/indices or pipeline state may be stale.
    q.copy_((torch.randn_like(q.float()) * 0.8).to(q.dtype))
    kv.copy_((torch.randn_like(kv.float()) * 0.8).to(kv.dtype))
    ids.copy_(torch.randint(257, ids.shape, device="cuda", dtype=ids.dtype))
    lengths.fill_(width)
    if not trusted:
        ids[:, :, 63::71] = 257
    graph.replay()
    check(result, reference(q, kv, ids, 0.25, 2.0, lengths, sink))
    if grid == "large" and not trusted:
        plain = native("prefill", q, kv, ids)
        noaux = native(
            "prefill", q.view(torch.uint8), kv.view(torch.uint8), ids, compute_aux=False
        )
        assert noaux[1].numel() == noaux[2].numel() == 0
        torch.testing.assert_close(noaux[0], plain[0], atol=0, rtol=0)
        graph, noaux = capture(lambda: native("prefill", q, kv, ids, compute_aux=False))
        q.copy_((torch.randn_like(q.float()) * 0.3).to(q.dtype))
        graph.replay()
        plain = native("prefill", q, kv, ids)
        torch.testing.assert_close(noaux[0], plain[0], atol=0, rtol=0)


@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("trusted", [False, True])
def test_running_max_rescaling(dim, trusted):
    width = 2176
    queries = 64
    q = torch.ones(queries, 64, dim, device="cuda").to(torch.float8_e4m3fn)
    # Later KV tiles repeatedly raise the running maximum. Some rows start
    # empty, contain a single strongly negative score, or stay entirely empty.
    kv = torch.linspace(-1, 1, width, device="cuda").reshape(width, 1, 1)
    kv = kv.expand(width, 1, dim).contiguous().to(torch.float8_e4m3fn)
    ids = (
        torch.arange(width, device="cuda", dtype=torch.int32)
        .repeat(queries, 1)
        .unsqueeze(1)
    )
    ids[0] = -1
    ids[1] = -1
    ids[1, 0, 0] = 0
    ids[2, :, 32:64] = -1
    lengths = torch.full((queries,), width, device="cuda", dtype=torch.int32)
    lengths[3] = width - 3
    if trusted:
        # The certificate applies to every index before topk_length.
        lengths[0] = 0
        lengths[1] = 1
        ids[2] = torch.arange(width, device="cuda", dtype=torch.int32)
    fn = lambda: native(
        "prefill", q, kv, ids, topk_length=lengths, trusted_full_topk=trusted
    )
    ans = fn()
    check(ans, reference(q, kv, ids, lengths=lengths))
    graph, ans = capture(fn)
    graph.replay()
    torch.cuda.synchronize()
    check(ans, reference(q, kv, ids, lengths=lengths))
    q.copy_(-q.float())  # Reverse the maximum progression with the same graph.
    graph.replay()
    torch.cuda.synchronize()
    check(ans, reference(q, kv, ids, lengths=lengths))


@pytest.mark.parametrize("queries", [64, 128])
@pytest.mark.parametrize("dim", [512, 576])
def test_decode_auto_high_query_contract(queries, dim):
    torch.manual_seed(3917)
    q = (torch.randn(queries, 64, dim, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    kv = (torch.randn(193, 1, dim, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    ids = torch.randint(0, 193, (queries, 1, 128), device="cuda", dtype=torch.int32)
    ids[0] = -1
    ids[2, :, 31::47] = torch.iinfo(torch.int32).min
    check(native("decode", q, kv, ids), reference(q, kv, ids))


@pytest.mark.parametrize("static_scale", [None, 1.0 / 16])
@pytest.mark.parametrize("storage_offset", [0, 1, 2])
def test_fused_output_rope_token_offset(static_scale, storage_offset):
    q, kv, ids = inputs(heads=64, dim=512)
    b, h, _ = q.shape
    plain = native("prefill", q, kv, ids, compute_aux=False)[0]
    angle = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    freqs = torch.stack([angle.cos(), angle.sin()], dim=-1).flatten(1)
    positions = torch.tensor([3, 0, 1, 2, 1, 0, 3, 2], device="cuda", dtype=torch.int64)
    offset = 2
    pos = positions[offset : offset + b]
    x = plain.float()
    a = x[:, :, -64:].reshape(b, h, 32, 2)
    f = freqs[pos].reshape(b, 1, 32, 2)
    rotated = torch.stack(
        [
            a[..., 0] * f[..., 0] + a[..., 1] * f[..., 1],
            a[..., 1] * f[..., 0] - a[..., 0] * f[..., 1],
        ],
        dim=-1,
    ).flatten(2)
    x[:, :, -64:] = rotated.to(torch.bfloat16).float()
    grouped = x.reshape(b, 8, 4096).transpose(0, 1).contiguous()
    maxima = grouped.abs().amax(-1)
    scale = (
        torch.where(maxima > 0, maxima / 448.0, 1.0)
        if static_scale is None
        else torch.full_like(maxima, static_scale)
    )
    expected = (
        (grouped / scale[:, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    )
    backing = torch.zeros(8 * 8 * 4096 + storage_offset, device="cuda", dtype=torch.float8_e4m3fn)
    out8 = backing[storage_offset:].view(8, 8, 4096)
    scales = torch.full((8, 8), -1.0, device="cuda", dtype=torch.float32)
    if storage_offset % 2:
        with pytest.raises(RuntimeError, match="out_fp8 must be 2-byte aligned"):
            native(
                "prefill", q, kv, ids, compute_aux=False, out_rope_freqs_cis=freqs,
                out_rope_positions=positions, out_fp8=out8, out_scale=scales,
                out_quant_heads_per_group=8, out_static_scale=static_scale,
            )
        torch.cuda.synchronize()
        return
    def call(out=None):
        return native(
            "prefill", q, kv, ids, out=out, compute_aux=False,
            out_rope_freqs_cis=freqs, out_rope_positions=positions,
            out_fp8=out8, out_scale=scales, out_quant_heads_per_group=8,
            out_token_offset=offset, out_static_scale=static_scale,
        )

    unused_out = torch.full_like(plain, 123)
    with pytest.raises(RuntimeError, match="fused output requires out=None"):
        call(out=unused_out)
    assert (unused_out == 123).all()
    assert (out8.view(torch.uint8) == 0).all() and (scales == -1).all()

    result = call()
    assert all(t.shape == (0,) and t.device == q.device for t in result)
    assert [t.dtype for t in result] == [torch.bfloat16, torch.float32, torch.float32]
    graph, result = capture(call)
    graph.replay()
    assert all(t.shape == (0,) for t in result)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        scales[:, offset : offset + b], scale, atol=2e-6, rtol=2e-5
    )
    actual = out8[:, offset : offset + b].float() * scale[:, :, None]
    want = expected * scale[:, :, None]
    assert (
        torch.linalg.vector_norm(actual - want) / torch.linalg.vector_norm(want)
    ).item() < 0.01
    assert (out8[:, :offset].view(torch.uint8) == 0).all() and (
        scales[:, :offset] == -1
    ).all()
    assert (out8[:, offset + b :].view(torch.uint8) == 0).all() and (
        scales[:, offset + b :] == -1
    ).all()


@pytest.mark.parametrize("queries", range(2, 10))
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("shared_history", [False, True])
@pytest.mark.parametrize(
    "query_dtype,flat_queries,dim",
    [(torch.float8_e4m3fn, False, 512), (torch.float8_e4m3fn, True, 512),
     (torch.bfloat16, False, 512), (torch.float8_e4m3fn, False, 576),
     (torch.float8_e4m3fn, True, 576), (torch.uint8, False, 576), (torch.uint8, True, 576)],
    ids=["q8-grouped", "q8-flat", "bf16-input", "d576-grouped", "d576-flat", "d576-raw-grouped", "d576-raw-flat"],
)
def test_verify_union_and_independent_fallback_graph(queries, heads, shared_history, query_dtype, flat_queries, dim):
    torch.manual_seed(14263)
    batch, width = 64, 2048
    value_dtype = torch.float8_e4m3fn if query_dtype == torch.uint8 else query_dtype
    q = (torch.randn(batch, queries, heads, dim, device="cuda") * 0.3).to(
        value_dtype
    ).view(query_dtype)
    kv = (torch.randn(64, 16, 1, dim, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    indices = torch.randint(
        1024, (batch, queries, width), device="cuda", dtype=torch.int32
    )
    if shared_history:
        indices.copy_(indices[:, :1].expand_as(indices).clone())
    indices[:, :, 17::37] = -1
    indices[0, 0] = -1
    sink = torch.zeros(heads, device="cuda")
    sink[-1] = torch.inf
    scale = 256**-0.5
    out = torch.empty(batch, queries, heads, 512, device="cuda", dtype=torch.bfloat16)

    def call():
        if query_dtype == torch.bfloat16 or not flat_queries:
            import flash_mla_cuda

            output, lse = flash_mla_cuda.native_fp8_verify_fwd(
                q.flatten(0, 1), kv.flatten(0, 1),
                indices.reshape(batch * queries, 1, width), queries, scale,
                sink, None, out.flatten(0, 1),
            )
            return output.view_as(out), flash_mla_cuda.native_fp8_lse_to_natural(lse, batch, queries)
        output, lse = flash_mla.flash_mla_with_kvcache(
            q.reshape(batch * queries, 1, heads, dim),
            kv,
            None,
            None,
            512,
            flash_mla.FlashMLASchedMeta(),
            softmax_scale=scale,
            indices=indices.reshape(batch * queries, 1, width),
            attn_sink=sink,
            out=out.reshape(batch * queries, 1, heads, 512),
            num_queries_per_request=queries,
        )
        lse = lse.reshape(batch, queries, heads).transpose(1, 2)
        return output.view_as(out), lse

    result = call()
    assert result[0].data_ptr() == out.data_ptr()
    check(
        result,
        reference(q.view(value_dtype).flatten(0, 1), kv, indices, sink=sink, scale=scale),
        natural_lse=True,
    )
    graph, result = capture(call)
    q.copy_((torch.randn_like(q.float()) * 0.8).to(value_dtype).view(query_dtype))
    kv.copy_((torch.randn_like(kv.float()) * 0.8).to(kv.dtype))
    # Force disagreement after the first classifier slice, including when
    # capture used shared lists. The graph must re-evaluate this contract.
    indices[:, 1, width // 2] = (indices[:, 0, width // 2] + 1) % 1024
    graph.replay()
    torch.cuda.synchronize()
    check(
        result,
        reference(q.view(value_dtype).flatten(0, 1), kv, indices, sink=sink, scale=scale),
        natural_lse=True,
    )
    indices.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert (result[0] == 0).all() and torch.isposinf(result[1]).all()


@pytest.mark.parametrize("heads", [8, 64])
def test_dspark_extra_combines_normalization(heads):
    torch.manual_seed(14863)
    q = (torch.randn(4, 8, heads, 512, device="cuda") * 0.8).to(torch.float8_e4m3fn)
    kv = (torch.randn(64, 16, 1, 512, device="cuda") * 0.8).to(q.dtype)
    extra = (torch.randn(32, 16, 1, 512, device="cuda") * 0.8).to(q.dtype)
    ids = torch.randint(1024, (4, 8, 128), device="cuda", dtype=torch.int32)
    extra_ids = torch.randint(512, (4, 8, 512), device="cuda", dtype=torch.int32)
    ids[0, 0] = -1
    extra_ids[0, 0] = -1
    extra_ids[:, :, 23::43] = -1
    sink = torch.linspace(-2, 2, heads, device="cuda")
    scale = 512**-0.5
    result = flash_mla.flash_mla_with_kvcache(
        q,
        kv,
        None,
        None,
        512,
        flash_mla.FlashMLASchedMeta(),
        softmax_scale=scale,
        indices=ids,
        attn_sink=sink,
        extra_k_cache=extra,
        extra_indices_in_kvcache=extra_ids,
    )
    keys = torch.cat((kv.reshape(-1, 1, 512), extra.reshape(-1, 1, 512)))
    shifted = torch.where(extra_ids >= 0, extra_ids + kv.numel() // 512, -1)
    all_ids = torch.cat((ids, shifted), dim=-1)
    check(
        result,
        reference(q.flatten(0, 1), keys, all_ids, sink=sink, scale=scale),
        natural_lse=True,
    )


@pytest.mark.parametrize("dim", [512, 576])
def test_packed_cache_preserves_legacy_dispatch(dim):
    # Packed caches expose E4M3 too, but include scales/rope beyond Q's dimension.
    # They must retain their original path when native paged dispatch is added.
    import lib
    import ref

    dtype = torch.get_default_dtype()
    try:
        with torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            extra = dim == 512
            params = lib.RawTestParamForDecode(
                b=4,
                h_q=64,
                s_q=1,
                h_kv=1,
                s_kv=4096,
                is_varlen=True,
                topk=512,
                is_fp8=True,
                enable_attn_sink=True,
                have_topk_length=extra,
                extra_s_k=1027 if extra else None,
                extra_topk=192 if extra else None,
                have_extra_topk_length=extra,
                d_qk=dim,
                num_runs=0,
                seed=1463,
            ).to_test_param()
            tensors = lib.generate_testcase_for_decode(params)
            metadata, splits = flash_mla.get_mla_metadata()
            actual = lib.run_flash_mla_decode(params, tensors, metadata, splits)
            expected = ref.ref_sparse_attn_decode(params, tensors)
            assert lib.check_is_allclose(
                "output", actual[0], expected[0], 8e-4, 2.01 / 128, 5e-6
            )
            assert lib.check_is_allclose(
                "lse", actual[1], expected[1], 1e-5, 8.01 / 65536, 5e-6
            )
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize(
    "tokens", [1, 2, 4, 8, 16, 27, 28, 32, 38, 39, 40, 48, 64, 96, 128, 192, 256, 512]
)
def test_decode_grid_and_graph_replay(tokens):
    q, kv, ids = inputs()
    copies = (tokens + len(q) - 1) // len(q)
    q = q.repeat(copies, 1, 1)[:tokens]
    ids = ids.repeat(copies, 1, 16)[:tokens]
    # Exercise automatic split/head tiling and full-head launches across device waves.
    def call():
        return native("decode", q, kv, ids)

    check(call(), reference(q, kv, ids))
    graph, result = capture(call)
    # Replay across nonempty/empty rows and changed Q values. This also covers
    # the transition from split-head to full-head segment tiles.
    q.copy_(q.float().mul_(0.75).to(q.dtype))
    ids[::3] = -1
    graph.replay()
    check(result, reference(q, kv, ids))


@pytest.mark.parametrize(
    "heads,extra_width,dim,tokens,width",
    [(h, e, 512, 256, 128) for h in (8, 16, 64, 128) for e in (0, 512)]
    + [(8, 0, 576, 48, 512), (8, 0, 576, 256, 512), (16, 0, 576, 512, 512)]
    + [(8, 0, 512, 32, 2048), (16, 0, 512, 32, 2048),
       (64, 0, 512, 32, 2048), (128, 0, 512, 16, 2048)],
)
def test_native_paged_lengths_and_graph_replay(heads, extra_width, dim, tokens, width):
    """Flat verify preserves per-query masks, two pools and changing graph inputs."""
    torch.manual_seed(61937)
    rows = 2048
    q = (torch.randn(tokens, 1, heads, dim, device="cuda") * .8).to(torch.float8_e4m3fn)
    kv = (torch.randn(rows, 1, 1, dim, device="cuda") * .8).to(q.dtype)
    ids = torch.randint(rows, (tokens, 1, width), device="cuda", dtype=torch.int32)
    ids[:, :, 7::19] = -1
    ids[::5, :, 11] = torch.iinfo(torch.int32).max
    lengths = torch.tensor([-1, 0, 1, 63, 64, 65, width - 1, width], device="cuda", dtype=torch.int32).repeat(tokens // 8)
    extra = extra_ids = extra_lengths = None
    if extra_width:
        extra = (torch.randn(rows, 1, 1, dim, device="cuda") * .8).to(q.dtype)
        extra_ids = torch.randint(rows, (tokens, 1, extra_width), device="cuda", dtype=torch.int32)
        extra_ids[:, :, 13::31] = torch.iinfo(torch.int32).min
        extra_lengths = torch.tensor([0, 0, 1, 255, 256, 257, 511, 512], device="cuda", dtype=torch.int32).repeat(tokens // 8)
    sink = torch.linspace(-2, 2, heads, device="cuda")
    sink[0], sink[-1] = -torch.inf, torch.inf
    out = torch.empty(tokens, 1, heads, 512, device="cuda", dtype=torch.bfloat16)

    def call():
        return flash_mla.flash_mla_with_kvcache(
            q, kv, None, None, 512, flash_mla.FlashMLASchedMeta(), causal=False,
            indices=ids, attn_sink=sink, topk_length=lengths, out=out,
            extra_k_cache=extra, extra_indices_in_kvcache=extra_ids,
            extra_topk_length=extra_lengths,
        )

    def expected():
        masked = ids.masked_fill(torch.arange(width, device="cuda") >= lengths[:, None, None], -1)
        keys = kv
        if extra_width:
            other = extra_ids.masked_fill(
                torch.arange(extra_width, device="cuda") >= extra_lengths[:, None, None], -1
            )
            other = torch.where((other >= 0) & (other < rows), other + rows, -1)
            masked = torch.cat((masked, other), dim=-1)
            keys = torch.cat((kv, extra))
        return reference(q.flatten(0, 1), keys, masked, sink=sink)

    result = call()
    assert result[0].data_ptr() == out.data_ptr()
    check(result, expected(), natural_lse=True)
    graph, result = capture(call)
    q.copy_((torch.randn_like(q.float()) * .3).to(q.dtype))
    lengths.copy_(lengths.roll(3))
    if extra_width:
        extra_lengths.copy_(extra_lengths.roll(5))
        extra_ids[:, :, 300] = (extra_ids[:, :, 300] + 17) % rows
    graph.replay()
    check(result, expected(), natural_lse=True)
    ids.fill_(-1)
    if extra_width:
        extra_ids.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert (result[0] == 0).all() and torch.isposinf(result[1]).all()


@pytest.mark.parametrize("queries", range(2, 10))
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("extra_width", [0, 512])
@pytest.mark.parametrize("flat_queries", [False, True])
def test_window_union_masks_and_fallback_replay(queries, heads, extra_width, flat_queries):
    import flash_mla_cuda

    torch.manual_seed(21789)
    batch, width = (64 if flat_queries else 3), 128
    rows = max(1024, batch * 256)
    q = (torch.randn(batch * queries, heads, 512, device="cuda") * .3).to(torch.float8_e4m3fn)
    kv = (torch.randn(rows, 1, 512, device="cuda") * .3).to(q.dtype)
    physical = torch.randperm(rows, device="cuda")
    positions = (torch.arange(batch, device="cuda")[:, None, None] * 256
                 + torch.arange(queries, device="cuda")[None, :, None]
                 + torch.arange(width, device="cuda")[None, None, :])
    ids = physical[positions].int().reshape(batch * queries, 1, width)
    ids[:, :, 17::29] = -1
    ids[0] = -1
    lengths = torch.full((batch * queries,), width, device="cuda", dtype=torch.int32)
    lengths[1] = 1
    lengths[-1] = 97
    extra = extra_ids = extra_lengths = None
    if extra_width:
        extra = (torch.randn(rows, 1, 512, device="cuda") * .3).to(q.dtype)
        extra_ids = torch.randint(rows, (batch, 1, extra_width), device="cuda", dtype=torch.int32)
        extra_ids = extra_ids.expand(-1, queries, -1).contiguous().reshape(batch * queries, 1, extra_width)
        extra_ids[:, :, 23::43] = -1
        extra_ids[0] = -1
        extra_lengths = torch.full_like(lengths, extra_width)
        extra_lengths[2] = 0
        extra_lengths[-1] = 319
    sink = torch.linspace(-2, 2, heads, device="cuda")
    sink[-1] = torch.inf
    out = torch.empty(batch * queries, heads, 512, device="cuda", dtype=torch.bfloat16)

    def call():
        if flat_queries:
            return flash_mla.flash_mla_with_kvcache(
                q[:, None], kv.view(-1, 64, 1, 512), None, None, 512,
                flash_mla.FlashMLASchedMeta(), indices=ids, attn_sink=sink,
                topk_length=lengths, out=out[:, None], num_queries_per_request=queries,
                extra_k_cache=None if extra is None else extra.view(-1, 64, 1, 512),
                extra_indices_in_kvcache=extra_ids, extra_topk_length=extra_lengths,
            )
        return flash_mla_cuda.native_fp8_verify_fwd(
            q, kv, ids, queries, 512**-.5, sink, lengths, out, True,
            extra, extra_ids, extra_lengths,
        )

    def expected():
        masked = ids.masked_fill(torch.arange(width, device="cuda") >= lengths[:, None, None], -1)
        keys = kv
        if extra_width:
            other = extra_ids.masked_fill(
                torch.arange(extra_width, device="cuda") >= extra_lengths[:, None, None], -1
            )
            other = torch.where(other >= 0, other + rows, -1)
            masked = torch.cat((masked, other), dim=-1)
            keys = torch.cat((kv, extra))
        return reference(q, keys, masked, sink=sink)

    result = call()
    assert result[0].data_ptr() == out.data_ptr()
    check(result, expected(), natural_lse=flat_queries)
    graph, result = capture(call)
    saved = ids.clone()
    # Switch a previously compatible request to an independent list. An extra
    # mismatch beyond the first classifier slice also exercises its late exit.
    ids[queries + 1, 0, width // 2] = (ids[queries, 0, width // 2 + 1] + 1) % rows
    if extra_width:
        extra_ids[2 * queries + 1, 0, 300] = (extra_ids[2 * queries, 0, 300] + 1) % rows
    q.copy_((torch.randn_like(q.float()) * .8).to(q.dtype))
    lengths[3] = 0
    graph.replay()
    torch.cuda.synchronize()
    check(result, expected(), natural_lse=flat_queries)
    ids.copy_(saved)
    if extra_width:
        grouped = extra_ids.view(batch, queries, extra_width)
        grouped.copy_(grouped[:, :1].expand_as(grouped).clone())
        extra_lengths.fill_(extra_width)
    lengths.fill_(width)
    graph.replay()
    torch.cuda.synchronize()
    check(result, expected(), natural_lse=flat_queries)
    ids.fill_(-1)
    if extra_width:
        extra_ids.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    is_empty_lse = torch.isposinf if flat_queries else torch.isneginf
    assert (result[0] == 0).all() and is_empty_lse(result[1]).all()


@pytest.mark.parametrize("batch,queries", [(0, 1), (4, 0)])
@pytest.mark.parametrize(
    "dim,width,extra_width",
    [(512, 128, 0), (512, 128, 512), (576, 128, 0), (576, 128, 512), (512, 2048, 512)],
)
def test_native_paged_empty_query_axes(batch, queries, dim, width, extra_width):
    # D576 and D512 with combined width > 2048 use the two-decode/merge fallback.
    heads = 8
    q = torch.empty(batch, queries, heads, dim, device="cuda", dtype=torch.float8_e4m3fn)
    kv = torch.empty(128, 1, 1, dim, device="cuda", dtype=q.dtype)
    indices = torch.empty(batch, queries, width, device="cuda", dtype=torch.int32)
    extra_indices = (torch.empty(batch, queries, extra_width, device="cuda", dtype=torch.int32)
                     if extra_width else None)
    output, lse = flash_mla.flash_mla_with_kvcache(
        q, kv, None, None, 512, flash_mla.FlashMLASchedMeta(), indices=indices,
        extra_k_cache=kv if extra_width else None, extra_indices_in_kvcache=extra_indices,
    )
    assert output.shape == (batch, queries, heads, 512)
    assert lse.shape == (batch, heads, queries)


@pytest.mark.parametrize("queries", [131071, 131072])
def test_prefill_two_head_groups_large_grid(queries):
    # H128/topk128 uses paired-query ordering with grid.y=65536, which the
    # PPU supports. Exercise both the odd-Q bounds check and the even-Q path.
    heads, dim, width = 128, 512, 128
    q = torch.zeros(queries, heads, dim, device="cuda", dtype=torch.float8_e4m3fn)
    kv = torch.zeros(1, 1, dim, device="cuda", dtype=q.dtype)
    ids = torch.full((queries, 1, width), -1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(queries, device="cuda", dtype=torch.int32)
    out = torch.full((queries, heads, 512), torch.nan, device="cuda", dtype=torch.bfloat16)
    result = native("prefill", q, kv, ids, topk_length=lengths, out=out)
    assert result[0].data_ptr() == out.data_ptr()
    assert (result[0] == 0).all() and torch.isneginf(result[-1]).all()


@pytest.mark.parametrize("width,extra_width", [(128, 0), (512, 0), (128, 512)])
def test_native_paged_grid_exceeds_y_limit(width, extra_width):
    # Full-head launchers must support more queries than grid.y can represent.
    tokens, heads, dim = 65537, 64, 512
    q = torch.empty(tokens, 1, heads, dim, device="cuda", dtype=torch.float8_e4m3fn)
    q.view(torch.uint8).zero_()
    kv = torch.empty(64, 1, 1, dim, device="cuda", dtype=q.dtype)
    kv.view(torch.uint8).zero_()
    indices = torch.full((tokens, 1, width), -1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    extra_ids = (torch.full((tokens, 1, extra_width), -1, device="cuda", dtype=torch.int32)
                 if extra_width else None)
    out = torch.full((tokens, 1, heads, dim), torch.nan, device="cuda", dtype=torch.bfloat16)
    output, lse = flash_mla.flash_mla_with_kvcache(
        q, kv, None, None, dim, flash_mla.FlashMLASchedMeta(), indices=indices,
        topk_length=lengths, out=out, extra_k_cache=kv if extra_width else None,
        extra_indices_in_kvcache=extra_ids, extra_topk_length=lengths if extra_width else None,
    )
    assert output.data_ptr() == out.data_ptr()
    assert (output == 0).all() and torch.isposinf(lse).all()


@pytest.mark.parametrize("batch,queries,grouping", [(5, 1, 2), (2, 3, 2), (2, 3, 0)])
def test_native_paged_rejects_invalid_request_grouping(batch, queries, grouping):
    q = torch.empty(batch, queries, 8, 512, device="cuda", dtype=torch.float8_e4m3fn)
    kv = torch.empty(1, 64, 1, 512, device="cuda", dtype=q.dtype)
    ids = torch.zeros(batch, queries, 128, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError):
        flash_mla.flash_mla_with_kvcache(
            q, kv, None, None, 512, flash_mla.FlashMLASchedMeta(),
            indices=ids, num_queries_per_request=grouping,
        )


@pytest.mark.parametrize("queries", [6, 8, 9])
@pytest.mark.parametrize("pool_l2_ratio", [0.25, 4, 32])
@pytest.mark.parametrize("batch", [64, 128])
@pytest.mark.parametrize("dim", [512, 576])
def test_verify_graph_across_kv_pool_sizes(queries, pool_l2_ratio, batch, dim):
    """KV allocation size must not change masks or fallback behavior on replay."""
    torch.manual_seed(17319)
    heads, width = 16, 2048
    cache_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    rows = max(width, int(cache_bytes * pool_l2_ratio) // dim)
    rows = (rows + 15) // 16 * 16
    q = (torch.randn(batch, queries, heads, dim, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    kv = (torch.randn(rows // 16, 16, 1, dim, device="cuda") * 0.3).to(q.dtype)
    indices = torch.randint(rows, (batch, 1, width), device="cuda", dtype=torch.int32)
    indices = indices.expand(batch, queries, width).clone()
    indices[:, :, 17::37] = -1
    metadata, _ = flash_mla.get_mla_metadata()
    scale = 256**-0.5

    def call():
        return flash_mla.flash_mla_with_kvcache(
            q, kv, None, None, 512, metadata,
            softmax_scale=scale, indices=indices,
        )

    graph, result = capture(call)
    for history in ("shared", "independent", "matching_prefix"):
        if history == "independent":
            indices[:, 1].copy_(
                torch.randint(rows, indices[:, 1].shape, device="cuda", dtype=indices.dtype)
            )
        elif history == "matching_prefix":
            # A reuse hint may match while later indices remain independent.
            # It must affect scheduling only, never the attention semantics.
            indices[:, 1, 0].copy_(indices[:, 0, 0])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(result[0]).all() and torch.isfinite(result[1]).all()
        # Keep the independent FP64 oracle bounded while testing the full grid.
        check(
            (result[0][:2], result[1][:2]),
            reference(q[:2].flatten(0, 1), kv, indices[:2], scale=scale),
            natural_lse=True,
        )
    indices.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert (result[0] == 0).all() and torch.isposinf(result[1]).all()


def test_gather_reuse_hint_graph_replay():
    """Gather hints may change on replay without changing attention semantics."""
    torch.manual_seed(19417)
    batch, queries, heads, dim, width = 64, 3, 64, 576, 1024
    # Exceed the streaming-pool threshold used by guarded gather selection.
    rows = batch * queries * (24 * width + 64)
    q = (torch.randn(batch, queries, heads, dim, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    kv = (torch.randn(rows // 64, 64, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.3).to(q.dtype)
    indices = torch.randint(rows, (batch, 1, width), device="cuda", dtype=torch.int32)
    indices = indices.expand(batch, queries, width).clone()
    lengths = torch.full((batch, queries), width, device="cuda", dtype=torch.int32)
    metadata, _ = flash_mla.get_mla_metadata()
    scale = 256**-0.5

    def call():
        return flash_mla.flash_mla_with_kvcache(
            q, kv, None, None, 512, metadata, softmax_scale=scale,
            indices=indices, topk_length=lengths,
        )

    graph, result = capture(call)
    for history in ("shared", "independent", "mixed", "matching_prefix", "empty_first"):
        if history == "independent":
            indices[:, 1].copy_(torch.randint(rows, indices[:, 1].shape, device="cuda", dtype=indices.dtype))
        elif history == "mixed":
            indices[::2, 1].copy_(indices[::2, 0])
        elif history == "matching_prefix":
            indices[:, 1, 0].copy_(indices[:, 0, 0])
        elif history == "empty_first":
            lengths[0, 0] = 0
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(result[0]).all()
        check(
            (result[0][:2], result[1][:2]),
            reference(q[:2].flatten(0, 1), kv, indices[:2],
                      lengths=lengths[:2].flatten(), scale=scale),
            natural_lse=True,
        )
    indices.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert (result[0] == 0).all() and torch.isposinf(result[1]).all()



@pytest.mark.parametrize("heads,width", [(64, 1024), (128, 2048), (64, 2176)])
@pytest.mark.parametrize("packed", [False, True])
def test_flat_split_statistics_and_graph_replay(heads, width, packed):
    """Split maxima, optional statistics and mutable masks retain the flat API contract."""
    q = torch.ones(4, heads, 512, device="cuda").to(torch.float8_e4m3fn)
    kv = torch.linspace(-1, 1, 1025, device="cuda")[:, None, None]
    kv = kv.expand(-1, 1, 512).contiguous().to(q.dtype)
    ids = (torch.arange(width, device="cuda", dtype=torch.int32) % len(kv))
    ids = ids.repeat(4, 1).unsqueeze(1)
    ids[1, :, 31::47] = torch.iinfo(torch.int32).max
    lengths = torch.tensor([0, width - 1, width // 3, width], device="cuda", dtype=torch.int32)
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[0], sink[-1] = -math.inf, math.inf
    out = torch.empty(4, heads, 512, device="cuda", dtype=torch.bfloat16)

    def call(aux=True):
        return native(
            "prefill", q.view(torch.uint8) if packed else q,
            kv.view(torch.uint8) if packed else kv, ids,
            sm_scale=1 / 512, q_scale=0.5, kv_scale=2.0,
            topk_length=lengths, attn_sink=sink, out=out if aux else None,
            compute_aux=aux,
        )

    def validate(result):
        check(result, reference(q, kv, ids, 0.5, 2.0, lengths, sink, 1 / 512))
        assert result[0].data_ptr() == out.data_ptr()
        assert result[1].shape == (4, heads) and result[1].dtype == torch.float32
        for row in range(4):
            index = ids[row, 0, :lengths[row].item()]
            index = index[(index >= 0) & (index < len(kv))].long()
            if len(index):
                scores = q[row].double() @ kv[index, 0].double().T / 512
                maximum = torch.ceil(scores.max(-1).values / math.log(2))
                torch.testing.assert_close(result[1][row].double(), maximum, atol=1e-5, rtol=0)
            else:
                assert torch.isfinite(result[1][row]).all()
                assert (result[1][row] < -1e20).all()
        noaux = call(False)
        assert noaux[1].numel() == noaux[2].numel() == 0
        torch.testing.assert_close(noaux[0], result[0], atol=0, rtol=0)

    validate(call())
    graph, result = capture(call)
    q.copy_(-q.float())
    lengths[0] = width
    ids[3].fill_(-1)
    graph.replay()
    validate(result)
    q.zero_()
    graph.replay()
    validate(result)
    # Equal scores make base-2 LSE exact and independently check its units.
    valid = (ids[:, 0] >= 0) & (ids[:, 0] < len(kv))
    valid &= torch.arange(width, device="cuda") < lengths[:, None]
    counts = valid.sum(-1)
    active = counts > 0
    expected = counts[active].float().log2()[:, None].expand(-1, heads)
    torch.testing.assert_close(result[2][active], expected, atol=1e-4, rtol=0)
