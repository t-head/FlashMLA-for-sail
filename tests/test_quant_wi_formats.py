"""SM89 H128 quantized WI regressions for V4.0 FP8 and V4.1 FP8/FP4.

Run after building the extension:
    python tests/test_quant_wi_formats.py --device 0

--list-cases imports no PyTorch and launches no GPU work. Execution requires
SM89, BF16 Q with 128 heads and D512, and the quantized-prefill API. A missing
V4.0 implementation, missing CUDA trace, or fallback route is a failure.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path


FORMATS = ("fp40", "fp8", "fp4")
SCALES = (("negative", -0.5), ("zero", 0.0), ("negative-zero", -0.0))


def list_cases(formats, suites):
    for fmt in formats:
        if "scales" in suites:
            for name, _ in SCALES:
                yield f"{fmt}/scale-{name}"
        if "graph" in suites:
            for name, _ in SCALES[:2]:
                yield f"{fmt}/graph-{name}"
    if "fp40" in formats and "layout" in suites:
        yield "fp40/byte-layout-rope-and-padding"


def traced_wi_call(fn, fmt, torch):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        answer = fn()
        torch.cuda.synchronize()
    names = sorted({event.name for event in prof.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA})
    mains = [name for name in names if "quant_prefill_wi_" in name]
    fallback = [name for name in names if "flash_sparse_prefill_fwd" in name
                or "flash_sparse_decode_fwd" in name]
    assert len(mains) == 1 and not fallback, (mains, fallback, names)
    traits = r"\b(?:QuantTraits|QuantNarrowAddressTraits)\s*<\s*{}\s*>"
    if fmt == "fp40":
        role_ok = bool(re.search(r"\bV40QuantNarrowTraits\b", mains[0]))
    elif fmt == "fp4":
        role_ok = bool(re.search(traits.format("true"), mains[0]))
    else:
        role_ok = bool(re.search(traits.format("false"), mains[0])) or bool(
            re.search(r"\bquant_prefill_wi_gpu_checked_fp8_kernel\b", mains[0]))
    assert role_ok, (fmt, mains)
    print("WI_ROUTE", fmt, mains[0], flush=True)
    return answer


def check_reference(answers, p, t, torch, kk, ref):
    _, expected_out, expected_max, expected_lse = ref.ref_sparse_attn_fwd(p, t)
    expected_lse[expected_lse == float("-inf")] = float("inf")
    # Keep the existing sparse Prefill thresholds and natural-log LSE units.
    correct = bool(kk.check_is_allclose(
        "out", answers[0].float(), expected_out,
        abs_tol=8e-4, rel_tol=3.01 / 128, cos_diff_tol=7e-6))
    correct &= bool(kk.check_is_allclose(
        "max_logits", answers[1], expected_max,
        abs_tol=1e-6, rel_tol=2.01 / 65536))
    correct &= bool(kk.check_is_allclose(
        "lse", answers[2], expected_lse,
        abs_tol=1e-6, rel_tol=2.01 / 65536))
    assert correct


def make_case(case, fmt, torch, lib, quant, validation):
    if fmt != "fp40":
        return validation.make_case(case, fmt, torch, lib, quant)
    p = lib.TestParam(case.s_q, case.s_kv, case.topk, h_q=128, h_kv=1,
                      d_qk=512, d_v=512, seed=case.seed, num_runs=0,
                      have_attn_sink=case.sink, is_fp8=False)
    original = lib.generate_testcase(p)
    if original.attn_sink is not None:
        original.attn_sink[0] = float("-inf")
        original.attn_sink[1] = float("inf")
    blocks = (case.s_kv + case.page_size - 1) // case.page_size
    dense = torch.zeros((blocks, case.page_size, 1, 512), dtype=torch.bfloat16)
    dense.reshape(-1, 1, 512)[:case.s_kv].copy_(original.kv)
    # These finite BF16 values include signed zeros/subnormals and values that
    # lose bits if the RoPE plane is mistakenly decoded as FP8.
    rope_bits = torch.tensor([0x3F81, 0xBF81, 0x3EAB, 0xBEAB,
                              0x0001, 0x8001, 0x0000, 0x8000],
                             dtype=torch.int32).to(torch.int16).repeat(8)
    dense[..., 448:].copy_(rope_bits.view(torch.bfloat16))
    layout = quant.FP8KVCacheLayout.MODEL1_FP8Sparse
    native = quant.quantize_k_cache(dense, layout)
    packed = native.view(torch.uint8)
    assert packed.shape == (blocks, case.page_size, 1, 584)
    assert packed.stride(1) == 584 and packed.stride(3) == 1
    pages = packed.view(blocks, case.page_size * 584)
    data = pages[:, :case.page_size * 576].view(blocks, case.page_size, 576)
    scale_bytes = pages[:, case.page_size * 576:].view(blocks, case.page_size, 8)
    rope = data[..., 448:].view(torch.bfloat16)
    assert torch.equal(rope.view(torch.int16), dense[:, :, 0, 448:].view(torch.int16))
    # Seven UE8 scales occupy the page tail; the eighth byte is unused padding.
    # Poison it with the NaN code to catch accidental scale-7 use in the kernel.
    scale_bytes[..., 7] = 255
    nope = data[..., :448].view(torch.float8_e4m3fn)
    scales = scale_bytes[..., :7].view(torch.float8_e8m0fnu)
    manual = dense[:, :, 0, :].clone()
    for tile in range(7):
        lo, hi = tile * 64, (tile + 1) * 64
        manual[..., lo:hi] = (nope[..., lo:hi].float()
                              * scales[..., tile, None].float()).to(torch.bfloat16)
    decoded = quant.dequantize_k_cache(native, layout).squeeze(2)
    assert torch.equal(decoded.view(torch.int16), manual.view(torch.int16))
    t = lib.Testcase(p, original.dOut, original.q,
                     manual.reshape(-1, 1, 512)[:case.s_kv], original.indices,
                     original.sm_scale, original.attn_sink, None)
    return p, t, packed


def pad_query(t, torch):
    # q.neg()/q.mul(0) can compact this view. The host must use q_eff strides.
    storage = torch.full((t.q.shape[0], 128, 544), float("nan"), dtype=t.q.dtype)
    view = storage[:, :, :512]
    view.copy_(t.q)
    t.q = view


def run_scales(fmt, rt):
    torch, kk, lib, quant, ref, flash_mla, validation = rt
    case = validation.Case("signed-scale", 129, 513, 640, 64,
                           sink=True, seed=2000)
    p, t, packed = make_case(case, fmt, torch, lib, quant, validation)
    pad_query(t, torch)
    for name, scale in SCALES:
        t.sm_scale = scale
        def call(out=None):
            return flash_mla.flash_mla_sparse_quant_fwd(
                t.q, packed, t.indices, scale, case.s_kv,
                attn_sink=t.attn_sink, topk_length=t.topk_length, out=out)
        answers = traced_wi_call(call, fmt, torch)
        output = torch.full((case.s_q, 128, 512), float("nan"), dtype=torch.bfloat16)
        reused = call(output)
        torch.cuda.synchronize()
        assert reused[0].data_ptr() == output.data_ptr()
        assert all(torch.equal(a, b) for a, b in zip(answers, reused)), name
        check_reference(answers, p, t, torch, kk, ref)
        print("PASS", fmt, "scale-" + name, flush=True)


def run_graph(fmt, rt):
    torch, kk, lib, quant, ref, flash_mla, validation = rt
    case = validation.Case("signed-scale-graph", 3, 153, 256, 128,
                           sink=True, seed=920)
    for name, scale in SCALES[:2]:
        p, t, packed = make_case(case, fmt, torch, lib, quant, validation)
        pad_query(t, torch)
        t.sm_scale = scale
        valid = torch.arange(case.topk, dtype=torch.int32) % case.s_kv
        t.indices.copy_(valid[None, None, :].expand_as(t.indices))
        saved_q, saved_indices = t.q.clone(), t.indices.clone()
        output = torch.empty((case.s_q, 128, 512), dtype=torch.bfloat16)
        def call():
            return flash_mla.flash_mla_sparse_quant_fwd(
                t.q, packed, t.indices, scale, case.s_kv,
                attn_sink=t.attn_sink, topk_length=t.topk_length, out=output)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = call()
        assert captured[0].data_ptr() == output.data_ptr()
        with torch.cuda.stream(stream):
            initial = None
            for state in ("valid", "changed-query-invalid", "restored"):
                t.q.copy_(saved_q.neg() if state == "changed-query-invalid" else saved_q)
                t.indices.copy_(saved_indices)
                if state == "changed-query-invalid":
                    t.indices[-1, 0, -1] = case.s_kv
                    t.indices[0, 0, 32] = -1
                traced_wi_call(graph.replay, fmt, torch)
                check_reference(captured, p, t, torch, kk, ref)
                if state == "valid":
                    initial = tuple(x.clone() for x in captured)
                elif state == "restored":
                    assert all(torch.equal(a, b) for a, b in zip(initial, captured)), state
                before = tuple(x.clone() for x in captured)
                graph.replay()
                stream.synchronize()
                assert all(torch.equal(a, b) for a, b in zip(before, captured)), state
        torch.cuda.current_stream().wait_stream(stream)
        print("PASS", fmt, "graph-" + name, flush=True)


def run_layout(rt):
    torch, kk, lib, quant, ref, flash_mla, validation = rt
    case = validation.Case("byte-layout-rope", 3, 153, 256, 128, seed=921)
    p, t, packed = make_case(case, "fp40", torch, lib, quant, validation)
    t.q.zero_()  # Isolate V, including the last RoPE chunk, from QK error.
    def call():
        return flash_mla.flash_mla_sparse_quant_fwd(
            t.q, packed, t.indices, t.sm_scale, case.s_kv)
    before = traced_wi_call(call, "fp40", torch)
    check_reference(before, p, t, torch, kk, ref)
    pages = packed.view(packed.size(0), case.page_size * 584)
    padding = pages[:, case.page_size * 576:].view(-1, case.page_size, 8)[..., 7]
    padding.zero_()
    after = call()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "padding byte was consumed"
    check_reference(after, p, t, torch, kk, ref)
    print("PASS fp40 byte-layout-rope-and-padding", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--format", choices=(*FORMATS, "all"), default="all")
    parser.add_argument("--only", choices=("scales", "graph", "layout"), action="append")
    parser.add_argument("--expect-module-sha")
    parser.add_argument("--source-dir", type=Path,
                        help="Use this source tree when the runner is copied elsewhere")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    formats = FORMATS if args.format == "all" else (args.format,)
    suites = set(args.only or ("scales", "graph", "layout"))
    if args.list_cases:
        print("\n".join(list_cases(formats, suites)))
        return
    root = args.source_dir.resolve() if args.source_dir else Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "tests"), str(root)]
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(args.device) != (8, 9):
        raise RuntimeError("This WI regression requires an available SM89 device")
    torch.cuda.set_device(args.device)
    torch.set_default_device(f"cuda:{args.device}")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("high")
    import kernelkit as kk
    import lib
    import quant
    import ref
    import flash_mla
    import flash_mla_cuda
    import test_v41_sparse_quant_prefill_validation as validation
    module_sha = hashlib.sha256(Path(flash_mla_cuda.__file__).read_bytes()).hexdigest()
    if args.expect_module_sha:
        assert module_sha == args.expect_module_sha
    assert hasattr(flash_mla_cuda, "sparse_quant_fwd")
    print("MODULE", flash_mla_cuda.__file__, module_sha, flush=True)
    rt = (torch, kk, lib, quant, ref, flash_mla, validation)
    with torch.inference_mode():
        for fmt in formats:
            if "scales" in suites:
                run_scales(fmt, rt)
            if "graph" in suites:
                run_graph(fmt, rt)
        if "fp40" in formats and "layout" in suites:
            run_layout(rt)
    print("VALIDATION", len(tuple(list_cases(formats, suites))), "PASS", flush=True)


if __name__ == "__main__":
    main()
