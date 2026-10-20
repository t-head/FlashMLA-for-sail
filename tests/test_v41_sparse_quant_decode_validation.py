"""Eight stock V4.1 generic WI decode regressions, including the H64 barrier hang.

Run after building the extension:
  FLASHMLA_V41_WI=1 python tests/test_v41_sparse_quant_decode_validation.py

--list-cases imports no torch and launches no GPU work. The first case preserves
the original failing seed28404. Every case uses the original decode fixture,
reference and tolerances, with HS64 disabled and an actual generic WI trace.
Use an external timeout when testing an older extension with the barrier defect.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


# IDs/seeds are from the original stock performance matrix. All eight use B4,
# Sq1, main64+extra64, page64, variable cache lengths, and an attention sink.
CASES = (
    ("744cb88bcc20a28f", 28404, 64, False, True),
    ("44f1b4ec64f70d34", 28396, 64, False, False),
    ("d73947695da198bc", 28412, 128, False, False),
    ("491051d9b6e0fc7d", 28420, 128, False, True),
    ("754e30792100797c", 28428, 64, True, False),
    ("3ab6e53db0bef496", 28436, 64, True, True),
    ("c217bbab37419ba7", 28444, 128, True, False),
    ("696d8179f267bf25", 28452, 128, True, True),
)


def case_params(heads, extra_length):
    return dict(b=4, h_q=heads, h_kv=1, s_q=1, s_kv=512, topk=64,
                extra_topk=64, extra_s_k=512, block_size=64, extra_block_size=64,
                d_qk=512, d_v=512, is_varlen=True, enable_attn_sink=True,
                have_topk_length=False, have_extra_topk_length=extra_length,
                have_zero_seqlen_k=False, is_all_indices_invalid=False)


def event(case, stage, **values):
    print("DECODE_VALIDATION_PHASE", json.dumps(dict(case=case, stage=stage, **values), sort_keys=True), flush=True)


def trace_call(call, case, heads, extra_fp4, torch):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        event(case, "quant-call_BEGIN")
        answer = call()
        event(case, "quant-call_ENQUEUED")
        torch.cuda.synchronize()
        event(case, "quant-sync_DONE")
    names = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    main = [n for n in names if "flash_sparse_decode_wg_kernel" in n]
    flag = "true" if extra_fp4 else "false"
    pattern = r"\bflash_sparse_decode_wg_kernel\s*<\s*Traits_v2\s*<\s*cutlass::bfloat16_t\s*,\s*512\s*,\s*true\s*,\s*" + str(heads) + r"\s*,\s*true\s*,\s*" + flag + r"\s*>"
    forbidden = [n for n in names if "flash_sparse_decode_fwd_kernel" in n]
    okay = len(main) == 1 and not forbidden and all(re.search(pattern, n) for n in main)
    trace = dict(case=case, status="WI_ACTUAL_GENERIC_MAIN" if okay else "WRONG_ROUTE",
                 heads=heads, FP4_extra=extra_fp4, main=main, forbidden_fallback=forbidden, CUDA_names=names)
    print("DECODE_VALIDATION_TRACE", json.dumps(trace, sort_keys=True), flush=True)
    assert okay, trace
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-cases", action="store_true")
    ap.add_argument("--case", choices=tuple(c[0] for c in CASES))
    ap.add_argument("--device", type=int, default=0, help="Logical visible CUDA device")
    ap.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--module-dir", type=Path, help="Optional explicitly built extension directory")
    ap.add_argument("--expect-module-sha", help="Optional required extension SHA256")
    args = ap.parse_args()
    cases = [c for c in CASES if args.case is None or c[0] == args.case]
    if args.list_cases:
        print(json.dumps([dict(id=n, seed=seed, heads=h, FP4_extra=fp4,
                               params=case_params(h, lengths)) for n, seed, h, fp4, lengths in cases], indent=2))
        return
    source = args.source_dir.resolve()
    sys.path[:0] = ([str(args.module_dir.resolve())] if args.module_dir else []) + [str(source / "tests"), str(source)]
    os.environ["FLASHMLA_V41_WI"] = "1"
    os.environ["FLASHMLA_V41_HS64"] = "0"
    import torch
    import kernelkit as kk
    import lib
    import quant
    import ref
    import flash_mla
    import flash_mla_cuda
    module = Path(flash_mla_cuda.__file__).resolve()
    module_sha = hashlib.sha256(module.read_bytes()).hexdigest()
    if args.module_dir:
        assert module.parent == args.module_dir.resolve(), module
    if args.expect_module_sha:
        assert module_sha == args.expect_module_sha, (module, module_sha)
    torch.cuda.set_device(args.device)
    torch.set_default_device(f"cuda:{args.device}")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(16)
    print("DECODE_VALIDATION_BINDING", json.dumps(dict(module=str(module), moduleSHA=module_sha,
          HS64="0", source=str(source), cases=len(cases)), sort_keys=True), flush=True)
    passed = []
    with torch.inference_mode():
        for name, seed, heads, fp4, lengths in cases:
            event(name, "fixture_begin", seed=seed, params=case_params(heads, lengths))
            raw = lib.RawTestParamForDecode(**case_params(heads, lengths), seed=seed, is_fp8=True,
                  fp8_kvcache_layout=quant.FP8KVCacheLayout.V41_FP8Sparse,
                  extra_fp8_kvcache_layout=quant.FP8KVCacheLayout.V41_FP4Sparse if fp4 else quant.FP8KVCacheLayout.V41_FP8Sparse)
            p = raw.to_test_param()
            p.num_runs = 0
            t = lib.generate_testcase_for_decode(p)
            torch.cuda.synchronize()
            event(name, "fixture_ready", extra_topk_length=t.extra_kv_scope.topk_length.tolist() if t.extra_kv_scope.topk_length is not None else None)
            metadata, _ = flash_mla.get_mla_metadata()
            out, lse = trace_call(lambda: lib.run_flash_mla_decode(p, t, metadata, None), name, heads, fp4, torch)
            event(name, "reference_BEGIN")
            out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)
            # Original tests/test_flash_mla_sparse_decoding.py comparisons.
            checks = dict(out=bool(kk.check_is_allclose("out", out, out_ref, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)),
                          lse=bool(kk.check_is_allclose("lse", lse, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)))
            assert all(checks.values()), (name, checks)
            event(name, "reference_checks_DONE", checks=checks)
            passed.append(name)
            print("PASS_DECODE_GENERIC_STOCK", name, flush=True)
    print("DECODE_VALIDATION_SUMMARY", json.dumps(dict(attempted=len(cases), passed=len(passed),
          failed=0, moduleSHA=module_sha, actual_generic_WI_required=True, original_thresholds=True), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
