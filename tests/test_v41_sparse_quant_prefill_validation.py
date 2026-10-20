"""Native V4.1 packed FP4/FP8 validation for the quantized Prefill API.

Run the small matrix after building the extension on M890P:
    python tests/test_v41_sparse_quant_prefill_validation.py

Include the measured H128/Sq4096/Sk8192/topk2048/P64 seed-0 workload:
    python tests/test_v41_sparse_quant_prefill_validation.py --target

--list-cases and --help do not import PyTorch or launch GPU work.
"""

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    s_q: int
    s_kv: int
    topk: int
    page_size: int
    sink: bool = False
    lengths: bool = False
    special: str = ""
    page_padding: int = 0
    seed: int = 0


# The eight shapes originate from the accepted native WI precision matrix.
# The partial P128 page and padded page stride exercise the public page API.
CASES = (
    Case("short", 3, 256, 128, 64, seed=900),
    Case("page128-sink", 62, 512, 256, 128, sink=True, seed=901),
    Case("partial-page64", 17, 95, 128, 64, seed=902),
    Case("partial-page128", 17, 153, 256, 128, sink=True, seed=903),
    Case("ragged", 62, 592, 128, 64, sink=True, lengths=True, seed=904),
    Case("zero-length", 19, 256, 128, 64, sink=True, lengths=True, special="zero", seed=905),
    Case("all-invalid", 19, 256, 128, 64, sink=True, special="invalid", seed=906),
    Case("padded-page-stride", 5, 1024, 512, 64, sink=True, page_padding=512, seed=907),
)
TARGET = Case("target", 4096, 8192, 2048, 64, sink=True, seed=0)


def make_case(case, fmt, torch, lib, quant):
    p = lib.TestParam(
        case.s_q, case.s_kv, case.topk, h_q=128, h_kv=1, d_qk=512, d_v=512,
        seed=case.seed, num_runs=0, have_attn_sink=case.sink,
        have_topk_length=case.lengths, is_fp8=False,
    )
    t = lib.generate_testcase(p)
    if case.name != "target" and t.attn_sink is not None:
        t.attn_sink[0] = float("-inf")
        t.attn_sink[1] = float("+inf")
    if case.special == "zero":
        t.topk_length = torch.zeros((case.s_q,), dtype=torch.int32, device=t.q.device)
    elif case.special == "invalid":
        t.indices.fill_(2147483647)
    elif case.lengths:
        # Cover the 32-token blocks and the even 64-token tail explicitly.
        borders = (0, 1, 31, 32, 33, 63, 64, 65, case.topk - 1, case.topk)
        t.topk_length.copy_(torch.tensor(
            [borders[i % len(borders)] for i in range(case.s_q)],
            dtype=torch.int32, device=t.q.device,
        ))

    blocks = (case.s_kv + case.page_size - 1) // case.page_size
    dense = torch.zeros((blocks * case.page_size, 1, 512), dtype=torch.bfloat16, device=t.kv.device)
    dense[:case.s_kv] = t.kv
    layout = (
        quant.FP8KVCacheLayout.V41_FP4Sparse if fmt == "fp4"
        else quant.FP8KVCacheLayout.V41_FP8Sparse
    )
    native = quant.quantize_k_cache(dense.reshape(blocks, case.page_size, 1, 512), layout)
    # Reference the exact dequantized BF16 values consumed by the attention MMA.
    decoded = quant.dequantize_k_cache(native, layout).reshape(blocks * case.page_size, 1, 512)[:case.s_kv]
    packed = native.view(torch.uint8)
    if case.page_padding:
        bytes_per_page = case.page_size * packed.size(-1)
        storage = torch.full(
            (blocks, bytes_per_page + case.page_padding), 0xA5,
            dtype=torch.uint8, device=packed.device,
        )
        storage[:, :bytes_per_page] = packed.reshape(blocks, bytes_per_page)
        packed = storage[:, :bytes_per_page].view(blocks, case.page_size, 1, packed.size(-1))
    reference_case = lib.Testcase(
        p, t.dOut, t.q, decoded, t.indices, t.sm_scale, t.attn_sink, t.topk_length,
    )
    return p, reference_case, packed


def check_case(case, fmt, torch, kk, lib, quant, ref, flash_mla):
    p, t, packed = make_case(case, fmt, torch, lib, quant)
    print(
        f"CASE {fmt}/{case.name} Sq={p.s_q} Sk={p.s_kv} topk={p.topk} "
        f"P={case.page_size} seed={case.seed} "
        f"q_stride={t.q.stride()} index_stride={t.indices.stride()} kv_stride={packed.stride()}",
        flush=True,
    )

    def call(out=None):
        return flash_mla.flash_mla_sparse_quant_fwd(
            t.q, packed, t.indices, t.sm_scale, case.s_kv,
            attn_sink=t.attn_sink, topk_length=t.topk_length, out=out,
        )

    # Produce both answers before allocating any reference answer.
    answers = call()
    output_buffer = torch.full(
        (p.s_q, 128, 512), float("nan"), dtype=torch.bfloat16, device=t.q.device,
    )
    reused = call(output_buffer)
    torch.cuda.synchronize()
    assert len(answers) == len(reused) == 3
    out, max_logits, lse = answers
    assert out.shape == (p.s_q, 128, 512) and out.dtype == torch.bfloat16
    assert max_logits.shape == lse.shape == (p.s_q, 128)
    assert max_logits.dtype == lse.dtype == torch.float32
    assert all(x.device == t.q.device for x in answers)
    assert reused[0].data_ptr() == output_buffer.data_ptr(), "out buffer was not reused"
    for name, original, repeated in zip(("out", "max_logits", "lse"), answers, reused):
        assert torch.equal(original, repeated), f"out reuse changed {name}"

    _, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
    ref_lse[ref_lse == float("-inf")] = float("+inf")
    torch.cuda.synchronize()
    # These are the existing sparse Prefill reference thresholds. LSE is
    # compared directly in natural-log units, with no log-base conversion.
    correct = bool(kk.check_is_allclose(
        "out", out.float(), ref_out_fp32,
        abs_tol=8e-4, rel_tol=3.01 / 128, cos_diff_tol=7e-6,
    ))
    correct &= bool(kk.check_is_allclose(
        "max_logits", max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01 / 65536,
    ))
    correct &= bool(kk.check_is_allclose(
        "lse", lse, ref_lse, abs_tol=1e-6, rel_tol=2.01 / 65536,
    ))
    assert correct, f"{fmt}/{case.name}"
    print(f"PASS {fmt}/{case.name} out-reuse=PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("fp4", "fp8", "both"), default="both")
    parser.add_argument("--only", action="append", help="Run a named case")
    parser.add_argument("--target", action="store_true", help="Include the H128/Sq4096/Sk8192/topk2048/P64 seed-0 case")
    parser.add_argument("--list-cases", action="store_true", help="List selected cases without importing PyTorch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expect-module-sha", help="Require this SHA256 of the loaded extension")
    args = parser.parse_args()
    cases = (*CASES, TARGET) if args.target else CASES
    selected = set(args.only or (case.name for case in cases))
    unknown = selected - {case.name for case in cases}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}; target requires --target")
    formats = ("fp4", "fp8") if args.format == "both" else (args.format,)
    if args.list_cases:
        for fmt in formats:
            for case in cases:
                if case.name in selected:
                    print(f"{fmt}/{case.name}")
        return

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "tests"), str(root)]
    import torch
    import kernelkit as kk
    import lib
    import quant
    import ref
    import flash_mla
    import flash_mla_cuda

    module_path = Path(flash_mla_cuda.__file__).resolve()
    module_sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
    print(f"MODULE {module_path} SHA256 {module_sha}", flush=True)
    if args.expect_module_sha and module_sha != args.expect_module_sha:
        raise RuntimeError("Loaded extension SHA256 does not match --expect-module-sha")
    if not hasattr(flash_mla_cuda, "sparse_quant_fwd"):
        raise RuntimeError("The loaded extension does not provide sparse_quant_fwd")
    device = torch.device(f"cuda:{args.device}")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    passed = 0
    with torch.inference_mode():
        for fmt in formats:
            for case in cases:
                if case.name not in selected:
                    continue
                check_case(case, fmt, torch, kk, lib, quant, ref, flash_mla)
                passed += 1
                torch.cuda.empty_cache()
    print(f"VALIDATION {passed}/{len(formats) * len(selected)} PASS", flush=True)


if __name__ == "__main__":
    main()
