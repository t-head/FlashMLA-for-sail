"""Validate eligible FP8 WI at a P128 boundary and changing Graph inputs.

Run after building flash_mla_cuda on SM89:
    python tests/test_v41_sparse_quant_gpu_checked_validation.py
"""
import argparse
import dataclasses
import hashlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--expect-module-sha')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / 'tests'), str(root)]
    import torch
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
    print('MODULE', flash_mla_cuda.__file__, module_sha, flush=True)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f'cuda:{args.device}')
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision('high')

    case = dataclasses.replace(validation.TARGET, name='gpu-checked-page128-boundary',
                               s_q=4097, s_kv=8207, topk=2176, page_size=128, seed=919)
    p, t, packed = validation.make_case(case, 'fp8', torch, lib, quant)

    def call():
        return flash_mla.flash_mla_sparse_quant_fwd(
            t.q, packed, t.indices, t.sm_scale, case.s_kv,
            attn_sink=t.attn_sink, topk_length=t.topk_length)

    def check_reference(answers):
        _, expected_out, expected_max, expected_lse = ref.ref_sparse_attn_fwd(p, t)
        expected_lse[expected_lse == float('-inf')] = float('inf')
        correct = bool(kk.check_is_allclose('out', answers[0].float(), expected_out,
                       abs_tol=8e-4, rel_tol=3.01 / 128, cos_diff_tol=7e-6))
        correct &= bool(kk.check_is_allclose('max_logits', answers[1], expected_max,
                        abs_tol=1e-6, rel_tol=2.01 / 65536))
        correct &= bool(kk.check_is_allclose('lse', answers[2], expected_lse,
                        abs_tol=1e-6, rel_tol=2.01 / 65536))
        assert correct

    # The original checker retains exact output-buffer reuse and reference
    # thresholds. Exercise the last query/column at a different eligible shape.
    original_make = validation.make_case
    row, col = case.s_q - 1, case.topk - 1
    saved_index = int(t.indices[row, 0, col].item())
    try:
        validation.make_case = lambda *_: (p, t, packed)
        for name, value in [('valid', saved_index), ('invalid', case.s_kv)]:
            t.indices[row, 0, col] = value
            validation.check_case(case, 'fp8', torch, kk, lib, quant, ref, flash_mla)
            print('PASS boundary', name, flush=True)
    finally:
        validation.make_case = original_make
        t.indices[row, 0, col] = saved_index

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for name, value in [('valid', saved_index), ('invalid', case.s_kv),
                        ('restored-valid', saved_index)]:
        t.indices[row, 0, col] = value
        graph.replay()
        torch.cuda.synchronize()
        check_reference(captured)
        before = tuple(x.clone() for x in captured)
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(before, captured)), name
        del before
        print('PASS graph-boundary', name, flush=True)
    print('VALIDATION FP8 boundary/reuse and Graph valid-invalid-valid PASS', flush=True)


if __name__ == '__main__':
    main()
