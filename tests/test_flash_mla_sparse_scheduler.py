"""Native precision and metadata regression for the SM89 BF16 WG128 route.

The large single-query case must schedule one partition per physical SM.
Boundary and unchanged-route cases protect the legacy scheduler contract.
"""
import argparse
import dataclasses
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import flash_mla
import kernelkit as kk
import lib
import ref


@torch.inference_mode()
def test_flash_mla_sparse_scheduler(raw, expected_parts):
    p = raw.to_test_param()
    print(f"Running scheduler regression: {p}", flush=True)
    t = lib.generate_testcase_for_decode(p)
    metadata, _ = flash_mla.get_mla_metadata()
    first = lib.run_flash_mla_decode(p, t, metadata, None)
    torch.cuda.synchronize()

    scheduler = metadata.tile_scheduler_metadata
    assert tuple(scheduler.shape) == (expected_parts, 8), (
        f"Expected {expected_parts} partitions, got {tuple(scheduler.shape)}")
    assert tuple(metadata.num_splits.shape) == (raw.b + 1,)
    pointers = (scheduler.data_ptr(), metadata.num_splits.data_ptr())
    # The last three metadata fields are unused/uninitialized padding.
    payload = scheduler[:, :5].clone()
    num_splits = metadata.num_splits.clone()
    reused = lib.run_flash_mla_decode(p, t, metadata, None)
    torch.cuda.synchronize()
    assert pointers == (metadata.tile_scheduler_metadata.data_ptr(),
                        metadata.num_splits.data_ptr())
    assert torch.equal(payload, metadata.tile_scheduler_metadata[:, :5])
    assert torch.equal(num_splits, metadata.num_splits)

    # Keep both answers alive before allocating the original native reference.
    out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)
    passed = True
    for label, (out, lse) in (("first", first), ("reused", reused)):
        passed &= kk.check_is_allclose(
            label + " out", out, out_ref,
            abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)
        passed &= kk.check_is_allclose(
            label + " lse", lse, lse_ref,
            abs_tol=1e-6, rel_tol=8.01/65536)
    assert passed, "Native output/LSE precision regression"


def main(loops=1, seed_offset=0):
    torch.cuda.set_device(0)
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9):
        print(f"SKIP SM89 sparse scheduler regression: capability={capability}")
        return
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    props = torch.cuda.get_device_properties(0)
    sm_count = 20 if "810E" in props.name else props.multi_processor_count
    legacy_h128_parts = sm_count // math.gcd(2, sm_count)
    legacy_h192_parts = sm_count // math.gcd(3, sm_count)
    base = lib.RawTestParamForDecode(
        b=74, h_q=128, s_q=1, h_kv=1, s_kv=8192, is_varlen=True,
        topk=512, is_fp8=False, enable_attn_sink=True, block_size=64,
        d_qk=576, d_v=512, check_correctness=True, num_runs=0)
    # B4 * TopK == SM_count * 256 must retain the old count because the
    # dispatch uses strict >. One additional 64-token block crosses it.
    boundary_k = sm_count * 64
    boundary_source_k = max(8192, boundary_k + 64)
    cases = [
        ("single_query_all_SMs", base, sm_count),
        ("work_boundary_equal", dataclasses.replace(
            base, b=4, topk=boundary_k, s_kv=boundary_source_k),
         legacy_h128_parts),
        ("work_boundary_above", dataclasses.replace(
            base, b=4, topk=boundary_k + 64, s_kv=boundary_source_k),
         sm_count),
        ("multi_query_unchanged", dataclasses.replace(base, s_q=3),
         legacy_h128_parts),
        ("page_fallback_unchanged", dataclasses.replace(base, block_size=61),
         legacy_h128_parts),
        ("H64_unchanged", dataclasses.replace(base, h_q=64), sm_count),
        ("H192_unchanged", dataclasses.replace(base, h_q=192),
         legacy_h192_parts),
    ]
    next_seed = seed_offset
    for loop in range(loops):
        for name, raw, expected_parts in cases:
            print(f"ROUND {loop} CASE {name}", flush=True)
            test_flash_mla_sparse_scheduler(
                dataclasses.replace(raw, seed=next_seed), expected_parts)
            next_seed += 1
    print(f"PASS {loops * len(cases)} SM89 native scheduler cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    if args.loops < 1:
        parser.error("--loops must be positive")
    main(args.loops, args.seed_offset)
