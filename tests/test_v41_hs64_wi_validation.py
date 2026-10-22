"""Correctness matrix for the opt-in V4.1 H64 WI2 experiment.

Run on M890P after building the sparse decode extension:
    FLASHMLA_V41_HS64=1 \
        python tests/test_v41_hs64_wi_validation.py
"""

import argparse
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import torch

from lib import RawTestParamForDecode
from quant import FP8KVCacheLayout
from test_flash_mla_sparse_decoding import test_flash_mla


# name, query heads, main topk, extra topk, extra FP4, ragged, invalid, zero
CASES = (
    ("fp8-short", 64, 64, 64, False, False, False, False),
    ("fp8-long", 64, 576, 576, False, False, False, False),
    ("fp4-short", 64, 64, 64, True, False, False, False),
    ("fp4-long", 64, 576, 576, True, False, False, False),
    ("fp8-ragged", 64, 192, 192, False, True, False, False),
    ("fp4-ragged", 64, 64, 64, True, True, False, False),
    ("fp8-invalid", 64, 576, 576, False, False, True, False),
    ("fp4-invalid", 64, 64, 64, True, False, True, False),
    ("fp8-zero", 64, 576, 576, False, False, False, True),
    ("fp8-no-extra", 64, 576, None, False, False, False, False),
    ("fp8-uneven", 64, 64, 576, False, False, False, False),
    ("fp4-uneven", 64, 576, 64, True, False, False, False),
    ("h96-fallback", 96, 576, 576, True, False, False, False),
)

# Exercise complete WI2 pipelines without the small-batch scheduler splitting
# them into short partitions. Trailing fields are batch, page size, cache size.
STRESS_CASES = (
    ("fp4-pipeline-long", 64, 576, 576, True, False, False, False, 74, 64, 4096),
    ("fp4-page128-long", 64, 2048, 2048, True, False, False, False, 74, 128, 4096),
    ("fp8-page128-long", 64, 2048, 2048, False, False, False, False, 74, 128, 4096),
    ("fp4-pipeline-ragged", 64, 576, 576, True, True, False, False, 74, 64, 4096),
    ("fp4-pipeline-invalid", 64, 576, 576, True, False, True, False, 74, 64, 4096),
    ("fp4-pipeline-zero", 64, 576, 576, True, False, False, True, 74, 64, 4096),
)

# H128 selects the generic WI decode path alongside the existing H64 cases.
H128_STRESS_CASES = (
    ("h128-fp4-pipeline-long", 128, 576, 576, True, False, False, False, 74, 64, 4096),
    ("h128-fp8-page128-long", 128, 2048, 2048, False, False, False, False, 74, 128, 4096),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="Run a named case")
    parser.add_argument("--stress", action="store_true", help="Include long batch-74 pipelines, page-128 and H128 generic WI cases")
    args = parser.parse_args()
    if os.getenv("FLASHMLA_V41_HS64") != "1":
        parser.error("set FLASHMLA_V41_HS64=1")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(torch.device("cuda:0"))
    torch.cuda.set_device(0)
    cases = [(*case, 2, 64, 1024) for case in CASES]
    if args.stress:
        cases.extend(STRESS_CASES)
        cases.extend(H128_STRESS_CASES)
    selected = set(args.only or (name for name, *_ in cases))
    unknown = selected - {name for name, *_ in cases}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")

    for index, (name, heads, topk, extra_topk, fp4, ragged, invalid, zero, batch, page, cache_size) in enumerate(cases):
        if name not in selected:
            continue
        raw = RawTestParamForDecode(
            batch, heads, 1, 1, cache_size, True, topk,
            is_fp8=True, is_all_indices_invalid=invalid,
            have_zero_seqlen_k=zero, have_topk_length=ragged,
            enable_attn_sink=True,
            extra_s_k=cache_size if extra_topk is not None else None,
            extra_topk=extra_topk,
            block_size=page,
            extra_block_size=page if extra_topk is not None else None,
            have_extra_topk_length=ragged and extra_topk is not None,
            d_qk=512, check_correctness=True, num_runs=0,
            seed=900 + index if index < len(CASES) else 1300 + index - len(CASES),
            fp8_kvcache_layout=FP8KVCacheLayout.V41_FP8Sparse,
            extra_fp8_kvcache_layout=(
                FP8KVCacheLayout.V41_FP4Sparse if fp4
                else FP8KVCacheLayout.V41_FP8Sparse
            ) if extra_topk is not None else None,
        )
        print(f"CASE {name}", flush=True)
        assert test_flash_mla(raw.to_test_param()).is_correct, name
        print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
