"""FlashMLA supplemental UTs.

Each UT occupies one row in CASES; add or remove coverage by modifying only this list.
The kind selects the validation path: dense uses flash_mla.flash_mla_with_kvcache,
and sparse uses flash_mla.flash_mla_sparse_fwd.
All UTs are correctness tests: performance measurement is disabled and existing numerical tolerances apply.

Run from the repository root:
    python -m pytest tests/test_flash_mla_supplemental.py -v
Append -k <case id> to select an individual UT; ACTest runs the entire file.
"""

from contextlib import contextmanager
from dataclasses import replace
from typing import NamedTuple
import random

import pytest
import torch

import flash_mla
import kernelkit as kk
import lib as _lib
import ref as _ref
import test_flash_mla_dense_decoding as _dense


class Case(NamedTuple):
    id: str
    kind: str     # Type: dense | sparse
    params: dict


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

CASES = [
    # id                                  type     parameters
    Case("dense_page32_kv38_noncontiguous", "dense",  dict(dtype="bf16", b=1,   s_q=1, s_k=38,   h_q=16,  block_size=32, is_causal=False, separated_pages=True)),
    Case("dense_fp16_b64_kv6120",           "dense",  dict(dtype="fp16", b=64,  s_q=1, s_k=6120, h_q=128, block_size=64, is_causal=False)),
    # With Sq=1, the backend normalizes causal to False, so this covers the API parameter combination.
    Case("dense_fp16_b128_kv8192",          "dense",  dict(dtype="fp16", b=128, s_q=1, s_k=8192, h_q=128, block_size=64, is_causal=True)),
    Case("sparse_prefill_int32_overflow",   "sparse", dict(dtype="bf16", s_q=30220, s_kv=30220, topk=2048, h_q=128, h_kv=1, d_qk=576, d_v=512)),
]


@contextmanager
def _cuda_case(dtype):
    """Provide defaults for legacy helpers without affecting other tests."""
    old_dtype = torch.get_default_dtype()
    old_random_state = random.getstate()
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_precision = torch.get_float32_matmul_precision()
    try:
        # The reused generator seeds all visible CUDA generators.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            with torch.device("cuda"):
                torch.set_default_dtype(dtype)
                torch.set_float32_matmul_precision("highest")
                torch.manual_seed(0)
                random.seed(0)
                yield
    finally:
        torch.set_default_dtype(old_dtype)
        random.setstate(old_random_state)
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.set_float32_matmul_precision(old_precision)


@torch.inference_mode()
def _check_dense(*, dtype, b, s_q, s_k, h_q, block_size, is_causal, separated_pages=False):
    torch_dtype = DTYPES[dtype]
    with _cuda_case(torch_dtype):
        p = _dense.TestParam(
            b=b, s_q=s_q, s_k=s_k, is_varlen=False, is_causal=is_causal,
            test_performance=False, block_size=block_size, h_q=h_q, seed=0,
        )
        cache_seqlens, q, block_table, blocked_k = _dense.generate_test_data(p)
        assert q.dtype == blocked_k.dtype == torch_dtype
        assert torch.all(cache_seqlens == p.s_k).item()
        if separated_pages:
            # Preserve logical KV values but fix their physical pages to [3, 1].
            # Page 1 has 6 valid tokens and a NaN tail; other pages stay poisoned.
            assert p.b == 1 and p.s_k == 38 and p.block_size == 32
            valid_pages = blocked_k[block_table[0, :2]].clone()
            blocked_k.fill_(float("nan"))
            blocked_k[3].copy_(valid_pages[0])
            blocked_k[1].copy_(valid_pages[1])
            block_table.fill_(2147480000)
            block_table[0, :2] = torch.tensor([3, 1], dtype=torch.int32)

        metadata, num_splits = flash_mla.get_mla_metadata()
        out, lse = flash_mla.flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, p.dv,
            metadata, num_splits, causal=p.is_causal,
        )
        torch.cuda.synchronize()
        out_ref, lse_ref = _dense.reference_torch(
            cache_seqlens, block_table, q, blocked_k, p.dv, p.is_causal,
        )
        assert torch.isfinite(out).all().item()
        assert torch.isfinite(lse).all().item()
        # Match the existing dense test's numerical criteria.
        assert kk.check_is_allclose(
            "out", out, out_ref, abs_tol=8e-4, rel_tol=2.01 / 128,
            cos_diff_tol=5e-6,
        )
        assert kk.check_is_allclose(
            "lse", lse, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536,
        )


@torch.inference_mode()
def _check_sparse_prefill(*, dtype, s_q, s_kv, topk, h_q, h_kv, d_qk, d_v):
    torch_dtype = DTYPES[dtype]
    with _cuda_case(torch_dtype):
        p = _lib.TestParam(
            s_q=s_q, s_kv=s_kv, topk=topk, h_q=h_q, h_kv=h_kv,
            d_qk=d_qk, d_v=d_v, seed=0, num_runs=0, is_fp8=False,
        )
        q = torch.randn(p.s_q, p.h_q, p.d_qk, dtype=torch_dtype).div_(10)
        kv = torch.randn(p.s_kv, p.h_kv, p.d_qk, dtype=torch_dtype).div_(10)
        # Unique, valid indices per query, including the end of the KV cache.
        # Avoid the legacy generator's Sq x Sk random matrix and topk operation.
        starts = torch.arange(p.s_q, dtype=torch.int32).unsqueeze(1) * 17
        offsets = torch.arange(p.topk, dtype=torch.int32).unsqueeze(0)
        indices = ((starts + offsets) % p.s_kv).unsqueeze(1).contiguous()
        assert (p.s_q - 1) * q.stride(0) > 2**31 - 1
        t = _lib.Testcase(
            p=p, q=q, kv=kv, indices=indices, sm_scale=0.5,
            # The shared reference does not use the backward-only dOut field.
            dOut=torch.empty(0), attn_sink=None, topk_length=None,
        )
        out, max_logits, lse = _lib.run_flash_mla_sparse_fwd(p, t, False)
        torch.cuda.synchronize()
        assert out.shape == (p.s_q, p.h_q, p.d_v)
        assert max_logits.shape == lse.shape == (p.s_q, p.h_q)

        # At 128 queries, gathered FP32 KV is 576 MiB instead of 132.8 GiB.
        for start in range(0, p.s_q, 128):
            end = min(start + 128, p.s_q)
            chunk_p = replace(p, s_q=end - start)
            chunk_t = replace(t, p=chunk_p, q=q[start:end], indices=indices[start:end])
            ref_bf16, ref_fp32, ref_max_logits, ref_lse = _ref.ref_sparse_attn_fwd(chunk_p, chunk_t)
            for value in (out[start:end], max_logits[start:end], lse[start:end]):
                assert torch.isfinite(value).all().item(), f"nonfinite rows {start}:{end}"
            # Use upstream tolerances per chunk, including the cosine check.
            # Each chunk must pass; errors in a few rows cannot be averaged away.
            assert kk.check_is_allclose(
                f"out[{start}:{end}]", out[start:end].float(), ref_fp32,
                abs_tol=8e-4, rel_tol=3.01 / 128, cos_diff_tol=7e-6,
            )
            assert kk.check_is_allclose(
                f"max_logits[{start}:{end}]", max_logits[start:end], ref_max_logits,
                abs_tol=1e-6, rel_tol=2.01 / 65536,
            )
            assert kk.check_is_allclose(
                f"lse[{start}:{end}]", lse[start:end], ref_lse,
                abs_tol=1e-6, rel_tol=2.01 / 65536,
            )
            del ref_bf16, ref_fp32, ref_max_logits, ref_lse, chunk_t


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_supplemental(case):
    if case.kind == "dense":
        _check_dense(**case.params)
    else:
        _check_sparse_prefill(**case.params)
