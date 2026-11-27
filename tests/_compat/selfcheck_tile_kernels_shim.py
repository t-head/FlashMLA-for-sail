#!/usr/bin/env python3
"""the shim round · independent self-check of the local ``tile_kernels`` shim.

Deliberately does NOT re-use the shim's own helpers to build expectations: the
reference here is written from the *published* semantics (DeepGEMM-style
per-block FP8 + UE8M0 power-of-two scales, as documented in the docstrings of
``flash_mla.fused_norm_rope_attn_rope_cast.permute_q_b_proj`` / ``permute_wv_proj``
and as used by the prefill FP8 study's verified quantize/dequantize path).

What is checked:
  1. round-trip error of ``per_token_cast`` -> ``per_token_cast_back``
  2. the fp32 per-channel form (gran == n)
  3. the packed int32 UE8M0 form, and that unpacking reproduces the byte scales
  4. ``utils.align``
  5. ``torch.cast_back`` == ``quant.per_token_cast_back``
  6. the no-overflow property: every quantized block stays within E4M3 range

Run:  python selfcheck_tile_kernels_shim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tile_kernels  # noqa: E402  (the shim under test)

FP8_MAX = 448.0


def ref_quant_per_block(x: torch.Tensor, gran: int, round_up: bool):
    """Independent reference: per-``gran`` amax -> scale -> FP8.

    Returns (data, scale[..., n//gran]) with scale in *plain float*, so it is
    comparable with the shim's ue8m0 bytes via 2**(byte-127).
    """
    n = x.shape[-1]
    xv = x.float().reshape(*x.shape[:-1], n // gran, gran)
    amax = xv.abs().amax(dim=-1).clamp_min(1e-4)
    scale = amax / FP8_MAX
    if round_up:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    data = (xv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(*x.shape[:-1], n)
    return data, scale


def main() -> int:
    torch.manual_seed(0)
    dev = "cpu"
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print("=== 1. round-trip: per_token_cast -> per_token_cast_back (fp32 scales) ===")
    for shape, gran in [((64, 512), 32), ((128, 1024), 128), ((8, 4096), 32)]:
        x = (torch.randn(shape, device=dev) / 10)
        data, sf = tile_kernels.quant.per_token_cast(x, "e4m3", gran)
        back = tile_kernels.quant.per_token_cast_back((data, sf), "fp32", gran)
        # reference expectation built independently
        ref_data, ref_scale = ref_quant_per_block(x, gran, round_up=False)
        max_scale_err = (sf - ref_scale).abs().max().item()
        rel = ((back - x).norm() / x.norm()).item()
        # E4M3 has a 3-bit mantissa => ~6.25% worst-case element error; for
        # random data the relative L2 of per-block quantization lands near 2-3%.
        check(f"shape={shape} gran={gran} relative L2 < 4e-2", rel < 4e-2,
              f"rel_L2={rel:.2e}")
        check(f"shape={shape} gran={gran} scale matches independent ref", max_scale_err < 1e-6,
              f"max|dscale|={max_scale_err:.2e}")
        check(f"shape={shape} gran={gran} data matches independent ref",
              torch.equal(data, ref_data))

    print("\n=== 2. per-channel form (gran == n) -> scale shape [..., 1] ===")
    x = torch.randn((32, 512), device=dev) / 10
    data, sf = tile_kernels.quant.per_token_cast(x, "e4m3", 512)
    check("scale shape is [..., 1]", tuple(sf.shape) == (32, 1), f"got {tuple(sf.shape)}")
    back = tile_kernels.quant.per_token_cast_back((data, sf), "fp32", 512)
    rel = ((back - x).norm() / x.norm()).item()
    check("per-channel round-trip relative L2 < 4e-2", rel < 4e-2, f"rel_L2={rel:.2e}")

    print("\n=== 3. packed int32 UE8M0 form ===")
    x = torch.randn((16, 512), device=dev) / 10
    data, sf = tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, use_packed_ue8m0=True, round_sf=True,
        use_tma_aligned_col_major_sf=True)
    check("packed scale dtype is int32", sf.dtype == torch.int32, f"got {sf.dtype}")
    check("packed scale last dim == n/gran/4", sf.shape[-1] == 512 // 32 // 4,
          f"got {sf.shape[-1]}")
    # unpack and compare against the ceil-rounded reference
    byte = sf.contiguous().view(torch.uint8).reshape(16, 512 // 32)
    scale = torch.pow(2.0, byte.to(torch.int32).float() - 127.0)
    _, ref_scale_ceil = ref_quant_per_block(x, 32, round_up=True)
    err = (scale - ref_scale_ceil).abs().max().item()
    check("unpacked ue8m0 scales match ceil-rounded reference", err < 1e-6,
          f"max|d|={err:.2e}")
    back = tile_kernels.quant.per_token_cast_back((data, sf), "fp32", 32)
    rel = ((back - x).norm() / x.norm()).item()
    check("packed round-trip relative L2 < 4e-2", rel < 4e-2, f"rel_L2={rel:.2e}")

    print("\n=== 4. utils.align ===")
    for a, b, want in [(0, 4, 0), (1, 4, 4), (3, 4, 4), (4, 4, 4), (5, 4, 8), (17, 16, 32)]:
        got = tile_kernels.utils.align(a, b)
        check(f"align({a},{b}) == {want}", got == want, f"got {got}")

    print("\n=== 5. torch.cast_back mirrors quant.per_token_cast_back ===")
    x = torch.randn((8, 256), device=dev) / 10
    d1, s1 = tile_kernels.quant.per_token_cast(x, "e4m3", 32)
    a = tile_kernels.quant.per_token_cast_back((d1, s1), "fp32", 32)
    b = tile_kernels.torch.cast_back((d1, s1), "fp32", (1, 32))
    check("identical results", torch.equal(a, b))

    print("\n=== 6. no E4M3 overflow: each element stays inside the FP8 range ===")
    x = (torch.randn((32, 512), device=dev) * 1000)      # deliberately large
    data, sf = tile_kernels.quant.per_token_cast(x, "e4m3", 32)
    back = tile_kernels.quant.per_token_cast_back((data, sf), "fp32", 32)
    # The real invariant: the block max maps *onto* 448 (the largest finite
    # E4M3 value), i.e. it lands at the top of the range rather than beyond it.
    # Note 448.0 itself is exactly representable, so `<=` is the correct bound.
    fp8_absmax = data.float().abs().max().item()
    check("stored FP8 lands at the top of the range, never beyond",
          fp8_absmax <= FP8_MAX, f"|fp8|max={fp8_absmax:.1f}")
    # And the dequantized magnitude tracks the original (block max ~ 448 * scale).
    ratio = back.abs().max().item() / x.abs().max().item()
    check("dequantized magnitude tracks the input (ratio in [0.9, 1.1])",
          0.9 <= ratio <= 1.1, f"ratio={ratio:.4f}")
    check("dequantized values finite", bool(torch.isfinite(back).all()))

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        return 1
    print("RESULT: all shim self-checks PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())