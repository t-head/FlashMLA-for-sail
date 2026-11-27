"""Local stand-in for the internal ``tile_kernels`` package.

WHY THIS FILE EXISTS
--------------------
``tests/fused_norm_rope_attn_rope_cast/test-fused-norm-rope-attn-rope-cast.py``
(the only official unit test covering the fused "five-in-one" MLA kernel used by
``flashmla_mega``) hard-imports ``tile_kernels`` at module scope::

    import tile_kernels                      # line 10
    tile_kernels.quant.per_token_cast(...)   # reference-side FP8 quantization
    tile_kernels.quant.per_token_cast_back(...)

That package is **not installable on PPU**.  It is absent from every PPU image
(the upstream-test round measured this), and the versions on the internal index all fail:

  * ``tile_kernels`` 2.x declares ``Requires-Dist: tilelang>=0.1.15``;
  * PPU tilelang tops out at ``0.1.12`` (``tilelang.utils.target`` does not exist);
  * 1.0.0 / 2.1.0 install but die at import on that same missing module.

So the version windows do not overlap and no pinned combination works.

WHAT THIS IS
------------
A **local, pure-PyTorch equivalent** of exactly the three symbols the test uses.
It is not a reimplementation of the upstream package and makes no claim to
bit-exactness with the real CUDA/TileLang kernels.  Its contract is the one the
*test* relies on, which is documented in the public docstrings of
``flash_mla.fused_norm_rope_attn_rope_cast.permute_q_b_proj`` /
``permute_wv_proj`` (DeepGEMM-style layout) plus the prefill FP8 study's verified
quantize/dequantize reference.

Scope: only what the fused test needs.
  * ``per_token_cast``      — 1-D/2-D/3-D FP8 cast with per-``gran`` UE8M0 scales
  * ``per_token_cast_back`` — inverse of the above
  * ``utils.align``         — round up to a multiple
  * ``torch.cast_back``     — dequantize a ``(data, scale)`` pair

Verified against an independent reference by ``selfcheck_tile_kernels_shim.py``
(see the shim round evidence); do not trust "the test passed" as proof this is right.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch as _torch_module

__all__ = ["quant", "utils", "torch"]

# EE4M3 (float8_e4m3fn) saturates at this magnitude; the scale is chosen so the
# block max maps here, matching DeepGEMM / tile_kernels conventions.
FP8_E4M3_MAX = 448.0
_FP8_MIN_SCALE = 1e-4


# --------------------------------------------------------------------------- #
# utils
# --------------------------------------------------------------------------- #
class _Utils:
    @staticmethod
    def align(x: int, n: int) -> int:
        """Round ``x`` up to the next multiple of ``n``."""
        assert n > 0
        return (x + n - 1) // n * n


# --------------------------------------------------------------------------- #
# scale helpers (UE8M0 = power-of-two scale stored as a biased exponent byte)
# --------------------------------------------------------------------------- #
def _amax_to_ue8m0_byte(amax: _torch_module.Tensor, round_sf: bool) -> _torch_module.Tensor:
    """amax -> ue8m0 exponent byte, i.e. the byte ``e`` such that scale = 2**(e-127).

    ``round_sf=True`` keeps the clipping guarantee by rounding **up** to the next
    power of two (ceil in log2), which is what the per-32 packed path expects.
    ``round_sf=False`` uses the plain quotient, matching the per-channel path.
    """
    scale = _torch_module.clamp_min(amax / FP8_E4M3_MAX, _FP8_MIN_SCALE)
    if round_sf:
        exp = _torch_module.ceil(_torch_module.log2(scale))
    else:
        exp = _torch_module.log2(scale)
    # 2**exp -> biased float32 exponent -> one byte
    pow2 = _torch_module.pow(2.0, exp)
    return (pow2.view(_torch_module.int32) >> 23).to(_torch_module.uint8)


def _ue8m0_byte_to_scale(byte: _torch_module.Tensor) -> _torch_module.Tensor:
    """ue8m0 exponent byte -> float32 scale 2**(e-127)."""
    return _torch_module.pow(2.0, byte.to(_torch_module.int32).float() - 127.0)


def _pack_ue8m0(byte_sf: _torch_module.Tensor) -> _torch_module.Tensor:
    """[..., k] uint8 -> [..., k//4] int32 (little-endian packing of 4 bytes)."""
    *lead, k = byte_sf.shape
    assert k % 4 == 0, f"scale dim {k} must be a multiple of 4 for int32 packing"
    return byte_sf.contiguous().view(*lead, k // 4, 4).view(_torch_module.int32).squeeze(-1)


def _unpack_ue8m0(packed: _torch_module.Tensor) -> _torch_module.Tensor:
    """[..., k] int32 -> [..., k*4] uint8."""
    return packed.contiguous().view(_torch_module.uint8)


# --------------------------------------------------------------------------- #
# quant
# --------------------------------------------------------------------------- #
class _Quant:
    @staticmethod
    def per_token_cast(
        x: _torch_module.Tensor,
        fmt: str = "e4m3",
        gran: Optional[int] = None,
        out: Optional[_torch_module.Tensor] = None,
        use_tma_aligned_col_major_sf: bool = False,
        round_sf: bool = False,
        use_packed_ue8m0: bool = False,
    ) -> Tuple[_torch_module.Tensor, _torch_module.Tensor]:
        """Per-``gran`` FP8 cast.

        Returns ``(fp8_data, scale_factors)`` where the data keeps ``x``'s shape
        and the scale factors are one value per ``gran`` contiguous elements
        along the last dim:

          * ``use_packed_ue8m0=False`` -> float32 scales, shape ``[..., n/gran]``
            (or ``1`` when ``gran == n``, i.e. the per-channel case)
          * ``use_packed_ue8m0=True``  -> int32 holding 4 ue8m0 bytes each,
            shape ``[..., n/gran/4]``
        """
        assert fmt == "e4m3", f"shim only implements e4m3, got {fmt!r}"
        assert x.dim() >= 1, "per_token_cast needs at least 1 dim"
        n = x.shape[-1]
        if gran is None:
            gran = n
        assert gran > 0 and n % gran == 0, f"last dim {n} not divisible by gran {gran}"

        xf = x.float()
        lead = xf.shape[:-1]
        xv = xf.reshape(*lead, n // gran, gran)
        amax = xv.abs().amax(dim=-1)                      # [..., n/gran]

        if use_packed_ue8m0:
            byte = _amax_to_ue8m0_byte(amax, round_sf=round_sf)
            scale_val = _ue8m0_byte_to_scale(byte)
            data = (xv / scale_val.unsqueeze(-1)).to(_torch_module.float8_e4m3fn)
            sf = _pack_ue8m0(byte)
            if use_tma_aligned_col_major_sf and sf.dim() == 2:
                # The fused host entry points assert `scale_factors.stride(0) == 1`
                # (see csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp), i.e. the
                # *first* dim must be the contiguous one.  Materialize that view.
                # 4-byte packing makes each int32 a group of 4 columns, so the
                # column-contiguous layout is built by transposing the packed
                # form directly.
                sf = sf.t().contiguous().t()
        else:
            scale_val = _torch_module.clamp_min(amax / FP8_E4M3_MAX, _FP8_MIN_SCALE)
            if round_sf:
                scale_val = _torch_module.pow(2.0, _torch_module.ceil(_torch_module.log2(scale_val)))
            data = (xv / scale_val.unsqueeze(-1)).to(_torch_module.float8_e4m3fn)
            sf = scale_val
            # per-channel (gran == n): `scale_val` is already [..., 1] because
            # n // gran == 1, which is exactly the shape the test expects.

        data = data.reshape(*lead, n)
        if out is not None:
            out.copy_(data)
            data = out
        return data, sf

    @staticmethod
    def per_token_cast_back(
        packed: Tuple[_torch_module.Tensor, _torch_module.Tensor],
        fmt: str = "fp32",
        gran: Optional[int] = None,
    ) -> _torch_module.Tensor:
        """Inverse of :meth:`per_token_cast` for the ``(data, sf)`` forms used here.

        Handles both scale layouts the test produces:

          * float32 with a trailing dim of 1 -> one scale per row (per-channel)
          * float32 / int32 with ``n/gran`` scales -> one scale per ``gran`` block
        """
        data, sf = packed
        assert fmt in ("fp32", "float32"), f"shim only dequantizes to fp32, got {fmt!r}"
        n = data.shape[-1]
        lead = data.shape[:-1]

        if sf.dtype == _torch_module.int32:
            assert gran is not None, "gran is required to unpack int32 ue8m0 scales"
            scale = _ue8m0_byte_to_scale(_unpack_ue8m0(sf))   # [..., n/gran]
        elif sf.dtype == _torch_module.float32:
            if sf.shape[-1] == 1:
                # per-channel: a single scale for the whole last dim
                return data.float() * sf.reshape(*sf.shape[:-1], 1)
            scale = sf
        else:
            raise TypeError(f"unsupported scale dtype {sf.dtype}")

        # Block scales: auto-detect the granularity from the scale count so the
        # helper also works when the caller's ``gran`` disagrees with the layout.
        nblk = scale.shape[-1]
        assert n % nblk == 0, f"last dim {n} not divisible by scale count {nblk}"
        blk = n // nblk
        if gran is not None and gran != blk:
            gran = blk
        return (data.float().reshape(*lead, nblk, blk)
                * scale.reshape(*lead, nblk, 1)).reshape(*lead, n)


# --------------------------------------------------------------------------- #
# torch submodule (cast_back / aliases used by projection_reference.py)
# --------------------------------------------------------------------------- #
class _Torch:
    @staticmethod
    def cast_back(packed, fmt: str = "fp32", gran=None) -> _torch_module.Tensor:
        """Alias of :meth:`_Quant.per_token_cast_back` (upstream keeps both)."""
        if gran is None:
            return _Quant.per_token_cast_back(packed, fmt, None)
        if isinstance(gran, (tuple, list)):
            gran = gran[-1]
        return _Quant.per_token_cast_back(packed, fmt, gran)


quant = _Quant()
utils = _Utils()
torch = _Torch()