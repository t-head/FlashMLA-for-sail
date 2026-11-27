#!/usr/bin/env python3
"""Diagnose where the extra=None arm-vs-arm O error comes from:
raw fp8 codes vs out_sf flips vs dequantized values."""
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR / "_compat"))
import importlib.util
spec = importlib.util.spec_from_file_location("tile_kernels", TESTS_DIR / "_compat" / "tile_kernels.py")
tk = importlib.util.module_from_spec(spec); sys.modules["tile_kernels"] = tk
spec.loader.exec_module(tk)
for sub in ("quant", "utils", "torch"):
    sys.modules[f"tile_kernels.{sub}"] = getattr(tk, sub)
import tile_kernels

from fused_norm_rope_attn_rope_cast import lib, quant
from fused_norm_rope_attn_rope_cast.lib import RawTestParamForDecode
import flash_mla

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(torch.device("cuda:0"))
torch.cuda.set_device(torch.device("cuda:0"))
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)

h_q, s_q, topk, s_kv = 64, 4, 128, 2048
p = RawTestParamForDecode(1, h_q, s_q, 1, s_kv, False, topk,
    is_all_indices_invalid=False, have_zero_seqlen_k=False, have_topk_length=False,
    enable_attn_sink=True, extra_s_k=None, extra_topk=None,
    block_size=(64+topk%8), extra_block_size=None, have_extra_topk_length=False,
    d_qk=512, kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse,
    extra_kvcache_layout=None, num_runs=0).to_test_param()
p.seed = 1234
t = lib.generate_testcase_for_decode(p)
b, sq = p.decode.b, p.s_q
tp = torch.zeros(b*sq, dtype=torch.int32, device="cuda")
nwg = h_q // 8
kv = t.kv_scope.get_kvcache_for_flash_mla()

# proper q via the official path: ramp-like deterministic q in DSA packed layout is
# unknown; use random q (t.q) reshaped like the swizzle-fix round's probe (standard layout, both arms
# equal) -- good enough to isolate sf-vs-code effects.
qr = t.q.reshape(b*sq, h_q, p.d_qk).to(torch.bfloat16).contiguous()

def run(fp8):
    return flash_mla.fused_norm_rope_attn_rope_cast.decode(
        enable_q_norm=False, rms_norm_eps=0.0, token_positions=tp,
        is_rope_neox_style=False, rope_dim=64,
        cos_sin_cache=torch.zeros(131072, 64, device="cuda", dtype=torch.float32),
        n_wv_group=nwg, num_per_channels=32, use_tma_aligned_col_major_sf=True,
        round_sf=True, use_packed_ue8m0=True, q=qr, k_cache=kv,
        indices_in_kvcache=t.kv_scope.indices_in_kvcache.reshape(b*sq, p.topk),
        sm_scale=t.sm_scale, d_v=p.d_v, attn_sink=t.attn_sink, topk_length=None,
        extra_k_cache=None, extra_indices_in_kvcache=None, extra_topk_length=None,
        fp8_compute=fp8)

ob = run(False); lib.device_synchronize()
of = run(True);  lib.device_synchronize()
(caf_b, sf_b, lse_b) = ob
(caf_f, sf_f, lse_f) = of

codes_b = caf_b.view(b*sq, -1).float()
codes_f = caf_f.view(b*sq, -1).float()
print(f"raw fp8 codes : mean|d|/mean|b| = {(codes_f-codes_b).abs().mean().item()/codes_b.abs().mean().item():.4%}")
print(f"raw lse       : mean|d|/mean|b| = {(lse_f-lse_b).abs().mean().item()/lse_b.abs().mean().item():.4%}")
sf_b_i = sf_b.view(b*sq, -1)
sf_f_i = sf_f.view(b*sq, -1)
ndiff = (sf_b_i != sf_f_i).sum().item()
print(f"out_sf int32  : {ndiff}/{sf_b_i.numel()} entries differ "
      f"({ndiff/sf_b_i.numel():.2%}); max ratio of differing = ", end="")
dmask = sf_b_i != sf_f_i
if dmask.any():
    import math
    r = (sf_f_i[dmask].float() / sf_b_i[dmask].float().clamp_min(1))
    print(f"{r.max().item():.3f}")
else:
    print("n/a")

def dequant(codes, sf):
    sf2d = sf.view(b*sq, -1)
    if sf2d.stride(0) != 1:
        sf2d = sf2d.as_strided(sf2d.shape, (1, sf2d.stride(1)))
    o = tile_kernels.quant.per_token_cast_back((codes.view(b*sq, -1), sf2d), 'fp32', 32)
    return o.view(b*sq, nwg, p.d_v // 32, 8, 32).transpose(2, 3).reshape(b*sq, p.h_q, p.d_v)

o_b_own = dequant(caf_b, sf_b)
o_f_own = dequant(caf_f, sf_f)
o_f_bsf = dequant(caf_f, sf_b)      # fp8 codes with bf16 arm's sf
print(f"dequant own sf    : mean|d|/mean|b| = {(o_f_own-o_b_own).abs().mean().item()/o_b_own.abs().mean().item():.4%}")
print(f"dequant common sf : mean|d|/mean|b| = {(o_f_bsf-o_b_own).abs().mean().item()/o_b_own.abs().mean().item():.4%}")
print(f"|O_bf16| mean = {o_b_own.abs().mean().item():.6f}")
