#!/usr/bin/env python3
"""Isolate which config dimension breaks the extra=None arm-vs-arm O comparison:
q path (official packed vs standard), rope (real vs zero), shape."""
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
from fused_norm_rope_attn_rope_cast.projection_reference import fp8_gemm_nt_reference
import flash_mla

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(torch.device("cuda:0"))
torch.cuda.set_device(torch.device("cuda:0"))
torch.set_float32_matmul_precision('high')

# official module for cos_sin_cache / get_q_b_weight
spec2 = importlib.util.spec_from_file_location(
    "official_fused_test", HERE / "test-fused-norm-rope-attn-rope-cast.py")
official = importlib.util.module_from_spec(spec2)
sys.modules["official_fused_test"] = official
spec2.loader.exec_module(official)

h_q = 64
nwg = h_q // 8

def run_config(s_q, topk, s_kv, q_mode, rope_mode, seed=1234, b=1):
    p = RawTestParamForDecode(b, h_q, s_q, 1, s_kv, False, topk,
        is_all_indices_invalid=False, have_zero_seqlen_k=False, have_topk_length=False,
        enable_attn_sink=True, extra_s_k=None, extra_topk=None,
        block_size=(64+topk%8), extra_block_size=None, have_extra_topk_length=False,
        d_qk=512, kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse,
        extra_kvcache_layout=None, num_runs=0).to_test_param()
    p.seed = seed
    t = lib.generate_testcase_for_decode(p)
    bs = b * s_q
    if rope_mode == "real":
        cs = official.cos_sin_cache
        tp = torch.randint(0, official.max_token_position, (bs,), dtype=torch.int32)
    else:
        cs = torch.zeros(131072, 64, device="cuda", dtype=torch.float32)
        tp = torch.zeros(bs, dtype=torch.int32)
    kv = t.kv_scope.get_kvcache_for_flash_mla()

    if q_mode == "official":
        q_lora_rank, gran = 1536, 128
        q_lora = torch.randn((bs, q_lora_rank)) / 10
        q_lora = tile_kernels.quant.per_token_cast(
            q_lora, 'e4m3', gran, None, use_tma_aligned_col_major_sf=True,
            round_sf=True, use_packed_ue8m0=True)
        q_b_proj, q_b_proj_perm = official.get_q_b_weight(h_q, 512, q_lora_rank, gran)
        def calc(weight):
            q = torch.empty((bs, h_q*512), dtype=torch.bfloat16)
            fp8_gemm_nt_reference(q_lora, weight, q, gran)
            return q.view(bs, h_q, 512)
        q = calc(q_b_proj_perm)
    else:
        q = t.q.reshape(bs, h_q, 512).to(torch.bfloat16).contiguous()

    def run(fp8):
        return flash_mla.fused_norm_rope_attn_rope_cast.decode(
            enable_q_norm=False, rms_norm_eps=0.0, token_positions=tp,
            is_rope_neox_style=False, rope_dim=64, cos_sin_cache=cs, n_wv_group=nwg,
            num_per_channels=32, use_tma_aligned_col_major_sf=True, round_sf=True,
            use_packed_ue8m0=True, q=q, k_cache=kv,
            indices_in_kvcache=t.kv_scope.indices_in_kvcache.reshape(bs, p.topk),
            sm_scale=t.sm_scale, d_v=p.d_v, attn_sink=t.attn_sink, topk_length=None,
            extra_k_cache=None, extra_indices_in_kvcache=None, extra_topk_length=None,
            fp8_compute=fp8)

    ob = run(False); lib.device_synchronize()
    of = run(True);  lib.device_synchronize()

    def dequant(codes, sf):
        sf2d = sf.view(bs, -1)
        if sf2d.stride(0) != 1:
            sf2d = sf2d.as_strided(sf2d.shape, (1, sf2d.stride(1)))
        o = tile_kernels.quant.per_token_cast_back((codes.view(bs, -1), sf2d), 'fp32', 32)
        return o.view(bs, nwg, 512 // 32, 8, 32).transpose(2, 3).reshape(bs, h_q, 512)

    o_b = dequant(ob[0], ob[1]); o_f = dequant(of[0], of[1])
    raw_rel = (of[0].view(bs,-1).float() - ob[0].view(bs,-1).float()).abs().mean().item() / \
              ob[0].view(bs,-1).float().abs().mean().item()
    deq_rel = (o_f - o_b).abs().mean().item() / o_b.abs().mean().item()
    lse_rel = (of[2] - ob[2]).abs().mean().item() / ob[2].abs().mean().item()
    sf_diff = (ob[1].view(bs,-1) != of[1].view(bs,-1)).sum().item()
    print(f"s_q={s_q:<3} topk={topk:<5} s_kv={s_kv:<5} q={q_mode:<9} rope={rope_mode:<5} "
          f"| raw {raw_rel:8.3%}  deq {deq_rel:8.3%}  lse {lse_rel:8.4%}  sfdiff {sf_diff}")

print("== shape sweep, standard q, zero rope (probe style) ==")
run_config(4, 128, 2048, "standard", "zero")
run_config(1, 64, 64, "standard", "zero")
print("== shape sweep, official q, real rope (test path) ==")
run_config(1, 64, 64, "official", "real")
run_config(4, 128, 2048, "official", "real")
print("== cross: official q zero rope / standard q real rope ==")
run_config(1, 64, 64, "official", "zero")
run_config(1, 64, 64, "standard", "real")
run_config(4, 128, 2048, "official", "zero")
run_config(4, 128, 2048, "standard", "real")
