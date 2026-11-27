#!/usr/bin/env python3
"""Verify predicate corner cells for defect-E boundary."""
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

h_q = 64
nwg = h_q // 8
cs = torch.zeros(131072, 64, device="cuda", dtype=torch.float32)

def run_config(b, s_q, topk, s_kv, seed=1234):
    p = RawTestParamForDecode(b, h_q, s_q, 1, s_kv, False, topk,
        is_all_indices_invalid=False, have_zero_seqlen_k=False, have_topk_length=False,
        enable_attn_sink=True, extra_s_k=None, extra_topk=None,
        block_size=(64+topk%8), extra_block_size=None, have_extra_topk_length=False,
        d_qk=512, kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse,
        extra_kvcache_layout=None, num_runs=0).to_test_param()
    p.seed = seed
    t = lib.generate_testcase_for_decode(p)
    bs = b * s_q
    tp = torch.zeros(bs, dtype=torch.int32)
    kv = t.kv_scope.get_kvcache_for_flash_mla()
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
    deq_rel = (o_f - o_b).abs().mean().item() / max(o_b.abs().mean().item(), 1e-12)
    lse_rel = (of[2] - ob[2]).abs().mean().item() / ob[2].abs().mean().item()
    verdict = "GREEN" if deq_rel < 0.05 else "BROKEN"
    print(f"b={b:<3} s_q={s_q:<3} topk={topk:<5} s_kv={s_kv:<5} | deq {deq_rel:9.3%}  lse {lse_rel:8.4%}  -> {verdict}")
    return deq_rel

print("== predicate corner cells ==")
run_config(1, 5, 512, 2048)    # b*s_q=5, topk=512  -> expect GREEN
run_config(1, 6, 120, 592)     # b*s_q=6, topk=120  -> expect BROKEN
run_config(5, 1, 128, 2048)    # b*s_q=5             -> expect GREEN
run_config(6, 1, 128, 2048)    # b*s_q=6             -> expect BROKEN
run_config(1, 1, 72, 2048)     # topk=72 lowest bs   -> expect GREEN
run_config(1, 4, 32, 2048)     # topk=32 (off-matrix, for the record)
run_config(2, 3, 128, 2048)    # b*s_q=6 via b=2     -> expect BROKEN
