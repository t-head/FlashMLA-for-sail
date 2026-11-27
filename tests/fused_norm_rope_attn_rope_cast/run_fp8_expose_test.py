#!/usr/bin/env python3
"""FP8-expose unit test for ``fused_norm_rope_attn_rope_cast.decode`` .

WHY THIS TEST EXISTS
--------------------
Every kernel probe in earlier rounds passed ``extra=None`` to the decode
kernel -- and, unknown until this task, only ever ran ONE shape
(b*s_q=4, topk=128).  End-to-end, the FP8 arm garbles every token after the
first.  This test closes both blind spots: the *same* testcase is run through
``flash_mla.fused_norm_rope_attn_rope_cast.decode`` twice -- with
``fp8_compute=False`` (BF16 arm) and ``fp8_compute=True`` (FP8 arm) -- and the
arms are compared across the official correctness-scan shape matrix,
including FP4 extra-KV combinations that no probe ever exercised.

KNOWN DEFECT MAP OF THE PRE-FIX WHEEL (the pre-fix wheel: .so sha256 c38e0245...,
measured by this task's diagnostics, extra=None unless noted):

  * defect D (the FP4-extra target): FP4 extra KV cache -> fp8 arm diverges
    wildly (O 9.5e4..1.1e6 %, lse 86..378 %, measured over the 214 broken
    cells of the 216-cell full sweep; the two remaining cells come out
    bitwise identical between arms) while the BF16 arm stays correct.
    Never covered before: all probes passed extra=None.
  * defect E (NEW, found by this task): extra=None fp8 arm diverges
    (O ~93..108 %, lse stays perfect at <= 0.004 %) iff topk < 72 or
    b*s_q >= 6.  The the swizzle-fix round probe shape (b*s_q=4, topk=128) sits inside the
    green island (b*s_q <= 5 and topk >= 72), which is why it looked fixed.
    Reported to the FP4-extra fix round; NOT fixed by this task (coordination boundary).

TWO-LEVEL CRITERIA (fixed BEFORE the first full run; do not tune after)
-----------------------------------------------------------------------
Main criterion -- arm-vs-arm (same testcase, same inputs):
    O   : after dequantizing (out_fp8 * sf expanded at num_per_channels=32),
          mean(|O_fp8 - O_bf16|) / mean(|O_bf16|)  <  ARM_O_REL_TOL
          (the swizzle-fix probe_layout convention; the green island measures ~2%,
          so 5% leaves quantization-level headroom)
    lse : mean(|lse_fp8 - lse_bf16|) / mean(|lse_bf16|) over entries finite
          in both arms  <  ARM_LSE_REL_TOL; any inf-mismatch between arms
          fails the case.
Calibration group -- BF16 arm vs the official pure-torch reference, judged
with the *official* criteria from test-fused-norm-rope-attn-rope-cast.py
(out: abs_tol 1.1e-3 / rel_tol 1.01/8 / cos_diff_tol 1e-3;  lse: 1e-5 /
4.01/65536).  If this group fails, the harness is broken -- fix the harness
before reading anything into the other tiers.

GROUP SEMANTICS
---------------
  cal      calibration: BF16 arm vs official reference (every case)
  tier 2   known-green:  extra=None, inside the swizzle-fix green island
           (topk >= 72 and b*s_q <= 5)
  tier 2b  defect-E cells: extra=None, outside the green island
           (topk < 72 or b*s_q >= 6) -- expected BROKEN on the pre-fix wheel
  tier 3   defect-D cells: FP4 extra KV cache -- expected BROKEN pre-fix

  --mode expose     (default) exit 0 iff cal AND tier 2 are fully green.
                    tier 2b / tier 3 are expected to FAIL on a pre-fix wheel
                    ("defects exposed"); if a group is green instead, the run
                    is flagged (fix already in, or the defect is not
                    reproducible at unit level -- record and notify the FP4-extra owners;
                    both are valid reportable outcomes per the task contract).
  --mode regression (for post-the FP4-extra fix round reruns) exit 0 iff EVERY case is green
                    ("all green" is part of the FP4-extra acceptance).

USAGE
-----
    # in the container, from tests/fused_norm_rope_attn_rope_cast/ :
    python run_fp8_expose_test.py                 # default subset (<10 min)
    python run_fp8_expose_test.py --full          # official-shape full sweep
    python run_fp8_expose_test.py --limit 2        # smoke
    python run_fp8_expose_test.py --mode regression   # post-fix gate

Both arms run the SAME testcase, hence a single h_q=64 (the pre-fix wheel's
fp8_compute only supports the H64 decode path -- host assert in
flash_mla_cuda; an h_q=128 fp8 run dies before any kernel work).

The runner installs the ``tests/_compat`` tile_kernels shim exactly like
``run_official_test_with_shim.py`` (see the shim round) and then *imports the
official test module* to reuse its reference implementation, quant helpers
and ``kk.check_is_allclose`` criteria verbatim -- no duplicated criteria
code.  Run ``_compat/selfcheck_tile_kernels_shim.py`` once before trusting
results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parent
COMPAT_DIR = TESTS_DIR / "_compat"
OFFICIAL = HERE / "test-fused-norm-rope-attn-rope-cast.py"

# --------------------------------------------------------------------------- #
# Criteria constants -- WRITTEN BEFORE THE FIRST FULL RUN.  Do not adjust
# after seeing results (SAFETY rule 9 / task discipline).
# --------------------------------------------------------------------------- #
ARM_O_REL_TOL = 0.05        # fp8-vs-bf16 O  : mean(|dO|)/mean(|O_bf16|), the swizzle-fix round convention
ARM_LSE_REL_TOL = 0.05       # fp8-vs-bf16 lse: mean(|dL|)/mean(|L_bf16|), finite entries

# Calibration group == the official decode criteria, verbatim numbers
CAL_OUT_ABS_TOL = 1.1e-3
CAL_OUT_REL_TOL = 1.01 / 8
CAL_OUT_COS_DIFF_TOL = 1e-3
CAL_LSE_ABS_TOL = 1e-5
CAL_LSE_REL_TOL = 4.01 / 65536

# Defect-E boundary, measured on the pre-fix wheel (diag3/4/5):
# extra=None fp8-arm O diverges iff topk < 72 or b*s_q >= 6 (green island:
# topk >= 72 and b*s_q <= 5; corner cells verified both sides of each edge).
DEFECT_E_MIN_GREEN_TOPK = 72
DEFECT_E_MAX_GREEN_BS = 5

# Case matrix (mirrors the official correctness scan, test:455-535)
#   main-cache (s_kv, topk) pairs, straight from the official scan list
MAIN_SHAPES_FULL = [
    (64, 64), (128, 128), (256, 256), (512, 512),          # regular
    (592, 120), (1840, 240), (1521, 600), (3412, 2896),    # irregular
    (95, 152), (153, 264), (2345, 5136),                   # irregular + OOB topk
    (32, 2048),                                            # blocks fully invalid
]
MAIN_SHAPES_DEFAULT = [(64, 64), (512, 512), (592, 120), (32, 2048)]
DECODE_SHAPES_DEFAULT = [(1, 1), (69, 3)]          # (b, s_q); --full adds (19, 80)
DECODE_SHAPES_FULL = [(1, 1), (69, 3), (19, 80)]
#   extra-cache combos: (extra_s_kv, extra_topk, extra_kvcache_layout)
EXTRA_COMBOS_DEFAULT = [
    (None, None, None),                       # control: extra=None (known green island)
    (32, 2048, "FP4"),                         # exposure: defect D
    (512, 512, "FP4"),                         # exposure: defect D
]
EXTRA_COMBOS_FULL = EXTRA_COMBOS_DEFAULT + [(3412, 2896, "FP4")]

SEED_BASE = 10_000             # p.seed = SEED_BASE + case index (deterministic)


# --------------------------------------------------------------------------- #
# shim installation (same pattern as run_official_test_with_shim.py, the shim round)
# --------------------------------------------------------------------------- #
def _install_shim() -> str:
    shim_path = COMPAT_DIR / "tile_kernels.py"
    real = importlib.util.find_spec("tile_kernels") if str(COMPAT_DIR) not in sys.path else None
    if real is not None and real.origin and Path(real.origin) != shim_path:
        import tile_kernels as _tk  # noqa: F401
        return f"real tile_kernels {getattr(_tk, '__file__', '?')}"
    spec = importlib.util.spec_from_file_location("tile_kernels", shim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shim from {shim_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tile_kernels"] = module
    spec.loader.exec_module(module)
    for sub in ("quant", "utils", "torch"):
        sys.modules[f"tile_kernels.{sub}"] = getattr(module, sub)
    module.__path__ = []
    return f"LOCAL SHIM {shim_path}"


def _load_official_module():
    """Import the official test module to reuse its reference code verbatim."""
    spec = importlib.util.spec_from_file_location("official_fused_test", OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_fused_test"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# case construction
# --------------------------------------------------------------------------- #
def build_cases(args):
    main_shapes = MAIN_SHAPES_FULL if args.full else MAIN_SHAPES_DEFAULT
    decode_shapes = DECODE_SHAPES_FULL if args.full else DECODE_SHAPES_DEFAULT
    extra_combos = EXTRA_COMBOS_FULL if args.full else EXTRA_COMBOS_DEFAULT

    cases = []
    for h_q in args.h_q:
        for (s_kv, topk) in main_shapes:
            for (b, s_q) in decode_shapes:
                for (extra_s_kv, extra_topk, extra_layout) in extra_combos:
                    for have_extra_topk_length in (False, True):
                        if have_extra_topk_length and extra_s_kv is None:
                            continue
                        # The fp4 extra KV cache requires topk % 8 == 0 (official filter)
                        if extra_layout == "FP4" and topk % 8 != 0:
                            continue
                        cases.append(dict(
                            h_q=h_q, b=b, s_q=s_q, s_kv=s_kv, topk=topk,
                            extra_s_kv=extra_s_kv, extra_topk=extra_topk,
                            extra_layout=extra_layout,
                            have_extra_topk_length=have_extra_topk_length,
                        ))
    # assign deterministic seeds
    for i, c in enumerate(cases):
        c["seed"] = SEED_BASE + i
    return cases


def case_to_raw_param(c, quant_mod):
    from fused_norm_rope_attn_rope_cast.lib import RawTestParamForDecode
    extra_layout = None if c["extra_layout"] is None else quant_mod.KVCacheLayout.V41_FP4Sparse
    return RawTestParamForDecode(
        c["b"], c["h_q"], c["s_q"], 1, c["s_kv"], False, c["topk"],
        is_all_indices_invalid=False,
        have_zero_seqlen_k=False,
        have_topk_length=False,
        enable_attn_sink=True,
        extra_s_k=c["extra_s_kv"],
        extra_topk=c["extra_topk"],
        block_size=(64 + c["topk"] % 8),
        extra_block_size=(128 + (c["extra_topk"] % 8) if c["extra_topk"] is not None else None),
        have_extra_topk_length=c["have_extra_topk_length"],
        d_qk=512,
        kvcache_layout=quant_mod.KVCacheLayout.V41_FP8Sparse,
        extra_kvcache_layout=extra_layout,
        num_runs=0,
        seed=c["seed"],
    )


def case_tag(c) -> str:
    ex = "None" if c["extra_layout"] is None else \
        f"({c['extra_s_kv']},{c['extra_topk']},FP4)"
    return (f"(b,s_q)=({c['b']},{c['s_q']}) main=({c['s_kv']},{c['topk']}) "
            f"h_q={c['h_q']} extra={ex} exlen={int(c['have_extra_topk_length'])}")


def classify(c) -> str:
    """Tier of a case.

    tier2  : extra=None inside the swizzle-fix green island -- fp8 arm must match bf16
             even on the pre-fix wheel (reproduces the swizzle-fix baseline).
    defectE: extra=None outside the island (topk < 72 or b*s_q >= 6) -- the
             NEW extra=None divergence found by this task.
    expose : FP4 extra KV cache -- defect D, the FP4-extra target.
    """
    if c["extra_layout"] is not None:
        return "expose"
    if c["topk"] < DEFECT_E_MIN_GREEN_TOPK or c["b"] * c["s_q"] > DEFECT_E_MAX_GREEN_BS:
        return "defectE"
    return "tier2"


# --------------------------------------------------------------------------- #
# per-case execution
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_case(official, lib, quant, kk, tile_kernels, flash_mla, p, c, verify_det=True):
    """Run one testcase through both arms + official reference.

    Returns a dict of metrics; raises on infrastructure errors (BF16 arm /
    reference broken == harness problem).
    """
    from fused_norm_rope_attn_rope_cast.projection_reference import fp8_gemm_nt_reference

    t = lib.generate_testcase_for_decode(p)
    b, s_q = p.decode.b, p.s_q

    # --- deterministic q construction (official path; fixed choices) ------- #
    q_lora_rank = 1536
    q_b_proj_scale_gran = 128
    q_lora = torch.randn((b * s_q, q_lora_rank)) / 10
    q_lora = tile_kernels.quant.per_token_cast(
        q_lora, 'e4m3', q_b_proj_scale_gran, None,
        use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
    q_b_proj, q_b_proj_permuted = official.get_q_b_weight(
        p.h_q, p.d_qk, q_lora_rank, q_b_proj_scale_gran)

    def calculate_q_b_proj(weight):
        q = torch.empty((b * s_q, p.h_q * p.d_qk), dtype=torch.bfloat16)
        fp8_gemm_nt_reference(q_lora, weight, q, q_b_proj_scale_gran)
        return q.view(b * s_q, p.h_q, p.d_qk)

    q_for_fused = calculate_q_b_proj(q_b_proj_permuted)
    token_positions = torch.randint(0, official.max_token_position, (b * s_q,), dtype=torch.int32)

    n_wv_group = p.h_q // 8
    o_scale_gran = 32            # per-32 packed ue8m0 output scales ("block32")
    kv_cache = t.kv_scope.get_kvcache_for_flash_mla()
    indices = t.kv_scope.indices_in_kvcache.reshape(b * s_q, p.topk)
    topk_length = t.kv_scope.topk_length.repeat_interleave(s_q, 0) \
        if t.kv_scope.topk_length is not None else None
    if t.extra_kv_scope is not None:
        extra_kv_cache = t.extra_kv_scope.get_kvcache_for_flash_mla()
        extra_indices = t.extra_kv_scope.indices_in_kvcache.reshape(b * s_q, p.decode.extra_topk)
        extra_topk_length = t.extra_kv_scope.topk_length.repeat_interleave(s_q, 0) \
            if t.extra_kv_scope.topk_length is not None else None
    else:
        extra_kv_cache = extra_indices = extra_topk_length = None

    def run_arm(fp8_compute):
        return flash_mla.fused_norm_rope_attn_rope_cast.decode(
            enable_q_norm=False,
            rms_norm_eps=0.0,
            token_positions=token_positions,
            is_rope_neox_style=False,
            rope_dim=64,
            cos_sin_cache=official.cos_sin_cache,
            n_wv_group=n_wv_group,
            num_per_channels=o_scale_gran,
            use_tma_aligned_col_major_sf=True,
            round_sf=True,
            use_packed_ue8m0=True,
            q=q_for_fused,
            k_cache=kv_cache,
            indices_in_kvcache=indices,
            sm_scale=t.sm_scale,
            d_v=p.d_v,
            attn_sink=t.attn_sink,
            topk_length=topk_length,
            extra_k_cache=extra_kv_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_length,
            fp8_compute=fp8_compute,
        )

    # --- dequantize O exactly like the official test ---------------------- #
    def dequant_out(out_fp8, out_sf):
        out_sf_2d = out_sf.view(b * s_q, -1)
        if o_scale_gran == 32 and out_sf_2d.stride(0) != 1:
            assert s_q == 1
            out_sf_2d = out_sf_2d.as_strided(out_sf_2d.shape, (1, out_sf_2d.stride(1)))
        o = tile_kernels.quant.per_token_cast_back(
            (out_fp8.view(b * s_q, -1), out_sf_2d), 'fp32', o_scale_gran)
        return o.view(b * s_q, n_wv_group, p.d_v // 32, 8, 32) \
            .transpose(2, 3).reshape(b * s_q, p.h_q, p.d_v)

    # --- stage 1: BF16 arm (+ determinism rerun) --------------------------- #
    lib.device_synchronize()
    out_b = run_arm(False)
    lib.device_synchronize()
    determinism_ok = True
    if verify_det:
        out_b2 = run_arm(False)
        lib.device_synchronize()
        for x, y in zip(out_b, out_b2):
            if not torch.equal(x, y):
                determinism_ok = False
    o_bf16 = dequant_out(out_b[0], out_b[1])
    lse_bf16 = out_b[2]

    # --- stage 2: calibration -- BF16 arm vs official reference ------------ #
    t.q = calculate_q_b_proj(q_b_proj).view(b, s_q, p.h_q, p.d_qk)
    ref_out, ref_lse = official.ref_fused_norm_rope_attn_rope_cast_decode(
        p, t, False, official.cos_sin_cache, token_positions)
    cal_out_ok = bool(kk.check_is_allclose(
        "out", o_bf16, ref_out.to(torch.float32),
        abs_tol=CAL_OUT_ABS_TOL, rel_tol=CAL_OUT_REL_TOL, cos_diff_tol=CAL_OUT_COS_DIFF_TOL))
    cal_lse_ok = bool(kk.check_is_allclose(
        "lse", lse_bf16, ref_lse,
        abs_tol=CAL_LSE_ABS_TOL, rel_tol=CAL_LSE_REL_TOL))

    # --- stage 3: FP8 arm (+ determinism rerun) and arm-vs-arm metrics ----- #
    # The FP8 arm crashing is itself an exposure datum (the smem overflow behind
    # defect D may surface as a device-side assert), so catch it separately --
    # unlike stages 1/2, whose failures mean the harness is broken.
    fp8_error = None
    try:
        out_f = run_arm(True)
        lib.device_synchronize()
        if verify_det:
            out_f2 = run_arm(True)
            lib.device_synchronize()
            for x, y in zip(out_f, out_f2):
                if not torch.equal(x, y):
                    determinism_ok = False
        o_fp8 = dequant_out(out_f[0], out_f[1])
        lse_fp8 = out_f[2]

        o_abs_mean = o_bf16.abs().mean().item()
        o_diff_mean = (o_fp8 - o_bf16).abs().mean().item()
        o_rel = o_diff_mean / max(o_abs_mean, 1e-12)
        o_max_diff = (o_fp8 - o_bf16).abs().max().item()

        fin_mask = torch.isfinite(lse_bf16) & torch.isfinite(lse_fp8)
        inf_mismatch = int((torch.isfinite(lse_bf16) != torch.isfinite(lse_fp8)).sum().item())
        if fin_mask.any():
            lse_rel = (lse_fp8[fin_mask] - lse_bf16[fin_mask]).abs().mean().item() / \
                max(lse_bf16[fin_mask].abs().mean().item(), 1e-12)
        else:
            lse_rel = float("nan")
        arm_ok = (o_rel < ARM_O_REL_TOL) and (lse_rel < ARM_LSE_REL_TOL) and inf_mismatch == 0
    except Exception as e:                                       # noqa: BLE001
        fp8_error = f"{type(e).__name__}: {e}"
        o_rel = o_max_diff = lse_rel = float("inf")
        o_abs_mean = 0.0
        inf_mismatch = -1
        arm_ok = False

    return dict(
        o_rel=o_rel, o_max_diff=o_max_diff, o_abs_mean=o_abs_mean,
        lse_rel=lse_rel, inf_mismatch=inf_mismatch,
        arm_ok=arm_ok, fp8_error=fp8_error,
        cal_out_ok=cal_out_ok, cal_lse_ok=cal_lse_ok,
        determinism_ok=determinism_ok,
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--full", action="store_true",
                        help="full sweep: all 12 official (s_kv, topk) shapes, "
                             "(19,80) decode shape and the (3412,2896,FP4) extra combo")
    parser.add_argument("--h-q", type=int, nargs="+", choices=[64, 128], default=[64],
                        help="head counts. Default 64: the pre-fix wheel's host code "
                             "asserts 'fp8_compute requires the H64 five-in-one decode "
                             "path', and the arm-vs-arm comparison needs one h_q for both "
                             "arms. h_q=128 can be given for bf16-side calibration only.")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N cases (smoke)")
    parser.add_argument("--mode", choices=("expose", "regression"), default="expose",
                        help="expose: exit 0 iff cal+tier2 green (defectE/expose reported); "
                             "regression: exit 0 iff everything green (post-the FP4-extra fix round gate)")
    parser.add_argument("--no-verify-determinism", action="store_true",
                        help="skip the bitwise rerun check (not recommended)")
    args = parser.parse_args()
    # NOTE --full stays at h_q=64: the pre-fix wheel's fp8_compute only supports the
    # H64 decode path (host assert in flash_mla_cuda), so an h_q=128 sweep would turn
    # every fp8-arm run into a capability error, not a defect measurement.

    for p_ in (str(TESTS_DIR), str(COMPAT_DIR)):
        if p_ not in sys.path:
            sys.path.insert(0, p_)
    origin = _install_shim()

    assert torch.cuda.is_available(), "This test requires a CUDA-class device"
    default_device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(default_device)
    torch.cuda.set_device(default_device)
    torch.set_float32_matmul_precision('high')

    import tile_kernels
    import kernelkit as kk
    import flash_mla
    from fused_norm_rope_attn_rope_cast import lib, quant

    # wheel fingerprint -- disambiguates pre-fix vs post-fix wheels
    so_path = Path(flash_mla.fused_norm_rope_attn_rope_cast.__file__)
    for cand in so_path.parent.parent.glob("flash_mla_cuda*.so"):
        so_path = cand
    so_sha = hashlib.sha256(so_path.read_bytes()).hexdigest()

    official = _load_official_module()

    cases = build_cases(args)
    if args.limit:
        cases = cases[:args.limit]

    print("=" * 100)
    print("FP8-expose unit test for fused_norm_rope_attn_rope_cast.decode (the exposure round)")
    print(f"  tile_kernels source : {origin}")
    print(f"  flash_mla version   : {flash_mla.__version__}   (so sha256 {so_sha[:16]}...)")
    print(f"  mode                : {args.mode}   full={args.full}   h_q={args.h_q}")
    print(f"  thresholds (fixed)  : arm O rel < {ARM_O_REL_TOL:.0%}, arm lse rel < {ARM_LSE_REL_TOL:.0%}; "
          f"cal out {CAL_OUT_ABS_TOL}/{CAL_OUT_REL_TOL:.4f}, cal lse {CAL_LSE_ABS_TOL}/{CAL_LSE_REL_TOL:.2e}")
    print(f"  defect-E boundary   : extra=None green island = topk >= {DEFECT_E_MIN_GREEN_TOPK} "
          f"and b*s_q <= {DEFECT_E_MAX_GREEN_BS} (measured on the pre-fix wheel)")
    print(f"  cases               : {len(cases)} "
          f"(determinism rerun {'ON' if not args.no_verify_determinism else 'OFF'})")
    print("=" * 100)

    tier = {"cal": [0, 0], "tier2": [0, 0], "defectE": [0, 0], "expose": [0, 0]}  # [pass, total]
    determinism_failures = 0
    crashed_cases = 0
    poisoned_skips = 0
    context_suspect = False        # set once an fp8-arm run has failed
    t_start = time.time()

    for idx, c in enumerate(cases):
        p = case_to_raw_param(c, quant).to_test_param()
        group = classify(c)
        tag = case_tag(c)
        print(f"[{idx + 1:4d}/{len(cases)}] {tag}  [{group}]")
        try:
            m = run_case(official, lib, quant, kk, tile_kernels, flash_mla, p, c,
                         verify_det=not args.no_verify_determinism)
        except Exception as e:                                   # noqa: BLE001
            if context_suspect:
                # A previous fp8-arm failure likely poisoned the device context
                # (defect D's smem overflow can escalate to a device-side
                # assert); do not blame the harness for what follows it.
                poisoned_skips += 1
                print(f"    CRASH after an earlier fp8-arm failure (context likely "
                      f"poisoned): {type(e).__name__}: {e}  -> skipped in group accounting")
                continue
            crashed_cases += 1
            print(f"    CRASH: {type(e).__name__}: {e}")
            traceback.print_exc()
            tier[group][1] += 1
            tier["cal"][1] += 1
            continue

        if m["fp8_error"] is not None:
            context_suspect = True

        # group bookkeeping
        tier["cal"][1] += 1
        tier["cal"][0] += int(m["cal_out_ok"] and m["cal_lse_ok"])
        tier[group][1] += 1
        tier[group][0] += int(m["arm_ok"])
        if not m["determinism_ok"]:
            determinism_failures += 1

        if m["fp8_error"] is not None:
            print(f"    FP8-ARM CRASH: {m['fp8_error']}")
            print(f"    ARM  -> FAIL (fp8 arm raised)   | CAL out={'PASS' if m['cal_out_ok'] else 'FAIL'} "
                  f"lse={'PASS' if m['cal_lse_ok'] else 'FAIL'}   | det={'OK' if m['determinism_ok'] else 'NONDET'}")
        else:
            verdict = "PASS" if m["arm_ok"] else "FAIL"
            print(f"    ARM  O: {m['o_rel']:12.4%} (max|dO|={m['o_max_diff']:.4f}, |O|={m['o_abs_mean']:.4f})   "
                  f"lse: {m['lse_rel']:10.4%} (infmm={m['inf_mismatch']})  -> {verdict}"
                  f"   | CAL out={'PASS' if m['cal_out_ok'] else 'FAIL'} lse={'PASS' if m['cal_lse_ok'] else 'FAIL'}"
                  f"   | det={'OK' if m['determinism_ok'] else 'NONDET'}")

    elapsed = time.time() - t_start
    print("=" * 100)
    print(f"SUMMARY ({args.mode} mode, wheel so sha256 {so_sha[:16]}..., {elapsed:.1f}s, "
          f"{crashed_cases} crashed, {poisoned_skips} context-poisoned skips, "
          f"{determinism_failures} nondeterministic)")
    print(f"  cal      calibration (bf16 arm vs official ref, official criteria): "
          f"{tier['cal'][0]}/{tier['cal'][1]} PASS")
    print(f"  tier 2   known-green  (extra=None, green island):                    "
          f"{tier['tier2'][0]}/{tier['tier2'][1]} PASS  [arm O<{ARM_O_REL_TOL:.0%}, lse<{ARM_LSE_REL_TOL:.0%}]")
    print(f"  tier 2b  defect-E     (extra=None, topk<{DEFECT_E_MIN_GREEN_TOPK} or b*s_q>{DEFECT_E_MAX_GREEN_BS}):        "
          f"{tier['defectE'][0]}/{tier['defectE'][1]} PASS")
    print(f"  tier 3   defect-D     (FP4 extra KV cache):                          "
          f"{tier['expose'][0]}/{tier['expose'][1]} PASS")

    cal_ok = tier["cal"][0] == tier["cal"][1] and tier["cal"][1] > 0
    green_ok = tier["tier2"][0] == tier["tier2"][1] and tier["tier2"][1] > 0
    e_broken = tier["defectE"][1] > 0 and tier["defectE"][0] < tier["defectE"][1]
    e_green = tier["defectE"][1] > 0 and tier["defectE"][0] == tier["defectE"][1]
    d_broken = tier["expose"][1] > 0 and tier["expose"][0] < tier["expose"][1]
    d_green = tier["expose"][1] > 0 and tier["expose"][0] == tier["expose"][1]

    if determinism_failures:
        print(f"\n  !! {determinism_failures} cases produced NON-DETERMINISTIC output; "
              f"readings are untrustworthy (check wheel install / environment).")
    if crashed_cases:
        print(f"\n  !! {crashed_cases} cases CRASHED before calibration (counted as failures).")
    if poisoned_skips:
        print(f"\n  !! {poisoned_skips} cases were skipped after an fp8-arm failure poisoned "
              f"the device context; their groups are under-counted.")

    if not cal_ok:
        print("\n  CALIBRATION FAILED: the harness itself is broken (bf16 arm vs official")
        print("  reference).  Do NOT read anything into the other groups until fixed.")
        return 1
    if not green_ok:
        print("\n  tier 2 FAILED: the swizzle-fix green island (extra=None) regressed.  This is a")
        print("  real failure independent of defects D/E.")
        return 1

    if args.mode == "regression":
        if e_green and d_green:
            print("\n  ALL GREEN (green island + defect-E cells + defect-D cells):")
            print("  regression gate PASSED.")
            return 0
        if not d_green:
            print("\n  regression gate FAILED: FP4-extra (defect D) cases still diverge.")
        if not e_green:
            print("\n  regression gate FAILED: extra=None defect-E cells still diverge.")
        return 1

    # expose mode: report each expected-broken group
    print()
    if d_broken:
        print("  DEFECT D EXPOSED at unit level: the fp8 arm diverges from the bf16 arm on")
        print("  FP4-extra cases (see the (1,1) rows with topk>=72 for cells unconfounded")
        print("  by defect E) -- exactly the e2e blind spot the FP4-extra fix round is fixing.")
    elif d_green:
        print("  defect-D cells are GREEN on this wheel. Either the FP4-extra fix round's fix is already")
        print("  in (check the so sha256 above against the pre-fix c38e0245... baseline),")
        print("  or defect D is NOT reproducible at unit level -- record it and notify")
        print("  the FP4-extra fix round (an independent experiment re-checks).")
    if e_broken:
        print("  DEFECT E EXPOSED at unit level (NEW, found by this test): the fp8 arm")
        print("  diverges on extra=None cells outside the green island (topk < 72 or")
        print("  b*s_q >= 6) while lse stays perfect. the swizzle-fix round's single-shape probe sat")
        print("  inside the island. Reported to the FP4-extra fix round -- not fixed by this task.")
    elif e_green:
        print("  defect-E cells are GREEN on this wheel (fixed, or the boundary moved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
