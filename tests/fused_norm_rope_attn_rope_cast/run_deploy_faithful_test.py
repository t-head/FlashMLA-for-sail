#!/usr/bin/env python3
"""Deployment-faithful fp8-expose unit test for ``fused_norm_rope_attn_rope_cast.decode``
.

WHY THIS TEST EXISTS
--------------------
An exposure test was built (``run_fp8_expose_test.py``) that separates the
pre-fix wheel's two defects at the kernel level:

  * defect D -- FP4 extra KV cache: the fp8 arm diverges wildly (O ~1e5..1e6 %)
  * defect E -- extra=None: the fp8 arm diverges (O ~93..108 %, lse perfect)
                iff ``topk < 72`` or ``b*s_q >= 6``

...but that *shape matrix* is not the deployment's shape.  This matrix first
adjudicated the deployment parameters from the sglang source and the model
config (see ``evidence/raw/s52_params.md``); the decisive finding is:

    The mega decode path NEVER clamps ``swa_topk_lengths`` up to ``SWA_WINDOW``.
    The ``clamp_min(SWA_WINDOW)`` at ``deepseek_v4_backend.py:613-615`` writes a
    *different* field (``trtllm_swa_lens``) consumed only by the trtllm path.
    The mega path reads ``core_attn_metadata.swa_topk_lengths``, which comes from
    ``build_page_table_positions`` = ``min(seq_len, SWA_WINDOW)``.

    => for an 8 x ~5-20 token short prompt, the effective attend counts are
       ``swa_topk_lengths`` in 5..20 AND the extra (c1) row length in 1..5.

Both are far below 72, so by that criterion the *real* end-to-end workload sits
squarely in **defect E's region** -- not in the green island.  But that matrix
stops at topk=64, so the 5..40 range was never actually measured.  This script
measures it, with the deployment's own parameters, one dimension at a time.

WHAT "DEPLOYMENT-FAITHFUL" MEANS HERE (each cell is a single-variable deviation
from the previous one, criteria fixed BEFORE the run)
-----------------------------------------------------
  d1  short-context single block : b*s_q=1, s_kv=topk=len (5/8/12/20), extra=None
                                   == the deployment's short-prompt effective
                                   attend count (the row is fully live here)
  d2  topk=512 (the deployment's ``index_topk``, never tested: the earlier matrix maxed at 256)
  d3  extra sized to the deployment: extra_topk=512 (what the indexer selects),
      extra_s_k=512; the ``candidate_topk_blocks=2048`` figure is a *candidate*
      block budget (x block_size 8), NOT the selected count -- see s52_params.md
  d4  output-scale granularity ``num_per_channels``: 32 (the exposure round) vs 4096 (the PPU
      deployment default in ``flash_mla_interface._MEGA_FIXED_PARAMS``)
  d5  the three sibling sf flags of that same PPU default set
      (``use_tma_aligned_col_major_sf`` / ``round_sf`` / ``use_packed_ue8m0``)
  d6  attn_sink: default official random sink vs the deployment-shaped sink
      (all-zero finite => no sink), plus an explicit +/-inf pattern
  d8  PARTIAL index row: s_kv < topk = 512, topk_length = min(s_kv, SWA_WINDOW)
      -- the mega path's real shape, a wide row that is mostly dead.  This is
      the aspect d1 cannot reach (there topk == s_kv, so topk_length is None)

Two candidate differences from the task contract were resolved by code reading
and are therefore NOT variables here (see s52_params.md section 3):

  * ``enable_q_norm``: the sglang mega decode call site
    (``deepseek_v4_mega_backend.py:83-95``) does NOT pass it, so the deployment
    runs the default ``False`` -- identical to the exposure round's cases.  Kept as an explicit
    ``--q-norm`` toggle for completeness, defaulting to off (= deployment).
  * Q layout: ``pack_query`` / ``interleave_query_rows`` exist in the mega module
    but the sglang call site passes ``q`` straight through, so the deployment
    layout is the same standard layout the exposure round uses.

CRITERIA -- FIXED BEFORE THE FIRST RUN (do not tune after; SAFETY rule 9)
------------------------------------------------------------------------
Identical to the exposure round, deliberately: arm-vs-arm (same case, ``fp8_compute`` False vs
True), O after dequantization ``mean(|dO|)/mean(|O_bf16|) < 5%``, lse likewise
``< 5%``, any inf-mismatch fails; BF16 arm vs the official reference judged with
the official criteria verbatim (calibration).  Using the exposure round's numbers unchanged is
what makes "does the deployment shape reproduce defect E?" a like-for-like
comparison rather than a new yardstick.

USAGE
-----
    # in the container, from tests/fused_norm_rope_attn_rope_cast/ :
    python run_deploy_faithful_test.py                 # full deploy matrix
    python run_deploy_faithful_test.py --limit 3       # smoke
    python run_deploy_faithful_test.py --only d1       # a single dimension
    python run_deploy_faithful_test.py --list          # show the matrix

Every case runs both arms and, unless ``--no-verify-determinism``, repeats each
arm once and asserts bitwise equality.
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
# Criteria constants -- WRITTEN BEFORE THE FIRST RUN.  Do not adjust after
# seeing results.  Deliberately identical to the exposure round's so the deployment-shape
# readings are comparable with the exposure-round matrix rather than judged by a new ruler.
# --------------------------------------------------------------------------- #
ARM_O_REL_TOL = 0.05
ARM_LSE_REL_TOL = 0.05

CAL_OUT_ABS_TOL = 1.1e-3
CAL_OUT_REL_TOL = 1.01 / 8
CAL_OUT_COS_DIFF_TOL = 1e-3
CAL_LSE_ABS_TOL = 1e-5
CAL_LSE_REL_TOL = 4.01 / 65536

# the exposure round's measured defect-E boundary, restated here only for labelling the output.
DEFECT_E_MIN_GREEN_TOPK = 72
DEFECT_E_MAX_GREEN_BS = 5        # the exposure round: b*s_q >= 6 is the second defect-E edge

# Deployment constants (sources in evidence/raw/s52_params.md).
DEPLOY_INDEX_TOPK = 512            # config.json text_config.index_topk
DEPLOY_SLIDING_WINDOW = 128        # == SWA_WINDOW (deepseek_v4_backend.py:169)
DEPLOY_CANDIDATE_TOPK_BLOCKS = 2048
DEPLOY_NUM_PER_CHANNELS = 4096     # _MEGA_FIXED_PARAMS on PPU (_is_sail)
DEPLOY_SHORT_PROMPT_LENS = (5, 8, 12, 20)   # 8 short prompts, ~5-20 tokens

SEED_BASE = 520_000


# --------------------------------------------------------------------------- #
# shim installation -- same pattern as run_official_test_with_shim.py (the shim helper)
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
    spec = importlib.util.spec_from_file_location("official_fused_test", OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_fused_test"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# the deployment-faithful matrix
#
# Each entry: (dim, label, case-dict).  ``dim`` groups cells by the ONE variable
# they deviate from the deployment baseline; the runner also prints, for every
# cell, the deployment baseline it is compared against.
# --------------------------------------------------------------------------- #
def _case(**kw) -> dict:
    base = dict(
        h_q=64, b=1, s_q=1, s_kv=20, topk=20,
        extra_s_kv=None, extra_topk=None, extra_layout=None,
        have_extra_topk_length=False,
        num_per_channels=32, tma_sf=True, round_sf=True, packed_ue8m0=True,
        enable_q_norm=False, rms_norm_eps=0.0,
        sink_mode="random",          # official default: randn + sparse +/-inf
        topk_length_mode="auto",     # auto: min(topk, s_kv) -- the deploy shape
        label="",
    )
    base.update(kw)
    return base


def build_cases(args):
    """The deploy matrix.

    ``topk_length_mode``:
      * ``auto``    -- ``clamp(min(topk, s_kv), max=SWA_WINDOW)``: exactly what
                       the mega path's ``build_page_table_positions`` produces for
                       a real short prompt (this is the deployment-faithful bit).
      * ``full``    -- ``topk``: the pre-the deploy-faithful round default (all indices live), kept as
                       an explicit control.
    """
    cases = []

    # ---- d1: short-context single block, extra=None ----------------------- #
    # The deployment's short prompt: b*s_q=1, s_kv=s_q=len, topk=len, no extra.
    #
    # NOTE on what this does and does not exercise (the adversarial review, item 1):
    # here ``topk == s_kv == len``, so ``eff == topk`` and the runner passes
    # ``topk_length=None`` -- i.e. these cells exercise "short context, extra
    # off, whole index row live", NOT a partially-filled row.  That is still
    # exactly the deployment's short-prompt *effective attend count* (the
    # quantity defect E keys on), but the "partial row" aspect is covered by d8
    # rather than here.  The earlier labelling of these cells as
    # ``topk_length = min(seq,128)`` over-claimed; see the module docstring.
    for L in DEPLOY_SHORT_PROMPT_LENS:
        cases.append(("d1_short_ctx_none", _case(
            s_kv=L, topk=L,
            label=f"short-prompt ctx={L}: effective attend count {L} (<72)")))

    # d1 green-island counterpart: short context but a LARGE topk row that the
    # deployment never has for short prompts; shows the difference is the
    # *effective* count, not the row width.
    cases.append(("d1_short_ctx_none", _case(
        s_kv=128, topk=128, label="ctx=128 (island: effective 128 >= 72)")))

    # ---- d8: PARTIAL index row -- the deployment's real shape --------------- #
    # A real short prompt keeps a ``topk = index_topk = 512``-wide index row but
    # only min(ctx, SWA_WINDOW) of it is live.  This is the aspect d1 does not
    # reach (there topk == s_kv), and the one the mega path's
    # ``min(seq_len, SWA_WINDOW)`` actually produces: a WIDE row that is mostly
    # dead.  ``topk_length_mode="auto"`` keeps eff < topk here, so the runner
    # really does pass a partial ``topk_length``.
    for L in (5, 20, 40):
        cases.append(("d8_partial_row", _case(
            s_kv=L, topk=DEPLOY_INDEX_TOPK,
            label=f"partial row: s_kv={L}, topk=512, topk_length={L} "
                  f"(mega shape, effective {L}<72)")))
    # the green-island counterpart of the same aspect: eff = 128 < topk = 512
    cases.append(("d8_partial_row", _case(
        s_kv=512, topk=DEPLOY_INDEX_TOPK,
        label="partial row: s_kv=512, topk=512 -> eff 128 (island)")))

    # ---- d2: topk=512, the deployment's index_topk (never tested) --------- #
    cases.append(("d2_topk512", _case(
        s_kv=512, topk=DEPLOY_INDEX_TOPK, extra_s_kv=None, extra_topk=None,
        label="topk=512 extra=None (b*s_q=1, island by the exposure round predicate)")))
    cases.append(("d2_topk512", _case(
        s_kv=4096, topk=DEPLOY_INDEX_TOPK, extra_s_kv=None, extra_topk=None,
        label="topk=512 extra=None, s_kv=4096 (long context)")))
    # multi-token decode at the deployment topk, inside the exposure round's island predicate
    cases.append(("d2_topk512", _case(
        b=1, s_q=4, s_kv=512, topk=DEPLOY_INDEX_TOPK,
        label="b=1 s_q=4 topk=512 (b*s_q=4<=5, island)")))
    # ... and just outside it
    cases.append(("d2_topk512", _case(
        b=2, s_q=3, s_kv=512, topk=DEPLOY_INDEX_TOPK,
        label="b=2 s_q=3 topk=512 (b*s_q=6>=6, E side)")))

    # ---- d3: extra sized to the deployment (FP4, extra_topk=512) ---------- #
    # deployment: indexer selects index_topk=512 entries out of the candidate
    # blocks -> extra_topk=512, extra_s_k must cover them.
    cases.append(("d3_extra_deploy", _case(
        s_kv=512, topk=DEPLOY_INDEX_TOPK,
        extra_s_kv=512, extra_topk=DEPLOY_INDEX_TOPK, extra_layout="FP4",
        have_extra_topk_length=False,
        label="extra=(512,512,FP4) exlen=0 @ topk=512 (deploy extra scale)")))
    cases.append(("d3_extra_deploy", _case(
        s_kv=512, topk=DEPLOY_INDEX_TOPK,
        extra_s_kv=512, extra_topk=DEPLOY_INDEX_TOPK, extra_layout="FP4",
        have_extra_topk_length=True,
        label="extra=(512,512,FP4) exlen=1 @ topk=512")))
    # deployment-sized extra on the SHORT context: the short prompt selects few
    # extra keys, so extra_topk_length is small while the extra row is 512 wide.
    cases.append(("d3_extra_deploy", _case(
        s_kv=512, topk=DEPLOY_INDEX_TOPK,
        extra_s_kv=512, extra_topk=DEPLOY_INDEX_TOPK, extra_layout="FP4",
        have_extra_topk_length=False, topk_length_mode="auto",
        label="extra=(512,512,FP4) exlen=0 @ topk=512, effective topk=512")))
    # the candidate-block budget itself, for scale reference
    cases.append(("d3_extra_deploy", _case(
        s_kv=DEPLOY_CANDIDATE_TOPK_BLOCKS, topk=DEPLOY_INDEX_TOPK,
        extra_s_kv=DEPLOY_CANDIDATE_TOPK_BLOCKS, extra_topk=DEPLOY_INDEX_TOPK,
        extra_layout="FP4",
        label=f"extra=({DEPLOY_CANDIDATE_TOPK_BLOCKS},512,FP4) @ topk=512 "
              f"(candidate_topk_blocks scale)")))

    # ---- d4: output-scale granularity ------------------------------------- #
    # The wheel documents two supported modes (see the decode docstring):
    #   num_per_channels=32   + tma/round/packed all True   ("per-32 UE8M0")
    #   num_per_channels=4096 + tma/round/packed all False  ("per-token FP32")
    # sglang's ``_MEGA_FIXED_PARAMS`` asks for the *second* on PPU
    # (``4096 if _is_sail``).  the deploy-faithful round's first full run showed the PRE-FIX wheel
    # rejects it outright ("Per-token output requires row-major, unrounded FP32
    # scales" and, for the flags alone, "Expected use_tma_aligned_col_major_sf &&
    # round_sf && use_packed_ue8m0 to be true, but got false") -- i.e. this wheel
    # implements only the per-32 form, so sglang's 4096 request is a *wheel-level*
    # gap, not a shape difference the tests can compare.  Kept here as explicit
    # probes so the rejection is recorded rather than silently assumed.
    cases.append(("d4_num_per_channels", _case(
        s_kv=512, topk=DEPLOY_INDEX_TOPK, num_per_channels=DEPLOY_NUM_PER_CHANNELS,
        tma_sf=False, round_sf=False, packed_ue8m0=False,
        label=f"num_per_channels={DEPLOY_NUM_PER_CHANNELS} + flags False "
              f"(sglang PPU default; wheel may reject), extra=None")))
    cases.append(("d4_num_per_channels", _case(
        s_kv=20, topk=20, num_per_channels=DEPLOY_NUM_PER_CHANNELS,
        tma_sf=False, round_sf=False, packed_ue8m0=False,
        label=f"num_per_channels={DEPLOY_NUM_PER_CHANNELS} + flags False, short ctx")))
    cases.append(("d4_num_per_channels", _case(
        s_kv=512, topk=DEPLOY_INDEX_TOPK, num_per_channels=DEPLOY_NUM_PER_CHANNELS,
        extra_s_kv=512, extra_topk=DEPLOY_INDEX_TOPK, extra_layout="FP4",
        tma_sf=False, round_sf=False, packed_ue8m0=False,
        label=f"num_per_channels={DEPLOY_NUM_PER_CHANNELS} + flags False, extra FP4")))

    # ---- d5: the three sibling sf flags, on their own --------------------- #
    # False flags with num_per_channels still 32 -- an inconsistent pairing, kept
    # to record the wheel's assert.
    for tma, rnd, packed in ((False, False, False),):
        cases.append(("d5_ppu_sf_flags", _case(
            s_kv=512, topk=DEPLOY_INDEX_TOPK, tma_sf=tma, round_sf=rnd,
            packed_ue8m0=packed,
            label=f"tma_sf={tma} round_sf={rnd} packed_ue8m0={packed} "
                  f"with npc=32 (inconsistent pairing; records the wheel assert)")))
    cases.append(("d5_ppu_sf_flags", _case(
        s_kv=20, topk=20, tma_sf=False, round_sf=False, packed_ue8m0=False,
        label="flags False with npc=32, short ctx")))

    # ---- d6: attn_sink variants ------------------------------------------- #
    cases.append(("d6_attn_sink", _case(
        s_kv=20, topk=20, sink_mode="zero",
        label="sink=all-zero finite (deployment-shaped: no sink)")))
    cases.append(("d6_attn_sink", _case(
        s_kv=20, topk=20, sink_mode="finite",
        label="sink=finite random (no +/-inf)")))
    cases.append(("d6_attn_sink", _case(
        s_kv=20, topk=20, sink_mode="inf",
        label="sink=all +inf (DEGENERATE: both arms output 0 -> trivially green, "
              "carries no evidence)")))

    # ---- d7: explicit enable_q_norm toggle (not a deploy difference) ------- #
    # eps must match the official module's global (test-...-cast.py:168 sets
    # rms_norm_eps = 1e-4 and the reference reads that global), otherwise this
    # cell fails *calibration* rather than testing the kernel.
    cases.append(("d7_q_norm", _case(
        s_kv=20, topk=20, enable_q_norm=True, rms_norm_eps=1e-4,
        label="enable_q_norm=True eps=1e-4 (deployment does NOT set this)")))

    # deterministic seeds
    for i, (_, c) in enumerate(cases):
        c["seed"] = SEED_BASE + i
    return cases


# --------------------------------------------------------------------------- #
# case -> official test param
# --------------------------------------------------------------------------- #
def case_to_raw_param(c, quant_mod):
    from fused_norm_rope_attn_rope_cast.lib import RawTestParamForDecode
    extra_layout = None if c["extra_layout"] is None else quant_mod.KVCacheLayout.V41_FP4Sparse
    return RawTestParamForDecode(
        c["b"], c["h_q"], c["s_q"], 1, c["s_kv"], False, c["topk"],
        is_all_indices_invalid=False,
        have_zero_seqlen_k=False,
        # Always True: lib only NaN-poisons the dead KV tail when this is set, and
        # run_case() then overwrites the drawn value with the deployment's
        # min(seq_len, SWA_WINDOW).  See the comment in run_case.
        have_topk_length=True,
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


def effective_topk_length(c) -> int:
    """The effective attend count the mega path would produce for this cell."""
    if c["topk_length_mode"] == "full":
        return c["topk"]
    return min(c["topk"], c["s_kv"], DEPLOY_SLIDING_WINDOW)


def case_tag(c) -> str:
    ex = "None" if c["extra_layout"] is None else f"({c['extra_s_kv']},{c['extra_topk']},FP4)"
    return (f"(b,s_q)=({c['b']},{c['s_q']}) main=({c['s_kv']},{c['topk']}) "
            f"h_q={c['h_q']} extra={ex} exlen={int(c['have_extra_topk_length'])} "
            f"npc={c['num_per_channels']} sf={'T' if c['tma_sf'] else 'F'}"
            f"{'T' if c['round_sf'] else 'F'}{'T' if c['packed_ue8m0'] else 'F'} "
            f"sink={c['sink_mode']} qnorm={int(c['enable_q_norm'])} "
            f"eff_topk={effective_topk_length(c)}")


def classify(c) -> str:
    """Deployment-region label -- the FULL the exposure round predicate, both edges.

    the exposure round measured defect E as ``topk < 72 OR b*s_q >= 6`` (green island =
    ``topk >= 72 AND b*s_q <= 5``).  An earlier version of this script kept only
    the topk edge, which mislabelled the deployment-topk cases that fail on the
    b*s_q edge (e.g. b=2, s_q=3, topk=512) as "island".  Both edges are applied
    here so the region label matches the criterion the reading is judged by.
    """
    if effective_topk_length(c) < DEFECT_E_MIN_GREEN_TOPK:
        return "E-region"
    if c["b"] * c["s_q"] > DEFECT_E_MAX_GREEN_BS:
        return "E-region"
    return "island"


# --------------------------------------------------------------------------- #
# per-case execution (mirrors run_fp8_expose_test.run_case, plus the deploy
# variables: num_per_channels / sf flags / q_norm / sink mode / topk_length)
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_case(official, lib, quant, kk, tile_kernels, flash_mla, p, c, verify_det=True):
    from fused_norm_rope_attn_rope_cast.projection_reference import fp8_gemm_nt_reference

    eff = effective_topk_length(c)

    # Deployment faithfulness for the effective attend count.
    #
    # lib.py draws ``topk_length = torch.randint(0, topk+1, (b,))`` and uses that
    # very tensor to (a) NaN-poison the KV slots its rows no longer reach and
    # (b) drive the reference.  The mega path instead yields
    # ``min(seq_len, SWA_WINDOW)`` (see s52_params.md section 1).  So the drawn
    # value itself has to be the deployment's, otherwise the KV is poisoned
    # against a different length and the bf16 arm cannot match the reference.
    #
    # The patch is scoped by *call signature* and only active while the library
    # builds the case.  A previous attempt patched ``torch.randint`` globally and
    # also swallowed unrelated draws (NaN from case 1) -- hence the strict match
    # below, plus a call counter that lets the draw through if it does not fit.
    _orig_randint = torch.randint

    def _patched_randint(*a, **kw):
        # exactly lib.py:274 -> (low=0, high=topk+1, size=(b,), dtype=int32)
        if (len(a) == 3 and int(a[0]) == 0 and int(a[1]) == c["topk"] + 1
                and isinstance(a[2], tuple) and a[2] == (c["b"],)
                and kw.get("dtype") == torch.int32):
            return torch.full((c["b"],), eff, dtype=torch.int32,
                              device=kw.get("device", torch.get_default_device()))
        return _orig_randint(*a, **kw)

    import fused_norm_rope_attn_rope_cast.lib as _lib_mod
    _lib_torch = _lib_mod.torch
    _lib_mod.torch = type("_T", (), {"__getattr__": staticmethod(
        lambda name: _patched_randint if name == "randint" else getattr(_lib_torch, name))})()
    try:
        t = lib.generate_testcase_for_decode(p)
    finally:
        _lib_mod.torch = _lib_torch
    b, s_q = p.decode.b, p.s_q

    q_lora_rank = 1536
    q_b_proj_scale_gran = 128
    q_lora = torch.randn((b * s_q, q_lora_rank)) / 10
    q_lora = tile_kernels.quant.per_token_cast(
        q_lora, 'e4m3', q_b_proj_scale_gran, None,
        use_tma_aligned_col_major_sf=c["tma_sf"], round_sf=c["round_sf"],
        use_packed_ue8m0=c["packed_ue8m0"])
    q_b_proj, q_b_proj_permuted = official.get_q_b_weight(
        p.h_q, p.d_qk, q_lora_rank, q_b_proj_scale_gran)

    def calculate_q_b_proj(weight):
        q = torch.empty((b * s_q, p.h_q * p.d_qk), dtype=torch.bfloat16)
        fp8_gemm_nt_reference(q_lora, weight, q, q_b_proj_scale_gran)
        return q.view(b * s_q, p.h_q, p.d_qk)

    q_for_fused = calculate_q_b_proj(q_b_proj_permuted)
    token_positions = torch.randint(0, official.max_token_position, (b * s_q,), dtype=torch.int32)

    n_wv_group = p.h_q // 8
    o_scale_gran = c["num_per_channels"]

    kv_cache = t.kv_scope.get_kvcache_for_flash_mla()
    indices = t.kv_scope.indices_in_kvcache.reshape(b * s_q, p.topk)
    if t.extra_kv_scope is not None:
        extra_kv_cache = t.extra_kv_scope.get_kvcache_for_flash_mla()
        extra_indices = t.extra_kv_scope.indices_in_kvcache.reshape(b * s_q, p.decode.extra_topk)
        extra_topk_length = t.extra_kv_scope.topk_length.repeat_interleave(s_q, 0) \
            if t.extra_kv_scope.topk_length is not None else None
    else:
        extra_kv_cache = extra_indices = extra_topk_length = None

    # ---- topk_length: the deployment's effective attend count -------------- #
    # The library already drew ``eff`` (via the patch above) and poisoned the KV
    # consistency-wise; just surface it in the shape the kernel wants.
    if t.kv_scope.topk_length is not None:
        t.kv_scope.topk_length.fill_(eff)
    topk_length = (t.kv_scope.topk_length.repeat_interleave(s_q, 0)
                   if t.kv_scope.topk_length is not None else None)
    if topk_length is not None and eff >= c["topk"]:
        topk_length = None

    # ---- attn_sink variants (set on ``t``: the reference reads t.attn_sink) -- #
    if c["sink_mode"] == "zero":
        t.attn_sink = torch.zeros((p.h_q,), dtype=torch.float32)
    elif c["sink_mode"] == "finite":
        t.attn_sink = torch.randn((p.h_q,), dtype=torch.float32)
    elif c["sink_mode"] == "inf":
        t.attn_sink = torch.full((p.h_q,), float("inf"), dtype=torch.float32)
    # "random" keeps the official mixed +/-inf sink generated by lib.py
    sink = t.attn_sink

    def run_arm(fp8_compute):
        return flash_mla.fused_norm_rope_attn_rope_cast.decode(
            enable_q_norm=c["enable_q_norm"],
            rms_norm_eps=c["rms_norm_eps"],
            token_positions=token_positions,
            is_rope_neox_style=False,
            rope_dim=64,
            cos_sin_cache=official.cos_sin_cache,
            n_wv_group=n_wv_group,
            num_per_channels=o_scale_gran,
            use_tma_aligned_col_major_sf=c["tma_sf"],
            round_sf=c["round_sf"],
            use_packed_ue8m0=c["packed_ue8m0"],
            q=q_for_fused,
            k_cache=kv_cache,
            indices_in_kvcache=indices,
            sm_scale=t.sm_scale,
            d_v=p.d_v,
            attn_sink=sink,
            topk_length=topk_length,
            extra_k_cache=extra_kv_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_length,
            fp8_compute=fp8_compute,
        )

    def dequant_out(out_fp8, out_sf):
        out_sf_2d = out_sf.view(b * s_q, -1)
        if o_scale_gran == 32 and out_sf_2d.stride(0) != 1:
            assert s_q == 1
            out_sf_2d = out_sf_2d.as_strided(out_sf_2d.shape, (1, out_sf_2d.stride(1)))
        o = tile_kernels.quant.per_token_cast_back(
            (out_fp8.view(b * s_q, -1), out_sf_2d), 'fp32', o_scale_gran)
        return o.view(b * s_q, n_wv_group, p.d_v // 32, 8, 32) \
            .transpose(2, 3).reshape(b * s_q, p.h_q, p.d_v)

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

    t.q = calculate_q_b_proj(q_b_proj).view(b, s_q, p.h_q, p.d_qk)
    ref_out, ref_lse = official.ref_fused_norm_rope_attn_rope_cast_decode(
        p, t, c["enable_q_norm"], official.cos_sin_cache, token_positions)
    cal_out_ok = bool(kk.check_is_allclose(
        "out", o_bf16, ref_out.to(torch.float32),
        abs_tol=CAL_OUT_ABS_TOL, rel_tol=CAL_OUT_REL_TOL, cos_diff_tol=CAL_OUT_COS_DIFF_TOL))
    cal_lse_ok = bool(kk.check_is_allclose(
        "lse", lse_bf16, ref_lse,
        abs_tol=CAL_LSE_ABS_TOL, rel_tol=CAL_LSE_REL_TOL))

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
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated dimension prefixes to run (e.g. d1,d2)")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    parser.add_argument("--list", action="store_true", help="print the matrix and exit")
    parser.add_argument("--no-verify-determinism", action="store_true",
                        help="skip the bitwise rerun check (not recommended)")
    args = parser.parse_args()

    cases = build_cases(args)
    if args.only:
        want = tuple(s.strip() for s in args.only.split(",") if s.strip())
        cases = [(d, c) for (d, c) in cases if d.startswith(want)]
    if args.limit:
        cases = cases[:args.limit]

    if args.list:
        for i, (d, c) in enumerate(cases):
            print(f"[{i + 1:3d}] {d:22s} {classify(c):9s} {case_tag(c)}  :: {c['label']}")
        print(f"total {len(cases)} cases")
        return 0

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

    so_path = Path(flash_mla.fused_norm_rope_attn_rope_cast.__file__)
    for cand in so_path.parent.parent.glob("flash_mla_cuda*.so"):
        so_path = cand
    so_sha = hashlib.sha256(so_path.read_bytes()).hexdigest()

    official = _load_official_module()

    print("=" * 110)
    print("Deployment-faithful fp8-expose test for fused_norm_rope_attn_rope_cast.decode (the deploy-faithful round)")
    print(f"  tile_kernels source : {origin}")
    print(f"  flash_mla version   : {flash_mla.__version__}   (so sha256 {so_sha[:16]}...)")
    print(f"  thresholds (fixed)  : arm O rel < {ARM_O_REL_TOL:.0%}, arm lse rel < {ARM_LSE_REL_TOL:.0%}; "
          f"cal out {CAL_OUT_ABS_TOL}/{CAL_OUT_REL_TOL:.4f}, cal lse {CAL_LSE_ABS_TOL}/{CAL_LSE_REL_TOL:.2e}")
    print(f"  deploy constants    : index_topk={DEPLOY_INDEX_TOPK} swa_window={DEPLOY_SLIDING_WINDOW} "
          f"candidate_topk_blocks={DEPLOY_CANDIDATE_TOPK_BLOCKS} num_per_channels={DEPLOY_NUM_PER_CHANNELS}")
    print(f"  E-region predicate  : effective topk < {DEFECT_E_MIN_GREEN_TOPK} (the exposure round boundary)")
    print(f"  cases               : {len(cases)} "
          f"(determinism rerun {'ON' if not args.no_verify_determinism else 'OFF'})")
    print("=" * 110)

    # [pass, total, crashed] -- crashed is tracked separately so the printed
    # denominator never mixes a wheel assert with a numeric verdict (review item 7).
    agg = {"island": [0, 0, 0], "E-region": [0, 0, 0]}
    per_dim = {}
    determinism_failures = 0
    crashed_cases = 0
    poisoned_skips = 0
    context_suspect = False
    t_start = time.time()
    rows = []

    for idx, (dim, c) in enumerate(cases):
        p = case_to_raw_param(c, quant).to_test_param()
        region = classify(c)
        tag = case_tag(c)
        print(f"[{idx + 1:3d}/{len(cases)}] {dim:22s} [{region}] {tag}")
        print(f"        {c['label']}")
        try:
            m = run_case(official, lib, quant, kk, tile_kernels, flash_mla, p, c,
                         verify_det=not args.no_verify_determinism)
        except Exception as e:                                   # noqa: BLE001
            if context_suspect:
                poisoned_skips += 1
                print(f"    CRASH after an earlier fp8-arm failure (context likely "
                      f"poisoned): {type(e).__name__}: {e}  -> skipped in group accounting")
                continue
            crashed_cases += 1
            print(f"    CRASH: {type(e).__name__}: {e}")
            traceback.print_exc()
            agg[region][2] += 1
            per_dim.setdefault(dim, [0, 0, 0])[2] += 1
            continue

        if m["fp8_error"] is not None:
            context_suspect = True
        if not m["determinism_ok"]:
            determinism_failures += 1

        agg[region][1] += 1
        agg[region][0] += int(m["arm_ok"])
        d = per_dim.setdefault(dim, [0, 0, 0])
        d[1] += 1
        d[0] += int(m["arm_ok"])
        rows.append((dim, region, c, m))

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
    print("=" * 110)
    print(f"SUMMARY (deploy mode, wheel so sha256 {so_sha[:16]}..., {elapsed:.1f}s, "
          f"{crashed_cases} crashed, {poisoned_skips} context-poisoned skips, "
          f"{determinism_failures} nondeterministic)")

    cal_tot = sum(1 for r in rows if r[3]["cal_out_ok"] and r[3]["cal_lse_ok"])
    print(f"  cal                 : {cal_tot}/{len(rows)} PASS (bf16 arm vs official ref; "
          f"CRASH cells excluded -- they never reach calibration)")
    for region in ("island", "E-region"):
        p_, t_, cr_ = agg[region]
        extra = f"  (+{cr_} CRASH, excluded from the denominator)" if cr_ else ""
        print(f"  {region:19s}: {p_}/{t_} PASS{extra}")
    print("  by dimension:")
    for dim in sorted(per_dim):
        p_, t_, cr_ = per_dim[dim]
        extra = f"  (+{cr_} CRASH)" if cr_ else ""
        print(f"    {dim:22s}: {p_}/{t_} PASS{extra}")

    if determinism_failures:
        print(f"\n  !! {determinism_failures} cases produced NON-DETERMINISTIC output; "
              f"readings are untrustworthy.")
    if crashed_cases:
        print(f"\n  !! {crashed_cases} cases CRASHED before calibration (counted as failures).")
    if poisoned_skips:
        print(f"\n  !! {poisoned_skips} cases skipped after an fp8-arm failure poisoned the context.")

    if cal_tot != len(rows):
        print("\n  CALIBRATION FAILED: the harness itself is broken (bf16 arm vs official")
        print("  reference).  Do NOT read anything into the other groups until fixed.")
        return 1

    e_p, e_t, _ = agg["E-region"]
    i_p, i_t, _ = agg["island"]
    # Distinguish a real numeric divergence from a crash and from the degenerate
    # all-inf-sink cell (both arms bitwise zero).  Counting either as a
    # "diverging cell" would overstate the result.
    e_crash = agg["E-region"][2]
    trivial = sum(1 for _, reg, c, m in rows
                  if reg == "E-region" and c["sink_mode"] == "inf"
                  and m["fp8_error"] is None and m["arm_ok"])
    # The degenerate cell is one of the PASSes, so e_fail already excludes it --
    # subtracting ``trivial`` again would undercount the numeric failures by one
    # (caught by the deploy-faithful round adversarial review).
    e_numeric = e_t - e_p
    print()
    print(f"  E-region breakdown: {e_t} judged cells = {e_p} PASS "
          f"(incl. {trivial} degenerate all-inf-sink cell) "
          f"+ {e_numeric} numeric FAIL; {e_crash} CRASH excluded")
    if e_numeric > 0:
        print(f"  DEPLOYMENT SHAPE REPRODUCES A DIVERGENCE: {e_numeric} E-region cells")
        print("  (effective attend count < 72, i.e. the real short-prompt workload) have the")
        print("  fp8 arm diverging NUMERICALLY from the bf16 arm, O broken while lse stays")
        print("  intact -- defect E's signature.  Per the deploy-faithful round this is the end-to-end landing zone.")
    elif e_t:
        print(f"  E-region cells show no numeric divergence ({e_p}/{e_t} PASS).")
    if i_t:
        print(f"  island cells: {i_p}/{i_t} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())