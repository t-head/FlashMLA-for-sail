# V4.0 FP8 and V4.1 FP4/FP8 warp interleave integration

## GitHub migration, 2026-10-20

This port targets `t-head/FlashMLA-for-sail` main at `3fa3224c` and carries
forward the quantized functionality from the two historical source commits
listed below. It preserves main's newer BF16 scheduling, paired-cube layouts,
combine dispatch and Prefill shared-V reader synchronization. In particular,
the BF16 Prefill entry stays in main's implementation instead of adopting the
older shared-driver adapter. Quantized Prefill continues to use its shared
three-format driver. The ACTL submodule stays at main's `0146d9e9`; it contains
the historical `9822f231` OddTile fix.

The performance and numerical results below belong to their stated historical
source, module, hardware and scope. They are not results from a rebuilt GitHub
merge. The final isolated source was `61e7cad99714`; its related native sparse
follow-up checked 99 existing case entries: 65 newly executed passes and 34
reused passes from original case logic. This was not the full native test
matrix. Dense Graph did not reach numerical comparison because that sparse
validation module lacked `dense_decode_fwd`. The three extreme UE8 failures
and the separate strict20 failures described below remain recorded failures.

The merged GitHub source requires its own build and native test qualification.
No full H128 BF16 parity or global no-regression claim is made by this port.
Local benchmark archives referenced below document the historical work and
are not included in the public source tree.

## Historical source commit scope

The release contains two commits above `13be8495`:

- `a5556c0b`: minimal V4.1 FP8 and FP4 extra-KV splitkv support.
- The WI commit: opt-in quantized Decode WI, quantized H128 Prefill,
  the accepted FP4/FP8 optimizations, and V4.0 FP8 Prefill support.

V4.0 FP8 now instantiates the same quantized WI main loop as V4.1 FP4/FP8.
Its load policy preserves the 448 FP8 NoPE values and the 64 raw BF16 RoPE
values. Two translation units provide bounded and wide addressing. V4.0
Decode is not routed to WI. FP4 Decode uses FP8 primary KV and FP4 extra KV;
it is not a pure FP4 primary-KV Decode implementation.

## Historical qualification, 2026-10-08–09

The fresh validation baseline is the published `df941c6e` source with pinned
ACTL `9822f231`. The candidate builds the WI dependency closure (16 objects),
including both new V4.0 instances. This is not a full production extension
build. Private SDK naming adaptations are excluded from the release source.
The initial three-format test module SHA256 is
`426096200265a1a2130390839f523f4aaf6161442967755d2f4ca1df7c391011`.
Its inherited device images match the freshly built baseline; only the two
V4.0 device instances are added. The final module additionally repairs the
generic Decode barrier described below, and has SHA256
`97b5c00e6af89278a5f929f99a941f995bc336c30dc553d8af52da06f8890836`.
That repair changes eight generic FP8 Decode device images: M64/M128 for
legacy D512/D576 and V4.1 with FP8/FP4 extra KV. All 85 other
images, including every quantized/native BF16 Prefill image, and the API
object are identical to the initial three-format module.

The full WI accuracy matrix on module `42609620` completed
**829 checks: 826 passed, 3 failed** (793 numerical WI checks and 36 host
rejection checks). Module `97b5c00e` inherits Prefill through the exact image/API
binding above and separately reruns generic Decode below.
It covers all three Prefill formats, V4.1 WI Decode, invalid indices, lengths,
partial pages, padding, wide addresses, streams, Graphs and host rejection.
Each numerical check requires an actual WI CUDA main trace. The 36 host
rejection checks are included in the 829 total. The 64
planned V4.0 Decode checks are unsupported and excluded. Splitkv results are
excluded from this qualification.

Full testing exposed incorrect masking for nonpositive softmax scales.
The host now uses `-Q` with `abs(scale)` for negative scales and `Q * 0`
with scale 1 for zero. Positive scales retain the original Q storage and
device code. Six original negative/zero checks and six additional Graph
checks (18 replay states with mutable Q) pass for all three formats.

The three remaining failures retain their original seeds and thresholds:
V4.0 all-finite UE8 scales, V4.1 all-finite UE8 scales, and V4.1 UE8 minimum/
maximum scales. Finite decoded values near `2^127` overflow the unnormalized
FP32 PV accumulator before output normalization. GPU diagnostics of all three
patterns confirm matching Inf locations in the same decoded BF16 WI while
the normalized references remain finite. The original 33 query rows are
repeated to Sq384 only to enable the native BF16 WI entry. These cases remain
**failed**. Six separate bounded UE8/payload-code checks pass with the original
thresholds; they do not replace the original failures.

The committed regression runner also passes all 16 checks, including negative
scale, positive/negative zero, changed-Q/index Graph replay and exact restore,
and V4.0 raw BF16 RoPE plus ignored scale padding:

```sh
python tests/test_quant_wi_formats.py --device 0
```

### Generic Decode synchronization repair

The performance matrix exposed a hang at H64/B4/Sq1/topk64/extra-topk64,
variable cache lengths and extra topk lengths (original seed 28404). Both
warpgroup callers pass local thread indices 0–255 to the staged loader.
Using that local index to select a barrier incorrectly assigned barrier 6
to both groups. Their arrivals could mix across two successive stages and
deadlock against the following CTA barrier. The one-line repair selects
barrier 6/7 from the global `threadIdx.x >> 8`; each still counts 256 threads.
Data indexing, conversion, tensor work and public arguments are unchanged.

The original hanging worker was stopped after more than 16 minutes; its
partial log and termination record are retained. The repaired module passes
all eight original stock cases, including seed 28404, and all 118 original
Decode precision cases with `FLASHMLA_V41_HS64=0`, actual generic WI traces
and original tolerances. These tests run on an otherwise idle GPU2 with
39 CUs. Earlier H64 checks using HS64=1 exercised a separate kernel.

The build recompiles only the affected generic Decode translation unit and
reuses 15 other objects. V4.1 H64 vector-register use increases by two with
zero spill/stack; V4.1 H128 and BF16 resources retain their baseline values.
Some legacy generic FP8 variants have a small increase in their pre-existing
spill/stack. The zero-spilling Prefill profiles below are unaffected.

The repository regression below independently passes all eight stock cases
on the final module with actual generic WI traces and the original thresholds.
It preserves the original single-call stock-test scope.

```sh
python tests/test_v41_sparse_quant_decode_validation.py --device 0
```

On GPU5, M890P, **39 CUs at 1300 MHz**, direct ACU attention-main counters
for B1/Hq128/Hkv1/Sq4096/Sk8192/topk2048/D512/P64/seed0/scale0.5, sink on,
no lengths, report:

| Format | Main duration | Effective TFLOPS | Direct TFU |
| --- | ---: | ---: | ---: |
| V4.1 FP4 | 15.118530 ms | 145.45 | 35.018584% |
| V4.1 FP8 | 14.906135 ms | 147.52 | 35.543471% |
| V4.0 FP8 | 16.759160 ms | 131.21 | 31.593829% |
| Matched decoded BF16 WI | about 14.27 ms | about 154.1 | about 37.11% |

All six matched profiles execute 268,435,456 tensor instructions with zero
spilling. These profiles use warm steady state, contiguous decoded BF16 KV,
and the original padded Q/index strides. The earlier 39.19% BF16 measurement
below used a different KV stride/conditioning harness; it is a historical
comparison. ACU selects one attention main and terminates the producer after
profiling; these numbers are not complete-call CUDA event samples.

The original performance checkpoint completed 592 worker jobs and 1184 of
1192 case IDs, retaining every sample: 30 event/wall samples per eligible
route from two opposite-order blocks. The final eight stock Decode cases
were interrupted by the hang above. The existing V4.1 Prefill event geomean
changes versus the fresh baseline are -0.358% for FP4 and +0.0218% for FP8.
V4.0 Prefill has no old quantized WI baseline and is reported separately.

The final performance qualification covers **all 1192 case IDs**: 768
Prefill cases retain their module `42609620` / GPU5 qualification through
the exact Prefill device-image and API-object binding above; all 424 Decode
cases are freshly measured on repaired module `97b5c00e` / GPU2.

The final Decode audit passes 106 clean worker jobs and 848 case records
(two blocks per case). It retains 25,320 CUDA event samples and 25,320
wall samples across 424 quantized routes and 420 eligible BF16 controls,
30 samples per measured route with no trimming. Four stock BF16 controls
are unsupported and excluded. These Decode measurements are current-only
on GPU2. Comparisons to the preserved GPU5 baseline remain historical.
Qualification uses the stated module/GPU bindings for each case.

Raw samples, actual kernel traces, guard records, independent statistics and
source/module bindings are retained under
`benchmark/v41_full_validation_20261008/`. The final Decode audit is
`AM_current_only424_decode_independent_review.json` (SHA256
`2b157fb9205f3c7b53855bb5a79b4edf4a71dcba0a66808d9417e9d66b7fd720`).

A separate GPU2 repeat checks the three original small-case regression
candidates, an adjacent FP8 P128 case and the FP4/FP8 target controls. All
four jobs pass with the same opaque fixtures and two opposite-order blocks;
the repeat is kept separate from the original GPU5 samples. One FP4 P64/Sq1
case retains a small regression: 22.635 to 23.398 microseconds, +0.763
microseconds (+3.372%), with +2.064% and +4.663% in the two orders. The
FP4 P128 and FP8 P64/P128 candidates do not exceed +2% in both orders.
The large target changes are -0.0136% for FP4 and -0.1378% for FP8. These
measurements do not establish an API-overhead cause. The repeat review is
SHA256 `ff2fdfa2a23302e5c84ba616e50907652a5318b1b235260b6f2c58ca3087b27e`.

### Sharing with BF16

Thirteen identical leaf math functions can be extracted into a common header.
A common main-loop template additionally needs compile-time load, address,
layout and validity policies: BF16 uses direct asynchronous copies, while
quantized KV uses register packets, decoding, BF16 TSM stores and publication.
The current release shares the quantized loop among the three formats and
leaves the BF16 implementation intact. No performance benefit is claimed for
a future common source implementation.

The sections below preserve the earlier optimization measurements and their
individual build and input conditions.

## API

`flash_mla_sparse_quant_fwd(q, kv, indices, sm_scale, num_tokens, ...)` consumes
the existing page layout directly, including the block-tail scale region:

- FP4: 256 payload bytes and 32 E4M3 scale bytes per token.
- FP8: 512 E4M3 payload bytes and 16 E8M0 scale bytes per token.
- V4.0 FP8: 448 E4M3 bytes and 64 BF16 values (576 data bytes), plus
  seven UE8M0 scales and one ignored padding byte per token.

It requires SM89, H128, D512, one KV head, and topk divisible by 128. The
logical `num_tokens` excludes final-page padding. The result is BF16 output,
FP32 scaled max logits, and FP32 natural-log LSE. Sink, per-query topk length,
page padding, invalid indices, and an optional output buffer are supported.
See the Python function docstring for the complete shape and stride contract.

The Decode experiment is enabled with:

```sh
FLASHMLA_V41_WI=1 FLASHMLA_V41_HS64=1 python tests/test_v41_hs64_wi_validation.py --stress
```

## Build compatibility

The historical qualification used ACTL gitlink `9822f231`. The HGGC compiler version is
reported as 2 in this SDK, while pinned CUTE's LDSM feature check expects an
NVIDIA-style major version of at least 11. That check generated eight `trap`
instructions in the quantized Prefill kernel instead of eight P matrix loads.

The PPU build adds `-DCUTE_ARCH_LDSM_PPU_ENABLED=1`. Its existing hardware
architecture guard still applies. CPU ISA comparison confirms that this single
flag reproduces the corrected kernel instructions exactly, with zero traps.
The broad dependency upgrade and SDK symbol renames are excluded from the
commit. The validation container uses a private SDK name adaptation for its
CUDA-compatible PyTorch installation.

## Validation

On M890P / SDK 13.3, the corrected module passed:

- 16 native Prefill cases: FP4 and FP8, P64/P128, partial pages, ragged and zero
  lengths, invalid indices, sink infinities, padded page strides, and output
  buffer reuse. Output, max logits, and natural-log LSE use the upstream
  reference tolerances.
- 21 Decode cases, including H64 WI2, H96 splitkv fallback, and H128 WI.
- Both formats on H128 / Sq4096 / Sk8192 / topk2048 / P64 / seed0 / sink on.

The full build uses the pinned dependency and excludes the unrelated FP4 LUT
and submodule changes. CPU evidence and guarded GPU logs are retained under
`benchmark/v41_wi_prod_integrate_results_20261007l/` locally.

## Optimization progress

Separate isolated native kernels retain complete KV/scale reads, dequantization,
and BF16 MMA operands. On the fixed large case, two opposite-order GPU0 rounds
gave these candidate timings (30 samples per round):

| Format / factor | Original | Candidate |
| --- | ---: | ---: |
| FP4, early register packets in two PV windows | 16.83 ms | 16.51 ms |
| FP4, additional WG0 first packet across QK | 16.51 ms | 16.37 ms |
| FP8, early register packets in two PV windows | 17.09 ms | 16.73 ms |

Each candidate passed its native eight-case matrix and the full target reference
check. The corresponding same-math BF16 kernel is about 13.61–13.62 ms; these
quantized timings have not reached BF16 performance. The additional all-main
FP4 packet factor was slower and was rejected. Candidate scheduling can change
when integrated into the production entry, so production timings require a
separate measured gate.

## Production packet prefetch

The follow-up engine change prefetches packed KV into registers before both PV
windows. FP4 also carries WG0's first packet across QK. All payload and scale
reads, native decoding, BF16 multiplication, invalid-index masking, and the
original publication points remain. FP8 invalid entries retain a zero float
scale rather than interpreting raw E8M0 zero as a nonzero scale.

The production extension passed 18 public API cases, including both target
formats, P128 partial pages, padded page strides, sink infinities, ragged/zero
lengths, invalid indices, and exact output-buffer reuse. Two opposite-order
GPU0 rounds with 30 samples each gave:

| Format | Before | Register prefetch | Change |
| --- | ---: | ---: | ---: |
| FP4 | 17.032 ms | 16.948 ms | -0.49% |
| FP8 | 17.055 ms | 16.868 ms | -1.10% |

These are full production-entry timings. The smaller isolated-kernel numbers
above should not be substituted for them. The published source is byte-identical
to the validated engine (`ed9c86c6`); the tested extension SHA256 is
`dcf8b3e282065a6c10f4ef3b59ee016752172fe2aa77879180aa25e8a1143788`.
Binding proofs are saved locally under
`benchmark/v41_wi_prod_integrate_results_20261007l/engine_publish_ready/`.

## Native BF16 reference

The experimental `same_math` BF16 kernel retains the newer validity and address
machinery. It is useful for isolating quantized loading/decoding costs, but it
does not establish the fastest native BF16 performance on this case.

Direct public-API comparisons use logical B1, Sq4096, Sk8192, H128/Hkv1,
D512, topk2048, seed0, scale0.5, sink enabled, and no topk lengths. Inputs retain
the official padded strides and independent random valid indices. Original BF16,
FP8-decoded BF16, and constant-0.5 KV give similar timings within each version:
the older paired experimental binary is about 12.72 ms, while main `c631b8f7`
is about 13.51 ms. On the same GPU1, elapsed TFU is 41.65% versus 39.19%.
Both GPUs used for these measurements expose 39 CUs at 1300 MHz.

The newer kernel includes per-token validity masking. A restricted diagnostic
that omits interior validity reads/masking while retaining producer stores and
tail handling reduces its time to 13.024 ms and raises TFU to 40.67%. That
diagnostic only supports the all-valid target with no topk lengths and is not a
production optimization. It is excluded from the packet-prefetch source.

## FP4 whole-word sign expansion

The FP4 decoder now expands signs once for all eight nibbles in each packed
word. Its four BF16 pairs match the previous decoder bit for bit, including
negative zero. The scale conversion, BF16 multiplication, payload reads,
masking, stores, and synchronization are unchanged. The FP8 device instructions
are byte-identical to the packet-prefetch version.

The production extension passed all 18 public API cases. Two opposite-order
GPU0 rounds, with 30 samples each and a target reference check in every process,
gave **16.950 ms → 16.253 ms (-4.11%)**. The isolated prototype improved from
16.371 ms to 15.686 ms; that result is not the production-entry timing.

On GPU1, with 39 CUs at 1300 MHz, production elapsed TFU increased from 31.24%
to **32.61%**. Average integer instructions per CU fell from 20,061,578 to
16,620,098. Tensor instructions (268,435,456), floating-point instructions,
load/store instructions, allocated registers, and zero spilling were unchanged.
This remains slower than the latest native BF16 reference.

The validated engine SHA256 is
`50f1c0eb45359339278cdd33b28f9addb967a444b825091fb347ff3b09705606`;
the extension SHA256 is
`47d2678a08ef1d07d090baeb34f60220be7bc4b31a11692636765eba14ba32ea`.
Source, ISA, independent review, and guarded GPU evidence are retained locally
under `benchmark/v41_wi_prod_integrate_results_20261007l/integrated_fp4_word_shared_sign/`.

## Bounded address specialization

The host now selects separate FP4/FP8 kernels when the complete physical byte
spans of Q, packed KV, indices, output, max logits, and LSE fit 2 GiB. The span
calculation uses 128-bit intermediates and includes unused physical KV pages.
Larger views retain the original 64-bit path. Public validation and the 152-byte
kernel parameter layout are unchanged. The address macro belongs only to the
new translation unit; default wide device instructions remain byte-identical.

Combined with the whole-word FP4 decoder, the production entry passed all 18
API cases and two additional large-stride cases. Those cases place three actual
packed pages at offsets 0, 2 GiB, and 4 GiB, read the last page through attention,
verify output reuse, and confirm the original wide kernel is selected.

Two opposite-order GPU0 rounds, 30 samples each, gave:

| Format | Whole-word version | Whole-word and bounded addresses | Change |
| --- | ---: | ---: | ---: |
| FP4 | 16.243 ms | 15.509 ms | -4.52% |
| FP8 | 16.844 ms | 16.637 ms | -1.23% |

Clean GPU3 ACU measurements report **34.01% FP4 TFU** and **31.87% FP8 TFU**,
with 39 CUs at 1300 MHz. Both execute 268,435,456 tensor instructions and have
zero spilling. Benchmark workers use an idle 640 MiB allocation to discourage
the local free-device selector from entering between serial workers; all other
GPU PIDs still invalidate measurements. Failed overlapping runs are retained.

The validated extension SHA256 is
`1f56e0516f57b6fe93771c4bca41f6ade6f9b065068dd662f0d8d3be17d6813a`.
Source and independent proof are retained under
`benchmark/v41_wi_prod_integrate_results_20261007l/integrated_word_narrow32_separate_tu/`.

## FP8 first packet across QK

FP8 now carries the WG0 first register packet across the preceding QK MMA,
alongside the existing second packet across PV. The initial and steady windows
cover 64 and 32 QK MMA instructions, respectively. Payload decoding, scales,
validity handling, the rear TSM drain, named barrier, and K publication retain
their original order. Both FP4 device images are byte-identical to the bounded
address version.

The final production extension passed 18 public API cases, both actual
large-stride fallback cases, and target references in all four timing workers.
On an otherwise idle GPU6, opposite-order rounds with 30 samples each gave
**16.627 ms → 16.342 ms (-1.71%)**. ACU reports **32.41% FP8 elapsed TFU**,
39 CUs at 1300 MHz, 268,435,456 tensor instructions, and zero spilling.
FP4 retains its previously measured 34.01% TFU. These results include real
payload reads, scale reads, and dequantization.

The validated engine SHA256 is
`e71938dbba596eca2429dc4d5b046c63f54afe81792709317af5735abd569903`;
the extension SHA256 is
`280784b57c8c0bef79fe1ac81e3426ce55667904f7c2093d429f008bef4ab561`.
Frozen source, independent ISA review, and guarded GPU records are retained
under `benchmark/v41_wi_prod_integrate_results_20261007l/integrated_word_narrow32_fp8_first_qk/`.

## FP8 E8M0 scale construction

FP8 now constructs the replicated BF16 scale bits directly from E8M0. Valid
raw 0 retains its true 2^-127 value; raw 255 retains the original canonical NaN;
invalid tokens retain scale bits 0. Actual device tests compare both scale and
multiplied payload bits for all 65,536 scale/payload combinations and 256
invalid-scale cases. Every result matches the previous implementation exactly.
The payload conversion and multiplication expressions, KV/scale reads, and
publication protocol are unchanged. Both FP4 device images remain byte-identical.

The final extension passed all 18 API cases, both actual large-stride fallback
cases, and four target references. Opposite-order GPU3 rounds gave
**16.343 ms → 16.213 ms (-0.79%)**. ACU reports **32.68% FP8 elapsed TFU**,
39 CUs at 1300 MHz, 268,435,456 tensor instructions, and zero spilling. Average
integer instructions per CU fell from 18,500,477 to 17,851,838; floating-point
instruction counts remain unchanged. The compiler reschedules some independent
instructions; the actual 64/32-MMA load windows and rear publication order are
verified in the linked extension.

The validated engine SHA256 is
`e65598ef1b08f42a0f09f3160b0e7de90e1c8885c819d28c78c3156a1d79f324`;
the extension SHA256 is
`5ccac09994a14554339229171ac30713adf0cbd37a1d9e55f9325287992ef959`.
Source, device bit tests, independent review, and guarded measurements are
retained under `benchmark/v41_wi_prod_integrate_results_20261007l/integrated_fp8_ue8_scale_bits/`.

## FP8 validity check and GPU selection

Large FP8 calls now check every sparse index once on the current GPU stream.
The checker writes 128 flags after the logical max-logits output, adding
**512 bytes** of backing storage. The returned output shape is unchanged.
Every warp of the following attention kernel reads the same immutable flags
and makes one uniform choice: all valid indices use a specialization that
omits repeated validity bookkeeping; any invalid index uses the complete
original computation. Payload loads, scale loads, dequantization, softmax and
MMA remain real in both paths. There is no D2H copy or host stream synchronization.

This route requires FP8, H128/Hkv1/D512, no lengths tensor, Sq >=4096,
topk >=2048, and the original six physical spans fitting bounded addresses.
The full max-logits backing, including the flags, participates in the span
check. Public topk validation still requires a multiple of 128. FP4, smaller
calls, lengths and wide spans retain their existing kernels. Default kernels
rebuilt from the current source are byte-identical to the previous four images.

The fixed B1/Sq4096/Sk8192/topk2048/P64 case, with seed 0, sink and official
padded Q/index strides, was measured in two opposite-order rounds of 30
samples each on an otherwise idle GPU3:

| Measurement | Previous FP8 | GPU checked FP8 | Change |
| --- | ---: | ---: | ---: |
| Complete-call CUDA event | 16.238 ms | 15.582 ms | -4.04% |
| Attention-main ACU TFU | 32.68% | 34.29% | +1.61 percentage points |

Complete-call timing includes the checker, checked attention, allocations,
launch gaps and extra backing storage; L2 conditioning is outside the call.
ACU TFU describes the attention main only. It reports 39 CUs at 1300 MHz,
268,435,456 dynamic BF16 tensor instructions and zero spilling. The wrapper
has 241 vector registers, rounded to 248 by the launch metric. Its two inline
bodies have 1024 static tensor instruction sites; only one body runs per CTA.
FP4 retains its previously measured 34.01% TFU. Latest native BF16 remains
39.19%, so the BF16 target is not reached.

Validation passed the original 18 API cases, two actual large-stride cases,
and 20 additional qualification cases. These include negative and upper-bound
indices, logical padding, all-invalid inputs, lengths, a Q span exceeding
2 GiB, and Graph replay with indices changing valid -> invalid -> valid.
The new portable regression test additionally covers
Sq4097/Sk8207/topk2176/P128, exact output-buffer reuse, original reference
thresholds and changing Graph inputs:

```bash
python tests/test_v41_sparse_quant_prefill_validation.py --target
python tests/test_v41_sparse_quant_gpu_checked_validation.py
```

The earlier host-decision experiment improved FP8 complete-call time by 3.14%
but introduced host synchronization; it was not promoted. Its FP4 route was
1.18% slower over the complete call and was rejected.

The validated engine SHA256 is
`8f28cf2c0e06529137132d2f567381ce462759bc60a3abfcc8952eb35a7733db`;
the extension SHA256 is
`176af935f8fcfa58b9a584230d850f281419bc8737506774ccd9291f5195876e`.
Frozen source, independent machine-code and source recompilation reviews,
guarded GPU records and counter CSV are retained under
`benchmark/v41_wi_prod_integrate_results_20261007l/integrated_fp8_gpu_checked_allvalid/`.

## FP8 register-load validity specialization

The GPU-proved all-valid FP8 body now treats register-packet loads as valid.
Its four load call sites remain bounded by the actual block range, including
the steady non-tail loop, so a fabricated tail token never reaches a load.
The transformation is enabled by a macro local to the checked FP8 translation
unit. The complete register-finish function retains the original conversion,
scale multiplication, stores and synchronization. The first finish region has
the same 145-opcode order, with conversion after the front reader drain;
register operands differ. Initial 64-MMA and steady 32-MMA prefetch windows
remain verified in the linked extension.

On the same fixed case, guarded GPU3 measurements retained all 60 samples
per route from two opposite-order rounds:

| Measurement | Previous checked FP8 | Load specialization |
| --- | ---: | ---: |
| Complete-call CUDA event | 15.581458 ms | 15.457820 ms (-0.79349%) |
| Attention-main ACU TFU | 34.290579% | 34.552414% |
| Average integer instructions per CU | 15,377,434.256 | 14,439,765.333 |

Dynamic BF16 tensor instructions remain 268,435,456. Floating-point
instructions remain 15,594,207.179 per CU. ACU reports 39 CUs at 1300 MHz,
248 allocated vector registers and zero spilling; compiler usage is 241V/122S.
TFU measures the attention main. Complete-call timing includes the checker.
FP4 retains 34.007797% TFU; latest native BF16 remains 39.189681%.

Validation passed the original 18 API cases, both actual large-stride cases,
10 FP8 qualification cases including changing Graph indices, and the portable
Sq4097/Sk8207/topk2176/P128 boundary test. The five inherited kernel images
are byte-identical to the previous extension. Default wide/narrow kernels
are additionally rebuilt from this source with the new macro disabled.

The tested engine SHA256 is
`0409001073182ba845d865a1ee9c4f27d6113dfda0a8a526df4e4b808aa8b0c8`;
the extension SHA256 is
`aed5e63da9fcf91a7c0f11fd0e692f122e23a148645d4452876543d01c1eef28`.
Frozen source, independent review and GPU evidence are retained under
`benchmark/v41_wi_fp8_load_valid_results_20261008t/`.

## Decoded stores: release before K publication

The narrow FP4/FP8 and checked FP8 translation units now replace the final
256-thread decode barrier with a block memory fence and an explicit compiler
memory boundary. Every producer completes its own decoded BF16 stores before
the original K-barrier arrival. That barrier still counts all 256 producers,
and consumers still wait for its parity before loading K into MMA. The front
reader drain, initial index publication and cross-group scale/P barriers remain
in place. The cross-group barriers also order index reads before reuse of the
index buffer. Wide kernels retain their previous implementation.

The linked machine code verifies all eight decoded-store publication sites
per body: TSM stores, fence, fence/TSM waits, then the original K commit.
There are eight fewer rear collective waits per body. The replacement is
not free: emitted block fences increase from 12 to 18 in FP4, 15 to 18 in
default FP8, and 30 to 38 across both checked FP8 bodies. FP4 also changes four
packed BF16 conversions into eight scalar RN conversions; the number of
converted lanes stays 240. Payloads, scale conversion, scale multiplication,
dequantization and tensor work remain complete.

The same B1/H128/Hkv1/Sq4096/Sk8192/topk2048/D512/P64 prefill case uses seed 0,
scale 0.5, sink, no lengths and the original padded Q/index strides. Two
opposite-order rounds retain 60 complete-call samples per route on idle GPU3:

| Measurement | Previous FP4 | Released FP4 | Previous FP8 | Released FP8 |
| --- | ---: | ---: | ---: | ---: |
| Complete-call CUDA event | 15.571674 ms | 15.262357 ms | 15.457024 ms | 15.253230 ms |
| Complete-call latency change | | -1.98640% | | -1.31845% |
| Attention-main ACU TFU | 34.007797% | 34.739779% | 34.552414% | 35.027783% |
| Attention-main ACU duration | | 15.242987 ms | | 15.127536 ms |
| Effective attention TFLOPS | | 144.26 | | 145.37 |

Effective TFLOPS uses the fixed 2.199023255552 TFLOP attention numerator and
the ACU main duration. TFU is the actual tensor-active counter, not a value
derived from complete-call time. Both profiles report 268,435,456 BF16 tensor
instructions, 39 CUs at 1300 MHz and zero spilling. Compiler usage is
242V/125S for FP4, 240V/121S for default FP8 and 240V/122S for checked FP8;
the checked FP8 launch allocation falls from 248 to 240 vector registers.
Latest native BF16 remains 39.189681% TFU.

The original 18 API cases, two actual wide-stride cases and 20 full-quant
qualification cases pass without changing precision thresholds. Qualification
includes invalid indices, logical padding, lengths, partial pages and large
spans. Both formats pass valid/repeated Graph replay; FP8 additionally passes
valid -> invalid -> valid replay and the portable
Sq4097/Sk8207/topk2176/P128 boundary/reuse test. The three inherited wide/checker
images match the prior extension. Recompiling the wide translation unit from
the new source with the macro disabled also reproduces both original images.

The tested engine SHA256 is
`e037df0c66af773a74e9f7562605ddfddc7a5d6079f3a3a68de0cc201c2b5a97`;
the extension SHA256 is
`c35f6367c9ccc333175b929e76f9734eaa3af8d7d486770bf8772f0e191befd2`.
Source, independent machine-code review, compiler records, all timing samples,
guard records and ACU counter CSV are retained under
`benchmark/v41_wi_store_release_results_20261008y/`.

## FP8 row offset and scale-plane delta

The bounded32 FP8 translation units carry an eight-byte row state: a payload
byte offset and the delta from that payload row to its scale row. Load issue
sites consume that state directly, avoiding repeated page/row address work.
The state is computed before the existing register-prefetch points; the FP8
payload and E8M0 scale loads, conversion, BF16 multiplication, 36-byte packet,
reader drain and decoded-store publication retain their previous behavior.
The host span guard and GPU validity check remain in place. Invalid indices,
lengths and Graph replay still select and execute the validated paths.

This applies only to FP8. The FP4, two wide kernels and validity-checker
machine images match the decoded-store-release baseline byte for byte.
The two FP8 images match the earlier combined address candidate; keeping its
FP4 change would regress the target and is therefore excluded.

The same B1/H128/Hkv1/Sq4096/Sk8192/topk2048/D512/P64 prefill case retains all
60 complete-call samples per route from two opposite-order rounds on idle
GPU3. There is no outlier filtering.

| Measurement | Previous FP8 | FP8 offset/delta |
| --- | ---: | ---: |
| Complete-call CUDA event | 15.270635 ms | 15.116591 ms |
| Complete-call wall time | 15.287126 ms | 15.121626 ms |
| Complete-call latency change | | -1.008758% |
| Attention-main ACU duration | 15.125043 ms | 14.973425 ms |
| Attention-main ACU TFU | 35.000839% | 35.382239% |
| Effective attention TFLOPS | 145.39 | 146.86 |

Both main profiles report 268,435,456 BF16 tensor instructions, 39 CUs at
1300 MHz and zero spilling. The candidate uses 242 vector registers and
111/112 scalar registers for default/checked FP8; the checked launch vector
allocation increases from 240 to 248. The actual counters above remain
separate from complete-call timing. FP4's previous 34.739779% TFU and latest
native BF16's 39.189681% TFU are historical comparisons, not new measurements
in this round.

The original 18 API cases and two actual wide-stride cases pass with the
original precision thresholds. The ten FP8 qualification checks cover valid
and invalid indices, physical padding, partial pages, lengths and wide spans,
including Graph valid -> invalid -> restored-valid replay. The portable
Sq4097/Sk8207/topk2176/P128 boundary and output reuse checks also pass. All ten
GPU jobs have clean foreign-process guards.

The tested engine SHA256 is
`ae8735ddf21cafca459e590c0959bcabf595fd378e3b69f083a286696c189619`;
the extension SHA256 is
`865cebe0b16ae342cae471a2e0b2578e7cd6c7faeadfab045df9f440e7296bb0`.
Frozen source, independent review, all samples, guards and main counter CSV
are retained under `benchmark/v41_wi_fp8_offset_delta_results_20261008ac2/`.

## FP4 signed high-byte LUT with native PRMT

The FP4 word decoder shares one selector over all eight nibbles. Repeated
high-byte LUT entries include the sign and preserve negative zero. The first
high-byte lookup uses native PPU PRMT with a 32-bit control operand, avoiding
an otherwise redundant low-16-bit selector conversion. Low-byte lookup,
E4M3 scale conversion, BF16 multiplication, KV loads and publication remain
unchanged. The measured helper costs 15 instructions instead of 16.

On the same fixed prefill case, two opposite-order rounds on idle GPU5 retain
all 60 complete-call event and wall samples per route, without filtering.

| Measurement | Previous FP4 | Native-PRMT FP4 |
| --- | ---: | ---: |
| Complete-call CUDA event | 15.252873 ms | 15.131232 ms |
| Complete-call wall time | 15.267269 ms | 15.137382 ms |
| Complete-call latency change | | -0.797495% |
| Attention-main ACU duration | 15.229440 ms | 15.128413 ms |
| Attention-main ACU TFU | 34.770021% | 35.019733% |
| Effective attention TFLOPS | 144.39 | 145.36 |

Both profiles report 268,435,456 BF16 tensor instructions, 39 CUs at 1300 MHz,
248 allocated vector registers and zero spilling. Actual integer-pipe work
falls 2.675026%; floating-pipe work is unchanged. Static FP4 instructions fall
30/31 in wide/narrow kernels; the compiler adds two/one waits respectively.
These code counts are distinct from the measured latency improvement.

The original 18 API cases, two actual wide-address cases and ten FP4 target
qualification checks pass with original thresholds, including valid/repeated
Graph replay and output reuse. This FP4 Graph test does not claim changed-index
replay coverage. All nine GPU guards are clean. Recompiled FP8 wide/narrow,
checked FP8 and validity-checker images match AC2 byte for byte; FP8's previous
35.382239% TFU and native BF16's 39.189681% remain historical comparisons.

The tested engine SHA256 is
`df6dbfa90b8ecb95d3f2d463d9eed94d34a6b12d6c8effc440ed9e7cb6a0d4f5`;
the extension SHA256 is
`b0c84c2249f6d0ad6e7a5d3e0b39259fd2009e7c7c0c3dc141fa960f65ca4a25`.
Evidence is retained under
`benchmark/v41_wi_round4_results_20261008/fp4_signed_high_native_AJ/`.

## FP8 WG1 first remote-PV operand prefetch

The checked, all-valid FP8 steady loop loads the original two P atoms and
first V atom immediately after `sP0Ready`. The packet carries 12 data vector
registers. The original next-K packet finish and future address calculation
follow; remote PV consumes those operands and loads the remaining V atoms.
Scale, dequantization, MMA, K publication and tail/invalid paths retain their
original work. The actual compiler coalesces the register copies, preserves
all load/store and math counts and removes two waits. The existing Y-release
TSM wait drains early reads; early reads can also enforce P readiness sooner.

Two opposite-order rounds on idle GPU5 retain all 60 complete-call samples
per route, including every outlier.

| Measurement | Previous checked FP8 | Remote-PV prefetch |
| --- | ---: | ---: |
| Complete-call CUDA event | 15.130415 ms | 15.048820 ms |
| Complete-call wall time | 15.135359 ms | 15.055264 ms |
| Complete-call latency change | | -0.539276% |
| Attention-main ACU duration | 14.977196 ms | 14.911098 ms |
| Attention-main ACU TFU | 35.357562% | 35.522623% |
| Effective attention TFLOPS | 146.82 | 147.48 |

Both profiles report identical integer/floating work, 268,435,456 BF16 tensor
instructions, 39 CUs at 1300 MHz, 248 allocated vector registers and zero
spilling. Compiler use remains 242V/112S. The five inherited images match AC2.
The original public18, FP8 qualify10 and P128/Sq4097/Sk8207/topk2176 portable
boundary checks pass with original thresholds and output reuse, including
Graph valid -> invalid -> restored-valid replay. All nine GPU5 guards are clean.

The measured standalone engine SHA256 is
`52c89e01f7f97a0470740b152b7ca43bb9fa664fcd637e7678cedc2301a28d4f`;
the extension SHA256 is
`c8a6c0336362a0e64f7f9a85fffb7a2165644cc74012259be474bbe3189ed61c`.
The first CPU compile exposed a static-assert expression syntax error. Its
source and logs are preserved; the successful retry changes only three
compile-time assertions. Evidence is retained under
`benchmark/v41_wi_round4_results_20261008/fp8_remote_pv_prefetch_AI/`.

The combined release uses engine
`57796518ea850fa4e0d09f02f739358a03beaaf42a8d9431d4eb7b0bcdf7834c`
and extension
`2c4570f713f74fccdfb0fac70ea8989b7f7ece763cda7cdb9990d195a04a730d`.
One checked-TU proof compile reproduces the measured AI checked image exactly;
the other five images match AJ exactly. A fresh combined-module public18
validation passes on clean GPU5. Each format therefore retains its separately
measured kernel; their percentage gains are not added together.

## H128 Sq1 live-length decode scheduler (L90, 2026-10-19)

The source delta is confined to `quant_decode_wi.cuh`; the public API and
shared prefill driver are unchanged. H128 Sq1, main topk128 and aligned extra
storage with live lengths use independently proven narrow cache addressing.
Storage up to 2048 uses the canonical split prefix to project each CTA onto
one existing task. If every row has one task, the row follows directly from
the CTA ordinal. Larger storage keeps the persistent scheduler. Live lengths
and indices still select the checked body when a complete-row proof fails;
no cached validity flag or numerical tolerance is introduced. Metadata must
be regenerated when lengths change, as required by the existing API.

The tested extension SHA256 is
`139c2026ab8ad154a7f7e23c0eec3284980aa2b0e0ff753870b1d3539af17e14`;
the private header SHA256 is
`90f9e10bd84026c1f886ac055781df9e94f0b57cdf622f08f971dce8c6450886`.
Only the generic translation unit was rebuilt; 15 other objects were inherited
with their original hashes. Compiler-recorded build dependencies and actual
loaded modules are audited; this is not a hermetic compiler-environment proof.

Fresh qualification against the original WI commit retains 1248 configurations
per variant (1236 H128 and 12 other-head controls), plus a separate fixed six.
Each route has 30 untrimmed complete warm API event and wall samples in two
opposite module orders. BF16 has 660 eligible comparisons and 588 explicit NA
entries under the original comparator gate. Known OOM cases were not rerun.
The full collection contains 115200 event and 115200 wall observations.
Within H128 alone, 650 configurations have BF16 comparisons and 586 are NA.
The candidate still trails BF16 by more than 3% in 611 event-time comparisons
and 605 wall-time comparisons; 39/45 respectively are within 3% or faster.
This optimization does not establish full H128 parity with BF16.
All 432 Sq1 decode configurations (424 H128 and eight other-head controls)
have no whole30 event or wall slowdown>3%.
Original full results retain 23 event and 43 wall flags, 46 cases in their union;
these include 45 Prefill controls and one MTP case. Integrity of collection is
distinct from a full no-regression claim. Eight Prefill block0 pairs span a
pause with separate clean reservations; their provenance remains explicit.

The 45 fixed Prefill flags received two additional opposite-order rounds:
original30, added60 and all six block means remain separate in the report.
Two V4.0 FP8 Prefill controls remain above 3% in the added60 mean: event/wall
+4.6252%/+5.3578% and+3.8509%/+3.4019%. Ten controls have either an added60
or six-block-median event/wall flag. Their inherited main-object bytes match,
but host/runtime/module placement and timing effects have not been isolated.
These are explicitly retained performance limitations, not waived outliers or
proof that environment alone caused the differences.

The fixed six fresh original-WI-to-L90 event improvements range from 8.1111% to
13.7388%; they do not measure the incremental L89-to-L90 change. The separate
attention-main hardware appendix uses L89 as its parent. For capacity1024,
actual 13–39 extra rows, that appendix records:

| Main profile | FP8 L89 -> L90 | Mixed FP4 L89 -> L90 | Same-run BF16 |
| --- | ---: | ---: | ---: |
| Duration, microseconds | 224.760 -> 201.487 | 237.078 -> 213.275 | 182.615 /183.635 |
| HBM read, GB/s | 299.255 -> 333.818 | 273.239 -> 303.729 | 489.148 /486.422 |
| TFU, percent | 16.175 -> 18.043 | 15.335 -> 17.046 | 19.908 /19.797 |

Read bytes are effectively unchanged. The main-profile improvement is about
10%, with a remaining BF16 duration gap of 10.3345%/16.1412%. Main-only hardware
profiles and complete API timings are separate measurements; API times are
not used to synthesize HBM throughput. No full hardware-counter sweep is claimed.

The clean cumulative numerical comparison covers 911 instances per variant:
908 pass and the same 3 inherited UE8 failures, with no new or changed failures.
The older strict20 appendix separately retains 4 inherited failed comparisons.
Neither set is relabelled a numerical PASS or given wider tolerances. The new
regression test covers short -> full -> restored-short lengths in eager and
Graph replay for both formats, checking outputs, LSE and scheduler metadata.
CPU task projection checks cover 13470 scenarios and 164103 task ordinals.

Closed raw evidence, exclusions, every timing sample, BF16 comparisons and
limited qualification decisions are under
`benchmark/v41_decode_memory_bound_20261016/release_l90_20261019/`.
The report is generated after the real commit receipt. Qualification applies
to the isolated committed source and frozen module; unrelated primary-worktree
experiments, including its older private-header edit, are preserved separately.
