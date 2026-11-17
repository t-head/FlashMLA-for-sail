# SM80 INT8 fused attention snapshot

This snapshot contains the selected stage90 implementation, based on
`575892e429513a299ce8b70a57a3be7c5237b5f3`. It preserves the measured computation
and restores the repository's HGGC SDK names from the CUDA compatibility build.
Rejected experiments and diagnostic instrumentation are excluded.

## Scope

Decode uses **INT8 main KV and packed INT4 extra KV**. Existing layout enum names
still contain `FP8Sparse` and `FP4Sparse`; those names do not describe the numeric
payload used in this experiment. The shared dequantization paths in this branch
interpret those payloads as integers. This is an INT8/INT4 variant, not an
SM89 FP8/E2M1 compatibility release.

The original 124 cases comprise 75 Decode cases and 49 BF16 KV Prefill cases.
Validation used SM80, Q normalization off, and
`FLASHMLA_FUSED_OUTPUT_CHANNELWISE=0` (block32 output). It does not establish
channelwise/macro1, Q normalization on, other architectures, or all optional
input combinations as validated configurations.

## Implementation

- Preserve integer KV decoding and the exact software output quantization.
- Use SM80 H64 Prefill hooks with the required Q publication synchronization,
  unsigned address calculations, and optimized output handling.
- Use static SplitKV partitions for suitable H8 cases, absolute partial-result
  slots, and a single temporary workspace allocation.
- Use complete-query dispatch for eligible small Decode cases. H64 with 16 total
  queries avoids a combine launch. H8 with 16 total queries uses M8/N64 and 512
  threads; Q and output ownership remains with the original first 256 threads.
- Allocate scale factors directly with the required strides and use cached
  device properties and kernel attributes.

## Acceptance results

The original inputs, correctness thresholds, and timing intervals were retained.
All 124 cases passed 966 correctness Boolean checks. A separate audit recomputed
744 performance and TFU values. The evidence archive contains 1,418 SHA-verified
files. Measurements used the same 810E GPU
at a fixed 1,700 MHz, with idle checks around each job.

The strict target is `fused / no_fused_main < 1.05`.

| Family | Cases | Target passed | Remaining | Faster than full |
|---|---:|---:|---:|---:|
| H64 Decode | 39 | 37 | 2 | 39 |
| H8 Decode | 36 | 33 | 3 | 36 |
| H64 BF16 Prefill | 37 | 0 | 37 | 37 |
| H8 BF16 Prefill | 12 | 10 | 2 | 12 |
| Total | 124 | 80 | 44 | 124 |

All cases are faster than no-fused full. The all-case main target remains
**incomplete**. H64 Prefill has a median main-relative overhead of 10.62484%.

See [all 124 cases](../benchmark/results/sm80-int8-fused-all124.csv) and
[the remaining 44 cases](../benchmark/results/sm80-int8-fused-remaining44.csv)
for each case's fused/full/main latency and all three TFU values.
The [validation manifest](../benchmark/results/sm80-int8-fused-validation.json)
records source hashes, result-table hashes, and the SDK-name mapping.

`fused` and `no_fused_full` use the original event-timing medians (five samples,
ten calls per sample). `no_fused_main` uses the original attention-main kernel
measurement, excluding preprocessing, postprocessing, metadata, and combine.
No ACU or newly collected diagnostic timing replaced the original acceptance
protocol; no-fused main retains its original profiler-based kernel timing.

TFU is effective logical QK+PV utilization:
`2 * valid_positions * heads * (Dqk + Dv) / time / 139.2e12 * 100`.
The 139.2 TFLOP/s denominator is the approved BF16 reference, not an INT8 peak
or a measured hardware utilization counter.

## Provenance

- Validated module SHA-256:
  `b2537a471466d653600883e5ab2751ad5d01074d4fe04ee481109c6e26ec2682`
- Frozen generated-source manifest SHA-256:
  `91760fcb076bba80b61982ce5a6ca68568299c286db042b6b35b93a7cfef4966`
- Original case catalog SHA-256:
  `969e122867de2c9eea3a0f1ed8d4d5a6e5b3c10d29051f346ebd940180d2d94a`
- Full124 evidence archive SHA-256:
  `3e61f10ead06e7d87049f296c4f19addfb6c3b7c4a59e5a8ea1461e95ea39f8a`
- actlize submodule:
  `0146d9e9462e38da4f455df1f71eae96465c45ff` (unchanged from the base).

The original runtime package has 4,630 entries; the candidate replaces only the
extension module. Kernel code in this snapshot is bound to the validated
generated source by the SDK-name round-trip audit. The result tables describe
that validated build, not a new native-HGGC build or a cross-architecture test.
