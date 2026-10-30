#pragma once

#include <algorithm>
#include <cstdint>

namespace flashmla::sparse_native_fp8 {

inline constexpr int ValueDim = 512;
inline constexpr int TileKeys = 64;
inline constexpr int MaxTopk = 2048;
inline constexpr int MaxVerifyQueries = 32;

// Measured M890P reuse crossover for sparse FP8 gathers. This is a launch
// policy budget, not a claim about physical cache capacity. Pools below it
// benefit from narrower gathers and more resident CTAs; streaming pools favor
// wider transactions and fewer competing CTAs.
inline bool prefer_kv_reuse(int64_t working_bytes, int l2_bytes) {
    constexpr int ReuseBudgetL2Multiple = 16;
    return l2_bytes > 0 && working_bytes <= int64_t(l2_bytes) * ReuseBudgetL2Multiple;
}

// Keep at least four key tiles per partition while exposing roughly two
// query/head CTAs per CU. Power-of-two splits bound the merge overhead.
inline int choose_splits(int queries, int heads, int width, int compute_units) {
    constexpr int MaxSplits = 8;
    constexpr int MinTilesPerSplit = 4;
    const int head_tile = heads <= 16 ? 16 : 32;
    const int groups = (heads + head_tile - 1) / head_tile;
    const int budget = std::min(MaxSplits, std::max(1, width / TileKeys / MinTilesPerSplit));
    const int desired = std::min<int64_t>(
        budget, std::max<int64_t>(1, 2LL * compute_units / (int64_t(queries) * groups)));
    int splits = 1;
    while (splits * 2 <= desired)
        splits *= 2;
    return splits;
}

// Small head tiles fit three double-buffered or two triple-buffered CTAs per
// M890P CU. Compare complete waves using their measured relative tile costs.
inline bool prefer_two_stages(int queries, int splits, int heads, int compute_units) {
    const int64_t tasks = int64_t(queries) * splits;
    const int64_t two_stage_wave = 3LL * compute_units;
    const int64_t three_stage_wave = 2LL * compute_units;
    constexpr int TwoStageCost = 10;
    constexpr int ThreeStageCost = 8;
    return heads <= 16 && ((tasks + two_stage_wave - 1) / two_stage_wave) * TwoStageCost <=
                              ((tasks + three_stage_wave - 1) / three_stage_wave) * ThreeStageCost;
}

} // namespace flashmla::sparse_native_fp8
