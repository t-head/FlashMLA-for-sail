#pragma once

#include <algorithm>
#include <cstdint>
#include "decode/sparse/sparse_decode_wg.h"

namespace flashmla::dsa {

enum class SparseDecodeRoute { Legacy, WG128, HS64 };

struct SparseDecodeDispatchPlan {
    SparseDecodeRoute route;
    int block_m;
    int num_m_tiles;
    int num_sm_parts;
};

inline int sparse_decode_partition_count(
        int num_m_tiles, int num_heads_k, int batch, int ngroups, int sm_count) {
    const int64_t tiles = int64_t(num_m_tiles) * num_heads_k;
    if (tiles <= 0) return 1;
    int parts;
    if (ngroups > 128 && batch < 4) {
        // Preserve the small-batch policy, using the selected launch's tiles.
        parts = int(std::max(int64_t{1}, int64_t(sm_count) / tiles));
    } else {
        int64_t a = tiles, b = sm_count;
        while (b != 0) {
            const int64_t remainder = a % b;
            a = b;
            b = remainder;
        }
        parts = int(sm_count / a);
    }
    return parts <= 320 ? parts : (320 / sm_count) * sm_count;
}

inline SparseDecodeDispatchPlan sparse_decode_dispatch_plan(
        const Flash_fwd_params &p, bool is_fp8, bool can_wg128, int sm_count) {
    // Match the decode launcher's existing 810E normalization.
    if (sm_count == 64) sm_count = 20;
    const auto is_pow2 = [](int n) { return n > 0 && (n & (n - 1)) == 0; };
    const bool pages_pow2 = is_pow2(p.page_block_size) &&
        (p.extra_page_block_size == 0 || is_pow2(p.extra_page_block_size));
    const int fallback_m = p.seqlen_q <= 16 ? 16 : p.seqlen_q <= 32 ? 32 : 64;
    const auto tiles_for_m = [&](int m) {
        return p.q_orig * ((p.ngroups + m - 1) / m);
    };
    SparseDecodeDispatchPlan plan{SparseDecodeRoute::Legacy, fallback_m,
                                 tiles_for_m(fallback_m), 0};

    if (!is_fp8 && can_wg128 && p.ngroups > 0 && p.ngroups % 128 == 0 &&
        pages_pow2 && sparse_decode_m128_index_tiles_supported(p)) {
        const int candidate_tiles = tiles_for_m(128);
        const int candidate_parts = sparse_decode_partition_count(
            candidate_tiles, p.h, p.b, p.ngroups, sm_count);
        const int64_t kv_tokens = int64_t(p.topk) + std::max(p.extra_topk, 0);
        const int min_kv_per_part = p.q_orig > 1 ? 512 : 256;
        // Test the candidate's own partitions, never the supplied metadata's
        // row count. Both metadata creation and dispatch then choose one route.
        if (int64_t(p.b) * kv_tokens > int64_t(candidate_parts) * min_kv_per_part) {
            plan = {SparseDecodeRoute::WG128, 128, candidate_tiles, candidate_parts};
        }
    }
    if (!is_fp8 && plan.route == SparseDecodeRoute::Legacy &&
        p.ngroups % 128 == 64 && pages_pow2 &&
        sparse_decode_hs64_addressing_supported(p)) {
        plan = {SparseDecodeRoute::HS64, 64, tiles_for_m(64), 0};
    }
    if (plan.num_sm_parts == 0) {
        // FP8 and older architectures retain their existing metadata policy.
        const int scheduler_tiles = !is_fp8 && can_wg128
            ? plan.num_m_tiles : (p.ngroups + 63) / 64;
        plan.num_sm_parts = sparse_decode_partition_count(
            scheduler_tiles, p.h, p.b, p.ngroups, sm_count);
    }
    return plan;
}

} // namespace flashmla::dsa
