#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>

namespace flashmla::v41prefill {

// This exact family keeps Q in [B, original Sq, original H, D]. Its API
// branch calls the paired quantized launcher directly; all other families
// retain the original folded-Q dispatcher and strides.
inline bool sparse_quant_decode_uses_raw_q_layout(
        bool is_v41_fp8, bool is_bf16_query, bool is_sm89_or_newer,
        int head_size, int head_size_v, int num_heads_k, int ngroups,
        int primary_page_size, int extra_page_size) {
    const char *wi = std::getenv("FLASHMLA_V41_WI");
    const auto pow2 = [](int value) {
        return value > 0 && (value & (value - 1)) == 0;
    };
    return is_v41_fp8 && is_bf16_query && is_sm89_or_newer
        && wi && wi[0] == '1' && wi[1] == '\0'
        && head_size == 512 && head_size_v == 512 && num_heads_k == 1
        && ngroups >= 128 && ngroups % 128 == 0
        && pow2(primary_page_size)
        && (extra_page_size == 0 || pow2(extra_page_size));
}

// AIU reads BF16 vectors with a 16-byte aligned row base and an int head
// stride. Batch/query addressing remains int64_t. An unsafe input is cloned
// contiguously in rank four, preserving this family's stride interpretation.
inline bool sparse_quant_decode_raw_q_alignment_supported(
        const void *q, int64_t batch_stride, int64_t query_stride,
        int64_t head_stride, int head_size) {
    return (reinterpret_cast<uintptr_t>(q) & uintptr_t(15)) == 0
        && batch_stride > 0 && batch_stride % 8 == 0
        && query_stride > 0 && query_stride % 8 == 0
        && head_stride >= head_size && head_stride % 8 == 0
        && head_stride <= std::numeric_limits<int>::max();
}

} // namespace flashmla::v41prefill
