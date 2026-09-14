#pragma once

#include "params.h"

template<typename InputT, int Arch, bool IsFP8 = true, int BlockM = 128>
void run_flash_sparse_decode_wg_kernel(Flash_fwd_params &params, hggcStream_t stream);

namespace flashmla {
namespace dsa {

namespace hs64 {

// HS64 sparse-decode entry for BF16. The translation unit instantiates only
// D=512 and D=576; FP8 remains excluded.
void run_flash_sparse_decode_wg_kernel_hs64(
    Flash_fwd_params &params,
    hggcStream_t stream);

} // namespace hs64

// HS64 consumes two 64-token blocks per scheduler iteration. Keep that
// metadata quantum limited to BF16 PPU1.5; FP8 and PPU1.0 retain the original
// 64-token scheduler contract. An HS64 fallback must use the same quantum
// when padding the main cache before extra KV, despite its 64-token tile.
inline bool sparse_decode_needs_128_token_quantum(
        int ngroups, bool is_fp8_kvcache, bool is_sm89_or_newer) {
    return !is_fp8_kvcache && is_sm89_or_newer &&
           ngroups % 128 == 64;
}

inline int sparse_decode_metadata_block_size_n(
        int ngroups, bool is_fp8_kvcache, bool is_sm89_or_newer) {
    return sparse_decode_needs_128_token_quantum(
               ngroups, is_fp8_kvcache, is_sm89_or_newer)
        ? 128
        : 64;
}

inline bool sparse_decode_hs64_cache_range_supported(
        int num_blocks, int page_size, int head_dim,
        Flash_fwd_params::index_t page_stride) {
    // HS64 exchanges uint32 offsets in 16-byte atoms. Keep every token,
    // including its final copy, inside that 64-GiB relative-address range.
    constexpr uint64_t kMaxElements = uint64_t{1} << 35; // BF16 elements
    if (num_blocks <= 0 || page_size <= 0 || page_stride < 0) return false;
    const uint64_t page_elements = uint64_t(page_size) * head_dim;
    return page_elements <= kMaxElements &&
           (page_stride == 0 ||
            uint64_t(num_blocks - 1) <=
                (kMaxElements - page_elements) / uint64_t(page_stride));
}

inline bool sparse_decode_hs64_addressing_supported(const Flash_fwd_params &p) {
    constexpr int64_t kMaxQueryStride = (int64_t{1} << 31) - 1;
    return p.indices_row_stride >= 0 && p.indices_row_stride <= kMaxQueryStride &&
           sparse_decode_hs64_cache_range_supported(
               p.num_blocks, p.page_block_size, p.d, p.k_batch_stride) &&
           (p.extra_topk < 0 ||
            (p.extra_indices_row_stride >= 0 &&
             p.extra_indices_row_stride <= kMaxQueryStride &&
             sparse_decode_hs64_cache_range_supported(
                 p.extra_num_blocks, p.extra_page_block_size, p.d,
                 p.extra_k_batch_stride)));
}

inline bool sparse_decode_m128_index_tiles_supported(const Flash_fwd_params &p) {
    // M128 prefetches both 32-token halves without element-level bounds checks.
    // Ragged storage uses the predicated legacy loader, not this hot pipeline.
    return p.topk > 0 && p.topk % 64 == 0 &&
           (p.extra_topk < 0 || p.extra_topk % 64 == 0);
}

} // namespace dsa
} // namespace flashmla
