#pragma once

#include "params.h"

template<typename InputT, int Arch, bool IsFP8 = true, int BlockM = 128,
         bool IsV41FP8 = false, bool IsExtraFP4 = false>
void run_flash_sparse_decode_wg_kernel(Flash_fwd_params &params, hggcStream_t stream);

namespace flashmla {
namespace dsa {

namespace hs64 {

// HS64 sparse-decode entries. BF16 covers D=512/576; the V4.1 experiment
// covers D=512 with FP8 main KV and FP8 or FP4 extra KV.
void run_flash_sparse_decode_wg_kernel_hs64(
    Flash_fwd_params &params,
    hggcStream_t stream);

// Experimental V4.1 FP8 path on the HS64/WI2 pipeline.
void run_flash_sparse_decode_wg_kernel_hs64_v41_fp8(
    Flash_fwd_params &params,
    hggcStream_t stream);
void run_flash_sparse_decode_wg_kernel_hs64_v41_fp4(
    Flash_fwd_params &params,
    hggcStream_t stream);

} // namespace hs64

// HS64 consumes two 64-token blocks per scheduler iteration. The default
// FP8 path retains the 64-token scheduler contract; the opt-in V4.1 HS64
// path selects 128 separately in sparse_decode.h.
inline bool sparse_decode_needs_128_token_quantum(
        int ngroups, bool is_fp8_kvcache) {
    return !is_fp8_kvcache &&
           ngroups % 128 == 64;
}

inline int sparse_decode_metadata_block_size_n(
        int ngroups, bool is_fp8_kvcache) {
    return sparse_decode_needs_128_token_quantum(
               ngroups, is_fp8_kvcache)
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

inline bool sparse_decode_hs64_v41_cache_range_supported(
        int num_blocks, int page_size, int data_bytes_per_token,
        int address_unit_bytes, Flash_fwd_params::index_t page_stride_bytes) {
    if (num_blocks <= 0 || page_size <= 0 || page_stride_bytes < 0) return false;
    // The two copy waves exchange uint32 offsets measured in 8-byte FP8 or
    // 4-byte FP4 units. Avoid truncating a page offset before the lane adds
    // its within-token address.
    const uint64_t limit = (uint64_t{1} << 32) * address_unit_bytes;
    const uint64_t row_bytes = uint64_t(page_size) * data_bytes_per_token;
    return row_bytes <= limit &&
        uint64_t(num_blocks - 1) <=
            (limit - row_bytes) / uint64_t(page_stride_bytes ? page_stride_bytes : 1);
}

inline bool sparse_decode_hs64_v41_addressing_supported(
        const Flash_fwd_params &p, bool extra_fp4) {
    return sparse_decode_hs64_v41_cache_range_supported(
               p.num_blocks, p.page_block_size, 512, 8, p.k_batch_stride) &&
           (p.extra_topk < 0 ||
            sparse_decode_hs64_v41_cache_range_supported(
                p.extra_num_blocks, p.extra_page_block_size,
                extra_fp4 ? 256 : 512, extra_fp4 ? 4 : 8,
                p.extra_k_batch_stride));
}

inline bool sparse_decode_m128_index_tiles_supported(const Flash_fwd_params &p) {
    // M128 prefetches both 32-token halves without element-level bounds checks.
    // Ragged storage uses the predicated legacy loader, not this hot pipeline.
    return p.topk > 0 && p.topk % 64 == 0 &&
           (p.extra_topk < 0 || p.extra_topk % 64 == 0);
}

} // namespace dsa
} // namespace flashmla
