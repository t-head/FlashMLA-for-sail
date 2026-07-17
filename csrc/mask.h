/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

namespace flash {

using namespace cute;

struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q)
        : max_seqlen_k(max_seqlen_k), max_seqlen_q(max_seqlen_q) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride,
                                               const int ngroups) {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10500
        static_assert(decltype(size<0>(tensor_))::value == 8, "First dimension must be 8");
#else
        // static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
#endif
        static constexpr bool Need_masking = Causal_mask || !Is_even_MN;
        // if (cute::thread0()) { printf("Causal_mask=%d, Is_even_MN = %d, Need_masking = %d\n", Causal_mask, Is_even_MN, Need_masking); }

        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !Causal_mask;
            const int lane_id = threadIdx.x % 32;
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4);
#else
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
#endif
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
                    const int col_idx_base = col_idx_offset + nj * 16;
#else
                    const int col_idx_base = col_idx_offset + nj * 8;
#endif
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
                        const int col_idx = col_idx_base + j * 4;
#else
                        const int col_idx = col_idx_base + j;
#endif
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit_right = std::min(max_seqlen_k, (row_idx / ngroups + 1 + max_seqlen_k - max_seqlen_q / ngroups));

                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
                            const int col_idx_base = col_idx_offset + nj * 16;
#else
                            const int col_idx_base = col_idx_offset + nj * 8;
#endif
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
                                const int col_idx = col_idx_base + j * 4;
#else
                                const int col_idx = col_idx_base + j;
#endif
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

template <typename Tensor0, typename Tensor1>
__forceinline__ __device__ void apply_indices_mask(Tensor0 &tensor_, Tensor1 &smem_valid_indices,  const int warpN_idx, const int buffer) {
    Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
    const int lane_id = threadIdx.x % 32;
    // const int col_base = warpN_idx * 4;
    // const int col_base_offset = col_base + (lane_id % 4) * 16;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        // const int col_nj = col_base_offset +  nj * 4;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
#if ACOMPUTE_VERSION ==10000
            // const int load_col_idx = warpN_idx * 16 + (lane_id % 4) + nj * 16 + j * 4;
            // ==> (warp_idx / AtomLayoutQ) * MMA_N_S * 16  + nj * 16 + (lane_id % 4) + j * 4
            // const int col_x = warpN_idx + nj;
            // const int col_y = j
            // const int col_z = lane_id % 4
            // (col_x, col_y, col_z) -> (col_z, col_x, col_y) = (warpN_idx + nj, j, lane_id % 4)
            // const int col_in_indices = col_nj + j;
            // const int col_in_indices = load_col_idx;
            // const int col_in_indices = (col%8)*8 + col/8;
            const int col_in_indices = (lane_id % 4) * 16 + warpN_idx *4 + nj * 4 + j;
            // const int col_in_indices = (load_col_idx % 4) * 16 + (load_col_idx / 16) * 4 + (load_col_idx % 16) / 4;
#else
            // const int col_idx = col_base + (lane_id % 4) * 2 + nj * 8 + j;
            const int col_in_indices = warpN_idx * 16 + (lane_id % 4) * 2 + nj * 8 + j;
#endif
            bool is_vaild = smem_valid_indices(buffer, col_in_indices);
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    if (!is_vaild) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
            }
        }
    }
}

} // namespace flash
