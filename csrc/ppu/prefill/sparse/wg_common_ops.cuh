/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include "acc_vreg_fraga.h"

// Shared by BF16 and quantized WI. Keep tile math independent of cache loading,
// scheduler, shared-memory ownership and publication/barrier policies.
namespace flashmla::sparse_wi {
using namespace cute;

__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup)
{
    // In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
    // You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
    int row_idx = (idx_in_warpgroup / 32) * 16 + local_row_idx * 8 + (idx_in_warpgroup % 32 / 4);
    return row_idx;
}

template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_acc(Tensor<Engine, Layout> const &tensor)
{
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    NumericArrayConverterPPU<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <
    typename Tensor0, typename Tensor1,
    typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(Tensor2 &tCrC, Tensor0 const &tCrA, Tensor1 const &tCrB, TiledMma &tiled_mma)
{
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i)
    {
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), tCrC);
    }
}

__forceinline__ __device__ void kernel_sleep_ns()
{
    __nanosleep(1);
}

template <
    typename TiledMMA,
    typename TiledCopyA,
    typename TiledCopyB,
    typename ThrCopyA,
    typename ThrCopyB,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4>
__forceinline__ __device__ void qkt_gemm_one_tile_sQ(
    TiledMMA &tiled_mma,
    TiledCopyA &smem_tiled_copy_Q,
    TiledCopyB &smem_tiled_copy_K,
    ThrCopyA &smem_thr_copy_Q,
    ThrCopyB &smem_thr_copy_K,
    Tensor<Engine0, Layout0> const &sQ_tiled,         // (MMA, 1, 4)
    Tensor<Engine1, Layout1> const &thr_mma_sQ_tile,  // (MMA, 1, 4)
    Tensor<Engine2, Layout2> const &sKV_tiled,        // (MMA, 1, 4)
    Tensor<Engine3, Layout3> const &thr_mma_sKV_tile, // (MMA, 1, 4)
    Tensor<Engine4, Layout4> &rP,                     // ((2, 2, 8), 1, 1)
    int idx_in_warpgroup)
{
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor rQ = thr_mma.partition_fragment_A(sQ_tiled);
    Tensor rK = thr_mma.partition_fragment_B(sKV_tiled);

    Tensor rQ_copy_view = smem_thr_copy_Q.retile_D(rQ);
    CUTE_STATIC_ASSERT_V(size<1>(thr_mma_sQ_tile) == size<1>(rQ_copy_view)); // M

    Tensor rK_copy_view = smem_thr_copy_K.retile_D(rK);
    CUTE_STATIC_ASSERT_V(size<1>(thr_mma_sKV_tile) == size<1>(rK_copy_view)); // M

    cute::copy(smem_tiled_copy_Q, thr_mma_sQ_tile, rQ_copy_view);
    cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile, rK_copy_view);

    cute::gemm(tiled_mma, rQ_copy_view(_, _, _0{}), rK_copy_view(_, _, _0{}), rP);
    cute::gemm(tiled_mma, rQ_copy_view(_, _, _1{}), rK_copy_view(_, _, _1{}), rP);
    cute::gemm(tiled_mma, rQ_copy_view(_, _, _2{}), rK_copy_view(_, _, _2{}), rP);
    cute::gemm(tiled_mma, rQ_copy_view(_, _, _3{}), rK_copy_view(_, _, _3{}), rP);
}

template <
    typename TiledMMA,
    typename TiledCopy,
    typename ThrCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3>
__forceinline__ __device__ void qkt_gemm_one_tile_rQ(
    TiledMMA &tiled_mma,
    TiledCopy &smem_tiled_copy_K,
    ThrCopy &smem_thr_copy_K,
    Tensor<Engine0, Layout0> const &thr_mma_rQ_tile,  // (MMA, 1, 4)
    Tensor<Engine1, Layout1> const &sKV_tiled,        // (MMA, 1, 4)
    Tensor<Engine2, Layout2> const &thr_mma_sKV_tile, // (MMA, 1, 4)
    Tensor<Engine3, Layout3> &rP,                     // ((2, 2, 8), 1, 1)
    int idx_in_warpgroup)
{
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor rK = thr_mma.partition_fragment_B(sKV_tiled);
    Tensor rK_copy_view = smem_thr_copy_K.retile_D(rK);
    CUTE_STATIC_ASSERT_V(size<1>(thr_mma_sKV_tile) == size<1>(rK_copy_view)); // M

    cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile, rK_copy_view);

    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _0{}), rK_copy_view(_, _, _0{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _1{}), rK_copy_view(_, _, _1{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _2{}), rK_copy_view(_, _, _2{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _3{}), rK_copy_view(_, _, _3{}), rP);
}

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void save_rP1_to_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> &sP,
    int idx_in_warpgroup
) {
    typename T::TiledMma tiled_mma;
    auto r2s_copy = make_tiled_copy_A(typename T::SmemCopyAtomS{}, tiled_mma);
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);

    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}

template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void save_rP0_to_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> &sP,
    int idx_in_warpgroup)
{
    typename T::TiledMma tiled_mma;
    auto r2s_copy = make_tiled_copy_C(typename T::SmemCopyAtomS{}, tiled_mma);
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);

    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void wg0_scale_rP0(
    Tensor<Engine0, Layout0> const &sScale1,	// (BLOCK_M)
    Tensor<Engine1, Layout1> const &rP0,		// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rPb,		// ((2, 2, 8), 1, 1)
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
        CUTLASS_PRAGMA_UNROLL
        if constexpr (T::Arch_value == 80) {
            for (int i = local_row_idx ? 4 : 0; i < size(rP0); i += 8) {
                rPb(i) = (typename T::InputT)(rP0(i)*scale_factor);
                rPb(i+1) = (typename T::InputT)(rP0(i+1)*scale_factor);
                rPb(i+2) = (typename T::InputT)(rP0(i+2)*scale_factor);
                rPb(i+3) = (typename T::InputT)(rP0(i+3)*scale_factor);
            }
        } else {
            for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
                rPb(i) = (typename T::InputT)(rP0(i)*scale_factor);
                rPb(i+1) = (typename T::InputT)(rP0(i+1)*scale_factor);
            }
        }
    }
}

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void wg0_rescale_rO0(
    Tensor<Engine0, Layout0> &rO0,
    Tensor<Engine1, Layout1> &sScale1,
    float rL[2],
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
        CUTLASS_PRAGMA_UNROLL
        if constexpr (T::Arch_value == 80) {
            for (int i = local_row_idx ? 4 : 0; i < size(rO0); i += 8) {
                rO0(i) *= scale_factor;
                rO0(i+1) *= scale_factor;
                rO0(i+2) *= scale_factor;
                rO0(i+3) *= scale_factor;
            }
        } else {
            for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
                rO0(i) = rO0(i)*scale_factor;
                rO0(i+1) = rO0(i+1)*scale_factor;
            }
        }
        rL[local_row_idx] *= scale_factor;
    }

}

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void wg1_scale0_rO1(
    Tensor<Engine0, Layout0> &rO1,
    Tensor<Engine1, Layout1> &sScale0,
    Tensor<Engine2, Layout2> &sScale1,
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale0(row_idx) * sScale1(row_idx);
        CUTLASS_PRAGMA_UNROLL
        if constexpr (T::Arch_value == 80) {
            for (int i = local_row_idx ? 4 : 0; i < size(rO1); i += 8) {
                rO1(i) *= scale_factor;
                rO1(i+1) *= scale_factor;
                rO1(i+2) *= scale_factor;
                rO1(i+3) *= scale_factor;
            }
        } else {
            for (int i = local_row_idx ? 2 : 0; i < size(rO1); i += 4) {
                rO1(i) = (rO1(i)*scale_factor);
                rO1(i+1) = (rO1(i+1)*scale_factor);
            }
        }
    }
}

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void wg0_scale0_rO0(
    Tensor<Engine0, Layout0> &rO0,
    Tensor<Engine1, Layout1> &sScale0,
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale0[row_idx];
        CUTLASS_PRAGMA_UNROLL
        if constexpr (T::Arch_value == 80) {
            for (int i = local_row_idx ? 4 : 0; i < size(rO0); i += 8) {
                rO0(i) *= scale_factor;
                rO0(i+1) *= scale_factor;
                rO0(i+2) *= scale_factor;
                rO0(i+3) *= scale_factor;
            }
        } else {
            for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
                rO0(i) *= scale_factor;
                rO0(i+1) *= scale_factor;
            }
        }
    }
}

template <
    typename T,
    bool IS_R,
    typename Engine0, typename Layout0>
__forceinline__ __device__ auto get_half_V(
    Tensor<Engine0, Layout0> &sK)
{
    Tensor sV = make_tensor(sK.data(), (typename T::SmemLayoutV){});
    Tensor sVL =  flat_divide(sV, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
    return sVL;
}

} // namespace flashmla::sparse_wi
