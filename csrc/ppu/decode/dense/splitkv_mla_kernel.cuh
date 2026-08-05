/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cute/util/debug.hpp>

#include <c10/cuda/CUDAException.h>

#include "params.h"
#include "config.h"
#include "kerutils/common/static_switch.h"
#include "kernel_traits.h"
#include "utils.h"
#include "kerutils/device/ppu/softmax.cuh"
#include "kerutils/device/ppu/mask.cuh"

#include "traits.h"
#include "acc_vreg_fraga.h"
#include "ppuxx/decode/combine/combine.cuh"

#include <hggc_ad.h>

using namespace cute;
using cutlass::arch::NamedBarrier;

// Here we use MAX_INIT_VAL_SM to initialize sM, and MAX_INIT_VAL for masking
// The reason is that, we need to calculate new_max = max(sM(row_idx), cur_max*scale_softmax_log2)
// so we must guarantee that MAX_INIT_VAL*scale_softmax_log2 < MAX_INIT_VAL_SM
static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

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
    // convert_op:: accum(tensor.data());
    // auto frag = convert_op(accum);
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// Launch TMA copy for a range of KV tile
// A tile has a shape of BlockN (32) x 64
template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_copy_aiu(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,       // (BLOCK_N, HEAD_DIM_K), swizzled
    const Flash_fwd_mla_params &params,
    __mbarrier_t *barriers_K,
    int warp_idx)
{
    // if (warp_idx == 0) {
    Tensor cur_gKV = gKV(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
    Tensor cur_sKV = sKV(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
    cute::copy(tiled_copy, cur_gKV, cur_sKV);
    // cutlass::arch::cpasync_barrier_arrive_noinc(&barriers_K[START_HEAD_DIM_TILE_IDX]);
    // }

    if constexpr (START_HEAD_DIM_TILE_IDX + 1 < END_HEAD_DIM_TILE_IDX)
    {
        launch_kv_tiles_copy_aiu<START_HEAD_DIM_TILE_IDX + 1, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV, params, barriers_K, warp_idx);
    }
}

// Launch TMA copy for a range of KV tile
// A tile has a shape of BLOCK_N (32) x 64
template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,       // (BLOCK_N, HEAD_DIM_K), swizzled
    const Flash_fwd_mla_params &params,
    __mbarrier_t *barriers_K,
    int warp_idx)
{
    launch_kv_tiles_copy_aiu<START_HEAD_DIM_TILE_IDX, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV, params, barriers_K, warp_idx);
    // if (warp_idx == 0)
    cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PPU: shared memory not support init by zero, need clear if not align.
template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = true, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                     Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                     Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0)
{

    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});

    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D)); // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D)); // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
    for (int m = 0; m < size<1>(S); ++m)
    {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)
        {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k)
            {
                if (Is_even_K || predicate_K(k))
                {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                }
                else if (Clear_OOB_K)
                {
                    cute::clear(D(_, m, k));
                }
            }
        }
        else if (Clear_OOB_MN)
        {
            cute::clear(D(_, m, _));
        }
    }
}

// Prefetch some KV tiles
// Currently this is not used because it leads to performance degradation
// template<
//     int START_HEAD_DIM_TILE_IDX,
//     int END_HEAD_DIM_TILE_IDX,
//     typename TMA_K_OneTile,
//     typename Engine0, typename Layout0
// >
// __forceinline__ __device__ void prefetch_kv_tiles(
//     Tensor<Engine0, Layout0> const &gKV,	// (BLOCK_N, HEAD_DIM_K)
//     TMA_K_OneTile &tma_K,
//     int idx_in_warpgroup
// ) {
//     if (idx_in_warpgroup == 0) {
//         auto thr_tma = tma_K.get_slice(_0{});
//         Tensor cur_gKV = thr_tma.partition_S(gKV)(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
//         cute::prefetch(tma_K, cur_gKV);
//         if constexpr (START_HEAD_DIM_TILE_IDX+1 < END_HEAD_DIM_TILE_IDX) {
//             prefetch_kv_tiles<START_HEAD_DIM_TILE_IDX+1, END_HEAD_DIM_TILE_IDX>(gKV, tma_K, idx_in_warpgroup);
//         }
//     }
// }

// Adapted from https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
// * Copyright (c) 2024, Tri Dao.
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
    // unsigned long long start = clock64();
    // 10ns ≈ 10,00000 cycles(if ppu freq is 1 GHz)
    // while (!(clock64() - start < 170000000ULL)); // 10us * 1700 cycles/us
    __nanosleep(1);
}

// Wait for one KV-tile to be ready, and then calculate P += Q K^T for one Q-tile (BLOCK_SIZE_Mx64) and one KV-tile (PAGE_BLOCK_SIZEx64)
// The Q-tile should be in shared memory
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

// Pipelined TMA wait and Q K^T gemm
// In order to overlap memory copy (G->S copy for K) and computation, we divide both Q and K into tiles of shape (BLOCK_SIZE_M, 64), and (BLOCK_N, 64) respectively, and then do the computation as follows:
// - Wait for the 0-th tile to be ready using `barrier.wait()`
// - Compute Q K^T for the 0-th tile
// - Wait for the 1-st tile to be ready
// - Compute Q K^T for the 1-st tile
// ...
// This gives latter tiles more time to be ready, and thus can overlap the memory copy and computation
template <
    typename T,    // Traits
    int PHASE_IDX, // See comments in the code
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3>
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm(
    Tensor<Engine0, Layout0> &sQ,   // (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV0, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV1, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,   // ((2, 2, 8), 1, 1)
    Tensor<Engine3, Layout3> &rQ8,  // The 8-th tile of Q. We store it separately to leave some room for storing sP1
    __mbarrier_t *barriers,
    bool &cur_phase,
    int idx_in_warpgroup,
    const int warp_idx)
{
    typename T::TiledMma tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename T::SmemCopyAtomK{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(warp_idx * 32);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(warp_idx * 32);

    Tensor sQ_tiled = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, _64>{})(_, _, _0{}, _); // (BLOCK_SIZE_M, 64, 9)
    Tensor sKV0_tiled = flat_divide(sKV0, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);  // (BLOCK_N, 64, 9)
    Tensor sKV1_tiled = flat_divide(sKV1, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);  // (BLOCK_N, 64, 9)
    Tensor thr_mma_sQ_tiled = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ_tiled));
    Tensor thr_mma_sKV0_tiled = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV0_tiled));
    Tensor thr_mma_sKV1_tiled = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV1_tiled));

    #define QKT_GEMM_ONE_TILE(TILE_IDX) \
        if constexpr(TILE_IDX == 8) { \
            qkt_gemm_one_tile_rQ(tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, \
                    rQ8, sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        } else if constexpr(TILE_IDX < 4) { \
            qkt_gemm_one_tile_sQ(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, \
                    smem_thr_copy_Q, smem_thr_copy_K, \
                    sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), \
                    sKV0_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV0_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        } else  { \
            qkt_gemm_one_tile_sQ(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, \
                    smem_thr_copy_Q, smem_thr_copy_K, \
                    sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), \
                    sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        }

// #define QKT_GEMM_ONE_TILE(TILE_IDX)                                                                                \
//     if constexpr (TILE_IDX == 8)                                                                                   \
//     {                                                                                                              \
//         qkt_gemm_one_tile_rQ(tiled_mma, smem_tiled_copy_K, smem_thr_copy_K,                                        \
//                              rQ8, sKV0_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV0_tiled(_, _, _, Int<TILE_IDX>{}), \
//                              rP, idx_in_warpgroup);                                                                \
//     }                                                                                                              \
//     else                                                                                                           \
//     {                                                                                                              \
//         qkt_gemm_one_tile_sQ(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,                                      \
//                              smem_thr_copy_Q, smem_thr_copy_K,                                                     \
//                              sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}),          \
//                              sKV0_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV0_tiled(_, _, _, Int<TILE_IDX>{}),      \
//                              rP, idx_in_warpgroup);                                                                \
//     }

    if constexpr (PHASE_IDX == 0)
    {
        // In PHASE-0, warpgroup 0 calculates Q K^T for the first 4 tiles
        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1))
        {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
        // cur_phase ^= 1;
    }
    else if constexpr (PHASE_IDX == 1)
    {
        // In PHASE-1, warpgroup 1 calculates Q K^T for all the 9 tiles
        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1))
        {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        QKT_GEMM_ONE_TILE(8);

        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1))
        {
            kernel_sleep_ns();
        };
        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
        cur_phase = (cur_phase + 1) & 1;
    }
    else
    {
        // In PHASE-2, warpgroup 0 calculates Q K^T for the last 5 tiles
        static_assert(PHASE_IDX == 2);

        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1))
        {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        QKT_GEMM_ONE_TILE(8);
        cur_phase = (cur_phase + 1) & 1;
    }
}

template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2>
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm_no_pipeline(
    Tensor<Engine0, Layout0> &rQ,  // (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV, // (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,  // ((2, 2, 8), 1, 1)
    int idx_in_warpgroup)
{
    typename T::TiledMma tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);

    Tensor rK = thr_mma.partition_fragment_B(sKV); // (MMA, 1, 576/16=36)
    auto smem_tiled_copy_K = make_tiled_copy_B(typename T::SmemCopyAtomK{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx_in_warpgroup);
    Tensor rK_copy_view = smem_thr_copy_K.retile_D(rK);
    auto tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV));
    CUTE_STATIC_ASSERT_V(size<1>(tSsK) == size<1>(rK_copy_view)); // M
    cute::copy(smem_tiled_copy_K, tSsK, rK_copy_view);

    // auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    // auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(idx_in_warpgroup);
    // Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));
    // Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    // Tensor rQ = smem_thr_copy_Q.retile_D(tSrQ);
    // CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(rQ));
    // cute::copy(smem_tiled_copy_Q, tSsQ, rQ);

    cute::clear(rP);
    gemm(rP, rQ, rK, tiled_mma);
}

// Compute O += PV, where P resides in register
template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2>
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_localP(
    Tensor<Engine0, Layout0> &rP,       // ((2, 2, 8), 1, 1), fragment A layout
    Tensor<Engine1, Layout1> &sKV_half, // (HEAD_DIM_V/2, BLOCK_N)
    Tensor<Engine2, Layout2> &rO,       // ((2, 2, 32), 1, 1)
    int idx_in_warpgroup,
    int warp_idx)
{
    // const int warp_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);
    typename T::TiledMma tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);

    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(warp_idx * 32);
    Tensor rVt = thr_mma.partition_fragment_B(sKV_half);
    Tensor rVt_copy_view = smem_thr_copy_Vt.retile_D(rVt);
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));

    CUTE_STATIC_ASSERT_V(size<1>(tSsVt) == size<1>(rVt_copy_view)); // M
    cute::copy(smem_tiled_copy_Vt, tSsVt, rVt_copy_view);

    gemm(rO, rP, rVt, tiled_mma);
}

// Compute O += PV, where P resides in shared memory
template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2>
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_remoteP(
    Tensor<Engine0, Layout0> &sP,
    Tensor<Engine1, Layout1> &sKV_half, // (HEAD_DIM_V/2, BLOCK_N)
    Tensor<Engine2, Layout2> &rO,       // ((2, 2, 32), 1, 1)
    int idx_in_warpgroup,
    int warp_idx)
{
    // TiledMMA tiled_mma = (typename T::TiledMMA_PV_RemoteP){};
    // const int warp_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

    typename T::TiledMma tiled_mma;
    auto smem_tiled_copy_P = make_tiled_copy_A(typename T::SmemCopyAtomP{}, tiled_mma);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(idx_in_warpgroup);
    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(warp_idx * 32);

    auto tSsP = smem_thr_copy_P.partition_S(sP);
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));

    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor thr_mma_sP = thr_mma.partition_fragment_A(sP);
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sKV_half); // (MMA, 1, 64/16=4)

    Tensor rP_copy_view = smem_thr_copy_P.retile_D(thr_mma_sP);
    Tensor rVt_copy_view = smem_thr_copy_Vt.retile_D(thr_mma_sKV_half);

    cute::copy(smem_tiled_copy_P, tSsP, rP_copy_view);
    cute::copy(smem_tiled_copy_Vt, tSsVt, rVt_copy_view);
    gemm(rO, rP_copy_view, rVt_copy_view, tiled_mma);
}

#if ACOMPUTE_VERSION == 10000
template<
    typename T,
    bool DO_OOB_FILLING,
    // typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4
>
__forceinline__ __device__ auto wg0_bunch_0(
    // Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,	// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,	// ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0,	// (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,	// (BLOCK_SIZE_M)
    float rL[2],
    int rRightBorderForQSeq[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup
) {

    // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rP0); i += 8) {
            if constexpr (DO_OOB_FILLING) {
                int token_idx = start_token_idx + (i/8)*16 + idx_in_warpgroup%4;
                rP0(i) = token_idx < rRightBorderForQSeq[local_row_idx] ? rP0(i) : MAX_INIT_VAL;
                rP0(i+1) = token_idx+4 < rRightBorderForQSeq[local_row_idx] ? rP0(i+1) : MAX_INIT_VAL;
                rP0(i+2) = token_idx+8 < rRightBorderForQSeq[local_row_idx] ? rP0(i+2) : MAX_INIT_VAL;
                rP0(i+3) = token_idx+12 < rRightBorderForQSeq[local_row_idx] ? rP0(i+3) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(max(rP0(i), rP0(i+1)), max(rP0(i+2), rP0(i+3))));
        }

        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        // Update sM and sL
        cur_max *= scale_softmax_log2;
        float new_max = max(sM(row_idx), cur_max);
        float scale_for_old = exp2f(sM(row_idx) - new_max);

        __syncwarp(); // Make sure all reads have finished before updating sM

        if (idx_in_warpgroup % 4 == 0)
        {
            sScale0(row_idx) = scale_for_old;
            sM(row_idx) = new_max;
        }

        // Scale-O
        // CUTLASS_PRAGMA_UNROLL
        // for (int i = local_row_idx ? 4 : 0; i < size(rO0); i += 8) {
        //     rO0(i) *= scale_for_old;
        //     rO0(i+1) *= scale_for_old;
        //     rO0(i+2) *= scale_for_old;
        //     rO0(i+3) *= scale_for_old;
        // }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rP0); i += 8) {
            rP0(i) = exp2f(rP0(i)*scale_softmax_log2 - new_max);
            rP0(i+1) = exp2f(rP0(i+1)*scale_softmax_log2 - new_max);
            rP0(i+2) = exp2f(rP0(i+2)*scale_softmax_log2 - new_max);
            rP0(i+3) = exp2f(rP0(i+3)*scale_softmax_log2 - new_max);
            cur_sum += (rP0(i) + rP0(i+1) + rP0(i+2) + rP0(i+3));
        }

        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }

    return convert_acc<typename T::InputT>(rP0);
}

template<
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    // typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5>
__forceinline__ __device__ auto wg1_bunch_0(
    // Tensor<Engine0, Layout0> &rP1b,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &sScale1, // (BLOCK_SIZE_M)
    Tensor<Engine2, Layout2> &rO1,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    int rRightBorderForQSeq[2],
    Tensor<Engine4, Layout4> const &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine5, Layout5> &rP1,           // ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup)
{
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx)
    {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8)
        {
            if constexpr (IS_BLK1_LAST || IS_BLK2_LAST)
            {
                // Need to apply the mask when either this block is the last one, or
                // the next block is the last one (because of the causal mask)
                // int token_idx = start_token_idx + (i/4)*8 + idx_in_warpgroup%4*2;
                int token_idx = start_token_idx + (i / 8) * 16 + idx_in_warpgroup % 4;
                rP1(i) = token_idx < rRightBorderForQSeq[local_row_idx] ? rP1(i) : MAX_INIT_VAL;
                rP1(i + 1) = token_idx + 4 < rRightBorderForQSeq[local_row_idx] ? rP1(i + 1) : MAX_INIT_VAL;
                rP1(i + 2) = token_idx + 8 < rRightBorderForQSeq[local_row_idx] ? rP1(i + 2) : MAX_INIT_VAL;
                rP1(i + 3) = token_idx + 12 < rRightBorderForQSeq[local_row_idx] ? rP1(i + 3) : MAX_INIT_VAL;
            }
            else if constexpr (IS_BLK0_LAST)
            {
                rP1(i) = rP1(i + 1) = rP1(i + 2) = rP1(i + 3) = MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(max(rP1(i), rP1(i + 1)), max(rP1(i + 2), rP1(i + 3))));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        cur_max *= scale_softmax_log2;

        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);

        __syncwarp();
        if (idx_in_warpgroup % 4 == 0)
        {
            sM(row_idx) = new_max;
            sScale1(row_idx) = scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        if constexpr (!IS_BLK0_LAST)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8)
            {
                rP1(i) = exp2f(rP1(i) * scale_softmax_log2 - new_max);
                rP1(i + 1) = exp2f(rP1(i + 1) * scale_softmax_log2 - new_max);
                rP1(i + 2) = exp2f(rP1(i + 2) * scale_softmax_log2 - new_max);
                rP1(i + 3) = exp2f(rP1(i + 3) * scale_softmax_log2 - new_max);
                cur_sum += (rP1(i) + rP1(i + 1) + rP1(i + 2) + rP1(i + 3));
            }
        }

        // Scale O
        float cur_scale_for_o1 = scale_for_old * sScale0(row_idx);
        // CUTLASS_PRAGMA_UNROLL
        // for (int i = local_row_idx ? 4 : 0; i < size(rO1); i += 8) {
        //     rO1(i) *= cur_scale_for_o1;
        //     rO1(i+1) *= cur_scale_for_o1;
        //     rO1(i+2) *= cur_scale_for_o1;
        //     rO1(i+3) *= cur_scale_for_o1;
        // }

        // // Update rL
        rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
    }

    return convert_acc<typename T::InputT>(rP1);
}
#else
template <
    typename T,
    bool DO_OOB_FILLING,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4>
__forceinline__ __device__ void wg0_bunch_0(
    Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,     // ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    int rRightBorderForQSeq[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup)
{
     // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            if constexpr (DO_OOB_FILLING) {
                int token_idx = start_token_idx + (i/4)*8 + idx_in_warpgroup%4*2;
                rP0(i) = token_idx < rRightBorderForQSeq[local_row_idx] ? rP0(i) : MAX_INIT_VAL;
                rP0(i+1) = token_idx+1 < rRightBorderForQSeq[local_row_idx] ? rP0(i+1) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(rP0(i), rP0(i+1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        // Update sM and sL
        cur_max *= scale_softmax_log2;
        float new_max = max(sM(row_idx), cur_max);
        float scale_for_old = exp2f(sM(row_idx) - new_max);
        __syncwarp();   // Make sure all reads have finished before updating sM
        if (idx_in_warpgroup%4 == 0) {
            sScale0(row_idx) = scale_for_old;
            sM(row_idx) = new_max;
        }

        // Scale-O
        // CUTLASS_PRAGMA_UNROLL
        // for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
        //     rO0(i) *= scale_for_old;
        //     rO0(i+1) *= scale_for_old;
        // }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            rP0(i) = exp2f(rP0(i)*scale_softmax_log2 - new_max);
            rP0(i+1) = exp2f(rP0(i+1)*scale_softmax_log2 - new_max);
            rPb(i) = (typename T::InputT)rP0(i);
            rPb(i+1) = (typename T::InputT)rP0(i+1);
            cur_sum += rP0(i) + rP0(i+1);
        }
        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5>
__forceinline__ __device__ auto wg1_bunch_0(
    Tensor<Engine0, Layout0> &rP1b,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &sScale1, // (BLOCK_SIZE_M)
    Tensor<Engine2, Layout2> &rO1,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    int rRightBorderForQSeq[2],
    Tensor<Engine4, Layout4> const &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine5, Layout5> &rP1,           // ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup)
{
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
            if constexpr (IS_BLK1_LAST || IS_BLK2_LAST) {
                // Need to apply the mask when either this block is the last one, or
                // the next block is the last one (because of the causal mask)
                int token_idx = start_token_idx + (i/4)*8 + idx_in_warpgroup%4*2;
                rP1(i) = token_idx < rRightBorderForQSeq[local_row_idx] ? rP1(i) : MAX_INIT_VAL;
                rP1(i+1) = token_idx+1 < rRightBorderForQSeq[local_row_idx] ? rP1(i+1) : MAX_INIT_VAL;

            } else if constexpr (IS_BLK0_LAST) {
                rP1(i) = rP1(i+1) = MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(rP1(i), rP1(i+1)));
        }

        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        cur_max *= scale_softmax_log2;


        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);
        __syncwarp();
        if (idx_in_warpgroup%4 == 0) {
            sM(row_idx) = new_max;
            sScale1(row_idx) = scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        if constexpr (!IS_BLK0_LAST) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                rP1(i) = exp2f(rP1(i)*scale_softmax_log2 - new_max);
                rP1(i+1) = exp2f(rP1(i+1)*scale_softmax_log2 - new_max);
                rP1b(i) = (typename T::InputT)rP1(i);
                rP1b(i+1) = (typename T::InputT)rP1(i+1);
                cur_sum += rP1(i) + rP1(i+1);
            }
        }

        // Scale O
        float cur_scale_for_o1 = scale_for_old * sScale0(row_idx);
        // CUTLASS_PRAGMA_UNROLL
        // for (int i = local_row_idx ? 2 : 0; i < size(rO1); i += 4) {
        //     rO1(i) *= cur_scale_for_o1;
        //     rO1(i+1) *= cur_scale_for_o1;
        // }

        // Update rL
        rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
    }
}
#endif

// Save rPb (64x64, bfloat16/half) to sP using the stmatrix instruction
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

// Save rPb (64x64, bfloat16/half) to sP using the stmatrix instruction
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

// Retrieve rPb (64x64, bfloat16/half) from sP using the ldmatrix instruction
template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void retrieve_rP_from_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> const &sP,
    int idx_in_warpgroup)
{
    typename T::TiledMma tiled_mma;
    const int warp_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

    auto thr_mma = tiled_mma.get_thread_slice(idx_in_warpgroup);
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(warp_idx * 32);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sP));
    // Tensor tSrQ  = thr_mma.partition_fragment_A(sP);
    // Tensor rQ8 = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(rPb));
    cute::copy(smem_tiled_copy_Q, tSsQ, rPb);
}


// Rescale rP0 and save the result to rPb
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
#if ACOMPUTE_VERSION == 10000
        for (int i = local_row_idx ? 4 : 0; i < size(rP0); i += 8) {
            rPb(i) = (typename T::InputT)(rP0(i)*scale_factor);
            rPb(i+1) = (typename T::InputT)(rP0(i+1)*scale_factor);
            rPb(i+2) = (typename T::InputT)(rP0(i+2)*scale_factor);
            rPb(i+3) = (typename T::InputT)(rP0(i+3)*scale_factor);
        }
#else
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            rPb(i) = (typename T::InputT)(rP0(i)*scale_factor);
            rPb(i+1) = (typename T::InputT)(rP0(i+1)*scale_factor);
        }
#endif
    }
}


// Rescale rO0 according to sScale1
template<
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
#if ACOMPUTE_VERSION == 10000
        for (int i = local_row_idx ? 4 : 0; i < size(rO0); i += 8) {
            rO0(i) *= scale_factor;
            rO0(i+1) *= scale_factor;
            rO0(i+2) *= scale_factor;
            rO0(i+3) *= scale_factor;
        }
#else
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            rO0(i) = rO0(i)*scale_factor;
            rO0(i+1) = rO0(i+1)*scale_factor;
        }
#endif
        rL[local_row_idx] *= scale_factor;
    }

}

// Rescale rO1 according to sScale0
template<
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
#if ACOMPUTE_VERSION == 10000
        for (int i = local_row_idx ? 4 : 0; i < size(rO1); i += 8) {
            rO1(i) *= scale_factor;
            rO1(i+1) *= scale_factor;
            rO1(i+2) *= scale_factor;
            rO1(i+3) *= scale_factor;
        }
#else
        for (int i = local_row_idx ? 2 : 0; i < size(rO1); i += 4) {
            rO1(i) = (rO1(i)*scale_factor);
            rO1(i+1) = (rO1(i+1)*scale_factor);
        }
#endif
        // rL[local_row_idx] *= scale_factor;
    }
}

// Rescale rO0 according to local scale
template<
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
#if ACOMPUTE_VERSION == 10000
        for (int i = local_row_idx ? 4 : 0; i < size(rO0); i += 8) {
            rO0(i) *= scale_factor;
            rO0(i+1) *= scale_factor;
            rO0(i+2) *= scale_factor;
            rO0(i+3) *= scale_factor;
        }
#else
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            rO0(i) *= scale_factor;
            rO0(i+1) *= scale_factor;
        }
#endif
    }
}

// Store O / OAccum
template<
    typename T,
    bool IS_NO_SPLIT,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void store_o(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    Tensor<Engine1, Layout1> &gOorAccum,	// (BLOCK_SIZE_M, HEAD_DIM_V)
    float rL[2],
    char* sO_addr,
    // TMAParams &tma_params,
    const Flash_fwd_mla_params &params,
    int batch_idx,
    int k_head_idx,
    int m_block_idx,
    int num_valid_seq_q,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    using InputT = typename T::InputT;
    using ElementO = std::conditional_t<IS_NO_SPLIT, typename T::InputT, typename T::ElementAccum>;

    using SmemTiledCopyO = std::conditional_t<
        IS_NO_SPLIT,
        typename T::SmemCopyAtomO,
        typename T::SmemCopyAtomOaccum
    >;
    Tensor sOutputBuf = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(sO_addr)),
        typename T::SmemLayoutO{});

    // (SMEM_M,SMEM_N) // Sw<3,3,3> o _0 o (_32,(_64,_8)):(_64,(_1,_2048))
    Tensor rOb = make_tensor_like<ElementO>(rO);

    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); ++idx) {
#if ACOMPUTE_VERSION == 10000
        rOb(idx) = (ElementO)(rO(idx) / rL[(idx / 4) % 2]);
#else
        rOb(idx) = (InputT)(rO(idx) / rL[idx%4 >= 2]);
#endif
    }

    // Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<_64, _256>{}, make_coord(_0{}, warpgroup_idx));
    Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<_128, _256>{}, make_coord(_0{}, warpgroup_idx));

    typename T::TiledMma tiled_mma;
    auto r2s_tiled_copy = make_tiled_copy_C(
        SmemTiledCopyO{}, tiled_mma);

    ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);
    Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
    Tensor r2s_thr_copy_sMyOutputBuf = r2s_thr_copy.partition_D(sMyOutputBuf);
    cute::copy(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sMyOutputBuf);

    __syncthreads();

    // tsm->global
    const int64_t row_offset_o = batch_idx * params.o_batch_stride + m_block_idx * T::kBlockM * params.o_row_stride + k_head_idx * params.o_head_stride;

    using GmemTiledCopyO = std::conditional_t<
        IS_NO_SPLIT,
        typename T::GmemTiledCopyO,
        typename T::GmemTiledCopyOaccum>;
    // T::GmemTiledCopyO gmem_tiled_copy_Oaccum;
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sOutputBuf); // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gOorAccum);

    Tensor tOrO = make_tensor<ElementO>(shape(tOgO));

    /// tsm -> reg
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOutputBuf), size<1>(sOutputBuf))); // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));

    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    copy<false, true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, num_valid_seq_q);
}

template <
    typename T,
    typename Tensor0>
__forceinline__ __device__ void launch_q_copy(
    const Flash_fwd_mla_params &params,
    int batch_idx,
    int m_block_idx,
    int k_head_idx,
    Tensor0 &sQ,
    const int tidx,
    const int warp_idx,
    __mbarrier_t* barrier_Q
) {
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<T::InputT*>(params.q_ptr)
                                            + batch_idx * params.q_batch_stride),
                            make_shape(params.seqlen_q, params.h_h_k_ratio, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(make_mix_tensor_like(mQ(_, k_head_idx, _)), Shape<Int<T::kBlockM>, Int<T::kHeadDim>>{},
                            make_coord(m_block_idx, 0));  // (kBlockM, kHeadDim)
    typename T::GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);

#if ACOMPUTE_VERSION == 10000
    int aiu_offset_q = 0;
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, params.seqlen_q, params.q_row_stride, T::kBlockM, T::kBlockKSmem, aiu_offset_q};
    // const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, params.seqlen_q, params.d, params.q_row_stride);
#endif
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    if (warp_idx == 0) {
        cute::copy(gmem_tiled_copy_Q, tQgQ, tQsQ);

        // __pipeline_arrive_on(barrier_Q);
        cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
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
    return flat_divide(sV, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
}

template <
    typename T,
    bool IS_R,
    typename Engine0, typename Layout0>
__forceinline__ __device__ auto get_half_V2(
    int block_idx,
    Tensor<Engine0, Layout0> sK0)
{
    const int blk_idx = (block_idx) % 3;
    Tensor sV = make_tensor(sK0.data(), (typename T::SmemLayoutV){});
    return flat_divide(sV, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
}

template <
    typename T>
__forceinline__ __device__ long get_block_index(
    int block_idx,
    const Flash_fwd_mla_params &params,
    int *block_table_ptr)
{
    // const int block_table_idx = block_idx * T::Page_In_BlockN;
    const int block_table_idx = block_idx * 1 / 2;
    const int block_table_offset = block_idx * T::kBlockN - block_table_idx * T::PAGE_BLOCK_SIZE;
    return long(__ldg(block_table_ptr + block_table_idx) * params.k_batch_stride + block_table_offset * params.k_row_stride + blockIdx.y * params.k_head_stride);
}

template <
    typename T,
    bool IS_BLK0_LAST, // "BLK0" means block_idx+0, "BLK1" means block_idx+1, ...
    bool IS_BLK1_LAST,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename Engine6, typename Layout6,
    typename Engine7, typename Layout7,
    typename Engine8, typename Layout8,
    typename Engine9, typename Layout9,
    typename Engine10, typename Layout10,
    typename Engine11, typename Layout11,
    typename Engine12, typename Layout12,
    typename Engine13, typename Layout13
>
__forceinline__ __device__ void wg0_subroutine(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> &tKgK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> sK,
    Tensor<Engine3, Layout3> &cur_sK0,
    Tensor<Engine4, Layout4> &cur_sK1,
    Tensor<Engine5, Layout5> &nxt_sK0,
    Tensor<Engine6, Layout6> &sP0,
    Tensor<Engine7, Layout7> &sP1,
    Tensor<Engine8, Layout8> &sM,
    Tensor<Engine9, Layout9> &sScale0,
    Tensor<Engine10, Layout10> &sScale1,
    Tensor<Engine11, Layout11> &rQ8,
    Tensor<Engine12, Layout12> &rP0,
    Tensor<Engine13, Layout13> &rO0,
    float rL[2],
    int rRightBorderForQSeq[2],
    __mbarrier_t barriers_K0[9],
    __mbarrier_t barriers_K1[9],
    bool &cur_phase_K0,
    const Flash_fwd_mla_params &params,
    int* block_table_ptr,
    int seqlen_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup,
    int wg_idx,
    int &kv_idx
) {
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx+2;
    int nxt_block1 = block_idx+3;

    Tensor sV0L = get_half_V<T, 0>(cur_sK0);
    Tensor sV1L = get_half_V<T, 0>(cur_sK1);

    auto nxt_sK1 = cur_sK0;

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if (wg_idx == 0) {
            tKgK.data().ptr_ = make_gmem_ptr(
                reinterpret_cast<T::InputT *>(params.k_ptr) + get_block_index<T>(nxt_block0, params, block_table_ptr));
            auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
            Tensor tKsK0 = gmem_thr_copy_K.partition_D(nxt_sK0);
            tiled_copy.desc_.dim_h = seqlen_k - (nxt_block0 * T::kBlockN);
            launch_kv_tiles_copy<0, 4>(tiled_copy, tKgK, tKsK0, params, &barriers_K0[0], wg_idx);
        }
    }
    // Calc P0 = softmax(P0)
#if ACOMPUTE_VERSION == 10000
    Tensor rPb = wg0_bunch_0< T, IS_BLK0_LAST || IS_BLK1_LAST > (rP0, rO0, sScale0, sM, rL, rRightBorderForQSeq, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup);
#else
    Tensor rPb = make_tensor<T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
    wg0_bunch_0< T, IS_BLK0_LAST || IS_BLK1_LAST > (rPb, rP0, rO0, sScale0, sM, rL, rRightBorderForQSeq, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup);
#endif
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale0Ready);

    // Issue rO0 += rPb @ sV0L
    wg0_scale0_rO0(rO0, sScale0, idx_in_warpgroup);
    warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, idx_in_warpgroup, wg_idx);

    //  if (!IS_BLK0_LAST && !IS_BLK1_LAST && __builtin_expect(block_idx + 3 < end_block_idx, true)) {
    //     if (wg_idx == 0) {
    //         tKgK.data().ptr_ = make_gmem_ptr(
    //             reinterpret_cast<T::InputT *>(params.k_ptr) + get_block_index<T>(nxt_block1, params, block_table_ptr));
    //         auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
    //         Tensor tKsK1 = gmem_thr_copy_K.partition_D(nxt_sK1);
    //         tiled_copy.desc_.dim_h = seqlen_k - (nxt_block1 * T::kBlockN);
    //         launch_kv_tiles_copy<0, 4>(tiled_copy, tKgK, tKsK1, params, &barriers_K1[0], wg_idx);
    //     }
    // }
    // Wait for warpgroup 1, rescale P0, notify warpgroup 1
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    if (!IS_BLK0_LAST && !IS_BLK1_LAST && __builtin_expect(block_idx + 3 < end_block_idx, true)) {
        if (wg_idx == 0) {
            tKgK.data().ptr_ = make_gmem_ptr(
                reinterpret_cast<T::InputT *>(params.k_ptr) + get_block_index<T>(nxt_block1, params, block_table_ptr));
            auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
            Tensor tKsK1 = gmem_thr_copy_K.partition_D(nxt_sK1);
            tiled_copy.desc_.dim_h = seqlen_k - (nxt_block1 * T::kBlockN);
            launch_kv_tiles_copy<0, 4>(tiled_copy, tKgK, tKsK1, params, &barriers_K1[0], wg_idx);
        }
    }

    wg0_scale_rP0<T>(sScale1, rP0, rPb, idx_in_warpgroup);
    save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);

    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sP0Ready);

    // Wait for warpgroup 1, rescale O0, issue rO0 += rPb @ sV1L
    if constexpr (!IS_BLK0_LAST)
    {
        NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::rO1sP0sV0RIssued);
        wg0_rescale_rO0(rO0, sScale1, rL, idx_in_warpgroup);
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, idx_in_warpgroup, wg_idx);
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST)
    {
        cute::clear(rP0);
        warpgroup_cooperative_qkt_gemm<T, 0>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
    }

    // Issue P0 = Q @ K0^T
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST)
    {
        warpgroup_cooperative_qkt_gemm<T, 2>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
    }

    kv_idx = (kv_idx + 2) % 3;
    cur_sK0 = sK(_, _, kv_idx);
    cur_sK1 = sK(_, _, (kv_idx + 1) % 3);
    nxt_sK0 = sK(_, _, (kv_idx + 2) % 3);
}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename Engine6, typename Layout6,
    typename Engine7, typename Layout7,
    typename Engine8, typename Layout8,
    typename Engine9, typename Layout9,
    typename Engine10, typename Layout10,
    typename Engine11, typename Layout11,
    typename Engine12, typename Layout12,
    typename Engine13, typename Layout13
>
__forceinline__ __device__ void wg1_subroutine(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> &tKgK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> sK,
    Tensor<Engine3, Layout3> &cur_sK1,
    Tensor<Engine4, Layout4> &cur_sK0,
    Tensor<Engine5, Layout5> &nxt_sK1,
    Tensor<Engine6, Layout6> &sP0,
    Tensor<Engine7, Layout7> &sP1,
    Tensor<Engine8, Layout8> &sM,
    Tensor<Engine9, Layout9> &sScale0,
    Tensor<Engine10, Layout10> &sScale1,
    Tensor<Engine11, Layout11> &rQ8,
    Tensor<Engine12, Layout12> &rP1,
    Tensor<Engine13, Layout13> &rO1,
    float rL[2],
    int rRightBorderForQSeq[2],
    __mbarrier_t barriers_K0[9],
    __mbarrier_t barriers_K1[9],
    bool &cur_phase_K1,
    const Flash_fwd_mla_params &params,
    int* block_table_ptr,
    int seqlen_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup,
    int wg_idx,
    int &kv_idx
) {
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx+2;
    int nxt_block1 = block_idx+3;

    auto nxt_sK0 = cur_sK1;
    Tensor sV0R = get_half_V<T, 1>(cur_sK0);
    Tensor sV1R = get_half_V<T, 1>(cur_sK1);

    // Wait for rO1 += rP1b @ sV1R, launch TMA for the next V1R
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST && !IS_BLK2_LAST)
    {
        if (wg_idx == 0)
        {
            tKgK.data().ptr_ = make_gmem_ptr(
                reinterpret_cast<T::InputT *>(params.k_ptr) + get_block_index<T>(nxt_block1, params, block_table_ptr));
            auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
            Tensor tKsK1 = gmem_thr_copy_K.partition_D(nxt_sK1);
            tiled_copy.desc_.dim_h = seqlen_k - (nxt_block1 * T::kBlockN);
            launch_kv_tiles_copy<4, 9>(tiled_copy, tKgK, tKsK1, params, &barriers_K1[1], wg_idx);
        }
    }
    // Wait for rP1 and warpgroup 0, run bunch 1, notify warpgroup 0
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);

#if ACOMPUTE_VERSION == 10000
    Tensor rP1b = wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST>(sScale1, rO1, sM, rL, rRightBorderForQSeq, sScale0, rP1, params.scale_softmax_log2, start_token_idx+T::kBlockN, idx_in_warpgroup);
#else
    Tensor rP1b = make_tensor<T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
    wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST>(rP1b, sScale1, rO1, sM, rL, rRightBorderForQSeq, sScale0, rP1, params.scale_softmax_log2, start_token_idx+T::kBlockN, idx_in_warpgroup);
#endif
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    // Save rPb to sP, and issue rO1 += rP1b @ sV1R
    // We do this after notifying warpgroup 1, since both "saving rPb to sP" and "issuing" WGMMA are high-latency operations
    if constexpr (!IS_BLK0_LAST) {
        save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
    }

    wg1_scale0_rO1(rO1, sScale0, sScale1, idx_in_warpgroup);
    if constexpr (!IS_BLK0_LAST) {
        warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV1R, rO1, idx_in_warpgroup, wg_idx);
    }

    // Wait for sP0, issue rO1 += sP0 @ sV0R, notify warpgroup 0
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sP0Ready);

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if (wg_idx == 0) {
            tKgK.data().ptr_ = make_gmem_ptr(
                reinterpret_cast<T::InputT *>(params.k_ptr) + get_block_index<T>(nxt_block0, params, block_table_ptr));
            auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
            Tensor tKsK0 = gmem_thr_copy_K.partition_D(nxt_sK0);
            tiled_copy.desc_.dim_h = seqlen_k-(nxt_block0*T::kBlockN);
            launch_kv_tiles_copy<4, 9>(tiled_copy, tKgK, tKsK0, params, &barriers_K0[1], wg_idx);
        }
    }

    warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx);

    if constexpr (!IS_BLK0_LAST) {
        NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::rO1sP0sV0RIssued);
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST && !IS_BLK2_LAST) {
        cute::clear(rP1);
        // Issue rP1 = sQ @ sK1, wait
        warpgroup_cooperative_qkt_gemm<T, 1>(sQ, nxt_sK0, nxt_sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
    }

    kv_idx = (kv_idx + 2) % 3;
    cur_sK1 = sK(_, _, kv_idx);
    cur_sK0 = sK(_, _, (kv_idx + 1) % 3);
    nxt_sK1 = sK(_, _, (kv_idx + 2) % 3);
}

// A helper function for determining the length of the causal mask for one q token
__forceinline__ __device__ int get_mask_len(const Flash_fwd_mla_params &params, int m_block_idx, int local_seq_q_idx) {
    int global_seq_q_idx = m_block_idx*Config::BLOCK_SIZE_M + local_seq_q_idx;
    if (global_seq_q_idx < params.seqlen_q) {
        int s_q_idx = global_seq_q_idx / params.ngroups;
        return params.q_orig - s_q_idx - 1;
    } else {
        // Out-of-bound request, regard as no masks
        return 0;
    }
}

template<typename T, bool Is_causal>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    // grid shape: [
    // 	num_m_blocks (=ceil_div(seqlen_q_ori*(num_q_heads//num_kv_heads))),
    // 	num_kv_heads,
    // 	num_sm_parts
    // ]
    // An "sm part" is responsible for all the BLOCK_SIZE_M q_heads in the m_block (as specified by m_block_idx), under one kv head (as specified by k_head_idx), of a segment (as specified by [start_block_idx, end_block_idx]) of one request (as specified by batch_idx).
    // If is_no_split is True, then this request is exclusively assigned to this sm_part, so we shall write the result directly into params.o_ptr and params.softmax_lse_ptr. Otherwise, write to oaccum_ptr and softmax_lseaccum_ptr, with the corresponding split idx being (n_split_idx + num_splits_ptr[batch_idx])
    // For the complete schedule of the kernel, please read our deep-dive write-up (link can be found in the README.md file).

    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int warpgroup_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 256);
    const int idx_in_warpgroup = threadIdx.x % 256;
    const int warp_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 32);

    const int tidx = threadIdx.x;
    using InputT = typename T::InputT;
    typename T::TiledMma tiled_mma;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    using SharedMemoryPlan = typename T::SharedMemoryPlan;
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan *>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_sQ.data()), (typename T::SmemLayoutQ){});
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_sK.data()), (typename T::SmemLayoutK){});
    Tensor sP0 = make_tensor(flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, _0{}, _8{}).data(), (typename T::SmemLayoutP0){}); // Overlap with sQ's 8-th tile
    Tensor sP1 = make_tensor(sP0.data() + sP0.size(), (typename T::SmemLayoutP0){});
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL_reduction_wksp = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<2 * T::BLOCK_SIZE_M>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    // char* sO_addr = (char*)plan.smem_sK0.data();	// Overlap with sK0 and sK1
    char *sO_addr = (char *)plan.smem_sQ.data(); // Overlap with sK0 and sK1

    // // Define TMA stuffs
    __mbarrier_t *barrier_Q = &(plan.barrier_Q);
    __mbarrier_t *barriers_K0 = plan.barriers_K0;
    __mbarrier_t *barriers_K1 = plan.barriers_K1;

    // // Initialize TMA barriers
    if (threadIdx.x == 0)
    {
        __mbarrier_init(barrier_Q, 32);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i)
        {
            __mbarrier_init(&barriers_K0[i], 32);
            __mbarrier_init(&barriers_K1[i], 32);
        }
    }
    bool cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;

    int4 tile_scheduler_metadata = *(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;

    if (begin_idx >= params.b)
        return;
    int begin_n_split_idx = *(tile_scheduler_metadata_ptr + 4);

    // Copy the first Q
    launch_q_copy<T>(params, begin_idx, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);

#pragma unroll 1
#pragma clang loop licm(disable)
    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx)
    {
        constexpr int kBlockN = T::kBlockN;
        const int n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
        int seqlen_k = __ldg(params.cu_seqlens_k + batch_idx);
        const int start_block_idx = batch_idx == begin_idx ? begin_seqlen / kBlockN : 0;
        int end_block_idx = batch_idx == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool is_no_split = start_block_idx == 0 && end_block_idx == cute::ceil_div(seqlen_k, kBlockN);

        int rRightBorderForQSeq[2];
        if constexpr (Is_causal)
        {
            // The causal mask looks like:
            // XXXX
            // XXXX
            // ...
            // XXXX
            //  XXX
            //  XXX
            //  ...
            //  XXX
            //   XX
            //   XX
            //  ...
            //   XX
            // Firstly, there is a common_mask_len, which is the minimum length of causal masks among all tokens. Since the length of the causal mask decreases monotonically, the common_mask_len is the length of the causal mask for the last token. We consider the common_mask_len as a "reduction in the length of the k-sequence.", and adjust end_block_idx based on it, to save some calculation.
            // Besides, a token may have some extra masks other than the common mask. We use rRightBorderForQSeq to denote it, which means the right border of the k-sequence for the particular q token. In this way, (seqlen_k-common_mask_len) - rRightBorderForQSeq < 64 holds, which means that we only need to apply the causal mask to the last two KV blocks
            // NOTE This may lead to start_block_idx >= end_block_idx which needs some special handling
            int common_mask_len = get_mask_len(params, m_block_idx, T::BLOCK_SIZE_M-1);
            end_block_idx = batch_idx == end_idx ? cute::ceil_div(min(end_seqlen, seqlen_k-common_mask_len), kBlockN) : cute::ceil_div(seqlen_k-common_mask_len, kBlockN);

            CUTLASS_PRAGMA_UNROLL
            for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
                int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
                rRightBorderForQSeq[local_row_idx] = min(seqlen_k-get_mask_len(params, m_block_idx, row_idx), end_block_idx*T::kBlockN);
            }
        } else {
            rRightBorderForQSeq[0] = rRightBorderForQSeq[1] = seqlen_k;
        }

        int* block_table_ptr = params.block_table + batch_idx*params.block_table_batch_stride;	// (/) : (1)

        Tensor gK = make_tensor(make_gmem_ptr(
                        reinterpret_cast<InputT *>(params.k_ptr) + get_block_index<T>(start_block_idx, params, block_table_ptr)),
                        Shape<Int<kBlockN>, Int<T::kHeadDim>>{},
                        make_stride(params.k_row_stride, _1{}));

        // Copy K0 and K1
        typename T::GmemTiledCopyKV gmem_tiled_copy_K;
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);
        Tensor tKgK = gmem_thr_copy_K.partition_S(make_mix_tensor_like(gK)); // (KCPY, KCPY_N, KCPY_K)

        Tensor cur_sK0 = sK(_, _, 0);
        Tensor cur_sK1 = sK(_, _, 1);
        Tensor nxt_sK0 = sK(_, _, 2);
        Tensor tKsK0 = gmem_thr_copy_K.partition_D(cur_sK0);
        Tensor tKsK1 = gmem_thr_copy_K.partition_D(cur_sK1);

        // gmem_tiled_copy_K.desc_ = AiuDesc{nullptr, kBlockN, params.k_row_stride, kBlockN, T::kBlockKSmem, 0};
#if ACOMPUTE_VERSION == 10000
        gmem_tiled_copy_K.desc_ = AiuDesc{nullptr, kBlockN, params.k_row_stride, kBlockN, T::kBlockKSmem, 0};
#else
        gmem_tiled_copy_K.desc_.init(nullptr, kBlockN, params.d, params.k_row_stride);
#endif
        if (seqlen_k != 0) {
            if (warp_idx == 0) {
                gmem_tiled_copy_K.desc_.dim_h = seqlen_k - (start_block_idx * kBlockN);
                launch_kv_tiles_copy<4, 9>(gmem_tiled_copy_K, tKgK, tKsK1, params, &barriers_K0[1], warp_idx);
                launch_kv_tiles_copy<0, 4>(gmem_tiled_copy_K, tKgK, tKsK0, params, &barriers_K0[0], warp_idx);
            }
         }

        if (start_block_idx+1 < end_block_idx) {
            if (warp_idx == 0) {
                tKgK.data().ptr_ = make_gmem_ptr(
                        reinterpret_cast<InputT *>(params.k_ptr) + get_block_index<T>(start_block_idx + 1, params, block_table_ptr));
                gmem_tiled_copy_K.desc_.dim_h = seqlen_k - ((start_block_idx + 1) * kBlockN);
                launch_kv_tiles_copy<4, 9>(gmem_tiled_copy_K, tKgK, tKsK0, params, &barriers_K1[1], warp_idx);
                launch_kv_tiles_copy<0, 4>(gmem_tiled_copy_K, tKgK, tKsK1, params, &barriers_K1[0], warp_idx);
            }
        }

        Tensor rO = partition_fragment_C((typename T::TiledMma){}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV / 2>>{});	// ((2, 2, 32), 1, 1)
        float rL[2];
        rL[0] = rL[1] = 0.0f;

        // Clear buffers
        cute::fill(rO, 0.);
        if (threadIdx.x < size(sM)) {
            sM[threadIdx.x] = MAX_INIT_VAL_SM;
        }

        while(!cutlass::arch::test_wait(barrier_Q, cur_phase_Q, 1)) {
            kernel_sleep_ns();
        }
        cur_phase_Q = (cur_phase_Q + 1) & 1;

        Tensor rQ8 = make_tensor<InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
        retrieve_rP_from_sP<T>(rQ8, local_tile(sQ, Shape<_128, _64>{}, Coord<_0, _8>{}), idx_in_warpgroup);

        if (warpgroup_idx == 0) {
            // Warpgroup 0
            // Tensor rP0 = make_tensor<float>((typename T::rP0Layout){});
            Tensor rP0 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
            const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

            // NOTE We don't use the pipelined version of Q K^T here since it leads
            // to a slow-down (or even register spilling, thanks to the great NVCC)
            // Issue P0 = Q @ K0^T, wait
            if (seqlen_k !=0) {
                cute::clear(rP0);
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK0, cur_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
            }

            int idx = 0;
            #define LAUNCH_WG0_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST)                     \
                wg0_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST>(                            \
                gmem_tiled_copy_K, tKgK, sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
                rP0, rO, rL, rRightBorderForQSeq,                                     \
                barriers_K0, barriers_K1, cur_phase_K0, params,                       \
                block_table_ptr, seqlen_k, block_idx, end_block_idx, idx_in_warpgroup, wg_idx, idx); \

            int block_idx = start_block_idx;

            #pragma unroll 1
            for (; block_idx < end_block_idx-2; block_idx += 2) {
                LAUNCH_WG0_SUBROUTINE(false, false);
            }

            if (block_idx+1 < end_block_idx) {
                LAUNCH_WG0_SUBROUTINE(false, true);
            } else if (block_idx < end_block_idx) {
                LAUNCH_WG0_SUBROUTINE(true, false);
            }
        } else {
            // // Warpgroup 1
            // Tensor rP1 = make_tensor<float>((typename T::rP0Layout){});
            Tensor rP1 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
            const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

            if (start_block_idx+1 < end_block_idx) {
                // Issue rP1 = sQ @ sK1, wait
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK1, cur_sK0, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
            }

            int idx = 0;
            #define LAUNCH_WG1_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST)       \
                wg1_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST>(              \
                gmem_tiled_copy_K, tKgK, sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
                rP1, rO, rL, rRightBorderForQSeq,                                     \
                barriers_K0, barriers_K1, cur_phase_K1, params,                       \
                block_table_ptr, seqlen_k, block_idx, end_block_idx, idx_in_warpgroup, wg_idx, idx); \

            int block_idx = start_block_idx;
            #pragma unroll 1
            for (; block_idx < end_block_idx-3; block_idx += 2) {
                LAUNCH_WG1_SUBROUTINE(false, false, false);
            }

            if (block_idx+2 < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(false, false, true);
                {
                    block_idx += 2;
                    LAUNCH_WG1_SUBROUTINE(true, false, false);
                }
            } else if (block_idx+1 < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(false, true, false);
            } else if (block_idx < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(true, false, false);
            }
        }

        // Reduce rL across threads within the same warp
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

        // Reduce rL across warpgroups
        int my_row = get_AorC_row_idx(0, idx_in_warpgroup);
        if (idx_in_warpgroup % 4 == 0)
        {
            sL_reduction_wksp[my_row + warpgroup_idx * 128] = rL[0];
            sL_reduction_wksp[my_row + 8 + warpgroup_idx * 128] = rL[1];
        }
        __syncthreads();

        if (warpgroup_idx == 0)
        {
            rL[0] += sL_reduction_wksp[my_row + 128];
            rL[1] += sL_reduction_wksp[my_row + 8 + 128];
        }
        else
        {
            if (idx_in_warpgroup % 4 == 0)
            {
                sL_reduction_wksp[my_row] += rL[0];
                sL_reduction_wksp[my_row + 8] += rL[1];
            }
            __syncwarp();
            rL[0] = sL_reduction_wksp[my_row];
            rL[1] = sL_reduction_wksp[my_row + 8];
        }

        // Prune out when rL is 0.0f or NaN
        // rL may be 0.0f if there are large values (~10^12) in QK^T, which leads
        // to exp2f(P(i)*scale-max) = 0.0f or +inf due to FMA error.
        // When this happens, we set rL to 1.0f. This aligns with the old version
        // of the MLA kernel.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i)
            rL[i] = (rL[i] == 0.0f || rL[i] != rL[i]) ? 1.0f : rL[i];

        // Epilogue
        int num_valid_seq_q = min(params.seqlen_q - m_block_idx * T::BLOCK_SIZE_M, T::BLOCK_SIZE_M);
        if (is_no_split)
        {
            InputT *o_ptr = (InputT *)params.o_ptr + batch_idx * params.o_batch_stride + m_block_idx * T::BLOCK_SIZE_M * params.o_row_stride + k_head_idx * params.o_head_stride; // (BLOCK_SIZE_M, HEAD_DIM_V) : (params.o_row_stride, 1)
            float *softmax_lse_ptr = (float *)params.softmax_lse_ptr + (batch_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M;                     // (BLOCK_SIZE_M) : (1)

            Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                                                              Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV>>{},
                                                              make_stride(params.o_row_stride, _1{})));
            Tensor gSoftmaxLse = make_tensor(make_gmem_ptr(softmax_lse_ptr), Layout<
                                                                                 Shape<Int<T::BLOCK_SIZE_M>>,
                                                                                 Stride<_1>>{});

            store_o<T, true>(rO, gO, rL, sO_addr, params, batch_idx, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            if (batch_idx + 1 <= end_idx)
            {
                launch_q_copy<T>(params, batch_idx + 1, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);
            }

            int i = threadIdx.x;
            if (i < num_valid_seq_q)
            {
                float cur_L = sL_reduction_wksp[i];
                gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L) ? INFINITY : logf(cur_L) + sM(i) / (float)M_LOG2E;
            }
        }
        else
        {
            // Don't use __ldg because of PDL and instruction reordering
            int split_idx = params.num_splits_ptr[batch_idx] + n_split_idx;

            float *oaccum_ptr = (float *)params.oaccum_ptr + ((split_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M) * T::kHeadDimV;    // (BLOCK_SIZE_M, HEAD_DIM_V) : (HEAD_DIM_V, 1)
            float *softmax_lseaccum_ptr = (float *)params.softmax_lseaccum_ptr + (split_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M; // (BLOCK_SIZE_M) : (1)
            Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), Layout<
                                                                        Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV>>,
                                                                        Stride<Int<T::kHeadDimV>, _1>>{});
            Tensor gSoftmaxLseAccum = make_tensor(make_gmem_ptr(softmax_lseaccum_ptr), Layout<
                                                                                           Shape<Int<T::BLOCK_SIZE_M>>,
                                                                                           Stride<_1>>{});

            int i = threadIdx.x;
            if (i < num_valid_seq_q)
            {
                float cur_L = sL_reduction_wksp[i];
                gSoftmaxLseAccum(i) = (cur_L == 0.0f || cur_L != cur_L) ? -INFINITY : log2f(cur_L) + sM(i);
            }

            store_o<T, false>(rO, gOAccum, rL, sO_addr, params, batch_idx, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            __syncthreads();

            if (batch_idx + 1 <= end_idx)
            {
                if (threadIdx.x == 0)
                {
                    __mbarrier_init(barrier_Q, 32);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i)
                    {
                        __mbarrier_init(&barriers_K0[i], 32);
                        __mbarrier_init(&barriers_K1[i], 32);
                    }
                }
                cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;
                launch_q_copy<T>(params, batch_idx + 1, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);
            }
        }
        if (batch_idx != end_idx)
            __syncthreads();
    }
}

template <typename InputT, int Arch>
void run_flash_splitkv_mla_kernel(Flash_fwd_mla_params &params, hggcStream_t stream)
{
    BOOL_SWITCH(params.is_causal, Is_causal, [&]
                {
        using T = Traits<InputT>;

        auto mla_kernel = &flash_fwd_splitkv_mla_kernel<T, Is_causal>;
        constexpr size_t smem_size = std::max(sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));

                hggcFuncSetAttribute(mla_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        const int num_m_block = cute::ceil_div(params.seqlen_q, T::kBlockM);

        int ctas_per_sm;
        hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, mla_kernel, T::NUM_THREADS, smem_size);

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            printf("[splitkv_mla]:\n");
            printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, mla_kernel);
            int sm_count = get_num_sm(get_current_device());
            if (sm_count == 64) sm_count = 20;

            printf("blockM:%d, blockN:%d, threads:%d, block_size:%d\n",
                    T::kBlockM, T::kBlockN, T::NUM_THREADS, params.page_block_size);
            printf("Is_causal:%d\n", Is_causal);
            printf("grid_n[%d, %d, %d]\n",
                    num_m_block, params.h, params.num_sm_parts);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f, Arch:%d\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                    float(num_m_block * params.h * params.num_sm_parts) / float(sm_count * ctas_per_sm), Arch);
        }

        // Use hggcLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
        hggcLaunchAttribute mla_kernel_attributes[1];
        mla_kernel_attributes[0].id = hggcLaunchAttributeProgrammaticStreamSerialization;
        mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
        hggcLaunchConfig_t mla_kernel_config = {
            dim3(num_m_block, params.h, params.num_sm_parts),
            dim3(T::NUM_THREADS, 1, 1),
            smem_size,
            stream,
            mla_kernel_attributes,
            1
        };
        hggcLaunchKernelEx(&mla_kernel_config, mla_kernel, params);
        // mla_kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), T::NUM_THREADS, smem_size, stream>>>(params);

        CHECK_CUDA_KERNEL_LAUNCH();

        run_flash_mla_combine_kernel<InputT>(params, stream); });
}
