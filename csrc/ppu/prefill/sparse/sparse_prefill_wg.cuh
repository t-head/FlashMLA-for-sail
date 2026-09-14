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

#include "params.h"
#include "config.h"
#include "kerutils/common/static_switch.h"
#include "kerutils/host/hardware_info.h"

#include "traits.h"
#include "acc_vreg_fraga.h"

#include <c10/cuda/CUDAException.h>
#include "kernel_traits.h"
#include "utils.h"
#include "kerutils/device/ppu/softmax.cuh"
#include "kerutils/device/ppu/mask.cuh"

using namespace cute;
using cutlass::arch::NamedBarrier;

#include "kerutils/device/ppu/dequant.cuh"

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
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_dsa_wg_copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV)       // (BLOCK_N, HEAD_DIM_K), swizzled
{
    Tensor cur_gKV = gKV(_, _, Int<START_HEAD_DIM_TILE_IDX>{});
    Tensor cur_sKV = sKV(_, _, Int<START_HEAD_DIM_TILE_IDX>{});
    cute::copy(tiled_copy, cur_gKV, cur_sKV);

    if constexpr (START_HEAD_DIM_TILE_IDX + 1 < END_HEAD_DIM_TILE_IDX)
    {
        launch_kv_tiles_dsa_wg_copy<START_HEAD_DIM_TILE_IDX + 1, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV);
    }
}

template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_dsa_wg(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,       // (BLOCK_N, HEAD_DIM_K), swizzled
    __mbarrier_t *barriers_K)
{
    launch_kv_tiles_dsa_wg_copy<START_HEAD_DIM_TILE_IDX, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV);
    cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
}

// mirrors decode sparse_decode_wg.cuh:1533-1591 (compute_K_addr_bf16) -- Phase 1
// of the K-address two-phase split (Opt-C): pure address computation + valid
// mask write. Does NOT issue cp.async, so it can be placed in TC idle windows.
// Key adaptation vs decode: prefill addressing is flat --
//   gK_base + token_idx * stride_kv_s_kv + (idx_in_warpgroup % 8) * 8
// with NO page-table two-level decomposition and no USE_EXTRA branch; only the
// two-phase STRUCTURE is ported, not decode's paged address math.
// WRITE_VI: whether this call site owns the block's flag slot (WG0 owns even
// blocks -> slots 0/1, WG1 owns odd blocks -> slots 2/3). The cross-WG call
// (same block, other half of the tiles) passes false and skips the flag store.
template<typename T, bool WRITE_VI, typename TensorVI>
__forceinline__ __device__ void dsa_compute_K_addr(
    const SparsePrefillParams &params,
    typename T::InputT *gK_base,
    int token_idx,                 // prefetched token index (this thread's column)
    int block_idx,                 // block being computed (border fold-in base)
    int seqlen_k,
    int idx_in_warpgroup,
    TensorVI &smem_valid_indices,
    int vi_buf,
    typename T::InputT *&precomp_ptr,
    bool &precomp_valid)
{
    using InputT = typename T::InputT;
    bool is_token_valid = token_idx >= 0 && token_idx < params.s_kv;
    precomp_ptr = gK_base + token_idx * (int64_t)params.stride_kv_s_kv + (idx_in_warpgroup % 8) * 8;
    precomp_valid = is_token_valid;
    // Same flag value as the legacy inline write points: per-token validity with
    // the topk_length right border folded in (absolute topk position
    // block_idx*kBlockN + col < seqlen_k). The store guard (idx%8==0) and the
    // value formula are identical to the pre-split code -- only the TIMING moves
    // one iteration earlier (vi slot invariant (b) is preserved by the matching
    // runtime guards at the call sites; see traits.h L193-202).
    if constexpr (WRITE_VI) {
        if (idx_in_warpgroup % 8 == 0) {
            smem_valid_indices(vi_buf, idx_in_warpgroup / 8) =
                is_token_valid && (block_idx * T::kBlockN + idx_in_warpgroup / 8 < seqlen_k);
        }
    }
}

// mirrors decode sparse_decode_wg.cuh:1620-1674 (issue_K_load_bf16) -- Phase 2
// of the split: issue cp.async with the precomputed pointer, fused with the
// __ldg prefetch of the NEXT block's token index. No address math, no vi store.
// DO_PREFETCH=false is for the prolog (which does its own guarded prefetch of
// the next two block indices).
template<int S, int E, typename T, bool DO_PREFETCH = true,
         typename TiledCopy, typename Engine0, typename Layout0,
         typename Engine1, typename Layout1>
__forceinline__ __device__ void dsa_issue_K_load(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> &tKgK,   // partitioned gmem src (data ptr overwritten)
    Tensor<Engine1, Layout1> &tKsK,   // partitioned smem dst
    __mbarrier_t *barriers_K,
    typename T::InputT *precomp_ptr,
    bool precomp_valid,
    int *gIndices_ptr,                // per-thread prefetch base (incl. idx/8 offset)
    int prefetch_block,               // block whose token index is prefetched next
    int real_end_block_idx,           // [Even-align] REAL topk block count -- the only
                                      // safe bound for gIndices reads (see kernel prolog)
    int &nxt_token_idx)
{
    tKgK.data() = precomp_ptr;
    tiled_copy.pred = precomp_valid;
    launch_kv_tiles_dsa_wg<S, E>(tiled_copy, tKgK, tKsK, barriers_K);
    if constexpr (DO_PREFETCH) {
        // [Even-align] guard by the REAL block count: a padding block (beyond
        // real_end_block_idx) has no backing gIndices memory, so fabricate
        // token_idx = -1 -> invalid -> pred=false cp.async + vi-flag=false.
        if (prefetch_block < real_end_block_idx) {
            nxt_token_idx = __ldg(gIndices_ptr + prefetch_block * T::kBlockN);
        } else {
            nxt_token_idx = -1;
        }
    }
}

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

template <
    typename T,    // Traits
    int PHASE_IDX, // See comments in the code
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3>
__forceinline__ __device__ void dsa_warpgroup_cooperative_qkt_gemm(
    Tensor<Engine0, Layout0> &sQ,   // (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV0, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV1, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,   // ((2, 2, 8), 1, 1)
    Tensor<Engine3, Layout3> &rQ8,  // The last tile of Q. We store it separately to leave some room for storing sP1
    __mbarrier_t *barriers,
    bool &cur_phase,
    int idx_in_warpgroup,
    int warp_idx)
{
    typename T::TiledMma tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename T::SmemCopyAtomK{}, tiled_mma);
#if DSA_SIM_AIU
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(warp_idx * 32);
#else
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx_in_warpgroup);
#endif

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(warp_idx * 32);

    Tensor sQ_tiled = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, _64>{})(_, _, _0{}, _); // (BLOCK_SIZE_M, 64, NUM_TILES)
    Tensor sKV0_tiled = flat_divide(sKV0, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);  // (BLOCK_N, 64, NUM_TILES)
    Tensor sKV1_tiled = flat_divide(sKV1, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);  // (BLOCK_N, 64, NUM_TILES)
    Tensor thr_mma_sQ_tiled = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ_tiled));
#if DSA_SIM_AIU
    Tensor thr_mma_sKV0_tiled = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV0_tiled));
    Tensor thr_mma_sKV1_tiled = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV1_tiled));
#else
    Tensor thr_mma_sKV0_tiled = smem_thr_copy_K.partition_S(sKV0_tiled);
    Tensor thr_mma_sKV1_tiled = smem_thr_copy_K.partition_S(sKV1_tiled);
#endif

    #define QKT_GEMM_ONE_TILE(TILE_IDX) \
        if constexpr(TILE_IDX == T::NUM_TILES - 1) { \
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

    if constexpr (PHASE_IDX == 0)
    {
        // In PHASE-0, warpgroup 0 calculates Q K^T for the first 4 tiles
        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
    } else if constexpr (PHASE_IDX == 1) {
        // In PHASE-1, warpgroup 1 calculates Q K^T for all the NUM_TILES tiles
        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        if constexpr (T::NUM_TILES > 8) {
            QKT_GEMM_ONE_TILE(8);
        }

        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
            kernel_sleep_ns();
        };
        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
        cur_phase = (cur_phase + 1) & 1;
    } else {
        // In PHASE-2, warpgroup 0 calculates Q K^T for the last (NUM_TILES - 4) tiles
        static_assert(PHASE_IDX == 2);

        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        if constexpr (T::NUM_TILES > 8) {
            QKT_GEMM_ONE_TILE(8);
        }
        cur_phase = (cur_phase + 1) & 1;
    }
}

template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2>
__forceinline__ __device__ void dsa_warpgroup_cooperative_pv_gemm_localP(
    Tensor<Engine0, Layout0> &rP,       // ((2, 2, 8), 1, 1), fragment A layout
    Tensor<Engine1, Layout1> &sKV_half, // (HEAD_DIM_V/2, BLOCK_N)
    Tensor<Engine2, Layout2> &rO,       // ((2, 2, 32), 1, 1)
    int idx_in_warpgroup,
    int warp_idx)
{
    typename T::TiledMma tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);

    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
#if DSA_SIM_AIU
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(warp_idx * 32);
    Tensor rVt = thr_mma.partition_fragment_B(sKV_half);
#else
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(idx_in_warpgroup);
    Tensor sVtNoSwizzle = make_tensor(sKV_half.data(), typename T::SmemLayoutVNoSwizzle{});
    Tensor rVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
#endif
    Tensor rVt_copy_view = smem_thr_copy_Vt.retile_D(rVt);
#if DSA_SIM_AIU
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));
#else
    auto tSsVt = smem_thr_copy_Vt.partition_S(sKV_half);
#endif

    CUTE_STATIC_ASSERT_V(size<1>(tSsVt) == size<1>(rVt_copy_view)); // M
    cute::copy(smem_tiled_copy_Vt, tSsVt, rVt_copy_view);

    gemm(rO, rP, rVt, tiled_mma);
}

template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2>
__forceinline__ __device__ void dsa_warpgroup_cooperative_pv_gemm_remoteP(
    Tensor<Engine0, Layout0> &sP,
    Tensor<Engine1, Layout1> &sKV_half, // (HEAD_DIM_V/2, BLOCK_N)
    Tensor<Engine2, Layout2> &rO,       // ((2, 2, 32), 1, 1)
    int idx_in_warpgroup,
    int warp_idx)
{
    typename T::TiledMma tiled_mma;
    auto smem_tiled_copy_P = make_tiled_copy_A(typename T::SmemCopyAtomP{}, tiled_mma);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(idx_in_warpgroup);
    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
#if DSA_SIM_AIU
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(warp_idx * 32);
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));
#else
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(idx_in_warpgroup);
    auto tSsVt = smem_thr_copy_Vt.partition_S(sKV_half);
#endif

    auto tSsP = smem_thr_copy_P.partition_S(sP);

    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor thr_mma_sP = thr_mma.partition_fragment_A(sP);
#if DSA_SIM_AIU
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sKV_half); // (MMA, 1, 64/16=4)
#else
    Tensor sVtNoSwizzle = make_tensor(sKV_half.data(), typename T::SmemLayoutVNoSwizzle{});
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, 1, 64/16=4)
#endif

    Tensor rP_copy_view = smem_thr_copy_P.retile_D(thr_mma_sP);
    Tensor rVt_copy_view = smem_thr_copy_Vt.retile_D(thr_mma_sKV_half);

    cute::copy(smem_tiled_copy_P, tSsP, rP_copy_view);
    cute::copy(smem_tiled_copy_Vt, tSsVt, rVt_copy_view);
    gemm(rO, rP_copy_view, rVt_copy_view, tiled_mma);
}

template<
    typename T,
    bool DO_OOB_FILLING,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ auto wg0_bunch_0(
    Tensor<Engine1, Layout1> &rP0,
    Tensor<Engine2, Layout2> &rO0,
    Tensor<Engine3, Layout3> &sScale0,
    Tensor<Engine4, Layout4> &sM,
    float rL[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf
) {
    int r_valid[8];
    if constexpr (T::Arch_value == 80) {
        int lane4 = idx_in_warpgroup % 4;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < 2; k++) {
            int base = (k * 16 + lane4) % T::kBlockN;
            r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
            r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
            r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
            r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
        }
    } else {
        // each thread needs 8 values (4 groups of 2)
        int lane4 = idx_in_warpgroup % 4;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < 4; k++) {
            int base = (k * 8 + lane4 * 2) % T::kBlockN;
            r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
            r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
        }
    }
    if constexpr (T::Arch_value == 80) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
            // Mask, and get row-wise max
            float cur_max = MAX_INIT_VAL;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 4 : 0; i < size(rP0); i += 8) {
                int k_base = ((i/8) % 2) * 4;
                rP0(i)   = r_valid[k_base]     ? rP0(i)   : MAX_INIT_VAL;
                rP0(i+1) = r_valid[k_base + 1] ? rP0(i+1) : MAX_INIT_VAL;
                rP0(i+2) = r_valid[k_base + 2] ? rP0(i+2) : MAX_INIT_VAL;
                rP0(i+3) = r_valid[k_base + 3] ? rP0(i+3) : MAX_INIT_VAL;
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
    } else {
        Tensor rPb = make_tensor<typename T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

            // Mask, and get row-wise max
            float cur_max = MAX_INIT_VAL;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
                int k_base = ((i/4) % 4) * 2;
                rP0(i)   = r_valid[k_base]     ? rP0(i)   : MAX_INIT_VAL;
                rP0(i+1) = r_valid[k_base + 1] ? rP0(i+1) : MAX_INIT_VAL;
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
        return rPb;
    }
}
template<
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ auto wg1_bunch_0(
    Tensor<Engine1, Layout1> &sScale1,
    Tensor<Engine2, Layout2> &rO1,
    Tensor<Engine3, Layout3> &sM,
    float rL[2],
    Tensor<Engine4, Layout4> const &sScale0,
    Tensor<Engine5, Layout5> &rP1,
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf, 
    float r_cur_max_in[2] = nullptr
)
{
    [[maybe_unused]] int r_valid[8];
    if constexpr (!IS_BLK0_LAST) {
        if (r_cur_max_in == nullptr) {
            if constexpr (T::Arch_value == 80) {
                int lane4 = idx_in_warpgroup % 4;
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < 2; k++) {
                    int base = (k * 16 + lane4) % T::kBlockN;
                    r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
                    r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
                    r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
                    r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
                }
            } else {
                // each thread needs 8 values (4 groups of 2)
                int lane4 = idx_in_warpgroup % 4;
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < 4; k++) {
                    int base = (k * 8 + lane4 * 2) % T::kBlockN;
                    r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
                    r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
                }
            }
        }
    }
    if constexpr (T::Arch_value == 80) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx)
        {
            int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

            float cur_max;
            if (r_cur_max_in) {
                cur_max = r_cur_max_in[local_row_idx];
            } else {
                cur_max = MAX_INIT_VAL;
                CUTLASS_PRAGMA_UNROLL
                for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8)
                {
                    if constexpr (IS_BLK0_LAST)
                    {
                        rP1(i) = rP1(i + 1) = rP1(i + 2) = rP1(i + 3) = MAX_INIT_VAL;
                    }

                    if constexpr (!IS_BLK0_LAST)
                    {
                        int k_base = ((i/8) % 2) * 4;
                        rP1(i)     = r_valid[k_base]     ? rP1(i)     : MAX_INIT_VAL;
                        rP1(i + 1) = r_valid[k_base + 1] ? rP1(i + 1) : MAX_INIT_VAL;
                        rP1(i + 2) = r_valid[k_base + 2] ? rP1(i + 2) : MAX_INIT_VAL;
                        rP1(i + 3) = r_valid[k_base + 3] ? rP1(i + 3) : MAX_INIT_VAL;
                    }
                    cur_max = max(cur_max, max(max(rP1(i), rP1(i + 1)), max(rP1(i + 2), rP1(i + 3))));
                }
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

                cur_max *= scale_softmax_log2;
            }

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

            // Update rL
            rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
        }

        return convert_acc<typename T::InputT>(rP1);
    } else {
        Tensor rP1b = make_tensor<typename T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

            // Mask, and get row-wise max
            float cur_max;
            if (r_cur_max_in) {
                cur_max = r_cur_max_in[local_row_idx];
            } else {
                cur_max = MAX_INIT_VAL;
                CUTLASS_PRAGMA_UNROLL
                for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                    if constexpr (IS_BLK0_LAST) {
                        rP1(i) = rP1(i+1) = MAX_INIT_VAL;
                    }

                    if constexpr (!IS_BLK0_LAST) {
                        int k_base = ((i/4) % 4) * 2;
                        rP1(i)   = r_valid[k_base]     ? rP1(i)   : MAX_INIT_VAL;
                        rP1(i+1) = r_valid[k_base + 1] ? rP1(i+1) : MAX_INIT_VAL;
                    }
                    cur_max = max(cur_max, max(rP1(i), rP1(i+1)));
                }

                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
                cur_max *= scale_softmax_log2;
            }

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

            // Update rL
            rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
        }
        return rP1b;
    }
}

// dsa_wg1_bunch_0_pre: compute cur_max before the sScale0Ready barrier
template<
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void dsa_wg1_bunch_0_pre(
    float r_cur_max[2],               // output: per-row cur_max * scale_softmax_log2
    Tensor<Engine5, Layout5> &rP1,    // ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf
)
{
    // Same preload as wg1_bunch_0: identical vi slot and column mapping.
    [[maybe_unused]] int r_valid[8];
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (T::Arch_value == 80) {
            int lane4 = idx_in_warpgroup % 4;
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < 2; k++) {
                int base = (k * 16 + lane4) % T::kBlockN;
                r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
                r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
                r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
                r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
            }
        } else {
            // each thread needs 8 values (4 groups of 2)
            int lane4 = idx_in_warpgroup % 4;
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < 4; k++) {
                int base = (k * 8 + lane4 * 2) % T::kBlockN;
                r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
                r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
            }
        }
    }
    if constexpr (T::Arch_value == 80) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            float cur_max = MAX_INIT_VAL;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8) {
                if constexpr (IS_BLK0_LAST)
                {
                    rP1(i) = rP1(i + 1) = rP1(i + 2) = rP1(i + 3) = MAX_INIT_VAL;
                }
                else
                {
                    int k_base = ((i/8) % 2) * 4;
                    rP1(i)     = r_valid[k_base]     ? rP1(i)     : MAX_INIT_VAL;
                    rP1(i + 1) = r_valid[k_base + 1] ? rP1(i + 1) : MAX_INIT_VAL;
                    rP1(i + 2) = r_valid[k_base + 2] ? rP1(i + 2) : MAX_INIT_VAL;
                    rP1(i + 3) = r_valid[k_base + 3] ? rP1(i + 3) : MAX_INIT_VAL;
                }
                cur_max = max(cur_max, max(max(rP1(i), rP1(i + 1)), max(rP1(i + 2), rP1(i + 3))));
            }
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
            r_cur_max[local_row_idx] = cur_max * scale_softmax_log2;
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            float cur_max = MAX_INIT_VAL;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                if constexpr (IS_BLK0_LAST) {
                    rP1(i) = rP1(i+1) = MAX_INIT_VAL;
                } else {
                    int k_base = ((i/4) % 4) * 2;
                    rP1(i)   = r_valid[k_base]     ? rP1(i)   : MAX_INIT_VAL;
                    rP1(i+1) = r_valid[k_base + 1] ? rP1(i+1) : MAX_INIT_VAL;
                }
                cur_max = max(cur_max, max(rP1(i), rP1(i+1)));
            }
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
            r_cur_max[local_row_idx] = cur_max * scale_softmax_log2;
        }
    }
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
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(rPb));
    cute::copy(smem_tiled_copy_Q, tSsQ, rPb);
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

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void dsa_store_o(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    Tensor<Engine1, Layout1> &gOorAccum,	// (BLOCK_SIZE_M, HEAD_DIM_V)
    float rL[2],
    char* sO_addr,
    int k_head_idx,
    int m_block_idx,
    int num_valid_seq_q,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    using InputT = typename T::InputT;
    using ElementO = typename T::InputT;

    using SmemTiledCopyO = typename T::SmemCopyAtomO;
    Tensor sOutputBuf = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(sO_addr)),
        typename T::SmemLayoutO{});

    // (SMEM_M,SMEM_N) // Sw<3,3,3> o _0 o (_32,(_64,_8)):(_64,(_1,_2048))
    Tensor rOb = make_tensor_like<ElementO>(rO);

    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); ++idx) {
        if constexpr (T::Arch_value == 80) {
            rOb(idx) = (ElementO)(rO(idx) / rL[(idx / 4) % 2]);
        } else {
            rOb(idx) = (InputT)(rO(idx) / rL[idx%4 >= 2]);
        }
    }

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
    const int64_t row_offset_o = (int64_t)m_block_idx * T::kBlockM * T::kHeadDimV;

    using GmemTiledCopyO = typename T::GmemTiledCopyO;
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
__forceinline__ __device__ void launch_q_dsa_prefill_wg(
    const SparsePrefillParams &params,
    int m_block_idx,
    Tensor0 &sQ,
    const int tidx,
    const int warp_idx,
    __mbarrier_t* barrier_Q
) {
    using Element = T::InputT;
    const int m_block = m_block_idx % (params.h_q / T::kBlockM);
    const int64_t s_q_idx = m_block_idx / (params.h_q / T::kBlockM);
    const int64_t row_offset_q = s_q_idx * params.stride_q_s_q + m_block * (T::kBlockM * params.stride_q_h_q);
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q) + row_offset_q),
                            Shape<Int<T::kBlockM>, Int<T::kHeadDim>>{},
                            make_stride(params.stride_q_h_q, _1{}));
    typename T::GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));

    if constexpr (T::Arch_value == 80) {
        gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, T::kBlockM, params.stride_q_h_q, T::kBlockM, T::kBlockKSmem, 0};
    } else {
        gmem_tiled_copy_Q.desc_.init(nullptr, T::kBlockM, params.d_qk, params.stride_q_h_q);
    }
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    if (warp_idx == 0) {
        cute::copy(gmem_tiled_copy_Q, tQgQ, tQsQ);
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
    Tensor sVL =  flat_divide(sV, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
    return sVL;
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
    typename Engine13, typename Layout13,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void dsa_wg0_subroutine(
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
    __mbarrier_t barriers_K0[T::kHeadDim/256],
    __mbarrier_t barriers_K1[T::kHeadDim/256],
    bool &cur_phase_K0,
    const SparsePrefillParams &params,
    int* gIndices_ptr,
    int seqlen_k,
    int block_idx,
    int end_block_idx,       // [Even-align] ROUNDED-up (even) block count: loop/issue/compute guards
    int real_end_block_idx,  // [Even-align] REAL topk block count: gIndices prefetch guards only
    int idx_in_warpgroup,
    int wg_idx,
    int &kv_idx,
    int& nxt_token_idx0,
    int& nxt_token_idx1, 
    typename T::InputT*& precomp_ptr0,
    typename T::InputT*& precomp_ptr1,
    bool& precomp_valid0,
    bool& precomp_valid1,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices
) {
    using InputT = typename T::InputT;
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx + 4;
    int nxt_block1 = block_idx + 5;
    InputT* gK_base = reinterpret_cast<InputT*>(params.kv);

    Tensor sV0L = get_half_V<T, 0>(cur_sK0);
    Tensor sV1L = get_half_V<T, 0>(cur_sK1);

    auto nxt_sK1 = cur_sK0;
#if DSA_SIM_AIU
    auto nxt_sKSim0 = make_tensor(nxt_sK0.data(), (typename T::SmemLayoutKSim){})(_, _, 0);
    auto nxt_sKSim1 = make_tensor(nxt_sK1.data(), (typename T::SmemLayoutKSim){})(_, _, 0);
    int sim_cross_tid;
    if constexpr (T::Arch_value == 80) {
        int cross_tid_h = (idx_in_warpgroup & 0xFFFFFFF8) >> 3;
        int cross_tid_l = idx_in_warpgroup & 0x7;
        int cross_bias = (cross_tid_l / 2 == 1) ? 2 : ((cross_tid_l / 2 == 2) ? 1 : cross_tid_l / 2);
        cross_tid_h = (cross_tid_h & 0xFFFFFFFC) | (((cross_tid_h & 0x3) + cross_bias) & 0x3);
        sim_cross_tid = (cross_tid_h << 3) | cross_tid_l;
    } else {
        sim_cross_tid = idx_in_warpgroup;
    }
#endif

    Tensor rPb = wg0_bunch_0< T, IS_BLK0_LAST || IS_BLK1_LAST > (rP0, rO0, sScale0, sM, rL,
        params.sm_scale_div_log2, start_token_idx, idx_in_warpgroup,
        smem_valid_indices, (block_idx/2)%2
    );
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale0Ready);

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
#if DSA_SIM_AIU
        auto gmem_thr_copy_K = tiled_copy.get_thread_slice(sim_cross_tid);
        Tensor tKsK0 = gmem_thr_copy_K.partition_D(nxt_sKSim0);
#else
        auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
        Tensor tKsK0 = gmem_thr_copy_K.partition_D(nxt_sK0);
#endif
        dsa_issue_K_load<0, 4, T>(tiled_copy, tKgK, tKsK0, &barriers_K0[0],
            precomp_ptr0, precomp_valid0, gIndices_ptr, nxt_block0, real_end_block_idx, nxt_token_idx0);
    }

    // Issue rO0 += rPb @ sV0L
    wg0_scale0_rO0<T>(rO0, sScale0, idx_in_warpgroup);
    dsa_warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, idx_in_warpgroup, wg_idx);

    // Wait for warpgroup 1, rescale P0, notify warpgroup 1
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    if (!IS_BLK0_LAST && !IS_BLK1_LAST && __builtin_expect(block_idx + 3 < end_block_idx, true)) {
#if DSA_SIM_AIU
        auto gmem_thr_copy_K = tiled_copy.get_thread_slice(sim_cross_tid);
        Tensor tKsK1 = gmem_thr_copy_K.partition_D(nxt_sKSim1);
#else
        auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
        Tensor tKsK1 = gmem_thr_copy_K.partition_D(nxt_sK1);
#endif
        dsa_issue_K_load<0, 4, T>(tiled_copy, tKgK, tKsK1, &barriers_K1[0],
            precomp_ptr1, precomp_valid1, gIndices_ptr, nxt_block1, real_end_block_idx, nxt_token_idx1);
    }

    wg0_scale_rP0<T>(sScale1, rP0, rPb, idx_in_warpgroup);
    save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);

    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sP0Ready);

    // Wait for warpgroup 1, rescale O0, issue rO0 += rPb @ sV1L
    if constexpr (!IS_BLK0_LAST)
    {
        wg0_rescale_rO0<T>(rO0, sScale1, rL, idx_in_warpgroup);
        dsa_warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, idx_in_warpgroup, wg_idx);
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST)
    {
        cute::clear(rP0);
        dsa_warpgroup_cooperative_qkt_gemm<T, 0>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        // block_idx+4 (even, WG0-owned): this compute point writes its flag.
        // vi slot (block_idx/2)%2 == ((block_idx+4)/2)%2 -- the same slot the
        // top-of-subroutine softmax read for block_idx; that read is already
        // done, so the in-place slot reuse is safe. 
        if (block_idx + 4 < end_block_idx) {
            dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx0, block_idx + 4, seqlen_k, idx_in_warpgroup,
                smem_valid_indices, (block_idx/2)%2, precomp_ptr0, precomp_valid0);
        }
        // block_idx+5 (odd, WG1-owned): address only 
        if (block_idx + 5 < end_block_idx) {
            dsa_compute_K_addr<T, false>(params, gK_base, nxt_token_idx1, block_idx + 5, seqlen_k, idx_in_warpgroup,
                smem_valid_indices, 0, precomp_ptr1, precomp_valid1);
        }
    }

    // Issue P0 = Q @ K0^T
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST)
    {
        dsa_warpgroup_cooperative_qkt_gemm<T, 2>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
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
    typename Engine13, typename Layout13,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void dsa_wg1_subroutine(
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
    __mbarrier_t barriers_K0[T::kHeadDim/256],
    __mbarrier_t barriers_K1[T::kHeadDim/256],
    bool &cur_phase_K1,
    const SparsePrefillParams &params,
    int* gIndices_ptr,
    int seqlen_k,
    int block_idx,
    int end_block_idx,       // [Even-align] ROUNDED-up (even) block count: loop/issue/compute guards
    int real_end_block_idx,  // [Even-align] REAL topk block count: gIndices prefetch guards only
    int idx_in_warpgroup,
    int wg_idx,
    int &kv_idx,
    int& nxt_token_idx0,
    int& nxt_token_idx1, 
    typename T::InputT*& precomp_ptr0,
    typename T::InputT*& precomp_ptr1,
    bool& precomp_valid0,
    bool& precomp_valid1,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices
) {
    using InputT = typename T::InputT;
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx + 4;
    int nxt_block1 = block_idx + 5;
    InputT* gK_base = reinterpret_cast<InputT*>(params.kv);

    auto nxt_sK0 = cur_sK1;
    Tensor sV0R = get_half_V<T, 1>(cur_sK0);
    Tensor sV1R = get_half_V<T, 1>(cur_sK1);
#if DSA_SIM_AIU
    auto nxt_sKSim1 = make_tensor(nxt_sK1.data(), (typename T::SmemLayoutKSim){})(_, _, 0);
    auto nxt_sKSim0 = make_tensor(nxt_sK0.data(), (typename T::SmemLayoutKSim){})(_, _, 0);
    int sim_cross_tid;
    if constexpr (T::Arch_value == 80) {
        int cross_tid_h = (idx_in_warpgroup & 0xFFFFFFF8) >> 3;
        int cross_tid_l = idx_in_warpgroup & 0x7;
        int cross_bias = (cross_tid_l / 2 == 1) ? 2 : ((cross_tid_l / 2 == 2) ? 1 : cross_tid_l / 2);
        cross_tid_h = (cross_tid_h & 0xFFFFFFFC) | (((cross_tid_h & 0x3) + cross_bias) & 0x3);
        sim_cross_tid = (cross_tid_h << 3) | cross_tid_l;
    } else {
        sim_cross_tid = idx_in_warpgroup;
    }
#endif

    // Wait for rO1 += rP1b @ sV1R, launch TMA for the next V1R
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
#if DSA_SIM_AIU
        auto gmem_thr_copy_K = tiled_copy.get_thread_slice(sim_cross_tid);
        Tensor tKsK1 = gmem_thr_copy_K.partition_D(nxt_sKSim1);
#else
        auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
        Tensor tKsK1 = gmem_thr_copy_K.partition_D(nxt_sK1);
#endif
        dsa_issue_K_load<4, T::NUM_TILES, T>(tiled_copy, tKgK, tKsK1, &barriers_K1[1],
            precomp_ptr0, precomp_valid0, gIndices_ptr, nxt_block1, real_end_block_idx, nxt_token_idx1);
    }

    float r_cur_max[2];
    dsa_wg1_bunch_0_pre<T, IS_BLK0_LAST, IS_BLK1_LAST>(r_cur_max, rP1,
        params.sm_scale_div_log2, start_token_idx+T::kBlockN, idx_in_warpgroup,
        smem_valid_indices, 2+(block_idx/2)%2
    );

    // Wait for rP1 and warpgroup 0, run bunch 1, notify warpgroup 0
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);

    Tensor rP1b = wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST>(sScale1, rO1, sM, rL,
        sScale0, rP1, params.sm_scale_div_log2, start_token_idx+T::kBlockN, idx_in_warpgroup,
        smem_valid_indices, 2+(block_idx/2)%2, r_cur_max
    );

    // Save rPb to sP before arriving sScale1Ready, so that sScale1Ready also guarantees
    // that sP1 is ready for warpgroup 0's remote P V gemm (which reads sP1 after waiting
    // sScale1Ready). rP1b is fully produced by wg1_bunch_0 above.
    if constexpr (!IS_BLK0_LAST) {
        save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
    }
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    wg1_scale0_rO1<T>(rO1, sScale0, sScale1, idx_in_warpgroup);
    if constexpr (!IS_BLK0_LAST) {
        dsa_warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV1R, rO1, idx_in_warpgroup, wg_idx);
    }

    // Wait for sP0, issue rO1 += sP0 @ sV0R, notify warpgroup 0
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sP0Ready);

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
#if DSA_SIM_AIU
        auto gmem_thr_copy_K = tiled_copy.get_thread_slice(sim_cross_tid);
        Tensor tKsK0 = gmem_thr_copy_K.partition_D(nxt_sKSim0);
#else
        auto gmem_thr_copy_K = tiled_copy.get_thread_slice(idx_in_warpgroup);
        Tensor tKsK0 = gmem_thr_copy_K.partition_D(nxt_sK0);
#endif
        dsa_issue_K_load<4, T::NUM_TILES, T>(tiled_copy, tKgK, tKsK0, &barriers_K0[1],
            precomp_ptr1, precomp_valid1, gIndices_ptr, nxt_block0, real_end_block_idx, nxt_token_idx0);
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        // block_idx+5 (odd, WG1-owned): this compute point writes its flag.
        // vi slot 2+(block_idx/2)%2 == 2+((block_idx+5)/2)%2 -- the same slot
        // this iteration's bunch_0_pre/wg1_bunch_0 read for block_idx+1; that
        // read is already done (top of the subroutine), so the in-place slot
        // reuse is safe. 
        if (block_idx + 5 < end_block_idx) {
            dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx1, block_idx + 5, seqlen_k, idx_in_warpgroup,
                smem_valid_indices, 2+(block_idx/2)%2, precomp_ptr0, precomp_valid0);
        }
        // block_idx+4 (even, WG0-owned): address only
        if (block_idx + 4 < end_block_idx) {
            dsa_compute_K_addr<T, false>(params, gK_base, nxt_token_idx0, block_idx + 4, seqlen_k, idx_in_warpgroup,
                smem_valid_indices, 0, precomp_ptr1, precomp_valid1);
        }
    }

    dsa_warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx);

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP1);
        // Issue rP1 = sQ @ sK1, wait
        dsa_warpgroup_cooperative_qkt_gemm<T, 1>(sQ, nxt_sK0, nxt_sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
    }

    kv_idx = (kv_idx + 2) % 3;
    cur_sK1 = sK(_, _, kv_idx);
    cur_sK0 = sK(_, _, (kv_idx + 1) % 3);
    nxt_sK1 = sK(_, _, (kv_idx + 2) % 3);
}

__forceinline__ __device__ int get_mask_len(const SparsePrefillParams &params, int m_block_idx, int local_seq_q_idx) {
    int global_seq_q_idx = m_block_idx * Config::BLOCK_SIZE_M + local_seq_q_idx;
    int groups = params.h_q / params.h_kv;
    if (global_seq_q_idx < params.s_q * params.h_q) {
        int s_q_idx = global_seq_q_idx / groups;
        return params.s_q - s_q_idx - 1;
    } else {
        // Out-of-bound request, regard as no masks
        return 0;
    }
}

template<typename T, bool Is_causal>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_sparse_prefill_fwd_wg_kernel(__grid_constant__ const SparsePrefillParams params) {
    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    const int warpgroup_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 256);
    const int idx_in_warpgroup = threadIdx.x % 256;
    const int warp_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 32);

    const int tidx = threadIdx.x;
    using InputT = typename T::InputT;
    typename T::TiledMma tiled_mma;
    // DSA Cache load
    using KVCacheGmem = flash::KVCacheGmemBf16<InputT, T::kBlockN, 256>; // use 256 thread to load N32
    using GmemTiledCopyKNoAiu = typename KVCacheGmem::GmemTiledCopy;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    using SharedMemoryPlan = typename T::SharedMemoryPlan;
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan *>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_sQ.data()), (typename T::SmemLayoutQ){});
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_sK.data()), (typename T::SmemLayoutK){});
#if DSA_SIM_AIU
    Tensor sKSim = make_tensor(sK.data(), (typename T::SmemLayoutKSim){});
#endif
    Tensor sP0 = make_tensor(flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, _0{}, Int<T::NUM_TILES - 1>{}).data(), (typename T::SmemLayoutP0){}); // Overlap with sQ's last tile
    Tensor sP1 = make_tensor(sP0.data() + sP0.size(), (typename T::SmemLayoutP0){});
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL_reduction_wksp = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<2 * T::BLOCK_SIZE_M>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(plan.smem_valid_indices.data()),
        Shape<_4, Int<T::kBlockN>>{}, Stride<Int<T::kBlockN>, _1>{});
    char *sO_addr = (char *)plan.smem_sQ.data(); // Overlap with sK0 and sK1
    int q_idx = m_block_idx / (params.h_q / T::kBlockM);
    int* gIndices_ptr = params.indices + (int64_t)q_idx * params.stride_indices_s_q + idx_in_warpgroup / 8;
    InputT* gK_base = reinterpret_cast<InputT*>(params.kv);
    int nxt_token_idx0 = -1;
    int nxt_token_idx1 = -1;
    InputT *precomp_ptr0 = nullptr, *precomp_ptr1 = nullptr;
    bool precomp_valid0 = false, precomp_valid1 = false;
    constexpr int kBlockN = T::kBlockN;

    // Define TMA stuffs
    __mbarrier_t *barrier_Q = &(plan.barrier_Q);
    __mbarrier_t *barriers_K0 = plan.barriers_K0;
    __mbarrier_t *barriers_K1 = plan.barriers_K1;

    // Initialize TMA barriers
    if (threadIdx.x == 0) {
        __mbarrier_init(barrier_Q, 32);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            __mbarrier_init(&barriers_K0[i], 256);
            __mbarrier_init(&barriers_K1[i], 256);
        }
    }
    __syncthreads();

    // [Even-align] seqlen_k and the block range are computed BEFORE the initial
    // token-index reads so those reads can be guarded by the REAL block count.
    // real_end_block_idx: true topk block count (ceil(seqlen_k/kBlockN)) -- the
    //   ONLY bound ever used to guard gIndices reads; the indices row has
    //   exactly topk entries, so the rounded-up padding block has no backing
    //   memory there.
    // end_block_idx: real_end rounded UP to an even block count (seqlen_k
    //   rounded to 2 blocks = 64 tokens
    int seqlen_k = params.topk_length ? __ldg(params.topk_length + q_idx) : params.topk;
    int start_block_idx = 0;
    int real_end_block_idx = cute::ceil_div(seqlen_k, kBlockN);
    int end_block_idx = cute::ceil_div(cute::round_up(seqlen_k, 2 * kBlockN), kBlockN);

    if (warpgroup_idx == 0) {
        nxt_token_idx0 = 0 < real_end_block_idx ? __ldg(gIndices_ptr) : -1;
    } else {
        nxt_token_idx1 = 1 < real_end_block_idx ? __ldg(gIndices_ptr + kBlockN) : -1;
    }

    bool cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;
    // Copy the first Q
    launch_q_dsa_prefill_wg<T>(params, m_block_idx, sQ, tidx, warp_idx, barrier_Q);

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<InputT*>(params.kv)),
                        Shape<Int<kBlockN>, Int<T::kHeadDim>>{},
                        make_stride(params.stride_kv_s_kv, _1{}));

    // Copy K0 and K1
    GmemTiledCopyKNoAiu gmem_tiled_copy_K;
#if DSA_SIM_AIU
    int sim_cross_tid;
    if constexpr (T::Arch_value == 80) {
        int cross_tid_h = (idx_in_warpgroup & 0xFFFFFFF8) >> 3;
        int cross_tid_l = idx_in_warpgroup & 0x7;
        int cross_bias = (cross_tid_l / 2 == 1) ? 2 : ((cross_tid_l / 2 == 2) ? 1 : cross_tid_l / 2);
        cross_tid_h = (cross_tid_h & 0xFFFFFFFC) | (((cross_tid_h & 0x3) + cross_bias) & 0x3);
        sim_cross_tid = (cross_tid_h << 3) | cross_tid_l;
    } else {
        sim_cross_tid = idx_in_warpgroup;
    }
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(sim_cross_tid);
#else
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(idx_in_warpgroup);
#endif
    Tensor tKgK = gmem_thr_copy_K.partition_S(gK); // (KCPY, KCPY_N, KCPY_K)

    Tensor cur_sK0 = sK(_, _, 0);
    Tensor cur_sK1 = sK(_, _, 1);
    Tensor nxt_sK0 = sK(_, _, 2);
#if DSA_SIM_AIU
    Tensor cur_sKSim0 = sKSim(_, _, 0);
    Tensor cur_sKSim1 = sKSim(_, _, 1);
    Tensor tKsK0 = gmem_thr_copy_K.partition_D(cur_sKSim0);
    Tensor tKsK1 = gmem_thr_copy_K.partition_D(cur_sKSim1);
#else
    Tensor tKsK0 = gmem_thr_copy_K.partition_D(cur_sK0);
    Tensor tKsK1 = gmem_thr_copy_K.partition_D(cur_sK1);
#endif

    if (warpgroup_idx == 0 && seqlen_k != 0) {
        InputT *addr_blk0; bool valid_blk0;
        dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx0, start_block_idx, seqlen_k, idx_in_warpgroup,
            smem_valid_indices, (start_block_idx/2)%2, addr_blk0, valid_blk0);
        dsa_issue_K_load<4, T::NUM_TILES, T, false>(gmem_tiled_copy_K, tKgK, tKsK1, &barriers_K0[1],
            addr_blk0, valid_blk0, gIndices_ptr, 0, real_end_block_idx, nxt_token_idx0);
        dsa_issue_K_load<0, 4, T, false>(gmem_tiled_copy_K, tKgK, tKsK0, &barriers_K0[0],
            addr_blk0, valid_blk0, gIndices_ptr, 0, real_end_block_idx, nxt_token_idx0);
        // [Even-align] guarded by the REAL block count; padding blocks -> -1 (invalid).
        nxt_token_idx0 = 2 < real_end_block_idx ? __ldg(gIndices_ptr + kBlockN * 2) : -1;
        nxt_token_idx1 = 3 < real_end_block_idx ? __ldg(gIndices_ptr + kBlockN * 3) : -1;
        // WG-scope barrier: makes the flag stores above visible to all 256 threads
        // of WG0 before iteration-0's wg0_bunch_0 reads them (no other barrier sits
        // between the prolog write and that read). WG0-only: does not couple the
        // WG0/WG1 arrive/wait skew, hence no cross-WG sync stall.
        NamedBarrier::arrive_and_wait(T::NUM_THREADS/2, NamedBarriers::mGroup0);
        if (start_block_idx + 2 < end_block_idx) {
            dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx0, start_block_idx + 2, seqlen_k, idx_in_warpgroup,
                smem_valid_indices, (start_block_idx/2+1)%2, precomp_ptr0, precomp_valid0);
        }
        if (start_block_idx + 3 < end_block_idx) {
            dsa_compute_K_addr<T, false>(params, gK_base, nxt_token_idx1, start_block_idx + 3, seqlen_k, idx_in_warpgroup,
                smem_valid_indices, 0, precomp_ptr1, precomp_valid1);
        }
    }

    if (start_block_idx+1 < end_block_idx) {
        if (warpgroup_idx == 1) {
            InputT *addr_blk1; bool valid_blk1;
            dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx1, start_block_idx + 1, seqlen_k, idx_in_warpgroup,
                smem_valid_indices, 2+(start_block_idx/2)%2, addr_blk1, valid_blk1);
            dsa_issue_K_load<4, T::NUM_TILES, T, false>(gmem_tiled_copy_K, tKgK, tKsK0, &barriers_K1[1],
                addr_blk1, valid_blk1, gIndices_ptr, 0, real_end_block_idx, nxt_token_idx0);
            dsa_issue_K_load<0, 4, T, false>(gmem_tiled_copy_K, tKgK, tKsK1, &barriers_K1[0],
                addr_blk1, valid_blk1, gIndices_ptr, 0, real_end_block_idx, nxt_token_idx0);
            // [Even-align] guarded by the REAL block count; padding blocks -> -1 (invalid).
            nxt_token_idx0 = 2 < real_end_block_idx ? __ldg(gIndices_ptr + kBlockN * 2) : -1;
            nxt_token_idx1 = 3 < real_end_block_idx ? __ldg(gIndices_ptr + kBlockN * 3) : -1;
            // [Opt-A] WG1-scope barrier, mirroring the mGroup0 one above:
            // iteration-0's dsa_wg1_bunch_0_pre now reads the block-1 flags
            // BEFORE the subroutine's sScale0Ready arrive_and_wait (which used
            // to be the sole visibility cover -- see the comment at the flag
            // store above), so the stores need their own WG1-scope barrier,
            // exactly like WG0's mGroup0. WG1-only (256 threads): does not
            // couple the WG0/WG1 arrive/wait skew, hence no cross-WG stall.
            NamedBarrier::arrive_and_wait(T::NUM_THREADS/2, NamedBarriers::mGroup1);
            if (start_block_idx + 3 < end_block_idx) {
                dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx1, start_block_idx + 3, seqlen_k, idx_in_warpgroup,
                    smem_valid_indices, 2+(start_block_idx/2+1)%2, precomp_ptr0, precomp_valid0);
            }
            if (start_block_idx + 2 < end_block_idx) {
                dsa_compute_K_addr<T, false>(params, gK_base, nxt_token_idx0, start_block_idx + 2, seqlen_k, idx_in_warpgroup,
                    smem_valid_indices, 0, precomp_ptr1, precomp_valid1);
            }
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

    // rQ8 stores the last tile of Q (tile 8 for 576, tile 7 for 512) to leave smem room for sP0/sP1
    Tensor rQ8 = make_tensor<InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
    retrieve_rP_from_sP<T>(rQ8, local_tile(sQ, Shape<_128, _64>{}, Coord<_0, Int<T::NUM_TILES - 1>>{}), idx_in_warpgroup);

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
            dsa_warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK0, cur_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
        }

        int idx = 0;
        #define DSA_LAUNCH_WG0_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST)                     \
            dsa_wg0_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST>(                            \
            gmem_tiled_copy_K, tKgK, sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
            rP0, rO, rL,                   \
            barriers_K0, barriers_K1, cur_phase_K0, params, gIndices_ptr, seqlen_k, \
            block_idx, end_block_idx, real_end_block_idx, idx_in_warpgroup, wg_idx, idx, nxt_token_idx0, nxt_token_idx1, precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1 \
            , smem_valid_indices);

        int block_idx = start_block_idx;

        #pragma unroll 1
        for (; block_idx < end_block_idx-2; block_idx += 2) {
            DSA_LAUNCH_WG0_SUBROUTINE(false, false);
        }

        // [Even-align] end_block_idx is always even, so the loop leaves exactly
        // one uniform 2-block tail; the IS_BLK0_LAST single-block branch is
        // unreachable and removed (mirrors decode's single tail form).
        if (block_idx < end_block_idx) {
            DSA_LAUNCH_WG0_SUBROUTINE(false, true);
        }
    }
    else {
        // // Warpgroup 1
        // Tensor rP1 = make_tensor<float>((typename T::rP0Layout){});
        Tensor rP1 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
        const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

        if (start_block_idx+1 < end_block_idx) {
            // Issue rP1 = sQ @ sK1, wait
            dsa_warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK1, cur_sK0, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
        }

        int idx = 0;
        #define DSA_LAUNCH_WG1_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST)                     \
            dsa_wg1_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST>(                            \
            gmem_tiled_copy_K, tKgK, sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
            rP1, rO, rL,                 \
            barriers_K0, barriers_K1, cur_phase_K1, params, gIndices_ptr, seqlen_k, \
            block_idx, end_block_idx, real_end_block_idx, idx_in_warpgroup, wg_idx, idx, nxt_token_idx0, nxt_token_idx1, precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1 \
            , smem_valid_indices);

        int block_idx = start_block_idx;
        // [Even-align] end_block_idx is always even: bound end-2 gives the same
        // trip count as the old end-3, and the loop leaves exactly one uniform
        // 2-block tail. 
        #pragma unroll 1
        for (; block_idx < end_block_idx-2; block_idx += 2) {
            DSA_LAUNCH_WG1_SUBROUTINE(false, false);
        }

        if (block_idx < end_block_idx) {
            DSA_LAUNCH_WG1_SUBROUTINE(false, true);
        }
    }

    // Reduce rL across threads within the same warp
    rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
    rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
    rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
    rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

    // Reduce rL across warpgroups
    int my_row = get_AorC_row_idx(0, idx_in_warpgroup);
    float pre_sink0 = 0.0f, pre_sink1 = 0.0f;
    if (params.attn_sink != nullptr && seqlen_k > 0) {
        int head_block_idx = m_block_idx % (params.h_q / T::kBlockM);
        pre_sink0 = __ldg(params.attn_sink + head_block_idx * T::BLOCK_SIZE_M + my_row);
        pre_sink1 = __ldg(params.attn_sink + head_block_idx * T::BLOCK_SIZE_M + my_row + 8);
    }
    if (idx_in_warpgroup % 4 == 0) {
        sL_reduction_wksp[my_row + warpgroup_idx * 128] = rL[0];
        sL_reduction_wksp[my_row + 8 + warpgroup_idx * 128] = rL[1];
    }
    __syncthreads();

    if (warpgroup_idx == 0) {
        rL[0] += sL_reduction_wksp[my_row + 128];
        rL[1] += sL_reduction_wksp[my_row + 8 + 128];
    }
    else {
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

    // Apply attention sink: adjust rL to account for the sink term in the normalization denominator.
    // rL_adjusted = rL + exp2f(sink_log2 - sM[row])
    // where sink_log2 = attn_sink[head_idx] * M_LOG2E (convert natural log to log2 space)
    // and sM[row] is the log2-space scaled max (max(raw_score) * sm_scale_div_log2)
    // This ensures output = rO / rL_adjusted = sum(P*V) / (sum(P) + exp(attn_sink))
    // LSE and max_logits outputs are NOT affected (they read from sL_reduction_wksp, not rL)
    if (params.attn_sink != nullptr && seqlen_k > 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            int row_idx = get_AorC_row_idx(i, idx_in_warpgroup);
            // attn_sink value pre-issued above (pre_sink0/pre_sink1, before the
            // rL-reduction __syncthreads); only the arithmetic stays here.
            float sink_log2 = (i == 0 ? pre_sink0 : pre_sink1) * (float)M_LOG2E;
            rL[i] += exp2f(sink_log2 - sM(row_idx));
        }
    }

    // Epilogue
    int num_valid_seq_q = min(params.s_q * params.h_q - m_block_idx * T::BLOCK_SIZE_M, T::BLOCK_SIZE_M);
    InputT *o_ptr = (InputT *)params.out + (int64_t)m_block_idx * T::BLOCK_SIZE_M * T::kHeadDimV;
    float *softmax_lse_ptr = (float *)params.lse + m_block_idx * T::BLOCK_SIZE_M;
    float *mlogits_ptr = (float *)params.max_logits + m_block_idx * T::BLOCK_SIZE_M;

    Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                                                        Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV>>{},
                                                        make_stride(T::kHeadDimV, _1{})));
    Tensor gSoftmaxLse = make_tensor(make_gmem_ptr(softmax_lse_ptr), Layout<
                                                                            Shape<Int<T::BLOCK_SIZE_M>>,
                                                                            Stride<_1>>{});
    Tensor gMLogits = make_tensor(make_gmem_ptr(mlogits_ptr), Layout<
                                                                            Shape<Int<T::BLOCK_SIZE_M>>,
                                                                            Stride<_1>>{});

    dsa_store_o<T>(rO, gO, rL, sO_addr, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

    int i = threadIdx.x;
    if (i < num_valid_seq_q) {
        float cur_L = sL_reduction_wksp[i];
        gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L) ? INFINITY : logf(cur_L) + sM(i) / (float)M_LOG2E;
        // sM is monotonically non-decreasing from its init MAX_INIT_VAL_SM; staying at the
        // init value means no column ever exceeded it, i.e. the row is entirely masked out
        // (seqlen_k==0, or all selected tokens invalid / beyond the border). Emit -inf then,
        // matching the reference max_logits semantics.
        gMLogits(i) = (seqlen_k == 0 || sM(i) <= MAX_INIT_VAL_SM) ? -INFINITY : sM(i) * M_LN2;
    }
}

template<typename InputT, int Arch, int HEAD_DIM>
void run_flash_sparse_prefill_fwd_wg(SparsePrefillParams &params) {
    using T = DSA_Traits<cutlass::bfloat16_t, HEAD_DIM, Arch>;

    auto mla_kernel = &flash_sparse_prefill_fwd_wg_kernel<T, true>;
    constexpr size_t smem_size = std::max(sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));
        C10_CUDA_CHECK(hggcFuncSetAttribute(mla_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    const int num_m_block = params.s_q * cute::ceil_div(params.h_q, T::kBlockM);

    int ctas_per_sm;
    hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, mla_kernel, T::NUM_THREADS, smem_size);

    char *pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        printf("[dsa_prefill_wg]:\n");
        printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

        hggcFuncAttributes attr;
        hggcFuncGetAttributes(&attr, mla_kernel);
        auto dprops = at::cuda::getCurrentDeviceProperties();

        int sm_count = dprops->multiProcessorCount;
        if (std::string(dprops->name).find("810E") != std::string::npos) {
            sm_count = 20;
        }

        printf("blockM:%d, blockN:%d, threads:%d\n",
                T::kBlockM, T::kBlockN, T::NUM_THREADS);
        // printf("Is_causal:%d\n", Is_causal);
        printf("grid_n[%d, %d, %d]\n", num_m_block, params.h_kv, 1);
        printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f, Arch:%d\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                float(num_m_block * params.h_kv) / float(sm_count * ctas_per_sm), Arch);
    }

    // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
    // hggcLaunchAttribute mla_kernel_attributes[1];
    // mla_kernel_attributes[0].id = hggcLaunchAttributeProgrammaticStreamSerialization;
    // mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    // hggcLaunchConfig_t mla_kernel_config = {
    //     dim3(num_m_block, params.h_kv, 1),
    //     dim3(T::NUM_THREADS, 1, 1),
    //     smem_size,
    //     params.stream,
    //     mla_kernel_attributes,
    //     1
    // };
    // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params);

    mla_kernel<<<dim3(num_m_block, params.h_kv, 1), T::NUM_THREADS, smem_size, params.stream>>>(params);

    CHECK_CUDA_KERNEL_LAUNCH();
}
