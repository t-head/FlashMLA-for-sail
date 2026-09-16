// =============================================================================
// Based on the accepted C5.17 source:
//   commit: c6f41064b31305e21a94555a3e688192052b0acb
//   source MD5: 519cca234ec6cfff4de429ff7450fce5
// The BF16 M64 pipeline is retained, with D576 shared-memory lifetime fixes.
// sparse_decode_wg_hs64.cuh
//
// Sparse decode warp-interleave (T2) variant, forked from
// csrc/flash_splitkv/splitkv_mla.cu. Reuses the double-warpgroup ping-pong
// skeleton; sparse-specific parts are the Q/K-load layer (cp.async + indices
// indirection) and the indices-driven mask.
// =============================================================================

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <hggc_runtime.h>
#include <hggc_pipeline.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "decode/sparse/sparse_decode_wg.h"
#include "kerutils/host/hardware_info.h"
#include "kerutils/common/common.h"
#include "decode/sparse/sparse_decode_wg_traits.h"
#include "acc_vreg_fraga.h"
#include "ppuxx/decode/combine/combine.h"

namespace cute {
// D576 retains explicit L2 hints but omits the real-copy prefetch-size hint.
struct HS64_CP_ASYNC_NO_PREF128_ZFILL {};

template <>
struct Copy_Traits<HS64_CP_ASYNC_NO_PREF128_ZFILL>
    : Copy_Traits<PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>> {
    template <class TS, class SLayout, class TD, class DLayout>
    CUTE_HOST_DEVICE friend void copy_unpack(
        Copy_Traits const& traits,
        Tensor<TS, SLayout> const& src, Tensor<TD, DLayout>& dst) {
        static_assert(is_gmem<TS>::value && is_smem<TD>::value);
        Tensor rS = recast<uint128_t>(src);
        Tensor rD = recast<uint128_t>(dst);
        CUTE_STATIC_ASSERT_V(size(rS) == Int<1>{});
        CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{});
#if defined(CUTE_ARCH_CP_ASYNC_PPU_ENABLED) || defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        // SDK zfill counts the bytes to clear, not the bytes to copy.
        // Policy 1 preserves the cg path without a bulk-prefetch-size hint.
        __ppu_pipeline_memcpy_async_zfill(
            &rD[0], &rS[0], sizeof(uint128_t),
            traits.pred ? 0 : sizeof(uint128_t), 1);
#else
        CUTE_RUNTIME_ASSERT("HS64 async copy requires PPU support");
#endif
    }
};

// HS64 K-only atom: preserve the original descriptor and bit layouts.
struct HS64_TSM_K_UNIT16 {
    CUTE_HOST_DEVICE static void copy(
        void* dst, void* base, unsigned coord_w, unsigned coord_h,
        unsigned cube = 0, unsigned stage = 0) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
        const uint32_t base_units = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(base) >> 4);
        const uint32_t unit_offset =
            512u * (cube + stage) + coord_h * 8u + (coord_w >> 3);
        const uint32_t units = base_units + unit_offset;
        PPU0015_TSM_LD_SWZL_IMPL<cutlass::bfloat16_t, false>()(
            reinterpret_cast<int*>(dst), static_cast<int>(units), 64, 1, 0);
#else
        CUTE_RUNTIME_ASSERT("HS64 K unit address requires PPU1.5");
#endif
    }
};

template <>
struct Copy_Traits<HS64_TSM_K_UNIT16>
    : Copy_Traits<PPU_TSM_LD_SWZL<cutlass::bfloat16_t, 64, 64, true, false, 1>> {
    template <class Coord, int... Is>
    CUTE_HOST_DEVICE void unpack(
        void* dst, void* base, Coord const& coord, seq<Is...>) const {
        HS64_TSM_K_UNIT16::copy(dst, base, static_cast<unsigned>(get<Is>(coord))...);
    }
    template <class TS, class LS, class TD, class LD>
    CUTE_HOST_DEVICE friend void copy_unpack(
        Copy_Traits const& traits, Tensor<TS, LS> const& src, Tensor<TD, LD>& dst) {
        static_assert(is_mix_iterator<typename TS::iterator>::value);
        traits.unpack(raw_pointer_cast(dst.data()), src.data().ptr_.get(),
                      src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    }
};
}  // namespace cute

namespace flashmla::dsa::hs64 {

using namespace cute;

// SM80 simulated AIU storage rotates the four rows of each 16-column slab.
// Match M128's sim_cross_tid mapping: slab 0/1/2/3 uses bias 0/2/1/3.
__forceinline__ __device__ int hs64_sm80_store_row(int row, int col) {
    const int slab = (col / 16) & 3;
    const int bias = ((slab & 1) << 1) | (slab >> 1);
    return (row & ~3) | ((row + bias) & 3);
}

template<typename T>
__forceinline__ __device__ int hs64_store_thread(int tid) {
    if constexpr (T::kArch == 80) {
        return hs64_sm80_store_row(tid / 8, (tid & 7) * 8) * 8 + (tid & 7);
    } else {
        return tid;
    }
}

// Expose the same two-row iteration on either native C fragment. The index
// permutation folds at compile time; no lane shuffle or format conversion.
template<typename T, typename Tensor>
__forceinline__ __device__ decltype(auto) hs64_acc(Tensor &acc, int i) {
    if constexpr (T::kArch == 80) {
        return acc((i & ~6) | ((i & 2) << 1) | ((i & 4) >> 1));
    } else {
        return acc(i);
    }
}

template<typename T, typename Layout>
__forceinline__ __device__ auto hs64_convert_layout_acc_rowcol(Layout acc_layout)
{
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto atom_div = logical_divide(acc_layout, Shape<_4>{});
    if constexpr (T::kArch == 80) {
        return make_layout(make_layout(get<0, 1>(atom_div), get<1>(atom_div)),
                           make_layout(get<0, 0>(atom_div), get<2>(atom_div)));
    } else {
    auto row_div = logical_divide(atom_div, Shape<Shape<_2>>{});
    return make_layout(
        make_layout(get<0, 0, 1>(row_div), get<1>(row_div)),
        make_layout(get<0, 0, 0>(row_div),
                    make_layout(get<0, 1>(row_div), get<2>(row_div))));
    }
}

// Build tag printed once per kernel launch, for log attribution.
inline constexpr char kHs64BuildTag[] = "HS64_BF16_LOCAL_K_PIPELINE";

// Shared max/scale/P publication needs TSM ordering and the full rendezvous.
// D576 scopes these exchanges independently of pending VMEM operations;
// K completion, buffer-reader drains, and global-output barriers remain explicit.
template <typename T>
__forceinline__ __device__ void hs64_shared_exchange_sync(int id, int count) {
    if constexpr (T::kHasExtraKTile) {
        __ppu_barrier_sync(id, count, 15u);
    } else {
        cutlass::arch::NamedBarrier::sync(
            count, static_cast<cutlass::arch::ReservedNamedBarriers>(id));
    }
}


// Here we use MAX_INIT_VAL_SM to initialize sM, and MAX_INIT_VAL for masking
// The reason is that, we need to calculate new_max = max(sM(row_idx), cur_max*scale_softmax_log2)
// so we must guarantee that MAX_INIT_VAL*scale_softmax_log2 < MAX_INIT_VAL_SM
static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

template <typename T, typename TensorVI>
__forceinline__ __device__ int hs64_load_valid(
    TensorVI &smem_valid_indices, int valid_indices_buf, int token_idx) {
    if constexpr (T::kHasExtraKTile) {
        const unsigned owner = (static_cast<unsigned>(token_idx) >> 2) & 7u;
        const unsigned bits = smem_valid_indices(valid_indices_buf, owner);
        const unsigned shift = (static_cast<unsigned>(token_idx) & 3u) * 8u
                             + (static_cast<unsigned>(token_idx) >> 5) * 4u;
        return (bits >> shift) & 1u;
    } else {
        unsigned int bits = smem_valid_indices(valid_indices_buf, token_idx >> 2);
        return (bits >> (token_idx & 3)) & 1u;
    }
}

template <typename T, typename TensorVI>
__forceinline__ __device__
void
hs64_load_valid_pair(
    TensorVI &smem_valid_indices, int valid_indices_buf, int token_idx,
    int &valid0, int &valid1) {
    if constexpr (T::kArch == 80) {
        valid0 = hs64_load_valid<T>(smem_valid_indices, valid_indices_buf, token_idx);
        valid1 = hs64_load_valid<T>(smem_valid_indices, valid_indices_buf, token_idx + 4);
    } else if constexpr (T::kHasExtraKTile) {
        const unsigned owner = (static_cast<unsigned>(token_idx) >> 2) & 7u;
        const unsigned bits = smem_valid_indices(valid_indices_buf, owner);
        const unsigned shift = (static_cast<unsigned>(token_idx) & 3u) * 8u
                             + (static_cast<unsigned>(token_idx) >> 5) * 4u;
        valid0 = (bits >> shift) & 1u;
        valid1 = (bits >> (shift + 8)) & 1u;
    } else {
        unsigned int bits = smem_valid_indices(valid_indices_buf, token_idx >> 2);
        int shift = token_idx & 3;
        valid0 = (bits >> shift) & 1u;
        valid1 = (bits >> (shift + 1)) & 1u;
    }
}

// Natural warp order: two scalar ballots cover both copy waves.
template <typename T, typename TensorVI>
__forceinline__ __device__ void hs64_load_valid_quad(
    TensorVI &smem_valid_indices, int buf, int lane4, int warp_n, int (&valid)[8]) {
    static_assert(T::kHasExtraKTile && T::kValidWords == 16);
    static_assert(offsetof(typename T::SharedMemoryPlan, smem_valid_indices) % 16 == 0);
    if constexpr (T::kArch == 80) {
        CUTLASS_PRAGMA_UNROLL
        for (int cg = 0; cg < 4; ++cg) {
            const int base = (cg / 2) * 32 + warp_n * 16 + (cg % 2) * 8 + lane4;
            hs64_load_valid_pair<T>(smem_valid_indices, buf, base,
                                   valid[2 * cg], valid[2 * cg + 1]);
        }
    } else {
    const unsigned first_word = static_cast<unsigned>(warp_n) * 4u
                              + (static_cast<unsigned>(lane4) >> 1);
    const unsigned words[2] = {smem_valid_indices(buf, first_word),
                               smem_valid_indices(buf, first_word + 2u)};
    const unsigned shift = (static_cast<unsigned>(lane4) & 1u) * 16u;
    CUTLASS_PRAGMA_UNROLL
    for (int cg = 0; cg < 4; ++cg) {
        valid[2 * cg] = (words[cg & 1] >> (shift + (cg / 2) * 4)) & 1u;
        valid[2 * cg + 1] = (words[cg & 1] >> (shift + (cg / 2) * 4 + 8)) & 1u;
    }
    }
}

template <int AtomLayoutM = 8>
__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup)
{
    // In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
    // You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
    int row_idx = ((idx_in_warpgroup / 32) % AtomLayoutM) * 16 + local_row_idx * 8 + (idx_in_warpgroup % 32 / 4);
    return row_idx;
}

template <typename T>
__forceinline__ __device__ int get_PV_row_idx(int local_row_idx, int idx_in_warpgroup)
{
    const int warp_m = (idx_in_warpgroup / 32) % T::kPvAtomLayoutM;
    const int lane_row = (idx_in_warpgroup % 32) / 4;
    return warp_m * 16 + lane_row + (local_row_idx & 1) * 8
           + (local_row_idx / 2) * (16 * T::kPvAtomLayoutM);
}

template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_acc(Tensor<Engine, Layout> const &tensor)
{
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    NumericArrayConverterPPU<To_type, From_type, numel> convert_op;
    // convert_op returns frag from reinterpret_cast of tensor data
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
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
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

// Adapted from https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
// * Copyright (c) 2024, Tri Dao.
struct Wg1ScaleCache {
    float row0;
    float row1;
    float sum0;
    float sum1;
};

struct Wg0SoftmaxSums {
    float row0;
    float row1;
};

template <
    typename Tensor0, typename Tensor1,
    typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(Tensor2 &tCrC, Tensor0 const &tCrA, Tensor1 const &tCrB, TiledMma &tiled_mma)
{
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), tCrC);
    }
}

__forceinline__ __device__ void kernel_sleep_ns()
{
    __nanosleep(1);
}

template <typename T>
__forceinline__ __device__ void kernel_k_wait_sleep_ns()
{
    if constexpr (T::kIsCrossCut && T::kUseQkWeave) {
        __nanosleep(4);
        return;
    }
    __nanosleep(1);
}

// Wait for one KV-tile to be ready, and then calculate P += Q K^T for one Q-tile (BLOCK_SIZE_Mx64) and one KV-tile (PAGE_BLOCK_SIZEx64)
// The Q-tile should be in shared memory
template <
    typename T,
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
    if constexpr (T::kIsCrossCut && T::kUseQkWeave) {
        constexpr int kReductionSlices = 4;
        auto sQ_seed = local_tile(
            sQ_tiled, Shape<Int<T::BLOCK_SIZE_M>, _16>{}, Coord<_0, _0>{});
        auto sK_seed = local_tile(
            sKV_tiled, Shape<Int<T::kBlockN>, _16>{}, Coord<_0, _0>{});
        Tensor rQ0 = thr_mma.partition_fragment_A(sQ_seed);
        Tensor rQ1 = thr_mma.partition_fragment_A(sQ_seed);
        Tensor rQ2 = thr_mma.partition_fragment_A(sQ_seed);
        Tensor rQ3 = thr_mma.partition_fragment_A(sQ_seed);
        Tensor rK0 = thr_mma.partition_fragment_B(sK_seed);
        Tensor rK1 = thr_mma.partition_fragment_B(sK_seed);
        Tensor rK2 = thr_mma.partition_fragment_B(sK_seed);
        Tensor rK3 = thr_mma.partition_fragment_B(sK_seed);
        Tensor cQ0 = smem_thr_copy_Q.retile_D(rQ0);
        Tensor cQ1 = smem_thr_copy_Q.retile_D(rQ1);
        Tensor cQ2 = smem_thr_copy_Q.retile_D(rQ2);
        Tensor cQ3 = smem_thr_copy_Q.retile_D(rQ3);
        Tensor cK0 = smem_thr_copy_K.retile_D(rK0);
        Tensor cK1 = smem_thr_copy_K.retile_D(rK1);
        Tensor cK2 = smem_thr_copy_K.retile_D(rK2);
        Tensor cK3 = smem_thr_copy_K.retile_D(rK3);
        CUTE_STATIC_ASSERT_V(size<0>(thr_mma_sQ_tile) == size<0>(cQ0));
        CUTE_STATIC_ASSERT_V(size<1>(thr_mma_sQ_tile) == size<1>(cQ0));
        CUTE_STATIC_ASSERT_V(size<2>(thr_mma_sQ_tile) == Int<kReductionSlices>{});
        CUTE_STATIC_ASSERT_V(size<2>(cQ0) == _1{});
        CUTE_STATIC_ASSERT_V(size<0>(thr_mma_sKV_tile) == size<0>(cK0));
        CUTE_STATIC_ASSERT_V(size<1>(thr_mma_sKV_tile) == size<1>(cK0));
        CUTE_STATIC_ASSERT_V(size<2>(thr_mma_sKV_tile) == Int<kReductionSlices>{});
        CUTE_STATIC_ASSERT_V(size<2>(cK0) == _1{});

        auto load0 = [&](auto k) {
            cute::copy(smem_tiled_copy_Q, thr_mma_sQ_tile(_, _, k), cQ0(_, _, _0{}));
            cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile(_, _, k), cK0(_, _, _0{}));
        };
        auto load1 = [&](auto k) {
            cute::copy(smem_tiled_copy_Q, thr_mma_sQ_tile(_, _, k), cQ1(_, _, _0{}));
            cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile(_, _, k), cK1(_, _, _0{}));
        };
        auto load2 = [&](auto k) {
            cute::copy(smem_tiled_copy_Q, thr_mma_sQ_tile(_, _, k), cQ2(_, _, _0{}));
            cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile(_, _, k), cK2(_, _, _0{}));
        };
        auto load3 = [&](auto k) {
            cute::copy(smem_tiled_copy_Q, thr_mma_sQ_tile(_, _, k), cQ3(_, _, _0{}));
            cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile(_, _, k), cK3(_, _, _0{}));
        };
        load0(_0{});
        load1(_1{});
        load2(_2{});
        load3(_3{});
        cute::gemm(tiled_mma, cQ0(_, _, _0{}), cK0(_, _, _0{}), rP);
        cute::gemm(tiled_mma, cQ1(_, _, _0{}), cK1(_, _, _0{}), rP);
        cute::gemm(tiled_mma, cQ2(_, _, _0{}), cK2(_, _, _0{}), rP);
        cute::gemm(tiled_mma, cQ3(_, _, _0{}), cK3(_, _, _0{}), rP);
        return;
    }
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

// Carry the four existing K=16 fragment slots across the already-ready low
// half-bank. Each slot is refilled only after its previous MMA has consumed it.
template <typename T, typename TiledMMA, typename CopyQ, typename CopyK,
          typename ThrQ, typename ThrK, typename SQ, typename SK,
          typename QSource, typename KSource, typename CachedQ, typename Accumulator>
__forceinline__ __device__ void qkt_gemm_low_tiles_sm80(
    TiledMMA &mma, CopyQ &copy_q, CopyK &copy_k, ThrQ &thr_q, ThrK &thr_k,
    SQ &sQ_tiled, SK &sK_tiled, QSource const &q_source,
    KSource const &k_source, CachedQ const &q_cached, Accumulator &rP, int tid)
{
    static_assert(T::kArch == 80);
    auto thread_mma = mma.get_slice(tid);
    auto q_seed = local_tile(sQ_tiled(_, _, _0{}),
        Shape<Int<T::kBlockM>, _16>{}, Coord<_0, _0>{});
    auto k_seed = local_tile(sK_tiled(_, _, _0{}),
        Shape<Int<T::kBlockN>, _16>{}, Coord<_0, _0>{});
    auto q_fragment = thread_mma.partition_fragment_A(q_seed);
    auto k_fragment = thread_mma.partition_fragment_B(k_seed);
    // Preserve two BF16 values per word across the loop backedge, avoiding
    // scalar half-word PHIs and the corresponding unpack/repack instructions.
    auto q_word_layout = recast<uint32_t>(q_fragment).layout();
    auto k_word_layout = recast<uint32_t>(k_fragment).layout();
    Tensor wQ0 = make_tensor<uint32_t>(q_word_layout);
    Tensor wQ1 = make_tensor<uint32_t>(q_word_layout);
    Tensor wQ2 = make_tensor<uint32_t>(q_word_layout);
    Tensor wQ3 = make_tensor<uint32_t>(q_word_layout);
    Tensor wK0 = make_tensor<uint32_t>(k_word_layout);
    Tensor wK1 = make_tensor<uint32_t>(k_word_layout);
    Tensor wK2 = make_tensor<uint32_t>(k_word_layout);
    Tensor wK3 = make_tensor<uint32_t>(k_word_layout);
    Tensor rQ0 = recast<typename T::InputT>(wQ0);
    Tensor rQ1 = recast<typename T::InputT>(wQ1);
    Tensor rQ2 = recast<typename T::InputT>(wQ2);
    Tensor rQ3 = recast<typename T::InputT>(wQ3);
    Tensor rK0 = recast<typename T::InputT>(wK0);
    Tensor rK1 = recast<typename T::InputT>(wK1);
    Tensor rK2 = recast<typename T::InputT>(wK2);
    Tensor rK3 = recast<typename T::InputT>(wK3);
    Tensor cQ0 = thr_q.retile_D(rQ0);
    Tensor cQ1 = thr_q.retile_D(rQ1);
    Tensor cQ2 = thr_q.retile_D(rQ2);
    Tensor cQ3 = thr_q.retile_D(rQ3);
    Tensor cK0 = thr_k.retile_D(rK0);
    Tensor cK1 = thr_k.retile_D(rK1);
    Tensor cK2 = thr_k.retile_D(rK2);
    Tensor cK3 = thr_k.retile_D(rK3);
    auto q_slot = [&](auto slot) -> decltype(auto) {
        if constexpr (decltype(slot)::value == 0) return (cQ0);
        else if constexpr (decltype(slot)::value == 1) return (cQ1);
        else if constexpr (decltype(slot)::value == 2) return (cQ2);
        else return (cQ3);
    };
    auto k_slot = [&](auto slot) -> decltype(auto) {
        if constexpr (decltype(slot)::value == 0) return (cK0);
        else if constexpr (decltype(slot)::value == 1) return (cK1);
        else if constexpr (decltype(slot)::value == 2) return (cK2);
        else return (cK3);
    };
    auto load = [&](auto step) {
        constexpr int i = decltype(step)::value;
        auto &q = q_slot(Int<i % 4>{});
        auto &k = k_slot(Int<i % 4>{});
        if constexpr (i >= 4) {
            cute::copy(copy_q, q_source(_, _, Int<i % 4>{}, Int<i / 4>{}), q(_, _, _0{}));
        }
        cute::copy(copy_k, k_source(_, _, Int<i % 4>{}, Int<i / 4>{}), k(_, _, _0{}));
    };
    for_each(make_int_sequence<4>{}, [&](auto i) { load(i); });
    for_each(make_int_sequence<4>{}, [&](auto step) {
        constexpr int i = decltype(step)::value;
        auto &k = k_slot(Int<i>{});
        cute::gemm(mma, q_cached(_, _, Int<i>{}), k(_, _, _0{}), rP);
    });
    auto load_tile_slice = [&](int tile, auto slice) {
        constexpr int i = decltype(slice)::value;
        auto &q = q_slot(Int<i % 2>{});
        auto &k = k_slot(Int<i % 2>{});
        cute::copy(copy_q, q_source(_, _, slice, tile), q(_, _, _0{}));
        cute::copy(copy_k, k_source(_, _, slice, tile), k(_, _, _0{}));
    };
    for_each(make_int_sequence<2>{}, [&](auto slice) { load_tile_slice(1, slice); });
    // Carry two slices across each K64 backedge. Refill the consumed slot
    // first from the current tile, then from the next already-ready tile.
    #pragma unroll 1
    for (int tile = 1; tile < 3; ++tile) {
        for_each(make_int_sequence<4>{}, [&](auto slice) {
            constexpr int i = decltype(slice)::value;
            auto &q = q_slot(Int<i % 2>{});
            auto &k = k_slot(Int<i % 2>{});
            cute::gemm(mma, q(_, _, _0{}), k(_, _, _0{}), rP);
            if constexpr (i < 2) load_tile_slice(tile, Int<i + 2>{});
            else load_tile_slice(tile + 1, Int<i - 2>{});
        });
    }
    for_each(make_int_sequence<4>{}, [&](auto slice) {
        constexpr int i = decltype(slice)::value;
        auto &q = q_slot(Int<i % 2>{});
        auto &k = k_slot(Int<i % 2>{});
        cute::gemm(mma, q(_, _, _0{}), k(_, _, _0{}), rP);
        if constexpr (i < 2) load_tile_slice(3, Int<i + 2>{});
    });
}

// Pipelined TMA wait and Q K^T gemm. Q and K are split into (BLOCK_SIZE_M, 64)
// and (BLOCK_N, 64) tiles; each tile is waited on and multiplied in turn, so
// later tiles get more time to arrive and the copy overlaps the computation.
template <typename T>
using Hs64MbarPhase =
    std::conditional_t<T::kHasExtraKTile, bool, uint32_t>;

// K1-high is issued and consumed by WG1. Wait every thread's committed
// async group, then publish completion across all eight producer warps.
__forceinline__ __device__ void hs64_wait_local_k1_high() {
    __pipeline_wait_prior(0);
    cutlass::arch::NamedBarrier::sync(
        256, static_cast<cutlass::arch::ReservedNamedBarriers>(15));
}

template <
    typename T,    // Traits
    int PHASE_IDX, // See comments in the code
    bool LOCAL_HIGH_GROUP = false,
    bool LOCAL_LOW_GROUP = false,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename EngineQlow0, typename LayoutQlow0,
    typename BeforeQk>
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm(
    Tensor<Engine0, Layout0> &sQ,   // (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV0, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV1, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,   // ((2, 2, 8), 1, 1)
    Tensor<Engine3, Layout3> &rQ8,  // The 8-th tile of Q. We store it separately to leave some room for storing sP1
    Tensor<Engine4, Layout4> &rQ6,  // Penultimate Q tile for the active d512 path
    Tensor<Engine5, Layout5> &rQ4,  // First high-half Q tile for the active d512 path
    Tensor<EngineQlow0, LayoutQlow0> &rQlow0,
    __mbarrier_t *barriers,
    Hs64MbarPhase<T> &cur_phase,
    int idx_in_warpgroup,
    BeforeQk before_qk)
{
    typename T::TiledMma tiled_mma;
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    // Per-warp uniform index via read_firstlane (no register spill, unlike
    // __builtin_ppu_to_uniform_b32). Shared by the Q and K/V copy slices.
    using QkCopyIndex = std::conditional_t<T::kArch == 80, unsigned, int>;
    const QkCopyIndex warp_base = __ppu_read_firstlane(
        static_cast<QkCopyIndex>(idx_in_warpgroup) / QkCopyIndex(32));
    // Q is M-partitioned: fold the N-warp with % kAtomLayoutM.
    const QkCopyIndex cute_warp_Q =
        (warp_base % QkCopyIndex(T::kAtomLayoutM)) * QkCopyIndex(32);
    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);

    auto smem_tiled_copy_K = make_tiled_copy_B(
        std::conditional_t<T::kArch == 80, typename T::SmemCopyAtomK,
            Copy_Atom<HS64_TSM_K_UNIT16, typename T::InputT>>{}, tiled_mma);

    // K/V slice keeps the full warp index (warp_base*32). M128's cute_warp_Q
    // coincides with this (kAtomLayoutM==8), but M64 crosscut needs the unfolded
    // value to distinguish the two N-warps. read_firstlane avoids the spill that
    // __builtin_ppu_to_uniform_b32 triggers (verg 256 spill 132 -> verg 248 stack 0).
    const QkCopyIndex cute_warp_kv = warp_base * QkCopyIndex(32);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(cute_warp_kv);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(cute_warp_Q);

    Tensor sQ_tiled = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, _64>{})(_, _, _0{}, _); // (BLOCK_SIZE_M, 64, 9)
    Tensor sKV0_tiled = flat_divide(sKV0, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);  // (BLOCK_N, 64, 9)
    Tensor sKV1_tiled = flat_divide(sKV1, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);  // (BLOCK_N, 64, 9)
    Tensor thr_mma_sQ_tiled = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ_tiled));
    Tensor thr_mma_sKV0_tiled = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV0_tiled));
    Tensor thr_mma_sKV1_tiled = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV1_tiled));

    // PHASE-2 and PHASE-6 wait for the high K half before consuming Q tile 5.
    // Load that Q fragment just before the poll so its TSM latency overlaps the
    // K wait, without extending the fragment lifetime into other QK phases.
    constexpr bool kPrefetchQ5BeforeHighWait =
        T::kIsCrossCut && T::kUseQkWeave &&
        (PHASE_IDX == 2 || PHASE_IDX == 6);
    constexpr int kPrefetchedQTile = kPrefetchQ5BeforeHighWait ? 5 : 0;
    Tensor rQ5_prefetch = thr_mma.partition_fragment_A(
        sQ_tiled(_, _, Int<kPrefetchedQTile>{}));
    if constexpr (kPrefetchQ5BeforeHighWait) {
        const QkCopyIndex warp_idx_Q_prefetch = __builtin_ppu_to_uniform_b32(
            static_cast<QkCopyIndex>(idx_in_warpgroup) / QkCopyIndex(32));
        const QkCopyIndex cute_warp_Q_prefetch =
            (warp_idx_Q_prefetch % QkCopyIndex(T::kAtomLayoutM)) * QkCopyIndex(32);
        auto smem_thr_copy_Q_prefetch =
            smem_tiled_copy_Q.get_thread_slice(cute_warp_Q_prefetch);
        Tensor sQ5_prefetch_src = smem_thr_copy_Q_prefetch.partition_S(
            make_mix_tensor_like(sQ_tiled(_, _, Int<kPrefetchedQTile>{})));
        Tensor rQ5_prefetch_copy =
            smem_thr_copy_Q_prefetch.retile_D(rQ5_prefetch);
        CUTE_STATIC_ASSERT_V(
            size<1>(sQ5_prefetch_src) == size<1>(rQ5_prefetch_copy));
        cute::copy(
            smem_tiled_copy_Q, sQ5_prefetch_src, rQ5_prefetch_copy);
    }
    before_qk();

    #define QKT_GEMM_ONE_TILE(TILE_IDX) \
        if constexpr(kPrefetchQ5BeforeHighWait && TILE_IDX == kPrefetchedQTile) { \
            qkt_gemm_one_tile_rQ(tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, \
                    rQ5_prefetch, sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        } else if constexpr(T::kCacheLastQTile && TILE_IDX == T::kCachedQTile) { \
            qkt_gemm_one_tile_rQ(tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, \
                    rQ8, sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        } else if constexpr(T::kCachePrevQTile && TILE_IDX == T::kCachedPrevQTile) { \
            qkt_gemm_one_tile_rQ(tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, \
                    rQ6, sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        } else if constexpr(T::kCacheFirstHighQTile && TILE_IDX == T::kCachedFirstHighQTile) { \
            qkt_gemm_one_tile_rQ(tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, \
                    rQ4, sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        } else if constexpr(TILE_IDX < 4) { \
            qkt_gemm_one_tile_sQ<T>(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, \
                    smem_thr_copy_Q, smem_thr_copy_K, \
                    sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), \
                    sKV0_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV0_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        } else  { \
            qkt_gemm_one_tile_sQ<T>(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, \
                    smem_thr_copy_Q, smem_thr_copy_K, \
                    sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), \
                    sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, idx_in_warpgroup); \
        }

    auto qkt_gemm_low = [&]() {
        if constexpr (T::kArch == 80) {
            qkt_gemm_low_tiles_sm80<T>(tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K,
                sQ_tiled, sKV0_tiled, thr_mma_sQ_tiled, thr_mma_sKV0_tiled,
                rQlow0, rP, idx_in_warpgroup);
        } else {
            QKT_GEMM_ONE_TILE(0);
            QKT_GEMM_ONE_TILE(1);
            QKT_GEMM_ONE_TILE(2);
            QKT_GEMM_ONE_TILE(3);
        }
    };

    if constexpr (PHASE_IDX == 0) {
        // In PHASE-0, warpgroup 0 calculates Q K^T for the first 4 tiles
        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
            kernel_k_wait_sleep_ns<T>();
        };

        qkt_gemm_low();
    } else if constexpr (PHASE_IDX == 1 || PHASE_IDX == 3) {
        // PHASE-1 computes the full WG1 QK. PHASE-3 computes only its high
        // half so independent PV/copy work can be woven before the low half.
        if constexpr (T::kUseEvenHighBank && LOCAL_LOW_GROUP) {
            // The even prologue publishes both halves through local groups.
            __pipeline_wait_prior(0);
            cutlass::arch::NamedBarrier::sync(
                256, static_cast<cutlass::arch::ReservedNamedBarriers>(6));
        } else if constexpr (LOCAL_HIGH_GROUP) {
            hs64_wait_local_k1_high();
        } else {
            while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1)) {
                kernel_k_wait_sleep_ns<T>();
            }
        }

        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        // The K0 prologue consumes K8 from the low completion. Wait at the
        // same math boundary, before K8, preserving the original QK order.
        if constexpr (PHASE_IDX == 1 && LOCAL_LOW_GROUP && !T::kUseEvenHighBank) {
            __pipeline_wait_prior(0);
            cutlass::arch::NamedBarrier::sync(
                256, static_cast<cutlass::arch::ReservedNamedBarriers>(6));
        }
        if constexpr (T::kHasExtraKTile) { QKT_GEMM_ONE_TILE(8); }

        if constexpr (PHASE_IDX == 1) {
            if constexpr (!LOCAL_LOW_GROUP) {
                while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
                    kernel_k_wait_sleep_ns<T>();
                }
            }
            qkt_gemm_low();
            if constexpr (T::kHasExtraKTile) {
                cur_phase = (cur_phase + 1) & 1;
            } else {
                cur_phase ^= 1u;
            }
        }
    } else if constexpr (PHASE_IDX == 4) {
        // Local low epochs do not complete the prologue's shared mbarrier.
        // Preserve its phase across unsplit batches when polling is omitted.
        if constexpr (LOCAL_LOW_GROUP) {
            // The high-only wait left this low group in flight during QK4..8.
            // Publish completion to every WG1 reader before QK0..3.
            __pipeline_wait_prior(0);
            __ppu_barrier_sync(15, 256, 15u);
        } else {
            while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
                kernel_k_wait_sleep_ns<T>();
            }
        }
        qkt_gemm_low();
        if constexpr (!LOCAL_LOW_GROUP) {
            if constexpr (T::kHasExtraKTile) {
                cur_phase = (cur_phase + 1) & 1;
            } else {
                cur_phase ^= 1u;
            }
        }
    } else if constexpr (PHASE_IDX == 5) {
        // PHASE-5 consumes a K0-low epoch already polled by the caller.
        qkt_gemm_low();
    } else if constexpr (PHASE_IDX == 6) {
        // PHASE-6 consumes a K1-high epoch already polled by the caller.
        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        if constexpr (T::kHasExtraKTile) { QKT_GEMM_ONE_TILE(8); }
    } else {
        // In PHASE-2, warpgroup 0 calculates Q K^T for the last 5 tiles
        static_assert(PHASE_IDX == 2);

        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1)) {
            kernel_k_wait_sleep_ns<T>();
        };

        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        if constexpr (T::kHasExtraKTile) { QKT_GEMM_ONE_TILE(8); }
        if constexpr (T::kHasExtraKTile) {
            cur_phase = (cur_phase + 1) & 1;
        } else {
            cur_phase ^= 1u;
        }
    }
}

template <
    typename T,
    int PHASE_IDX,
    bool LOCAL_HIGH_GROUP = false,
    bool LOCAL_LOW_GROUP = false,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename EngineQlow0, typename LayoutQlow0>
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm(
    Tensor<Engine0, Layout0> &sQ,
    Tensor<Engine1, Layout1> &sKV0,
    Tensor<Engine1, Layout1> &sKV1,
    Tensor<Engine2, Layout2> &rP,
    Tensor<Engine3, Layout3> &rQ8,
    Tensor<Engine4, Layout4> &rQ6,
    Tensor<Engine5, Layout5> &rQ4,
    Tensor<EngineQlow0, LayoutQlow0> &rQlow0,
    __mbarrier_t *barriers,
    Hs64MbarPhase<T> &cur_phase,
    int idx_in_warpgroup)
{
    auto no_op = []() {};
    warpgroup_cooperative_qkt_gemm<T, PHASE_IDX, LOCAL_HIGH_GROUP, LOCAL_LOW_GROUP>(
        sQ, sKV0, sKV1, rP, rQ8, rQ6, rQ4, rQlow0,
        barriers, cur_phase, idx_in_warpgroup, no_op);
}

// D576 high-half QK over a selectable four-tile bank plus the fixed buf0/buf1
// tile8 tail. Q4..Q7 are persistent register fragments; Q8 remains in sQ.
// The caller owns the mbarrier wait/producer callback so WG1 can publish the
// following even block while computing the current odd block.
template <
    typename T,
    bool ADVANCE_PHASE,
    typename EngineQ, typename LayoutQ,
    typename EngineHigh, typename LayoutHigh,
    typename EngineTail, typename LayoutTail,
    typename EngineP, typename LayoutP,
    typename EngineQ4, typename LayoutQ4,
    typename EngineQ5, typename LayoutQ5,
    typename EngineQ6, typename LayoutQ6,
    typename EngineQ7, typename LayoutQ7,
    typename BeforeQk>
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm_high4_tail(
    Tensor<EngineQ, LayoutQ> &sQ,
    Tensor<EngineHigh, LayoutHigh> &sK_high4,
    Tensor<EngineTail, LayoutTail> &sK_tail_buf,
    Tensor<EngineP, LayoutP> &rP,
    Tensor<EngineQ4, LayoutQ4> &rQ4,
    Tensor<EngineQ5, LayoutQ5> &rQ5,
    Tensor<EngineQ6, LayoutQ6> &rQ6,
    Tensor<EngineQ7, LayoutQ7> &rQ7,
    Hs64MbarPhase<T> &cur_phase,
    int idx_in_warpgroup,
    BeforeQk before_qk)
{
    static_assert(T::kUseEvenHighBank);
    typename T::TiledMma tiled_mma;
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    using QkCopyIndex = std::conditional_t<T::kArch == 80, unsigned, int>;
    const QkCopyIndex warp_base = __ppu_read_firstlane(
        static_cast<QkCopyIndex>(idx_in_warpgroup) / QkCopyIndex(32));
    const QkCopyIndex cute_warp_Q =
        (warp_base % QkCopyIndex(T::kAtomLayoutM)) * QkCopyIndex(32);
    const QkCopyIndex cute_warp_kv = warp_base * QkCopyIndex(32);

    auto smem_tiled_copy_Q =
        make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q =
        smem_tiled_copy_Q.get_thread_slice(cute_warp_Q);
    auto smem_tiled_copy_K =
        make_tiled_copy_B(std::conditional_t<T::kArch == 80, typename T::SmemCopyAtomK,
            Copy_Atom<HS64_TSM_K_UNIT16, typename T::InputT>>{}, tiled_mma);
    auto smem_thr_copy_K =
        smem_tiled_copy_K.get_thread_slice(cute_warp_kv);

    Tensor sQ_tiled =
        flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, _64>{})(_, _, _0{}, _);
    Tensor sK_high_tiled =
        flat_divide(sK_high4, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);
    Tensor sK_tail_tiled =
        flat_divide(sK_tail_buf, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);
    Tensor thr_mma_sQ_tiled =
        smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ_tiled));
    Tensor thr_mma_sK_high_tiled =
        smem_thr_copy_K.partition_S(make_mix_tensor_like(sK_high_tiled));
    Tensor thr_mma_sK_tail_tiled =
        smem_thr_copy_K.partition_S(make_mix_tensor_like(sK_tail_tiled));

    // Keep D512 Q5 transient, as in its original PHASE-2/6 helper. Fetch
    // before the callback so WG1 still overlaps Q5 with the local K1 wait.
    auto thr_mma = tiled_mma.get_slice(cute_idx);
    Tensor rQ5_transient = thr_mma.partition_fragment_A(
        sQ_tiled(_, _, Int<5>{}));
    if constexpr (!T::kHasExtraKTile) {
        const QkCopyIndex warp_q5 = __builtin_ppu_to_uniform_b32(
            static_cast<QkCopyIndex>(idx_in_warpgroup) / QkCopyIndex(32));
        auto copy_q5 = smem_tiled_copy_Q.get_thread_slice(
            (warp_q5 % QkCopyIndex(T::kAtomLayoutM)) * QkCopyIndex(32));
        Tensor src_q5 = copy_q5.partition_S(
            make_mix_tensor_like(sQ_tiled(_, _, Int<5>{})));
        Tensor dst_q5 = copy_q5.retile_D(rQ5_transient);
        cute::copy(smem_tiled_copy_Q, src_q5, dst_q5);
    }

    before_qk();

    qkt_gemm_one_tile_rQ(
        tiled_mma, smem_tiled_copy_K, smem_thr_copy_K,
        rQ4, sK_high_tiled(_, _, _0{}),
        thr_mma_sK_high_tiled(_, _, _, _0{}), rP, idx_in_warpgroup);
    if constexpr (T::kHasExtraKTile) {
        qkt_gemm_one_tile_rQ(
            tiled_mma, smem_tiled_copy_K, smem_thr_copy_K,
            rQ5, sK_high_tiled(_, _, _1{}),
            thr_mma_sK_high_tiled(_, _, _, _1{}), rP, idx_in_warpgroup);
    } else {
        qkt_gemm_one_tile_rQ(
            tiled_mma, smem_tiled_copy_K, smem_thr_copy_K,
            rQ5_transient, sK_high_tiled(_, _, _1{}),
            thr_mma_sK_high_tiled(_, _, _, _1{}), rP, idx_in_warpgroup);
    }
    qkt_gemm_one_tile_rQ(
        tiled_mma, smem_tiled_copy_K, smem_thr_copy_K,
        rQ6, sK_high_tiled(_, _, _2{}),
        thr_mma_sK_high_tiled(_, _, _, _2{}), rP, idx_in_warpgroup);
    qkt_gemm_one_tile_rQ(
        tiled_mma, smem_tiled_copy_K, smem_thr_copy_K,
        rQ7, sK_high_tiled(_, _, _3{}),
        thr_mma_sK_high_tiled(_, _, _, _3{}), rP, idx_in_warpgroup);

    if constexpr (T::kHasExtraKTile) {
        qkt_gemm_one_tile_sQ<T>(
            tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K,
            sQ_tiled(_, _, Int<8>{}), thr_mma_sQ_tiled(_, _, _, Int<8>{}),
            sK_tail_tiled(_, _, Int<8>{}),
            thr_mma_sK_tail_tiled(_, _, _, Int<8>{}),
            rP, idx_in_warpgroup);
    }

    if constexpr (ADVANCE_PHASE) {
        if constexpr (T::kHasExtraKTile) {
            cur_phase = (cur_phase + 1) & 1;
        } else {
            cur_phase ^= 1u;
        }
    }
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
    // localP is only valid for non-cross-cut (BlockM=128, kAtomLayoutN==1).
    // Cross-cut (BlockM=64) routes P through SMEM via remoteP instead; early
    // return here prevents nvcc from eagerly instantiating the dead else-branch
    // whose K-tile count would mismatch.
    if constexpr (T::kIsCrossCut) { return; }
    typename T::TiledMma tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);

    // Vt is the B operand: keep the N-warp dimension in the slice so each
    // N-warp reads its own half of the kBlockN tokens under (4,2).
    // localP is non-cross-cut only (early return above), so it always uses the
    // validated warp-folded slice used by the warp-cooperative TSM load.
    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice((warp_idx % T::kAtomLayoutM) * 32);
    Tensor rVt = thr_mma.partition_fragment_B(sKV_half);
    Tensor rVt_copy_view = smem_thr_copy_Vt.retile_D(rVt);
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));

    CUTE_STATIC_ASSERT_V(size<1>(tSsVt) == size<1>(rVt_copy_view)); // M
    cute::copy(smem_tiled_copy_Vt, tSsVt, rVt_copy_view);

    gemm(rO, rP, rVt, tiled_mma);
}

struct Hs64NoPvPrimedWork {
    __forceinline__ __device__ void operator()() const {}
};

// Compute O += PV, where P resides in shared memory
template <
    typename T,
    bool PRIME_FIRST_P = false,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename AfterFirstP = Hs64NoPvPrimedWork>
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_remoteP(
    Tensor<Engine0, Layout0> &sP,
    Tensor<Engine1, Layout1> &sKV_half, // (HEAD_DIM_V/2, BLOCK_N)
    Tensor<Engine2, Layout2> &rO,       // ((2, 2, 32), 1, 1)
    int idx_in_warpgroup,
    int warp_idx,
    AfterFirstP after_first_p = {})
{
    typename T::TiledMmaPV tiled_mma;
    // P is the A operand (M x K only): fold the N-warp out with % kAtomLayoutM.
    // NOTE: P MUST use __builtin_ppu_to_uniform_b32 here; __ppu_read_firstlane
    // breaks M64 correctness for this A-operand slice (verified is_correct=False),
    // even though read_firstlane is correct+spill-free for the K/Vt B-operand slice.
    // (8,1) has a single N-warp, so it keeps the un-folded per-thread slice and
    // reads sP bare with Base's LDSM_N atom -- the pairing M128 was validated on.
    // All slice IDs are nonnegative. SM80 and D576 avoid CUTE's signed
    // correction chain; SM89 D512 preserves its existing code generation.
    using PvCopyIndex = std::conditional_t<T::kArch == 80 || T::kHasExtraKTile, unsigned, int>;
    const PvCopyIndex warp_idx_P = __builtin_ppu_to_uniform_b32(
        static_cast<PvCopyIndex>(idx_in_warpgroup) / PvCopyIndex(32));
    const PvCopyIndex cute_warp_P = T::kIsCrossCut
        ? (warp_idx_P % PvCopyIndex(T::kPvAtomLayoutM)) * PvCopyIndex(32)
        : static_cast<PvCopyIndex>(idx_in_warpgroup);
    auto smem_tiled_copy_P = make_tiled_copy_A(typename T::SmemCopyAtomP{}, tiled_mma);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(cute_warp_P);
    // Vt is the B operand: keep the full warp index (warp*32) so each N-warp
    // reads its own half of the kBlockN tokens under (4,2). M128's folded
    // (warp%8)*32 coincides with warp*32 (kAtomLayoutM==8). read_firstlane here
    // is correct AND spill-free (unlike the builtin, which the K slice showed spills).
    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
    const PvCopyIndex cute_warp_kv = __ppu_read_firstlane(
        static_cast<PvCopyIndex>(idx_in_warpgroup) / PvCopyIndex(32)) * PvCopyIndex(32);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(cute_warp_kv);

    // TSM_LD_SWZL needs a mix tensor source; LDSM_N needs the bare tensor.
    auto tSsP = [&]() {
        if constexpr (T::kArch == 80) {
            auto descriptor_view = make_tensor(sP.data(), typename T::SmemLayoutPTsm{});
            return smem_thr_copy_P.partition_S(make_mix_tensor_like(descriptor_view));
        } else if constexpr (T::kIsCrossCut) {
            return smem_thr_copy_P.partition_S(make_mix_tensor_like(sP));
        } else {
            return smem_thr_copy_P.partition_S(sP);
        }
    }();
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));

    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    if constexpr (T::kIsCrossCut && T::kUsePv2x4) {
        constexpr int kReductionSlices = T::kBlockN / 16;
        static_assert(kReductionSlices == 4);
        auto sP_seed = local_tile(
            sP, Shape<Int<T::BLOCK_SIZE_M>, _16>{}, Coord<_0, _0>{});
        auto sV_seed = local_tile(
            sKV_half, Shape<Int<T::kHeadDimV / 2>, _16>{}, Coord<_0, _0>{});
        Tensor rP0 = thr_mma.partition_fragment_A(sP_seed);
        Tensor rP1 = thr_mma.partition_fragment_A(sP_seed);
        Tensor rV0 = thr_mma.partition_fragment_B(sV_seed);
        Tensor rV1 = thr_mma.partition_fragment_B(sV_seed);
        Tensor cP0 = smem_thr_copy_P.retile_D(rP0);
        Tensor cP1 = smem_thr_copy_P.retile_D(rP1);
        Tensor cV0 = smem_thr_copy_Vt.retile_D(rV0);
        Tensor cV1 = smem_thr_copy_Vt.retile_D(rV1);
        CUTE_STATIC_ASSERT_V(size<0>(tSsP) == size<0>(cP0));
        CUTE_STATIC_ASSERT_V(size<1>(tSsP) == size<1>(cP0));
        CUTE_STATIC_ASSERT_V(size<2>(tSsP) == Int<kReductionSlices>{});
        CUTE_STATIC_ASSERT_V(size<2>(cP0) == _1{});
        CUTE_STATIC_ASSERT_V(size<0>(tSsVt) == size<0>(cV0));
        CUTE_STATIC_ASSERT_V(size<1>(tSsVt) == size<1>(cV0));
        CUTE_STATIC_ASSERT_V(size<2>(tSsVt) == Int<kReductionSlices>{});
        CUTE_STATIC_ASSERT_V(size<2>(cV0) == _1{});

        auto load0 = [&](auto k) {
            if constexpr (T::kArch == 80) {
                cute::copy(smem_tiled_copy_P, tSsP(_, _, k), cP0(_, _, _0{}));
            }
            cute::copy(smem_tiled_copy_Vt, tSsVt(_, _, k), cV0(_, _, _0{}));
            if constexpr (T::kArch != 80) {
                cute::copy(smem_tiled_copy_P, tSsP(_, _, k), cP0(_, _, _0{}));
            }
        };
        auto load1 = [&](auto k) {
            if constexpr (T::kArch == 80) {
                cute::copy(smem_tiled_copy_P, tSsP(_, _, k), cP1(_, _, _0{}));
            }
            cute::copy(smem_tiled_copy_Vt, tSsVt(_, _, k), cV1(_, _, _0{}));
            if constexpr (T::kArch != 80) {
                cute::copy(smem_tiled_copy_P, tSsP(_, _, k), cP1(_, _, _0{}));
            }
        };
        if constexpr (PRIME_FIRST_P) {
            static_assert(T::kHasExtraKTile);
            CUTE_STATIC_ASSERT_V(size<1>(cP0) == Int<2>{});
            CUTE_STATIC_ASSERT_V(size(cP0(_, _0{}, _0{})) == Int<8>{});
            // Publish only the first 8-BF16 P operand into its existing slot.
            cute::copy(smem_tiled_copy_P,
                tSsP(_, _0{}, _0{}), cP0(_, _0{}, _0{}));
            after_first_p();
            cute::copy(smem_tiled_copy_Vt, tSsVt(_, _, _0{}), cV0(_, _, _0{}));
            cute::copy(smem_tiled_copy_P,
                tSsP(_, _1{}, _0{}), cP0(_, _1{}, _0{}));
        } else {
            load0(_0{});
        }
        for_each(make_int_sequence<kReductionSlices>{}, [&](auto k) {
            constexpr int ki = decltype(k)::value;
            if constexpr (ki + 1 < kReductionSlices) {
                if constexpr (((ki + 1) & 1) == 0) {
                    load0(Int<ki + 1>{});
                } else {
                    load1(Int<ki + 1>{});
                }
            }
            if constexpr ((ki & 1) == 0) {
                cute::gemm(tiled_mma, rP0(_, _, _0{}), rV0(_, _, _0{}), rO);
            } else {
                cute::gemm(tiled_mma, rP1(_, _, _0{}), rV1(_, _, _0{}), rO);
            }
        });
        return;
    }
    Tensor thr_mma_sP = thr_mma.partition_fragment_A(sP);
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sKV_half); // (MMA, 1, 64/16=4)

    Tensor rP_copy_view = smem_thr_copy_P.retile_D(thr_mma_sP);
    Tensor rVt_copy_view = smem_thr_copy_Vt.retile_D(thr_mma_sKV_half);


    cute::copy(smem_tiled_copy_P, tSsP, rP_copy_view);
    cute::copy(smem_tiled_copy_Vt, tSsVt, rVt_copy_view);
    // gemm must consume the MMA fragments, not the copy views: retile_D reorders
    // for the copy and under (4,2) that layout is no longer isomorphic to the
    // MMA fragment layout (the N-atom adds a split that (8,1) does not have).
    gemm(rO, thr_mma_sP, thr_mma_sKV_half, tiled_mma);
}

template <
    typename T,
    bool DO_OOB_FILLING,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ Wg0SoftmaxSums wg0_bunch_0(
    Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,     // ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf,
    float* smem_cross_n_reduction)
{
    Wg0SoftmaxSums sums;
    // D576 accumulates rL independently in both N-warps. Read both old row
    // maxima before either N-warp can publish an update; the existing cross-N
    // max barrier below completes this read-before-write handoff.
    float old_max[2];
    if constexpr (!T::kUsePv2x4) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(
                local_row_idx, idx_in_warpgroup);
            old_max[local_row_idx] = sM(row_idx);
        }
    }

    // Preload smem_valid_indices into registers to hide SMEM latency
    // from the __shfl_xor_sync critical path.
    // Cross-cut: each N-warp only needs its own 16 cols -> 2 groups of 2 = 4 values.
    // Non-cross-cut: full 32 cols -> 4 groups of 2 = 8 values.
    int lane4 = idx_in_warpgroup % 4;
    int r_valid[T::kIsCrossCut ? (T::kBlockN/16*2) : 8];
    if constexpr (T::kIsCrossCut) {
        int warp_n_idx = (idx_in_warpgroup / 32) / T::kAtomLayoutM;
        constexpr int NCG = T::kBlockN / 16;   // column groups: 2 (N32) / 4 (N64)
        #pragma unroll
        for (int cg = 0; cg < NCG; cg++) {
            int base = (cg / 2) * 32 + warp_n_idx * 16 + (cg % 2) * 8 + lane4 * T::kScoreLaneStride;
            hs64_load_valid_pair<T>(smem_valid_indices, valid_indices_buf, base,
                                  r_valid[cg*2], r_valid[cg*2+1]);
        }
    } else {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int base = (k * 8 + lane4 * T::kScoreLaneStride) % T::kBlockN;
            hs64_load_valid_pair<T>(smem_valid_indices, valid_indices_buf, base,
                                  r_valid[k*2], r_valid[k*2+1]);
        }
    }
     // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        if constexpr (T::kIsCrossCut) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
                int g = i / 4;
                int rv_base = g * 2;
                hs64_acc<T>(rP0, i)   = r_valid[rv_base]     ? hs64_acc<T>(rP0, i)   : MAX_INIT_VAL;
                hs64_acc<T>(rP0, i+1) = r_valid[rv_base + 1] ? hs64_acc<T>(rP0, i+1) : MAX_INIT_VAL;
                cur_max = max(cur_max, max(hs64_acc<T>(rP0, i), hs64_acc<T>(rP0, i+1)));
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
                int k_base = ((i/4) % 4) * 2;
                hs64_acc<T>(rP0, i)   = r_valid[k_base]     ? hs64_acc<T>(rP0, i)   : MAX_INIT_VAL;
                hs64_acc<T>(rP0, i+1) = r_valid[k_base + 1] ? hs64_acc<T>(rP0, i+1) : MAX_INIT_VAL;
                cur_max = max(cur_max, max(hs64_acc<T>(rP0, i), hs64_acc<T>(rP0, i+1)));
            }
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        // Cross-N-warp max reduction (4,2): merge the two N-warps' partials.
        if constexpr (T::kIsCrossCut) {
            int warp_n_idx = (idx_in_warpgroup / 32) / T::kAtomLayoutM;
            if (idx_in_warpgroup % 4 == 0) {
                smem_cross_n_reduction[((int)(threadIdx.x >> 8) * 2 + warp_n_idx) * T::kBlockM + row_idx] = cur_max;
            }
            {
                int _wm = (idx_in_warpgroup / 32) % T::kAtomLayoutM;
                int _bar_id = 7 + (int)(threadIdx.x >> 8) * 4 + _wm;
                hs64_shared_exchange_sync<T>(_bar_id, 64);
            }
            if (idx_in_warpgroup % 4 == 0) {
                float partner_max = smem_cross_n_reduction[((int)(threadIdx.x >> 8) * 2 + (1 - warp_n_idx)) * T::kBlockM + row_idx];
                cur_max = max(cur_max, partner_max);
            }
            {
                int lane = threadIdx.x & 31;
                cur_max = __shfl_sync(0xffffffff, cur_max, (lane / 4) * 4);
            }
        }

        // Update sM and sL
        cur_max *= scale_softmax_log2;
        float new_max;
        float scale_for_old;
        if constexpr (T::kUsePv2x4) {
            new_max = max(sM(row_idx), cur_max);
            scale_for_old = exp2f(sM(row_idx) - new_max);
            __syncwarp();
        } else {
            new_max = max(old_max[local_row_idx], cur_max);
            scale_for_old = exp2f(old_max[local_row_idx] - new_max);
        }
        if (idx_in_warpgroup%4 == 0 && (idx_in_warpgroup / 32) < T::kAtomLayoutM) {
            sScale0(row_idx) = scale_for_old;
            sM(row_idx) = new_max;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            hs64_acc<T>(rP0, i) = exp2f(hs64_acc<T>(rP0, i)*scale_softmax_log2 - new_max);
            hs64_acc<T>(rP0, i+1) = exp2f(hs64_acc<T>(rP0, i+1)*scale_softmax_log2 - new_max);
            hs64_acc<T>(rPb, i) = (typename T::InputT)hs64_acc<T>(rP0, i);
            hs64_acc<T>(rPb, i+1) = (typename T::InputT)hs64_acc<T>(rP0, i+1);
            cur_sum += hs64_acc<T>(rP0, i) + hs64_acc<T>(rP0, i+1);
        }
        if constexpr (T::kUsePv2x4) {
            if (local_row_idx == 0) {
                sums.row0 = cur_sum;
            } else {
                sums.row1 = cur_sum;
            }
        } else {
            rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
        }
    }
    return sums;
}

template <
    typename T,
    bool DO_OOB_FILLING,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ Wg0SoftmaxSums wg0_bunch_0_uniform_geometry(
    Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,     // ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    int wg_idx,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf,
    float* smem_cross_n_reduction)
{
    Wg0SoftmaxSums sums;
    // WG0 already has a warp-uniform index. Keep warp geometry scalar instead
    // of rebuilding it from vector thread indices in each softmax iteration.
    const int softmax_warp = [&]() {
        // The uniform builtin does not carry threadIdx's range into this helper.
        // A warpgroup has eight warps; preserve that bound for address folding.
        if constexpr (T::kHasExtraKTile) return wg_idx & 7;
        else return idx_in_warpgroup / 32;
    }();
    const int softmax_wg = T::kHasExtraKTile ? 0 : int(threadIdx.x >> 8);
    auto row_of = [&](int local_row_idx) {
        if constexpr (T::kHasExtraKTile) {
            return (softmax_warp % T::kAtomLayoutM) * 16
                 + local_row_idx * 8 + ((idx_in_warpgroup & 31) >> 2);
        } else {
            return get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
        }
    };
    // D576 accumulates rL independently in both N-warps. Read both old row
    // maxima before either N-warp can publish an update; the existing cross-N
    // max barrier below completes this read-before-write handoff.
    float old_max[2];
    if constexpr (!T::kUsePv2x4) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = row_of(local_row_idx);
            old_max[local_row_idx] = sM(row_idx);
        }
    }

    // Preload smem_valid_indices into registers to hide SMEM latency
    // from the __shfl_xor_sync critical path.
    // Cross-cut: each N-warp only needs its own 16 cols -> 2 groups of 2 = 4 values.
    // Non-cross-cut: full 32 cols -> 4 groups of 2 = 8 values.
    int lane4 = idx_in_warpgroup % 4;
    int r_valid[T::kIsCrossCut ? (T::kBlockN/16*2) : 8];
    if constexpr (T::kIsCrossCut) {
        int warp_n_idx = softmax_warp / T::kAtomLayoutM;
        if constexpr (T::kHasExtraKTile) {
            hs64_load_valid_quad<T>(smem_valid_indices, valid_indices_buf,
                lane4, warp_n_idx, r_valid);
        } else {
            constexpr int NCG = T::kBlockN / 16;
            #pragma unroll
            for (int cg = 0; cg < NCG; cg++) {
                int base = (cg / 2) * 32 + warp_n_idx * 16 + (cg % 2) * 8 + lane4 * T::kScoreLaneStride;
                hs64_load_valid_pair<T>(smem_valid_indices, valid_indices_buf, base,
                                      r_valid[cg*2], r_valid[cg*2+1]);
            }
        }
    } else {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int base = (k * 8 + lane4 * T::kScoreLaneStride) % T::kBlockN;
            hs64_load_valid_pair<T>(smem_valid_indices, valid_indices_buf, base,
                                  r_valid[k*2], r_valid[k*2+1]);
        }
    }
     // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = row_of(local_row_idx);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        if constexpr (T::kIsCrossCut) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
                int g = i / 4;
                int rv_base = g * 2;
                hs64_acc<T>(rP0, i)   = r_valid[rv_base]     ? hs64_acc<T>(rP0, i)   : MAX_INIT_VAL;
                hs64_acc<T>(rP0, i+1) = r_valid[rv_base + 1] ? hs64_acc<T>(rP0, i+1) : MAX_INIT_VAL;
                cur_max = max(cur_max, max(hs64_acc<T>(rP0, i), hs64_acc<T>(rP0, i+1)));
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
                int k_base = ((i/4) % 4) * 2;
                hs64_acc<T>(rP0, i)   = r_valid[k_base]     ? hs64_acc<T>(rP0, i)   : MAX_INIT_VAL;
                hs64_acc<T>(rP0, i+1) = r_valid[k_base + 1] ? hs64_acc<T>(rP0, i+1) : MAX_INIT_VAL;
                cur_max = max(cur_max, max(hs64_acc<T>(rP0, i), hs64_acc<T>(rP0, i+1)));
            }
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        // Cross-N-warp max reduction (4,2): merge the two N-warps' partials.
        if constexpr (T::kIsCrossCut) {
            int warp_n_idx = softmax_warp / T::kAtomLayoutM;
            if (idx_in_warpgroup % 4 == 0) {
                smem_cross_n_reduction[(softmax_wg * 2 + warp_n_idx) * T::kBlockM + row_idx] = cur_max;
            }
            {
                int _wm = softmax_warp % T::kAtomLayoutM;
                int _bar_id = 7 + softmax_wg * 4 + _wm;
                hs64_shared_exchange_sync<T>(_bar_id, 64);
            }
            if (idx_in_warpgroup % 4 == 0) {
                float partner_max = smem_cross_n_reduction[(softmax_wg * 2 + (1 - warp_n_idx)) * T::kBlockM + row_idx];
                cur_max = max(cur_max, partner_max);
            }
            {
                int lane = threadIdx.x & 31;
                cur_max = __shfl_sync(0xffffffff, cur_max, (lane / 4) * 4);
            }
        }

        // Update sM and sL
        cur_max *= scale_softmax_log2;
        float new_max;
        float scale_for_old;
        if constexpr (T::kUsePv2x4) {
            new_max = max(sM(row_idx), cur_max);
            scale_for_old = exp2f(sM(row_idx) - new_max);
            __syncwarp();
        } else {
            new_max = max(old_max[local_row_idx], cur_max);
            scale_for_old = exp2f(old_max[local_row_idx] - new_max);
        }
        if (idx_in_warpgroup%4 == 0 && softmax_warp < T::kAtomLayoutM) {
            sScale0(row_idx) = scale_for_old;
            sM(row_idx) = new_max;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            hs64_acc<T>(rP0, i) = exp2f(hs64_acc<T>(rP0, i)*scale_softmax_log2 - new_max);
            hs64_acc<T>(rP0, i+1) = exp2f(hs64_acc<T>(rP0, i+1)*scale_softmax_log2 - new_max);
            hs64_acc<T>(rPb, i) = (typename T::InputT)hs64_acc<T>(rP0, i);
            hs64_acc<T>(rPb, i+1) = (typename T::InputT)hs64_acc<T>(rP0, i+1);
            cur_sum += hs64_acc<T>(rP0, i) + hs64_acc<T>(rP0, i+1);
        }
        if constexpr (T::kUsePv2x4) {
            if (local_row_idx == 0) {
                sums.row0 = cur_sum;
            } else {
                sums.row1 = cur_sum;
            }
        } else {
            rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
        }
    }
    return sums;
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
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ auto wg1_bunch_0(
    Tensor<Engine0, Layout0> &rP1b,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &sScale1, // (BLOCK_SIZE_M)
    Tensor<Engine2, Layout2> &rO1,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    Tensor<Engine4, Layout4> const &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine5, Layout5> &rP1,           // ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf,
    float* smem_cross_n_reduction,
    float r_cur_max_in[2] = nullptr)  // if non-null, skip mask+max (pre-computed)
{
    Wg1ScaleCache cache;
    const bool is_primary_n_warp =
        (idx_in_warpgroup / 32) < T::kAtomLayoutM;
    // For D576, bar1 makes WG0's sM update visible. Snapshot both shared rows,
    // then make the paired N-warps rendezvous before the primary overwrites sM.
    float old_max[2];
    if constexpr (!T::kUsePv2x4) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(
                local_row_idx, idx_in_warpgroup);
            old_max[local_row_idx] = sM(row_idx);
        }
        int warp_m_idx = (idx_in_warpgroup / 32) % T::kAtomLayoutM;
        int bar_id = 7 + (int)(threadIdx.x >> 8) * 4 + warp_m_idx;
        cutlass::arch::NamedBarrier::sync(
            64, static_cast<cutlass::arch::ReservedNamedBarriers>(bar_id));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);

        float cur_max;
        if (r_cur_max_in) {
            cur_max = r_cur_max_in[local_row_idx];
        } else {
            // Preload smem_valid_indices into registers.
            // Cross-cut: each N-warp only needs its own 16 cols -> 4 values.
            int lane4 = idx_in_warpgroup % 4;
            int r_valid[T::kIsCrossCut ? (T::kBlockN/16*2) : 8];
            if constexpr (T::kIsCrossCut) {
                int warp_n_idx = (idx_in_warpgroup / 32) / T::kAtomLayoutM;
                constexpr int NCG = T::kBlockN / 16;   // column groups: 2 (N32) / 4 (N64)
                #pragma unroll
                for (int cg = 0; cg < NCG; cg++) {
                    int base = (cg / 2) * 32 + warp_n_idx * 16 + (cg % 2) * 8 + lane4 * T::kScoreLaneStride;
                    hs64_load_valid_pair<T>(smem_valid_indices, valid_indices_buf, base,
                                          r_valid[cg*2], r_valid[cg*2+1]);
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int base = (k * 8 + lane4 * T::kScoreLaneStride) % T::kBlockN;
                    hs64_load_valid_pair<T>(smem_valid_indices, valid_indices_buf, base,
                                          r_valid[k*2], r_valid[k*2+1]);
                }
            }
            // Mask, and get row-wise max
            cur_max = MAX_INIT_VAL;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                if constexpr (IS_BLK0_LAST) {
                    hs64_acc<T>(rP1, i) = hs64_acc<T>(rP1, i+1) = MAX_INIT_VAL;
                } else if constexpr (T::kIsCrossCut) {
                    int g = i / 4;
                    int rv_base = g * 2;
                    hs64_acc<T>(rP1, i)   = r_valid[rv_base]     ? hs64_acc<T>(rP1, i)   : MAX_INIT_VAL;
                    hs64_acc<T>(rP1, i+1) = r_valid[rv_base + 1] ? hs64_acc<T>(rP1, i+1) : MAX_INIT_VAL;
                } else {
                    int k_base = ((i/4) % 4) * 2;
                    hs64_acc<T>(rP1, i)   = r_valid[k_base]     ? hs64_acc<T>(rP1, i)   : MAX_INIT_VAL;
                    hs64_acc<T>(rP1, i+1) = r_valid[k_base + 1] ? hs64_acc<T>(rP1, i+1) : MAX_INIT_VAL;
                }
                cur_max = max(cur_max, max(hs64_acc<T>(rP1, i), hs64_acc<T>(rP1, i+1)));
            }

            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

            // Cross-N-warp max reduction (4,2): merge the two N-warps' partials.
            if constexpr (T::kIsCrossCut) {
                int warp_n_idx = (idx_in_warpgroup / 32) / T::kAtomLayoutM;
                if (idx_in_warpgroup % 4 == 0) {
                    smem_cross_n_reduction[((int)(threadIdx.x >> 8) * 2 + warp_n_idx) * T::kBlockM + row_idx] = cur_max;
                }
                {
                    int _wm = (idx_in_warpgroup / 32) % T::kAtomLayoutM;
                    int _bar_id = 7 + (int)(threadIdx.x >> 8) * 4 + _wm;
                    cutlass::arch::NamedBarrier::sync(
                        64, static_cast<cutlass::arch::ReservedNamedBarriers>(_bar_id));
                }
                if (idx_in_warpgroup % 4 == 0) {
                    float partner_max = smem_cross_n_reduction[((int)(threadIdx.x >> 8) * 2 + (1 - warp_n_idx)) * T::kBlockM + row_idx];
                    cur_max = max(cur_max, partner_max);
                }
                {
                    int lane = threadIdx.x & 31;
                    cur_max = __shfl_sync(0xffffffff, cur_max, (lane / 4) * 4);
                }
            }
            cur_max *= scale_softmax_log2;
        }

        float new_max;
        float scale_for_old;
        if constexpr (T::kUsePv2x4) {
            float old_max_pv = sM(row_idx);
            new_max = max(old_max_pv, cur_max);
            scale_for_old = exp2f(old_max_pv - new_max);
            __syncwarp();
        } else {
            new_max = max(old_max[local_row_idx], cur_max);
            scale_for_old = exp2f(old_max[local_row_idx] - new_max);
        }
        // Only primary M-warps write sM/sScale (prevents aliased warp double-write).
        if (idx_in_warpgroup%4 == 0 && (idx_in_warpgroup / 32) < T::kAtomLayoutM) {
            sM(row_idx) = new_max;
            sScale1(row_idx) = scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        if constexpr (!IS_BLK0_LAST) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                if constexpr (T::kUsePv2x4) {
                    hs64_acc<T>(rP1, i) = exp2f(hs64_acc<T>(rP1, i) - new_max);
                    hs64_acc<T>(rP1, i + 1) = exp2f(hs64_acc<T>(rP1, i + 1) - new_max);
                } else {
                    hs64_acc<T>(rP1, i) = exp2f(hs64_acc<T>(rP1, i) * scale_softmax_log2 - new_max);
                    hs64_acc<T>(rP1, i + 1) = exp2f(hs64_acc<T>(rP1, i + 1) * scale_softmax_log2 - new_max);
                }
                hs64_acc<T>(rP1b, i) = (typename T::InputT)hs64_acc<T>(rP1, i);
                hs64_acc<T>(rP1b, i+1) = (typename T::InputT)hs64_acc<T>(rP1, i+1);
                cur_sum += hs64_acc<T>(rP1, i) + hs64_acc<T>(rP1, i+1);
            }
        }

        float scale0_for_o1 = sScale0(row_idx);
        float cur_scale_for_o1 = scale_for_old * scale0_for_o1;
        // Only the primary N-warp owns the pre-update sM value. Secondary
        // N-warps cache stable scale0 and combine it with sScale1 after bar2.
        if (local_row_idx == 0) {
            cache.row0 = is_primary_n_warp ? cur_scale_for_o1 : scale0_for_o1;
        } else {
            cache.row1 = is_primary_n_warp ? cur_scale_for_o1 : scale0_for_o1;
        }
        if constexpr (!T::kUsePv2x4) {
            // Reuse the factor already live for rL instead of reloading both
            // shared scales after bar2.
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rO1); i += 4) {
                hs64_acc<T>(rO1, i) = hs64_acc<T>(rO1, i)*cur_scale_for_o1;
                hs64_acc<T>(rO1, i+1) = hs64_acc<T>(rO1, i+1)*cur_scale_for_o1;
            }
        }

        if constexpr (T::kUsePv2x4) {
            if (local_row_idx == 0) {
                cache.sum0 = cur_sum;
            } else {
                cache.sum1 = cur_sum;
            }
        } else {
            rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
        }
    }
    return cache;
}

// wg1_bunch_0_pre: compute cur_max before barrier (doesn't depend on sM)
// Masks rP1 and computes per-row cur_max via warp shuffle reduce.
template<
    typename T,
    bool IS_BLK0_LAST,
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ void wg1_bunch_0_pre(
    float r_cur_max[2],                // output: per-row cur_max * scale
    Tensor<Engine5, Layout5> &rP1,           // ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf)
{
    int r_valid[8];
    int lane4 = idx_in_warpgroup % 4;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int base = (k * 8 + lane4 * T::kScoreLaneStride) % T::kBlockN;
        hs64_load_valid_pair<T>(smem_valid_indices, valid_indices_buf, base,
                              r_valid[k*2], r_valid[k*2+1]);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
            if constexpr (IS_BLK0_LAST) {
                hs64_acc<T>(rP1, i) = hs64_acc<T>(rP1, i+1) = MAX_INIT_VAL;
            } else {
                int k_base = ((i/4) % 4) * 2;
                hs64_acc<T>(rP1, i)   = r_valid[k_base]     ? hs64_acc<T>(rP1, i)   : MAX_INIT_VAL;
                hs64_acc<T>(rP1, i+1) = r_valid[k_base + 1] ? hs64_acc<T>(rP1, i+1) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(hs64_acc<T>(rP1, i), hs64_acc<T>(rP1, i+1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        r_cur_max[local_row_idx] = cur_max * scale_softmax_log2;
    }
}

// Cross-cut variant of wg1_bunch_0_pre. The mask/max reduction is independent
// of sM/sScale0, so it can fill WG1's CTA-bar1 wait window.
template<
    typename T,
    bool IS_BLK0_LAST,
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ void wg1_bunch_0_pre_crosscut(
    float r_cur_max[2],
    Tensor<Engine5, Layout5> &rP1,
    float scale_softmax_log2,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf,
    float* smem_cross_n_reduction)
{
    static_assert(T::kIsCrossCut);
    constexpr int NCG = T::kBlockN / 16;
    int r_valid[NCG * 2];
    int lane4 = idx_in_warpgroup % 4;
    int warp_n_idx = (idx_in_warpgroup / 32) / T::kAtomLayoutM;

    if constexpr (T::kHasExtraKTile) {
        hs64_load_valid_quad<T>(smem_valid_indices, valid_indices_buf,
            lane4, warp_n_idx, r_valid);
    } else {
        #pragma unroll
        for (int cg = 0; cg < NCG; ++cg) {
            int base = (cg / 2) * 32 + warp_n_idx * 16 + (cg % 2) * 8 + lane4 * T::kScoreLaneStride;
            hs64_load_valid_pair<T>(smem_valid_indices, valid_indices_buf, base,
                                  r_valid[cg * 2], r_valid[cg * 2 + 1]);
        }
    }

    // Publish both rows before the caller's existing CTA bar1.

    int row_idx0 = get_AorC_row_idx<T::kAtomLayoutM>(0, idx_in_warpgroup);
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = row_idx0 + local_row_idx * 8;
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
            if constexpr (IS_BLK0_LAST) {
                hs64_acc<T>(rP1, i) = hs64_acc<T>(rP1, i + 1) = MAX_INIT_VAL;
            } else {
                int rv_base = (i / 4) * 2;
                hs64_acc<T>(rP1, i) = r_valid[rv_base] ? hs64_acc<T>(rP1, i) : MAX_INIT_VAL;
                hs64_acc<T>(rP1, i + 1) = r_valid[rv_base + 1] ? hs64_acc<T>(rP1, i + 1) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(hs64_acc<T>(rP1, i), hs64_acc<T>(rP1, i + 1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        // D512: move the existing score-scale multiplies ahead of CTA bar1 so
        // independent register work overlaps WG0's scale publication latency.
        if constexpr (T::kUsePv2x4 && !IS_BLK0_LAST) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                hs64_acc<T>(rP1, i) *= scale_softmax_log2;
                hs64_acc<T>(rP1, i + 1) *= scale_softmax_log2;
            }
        }
        r_cur_max[local_row_idx] = cur_max;
        if (idx_in_warpgroup % 4 == 0) {
            smem_cross_n_reduction[((int)(threadIdx.x >> 8) * 2 + warp_n_idx) * T::kBlockM + row_idx] = cur_max;
        }
    }

    // The caller immediately enters CTA bar1. Let that existing barrier
    // publish both cross-N halves and finish the partner reduction after it.
}

template<typename T>
__forceinline__ __device__ void wg1_bunch_0_pre_crosscut_finish(
    float r_cur_max[2],
    float scale_softmax_log2,
    int idx_in_warpgroup,
    float* smem_cross_n_reduction)
{
    static_assert(T::kIsCrossCut);
    int warp_n_idx = (idx_in_warpgroup / 32) / T::kAtomLayoutM;
    int row_idx0 = get_AorC_row_idx<T::kAtomLayoutM>(0, idx_in_warpgroup);
    float partner_max[2];
    if (idx_in_warpgroup % 4 == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = row_idx0 + local_row_idx * 8;
            partner_max[local_row_idx] =
                smem_cross_n_reduction[((int)(threadIdx.x >> 8) * 2 + (1 - warp_n_idx)) * T::kBlockM + row_idx];
        }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        float cur_max = r_cur_max[local_row_idx];
        if (idx_in_warpgroup % 4 == 0) {
            cur_max = max(cur_max, partner_max[local_row_idx]);
        }
        int lane = threadIdx.x & 31;
        cur_max = __shfl_sync(0xffffffff, cur_max, (lane / 4) * 4);
        r_cur_max[local_row_idx] = cur_max * scale_softmax_log2;
    }
}

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
    if constexpr (T::kArch == 80) {
        auto coords = typename T::TiledMma{}.get_slice(idx_in_warpgroup).partition_C(
            make_identity_tensor(Shape<Int<T::kBlockM>, Int<T::kBlockN>>{}));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(rPb); ++i) {
            int row = get<0>(coords(i));
            int col = get<1>(coords(i));
            sP(hs64_sm80_store_row(row, col), col) = rPb(i);
        }
        return;
    }
    typename T::TiledMma tiled_mma;
    // C copy, same rationale as save_rP0_to_sP.
    auto r2s_copy = make_tiled_copy_C(typename T::SmemCopyAtomS{}, tiled_mma);
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);

    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}

template <typename T, typename Engine, typename Layout>
__forceinline__ __device__ auto hs64_prepare_p_store(
    Tensor<Engine, Layout> &sP, int idx_in_warpgroup)
{
    if constexpr (T::kArch == 80) {
        auto coords = typename T::TiledMma{}.get_slice(idx_in_warpgroup).partition_C(
            make_identity_tensor(Shape<Int<T::kBlockM>, Int<T::kBlockN>>{}));
        cute::array<uint32_t, decltype(size(coords))::value> addresses;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(coords); ++i) {
            int row = get<0>(coords(i));
            int col = get<1>(coords(i));
            addresses[i] = cast_smem_ptr_to_uint(&sP(hs64_sm80_store_row(row, col), col));
        }
        return addresses;
    } else {
    auto r2s_copy = make_tiled_copy_C(typename T::SmemCopyAtomS{}, typename T::TiledMma{});
    auto thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    auto dst = recast<typename T::PStoreWord>(thr_copy.partition_D(sP));
    cute::array<uint32_t, decltype(size(dst))::value> addresses;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(dst); ++i) {
        addresses[i] = cast_smem_ptr_to_uint(&dst(i));
        // Keep packed P destinations live across the source-level K-copy burst.
        asm volatile("" : "+r"(addresses[i]));
    }
    return addresses;
    }
}

template <typename T, typename Engine, typename Layout, size_t N>
__forceinline__ __device__ void hs64_save_prepared_p(
    Tensor<Engine, Layout> &rPb, cute::array<uint32_t, N> const &addresses,
    int idx_in_warpgroup)
{
    if constexpr (T::kArch == 80) {
        auto src = recast<uint16_t>(rPb);
        static_assert(decltype(size(src))::value == N);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            cutlass::arch::shared_store<2>(addresses[i], &src(i));
        }
        return;
    }
    auto r2s_copy = make_tiled_copy_C(typename T::SmemCopyAtomS{}, typename T::TiledMma{});
    auto thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    auto src = recast<typename T::PStoreWord>(thr_copy.retile_S(rPb));
    static_assert(decltype(size(src))::value == N);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        cutlass::arch::shared_store<sizeof(typename T::PStoreWord)>(addresses[i], &src(i));
    }
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
    if constexpr (T::kArch == 80) {
        save_rP1_to_sP<T>(rPb, sP, idx_in_warpgroup);
        return;
    }
    typename T::TiledMma tiled_mma;
    // rPb is the QK GEMM's C fragment, so the write side is a C copy (matches
    // the non-WI kernel and FlashMLA HS64). rPb MUST be shaped from
    // partition_fragment_C -- a hard-coded (8,1) 16-element rPb makes retile_S
    // mis-map onto the 8-element (4,2) C slice (probe: sP read back all zeros).
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
    const int cute_warp = (warp_idx % T::kAtomLayoutM) * 32;

    auto thr_mma = tiled_mma.get_thread_slice(idx_in_warpgroup);
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(cute_warp);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sP));
    // retile the destination fragment into the copy TV-layout: under (4,2) the
    // A-fragment is not isomorphic to the copy view, so a direct copy into rPb
    // mis-orders the elements.
    Tensor rPb_copy_view = smem_thr_copy_Q.retile_D(rPb);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(rPb_copy_view));
    cute::copy(smem_tiled_copy_Q, tSsQ, rPb_copy_view);
}


struct Wg0ScaleFactors {
    float row0;
    float row1;
};

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void scale_rO_pv_one(
    Tensor<Engine0, Layout0> &rO,
    Tensor<Engine1, Layout1> const &sScale,
    int idx_in_warpgroup,
    Wg0ScaleFactors *qk_factors = nullptr
) {
    Tensor rO_rowcol = make_tensor(
        rO.data(), hs64_convert_layout_acc_rowcol<T>(rO.layout()));
    static_assert(decltype(size<0>(rO_rowcol))::value == 4);
    static_assert(decltype(size<1>(rO_rowcol))::value == 16);
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < 4; ++mi) {
        float scale = sScale(get_PV_row_idx<T>(mi, idx_in_warpgroup));
        if (qk_factors != nullptr) {
            const bool owns_low_repeat =
                ((idx_in_warpgroup / 32) % T::kAtomLayoutM) < T::kPvAtomLayoutM;
            if (mi == (owns_low_repeat ? 0 : 2)) {
                qk_factors->row0 = scale;
            } else if (mi == (owns_low_repeat ? 1 : 3)) {
                qk_factors->row1 = scale;
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < 16; ++ni) {
            rO_rowcol(mi, ni) *= scale;
        }
    }
}

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void scale_rO_pv_product(
    Tensor<Engine0, Layout0> &rO,
    Tensor<Engine1, Layout1> const &sScale0,
    Tensor<Engine2, Layout2> const &sScale1,
    int idx_in_warpgroup
) {
    Tensor rO_rowcol = make_tensor(
        rO.data(), hs64_convert_layout_acc_rowcol<T>(rO.layout()));
    static_assert(decltype(size<0>(rO_rowcol))::value == 4);
    static_assert(decltype(size<1>(rO_rowcol))::value == 16);
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < 4; ++mi) {
        int row = get_PV_row_idx<T>(mi, idx_in_warpgroup);
        float scale = sScale0(row) * sScale1(row);
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < 16; ++ni) {
            rO_rowcol(mi, ni) *= scale;
        }
    }
}

template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void scale_rO_pv_product_with_cache(
    Tensor<Engine0, Layout0> &rO,
    Tensor<Engine1, Layout1> const &sScale0,
    Tensor<Engine2, Layout2> const &sScale1,
    Wg1ScaleCache const &cache,
    float rL[2],
    int idx_in_warpgroup
) {
    Tensor rO_rowcol = make_tensor(
        rO.data(), hs64_convert_layout_acc_rowcol<T>(rO.layout()));
    static_assert(decltype(size<0>(rO_rowcol))::value == 4);
    static_assert(decltype(size<1>(rO_rowcol))::value == 16);
    static_assert(T::kAtomLayoutM == 4 && T::kPvAtomLayoutM == 2);
    const int qk_m = (idx_in_warpgroup / 32) % 4;
    const bool is_primary_n_warp =
        (idx_in_warpgroup / 32) < T::kAtomLayoutM;
    const bool owns_low_repeat = qk_m < 2;
    const int qk_row0 = get_AorC_row_idx<4>(0, idx_in_warpgroup);
    float own0 = cache.row0;
    float own1 = cache.row1;
    if (!is_primary_n_warp) {
        const float scale1_row0 = sScale1(qk_row0);
        const float scale1_row1 = sScale1(qk_row0 + 8);
        own0 *= scale1_row0;
        own1 *= scale1_row1;
    }
    rL[0] = rL[0] * own0 + cache.sum0;
    rL[1] = rL[1] * own1 + cache.sum1;

    const int peer_row0 = qk_row0 + (owns_low_repeat ? 32 : -32);
    const float peer0 = sScale0(peer_row0) * sScale1(peer_row0);
    const float scale0 = owns_low_repeat ? own0 : peer0;
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < 16; ++ni) {
        rO_rowcol(0, ni) *= scale0;
    }
    const float scale2 = owns_low_repeat ? peer0 : own0;
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < 16; ++ni) {
        rO_rowcol(2, ni) *= scale2;
    }

    const int peer_row1 = peer_row0 + 8;
    const float peer1 = sScale0(peer_row1) * sScale1(peer_row1);
    const float scale1 = owns_low_repeat ? own1 : peer1;
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < 16; ++ni) {
        rO_rowcol(1, ni) *= scale1;
    }
    const float scale3 = owns_low_repeat ? peer1 : own1;
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < 16; ++ni) {
        rO_rowcol(3, ni) *= scale3;
    }
}

// Rescale rP0 and save the result to rPb
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ Wg0ScaleFactors wg0_scale_rP0(
    Tensor<Engine0, Layout0> const &sScale1,	// (BLOCK_M)
    Tensor<Engine1, Layout1> const &rP0,		// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rPb,		// ((2, 2, 8), 1, 1)
    int idx_in_warpgroup,
    Wg0ScaleFactors const *cached_factors = nullptr
) {
    Wg0ScaleFactors factors;
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
        float scale_factor;
        if constexpr (T::kUsePv2x4) {
            scale_factor = cached_factors != nullptr
                ? (local_row_idx == 0 ? cached_factors->row0 : cached_factors->row1)
                : sScale1(row_idx);
        } else {
            scale_factor = sScale1(row_idx);
        }
        if (local_row_idx == 0) {
            factors.row0 = scale_factor;
        } else {
            factors.row1 = scale_factor;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            hs64_acc<T>(rPb, i) = (typename T::InputT)(hs64_acc<T>(rP0, i)*scale_factor);
            hs64_acc<T>(rPb, i+1) = (typename T::InputT)(hs64_acc<T>(rP0, i+1)*scale_factor);
        }
    }
    return factors;
}


// Rescale rO0 according to sScale1
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
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            hs64_acc<T>(rO0, i) = hs64_acc<T>(rO0, i)*scale_factor;
            hs64_acc<T>(rO0, i+1) = hs64_acc<T>(rO0, i+1)*scale_factor;
        }
        rL[local_row_idx] *= scale_factor;
    }

}

template<
    typename T,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void wg0_rescale_rO0_with_factors(
    Tensor<Engine0, Layout0> &rO0,
    Wg0ScaleFactors const &factors,
    float rL[2]
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        float scale_factor = local_row_idx == 0 ? factors.row0 : factors.row1;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            hs64_acc<T>(rO0, i) = hs64_acc<T>(rO0, i)*scale_factor;
            hs64_acc<T>(rO0, i+1) = hs64_acc<T>(rO0, i+1)*scale_factor;
        }
        rL[local_row_idx] *= scale_factor;
    }
}

// Rescale rO1 according to sScale0
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
    if constexpr (T::kUsePv2x4) {
        scale_rO_pv_product<T>(rO1, sScale0, sScale1, idx_in_warpgroup);
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
            float scale_factor = sScale0(row_idx) * sScale1(row_idx);
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rO1); i += 4) {
                hs64_acc<T>(rO1, i) = (hs64_acc<T>(rO1, i)*scale_factor);
                hs64_acc<T>(rO1, i+1) = (hs64_acc<T>(rO1, i+1)*scale_factor);
            }
        }
    }
}

// Rescale rO0 according to local scale
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void wg0_scale0_rO0(
    Tensor<Engine0, Layout0> &rO0,
    Tensor<Engine1, Layout1> &sScale0,
    int idx_in_warpgroup,
    Wg0SoftmaxSums const *sums = nullptr,
    float rL[2] = nullptr
) {
    if constexpr (T::kUsePv2x4) {
        Wg0ScaleFactors factors;
        scale_rO_pv_one<T>(rO0, sScale0, idx_in_warpgroup, &factors);
        rL[0] = rL[0] * factors.row0 + sums->row0;
        rL[1] = rL[1] * factors.row1 + sums->row1;
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
            float scale_factor = sScale0[row_idx];
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
                hs64_acc<T>(rO0, i) *= scale_factor;
                hs64_acc<T>(rO0, i+1) *= scale_factor;
            }
        }
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
    float const *rNorm,
    char* sO_addr,
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

    // (SMEM_M,SMEM_N) // Sw<3,3,3> o _0 o (_32,(_64,_8)):(_64,(_1,_2048))
    Tensor rOb = make_tensor_like<ElementO>(rO);

    if constexpr (T::kUsePv2x4) {
        Tensor rO_rowcol = make_tensor(
            rO.data(), hs64_convert_layout_acc_rowcol<T>(rO.layout()));
        Tensor rOb_rowcol = make_tensor(
            rOb.data(), hs64_convert_layout_acc_rowcol<T>(rOb.layout()));
        static_assert(decltype(size<0>(rO_rowcol))::value == 4);
        static_assert(decltype(size<1>(rO_rowcol))::value == 16);
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < 4; ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < 16; ++ni) {
                rOb_rowcol(mi, ni) = (ElementO)(rO_rowcol(mi, ni) / rNorm[mi]);
            }
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < size(rO); ++idx) {
            rOb(idx) = (ElementO)(rO(idx) / rNorm[idx%4 >= 2]);
        }
    }

    if constexpr (!IS_NO_SPLIT && T::kBlockM >= 128) {
        // Two-pass float32 output for BlockM>=128 to fit PPU M890P SMEM limit.
        using SmemLayoutO_Half = typename T::SmemLayoutO_Half;

        Tensor sHalfBuf = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(sO_addr)),
            SmemLayoutO_Half{});  // (kBlockM, kHeadDimV/2) = 128x256 floats

        typename T::TiledMmaPV tiled_mma;
        auto r2s_tiled_copy = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
        ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);
        Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
        Tensor r2s_thr_copy_sHalfBuf = r2s_thr_copy.partition_D(sHalfBuf);

        // GMEM copy infrastructure (partitioned over 128x256)
        using GmemTiledCopyOHalf = typename T::GmemTiledCopyOaccum;
        GmemTiledCopyOHalf gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sHalfBuf);

        // Identity tensor for M-boundary check (reused across passes)
        Tensor cO = make_identity_tensor(make_shape(size<0>(sHalfBuf), size<1>(sHalfBuf)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

        CUTLASS_PRAGMA_UNROLL
        for (int pass = 0; pass < 2; ++pass) {
            // r2s: only the WG whose data matches this pass writes
            if (warpgroup_idx == pass) {
                cute::copy(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sHalfBuf);
            }
            __syncthreads();

            // s2g: all threads copy 128x256 half to correct GMEM columns
            // gOorAccum has stride (kHeadDimV, 1); offset by pass*(kHeadDimV/2)
            Tensor gO_half = make_tensor(
                make_gmem_ptr(reinterpret_cast<ElementO *>(gOorAccum.data().get()) + pass * (T::kHeadDimV / 2)),
                Layout<Shape<Int<T::kBlockM>, Int<T::kHeadDimV / 2>>,
                       Stride<Int<T::kHeadDimV>, _1>>{});
            Tensor tOgO = gmem_thr_copy_O.partition_D(gO_half);
            Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
            Tensor tOrO_local = make_tensor<ElementO>(shape(tOgO));

            // smem -> reg
            cute::copy(gmem_tiled_copy_O, tOsO, tOrO_local);
            // reg -> gmem (with M-boundary mask)
            copy<false, true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO_local, tOgO, tOcO, tOpO, num_valid_seq_q);

            // Sync before next pass's r2s overwrites the buffer
            if (pass == 0) __syncthreads();
        }
    } else {
        // ==================================================================
        // Original single-pass path (BF16 output or BlockM<128)
        // ==================================================================
        Tensor sOutputBuf = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(sO_addr)),
            typename T::SmemLayoutO{});

        Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<Int<T::kBlockM>, _256>{}, make_coord(_0{}, warpgroup_idx));

        typename T::TiledMmaPV tiled_mma;
        auto r2s_tiled_copy = make_tiled_copy_C(
            SmemTiledCopyO{}, tiled_mma);

        ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);
        Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
        Tensor r2s_thr_copy_sMyOutputBuf = r2s_thr_copy.partition_D(sMyOutputBuf);
        cute::copy(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sMyOutputBuf);

        __syncthreads();

        const int64_t row_offset_o = batch_idx * params.o_batch_stride
            + m_block_idx * T::kBlockM * params.o_row_stride
            + k_head_idx * params.o_head_stride;

        using GmemTiledCopyO = std::conditional_t<
            IS_NO_SPLIT,
            typename T::GmemTiledCopyO,
            typename T::GmemTiledCopyOaccum>;
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
}


// Phase 1: Compute the per-copy-wave token_ptr set and write
// smem_valid_indices. Does NOT issue cp.async, does NOT write to sK buffer.
// Can be placed anywhere (fills SIMT gaps during TC execution).
//
// M64 issues two physical copy waves of 32 tokens. XOR4 computes both waves'
// pointers from one metadata pass.
// Compile-time version: for mainloop (USE_EXTRA as template param)
template<int S, bool USE_EXTRA, typename T, bool PREFETCH_K = false, typename TensorVI>
__forceinline__ __device__ void compute_K_addr_bf16(
    const Flash_fwd_mla_params &params,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    int tidx,
    int ori_block_max,
    TensorVI &smem_valid_indices,
    int vi_buf,
    int *pre_token_idx,
    int extra_seqlen_k,
    typename T::InputT* (&token_ptr)[T::kGmemPasses],
    bool (&is_valid)[T::kGmemPasses]
) {
    using InputT = typename T::InputT;
    constexpr int kBlockN        = T::kBlockN;
    constexpr int kBytesPerToken = T::kBytesPerToken;

    // Use if constexpr to force compile-time branch elimination (PPU compiler doesn't optimize ternary with template params)
    int effective_block, page_block_size, valid_len;
    InputT* k_base_ptr;
    size_t k_stride_elems;
    if constexpr (USE_EXTRA) {
        effective_block = block_idx_kv - ori_block_max;
        page_block_size = params.extra_page_block_size;
        valid_len = extra_seqlen_k;
        k_base_ptr = reinterpret_cast<InputT*>(params.extra_k_ptr);
        k_stride_elems = static_cast<size_t>(params.extra_k_batch_stride);
    } else {
        effective_block = block_idx_kv;
        page_block_size = params.page_block_size;
        valid_len = seqlen_k;
        k_base_ptr = reinterpret_cast<InputT*>(params.k_ptr);
        k_stride_elems = static_cast<size_t>(params.k_batch_stride);
    }
    int tile_valid = valid_len - effective_block * kBlockN;

    // Interleave the two metadata halves inside each 8-lane copy group. Lanes
    // 0..3 own pass 0 and lanes 4..7 own pass 1; one 32-bit xor4 shuffle shares
    // the other pass's 16-byte-atom offset. The invalid sentinel carries valid.
    // The downstream 8 lanes x 16 B contiguous 128 B copy remains unchanged.
    static_assert(kBlockN == 64 && T::kGmemPasses == 2 && T::kGmemThrPerTok == 8,
                  "the one-pass offset mapping requires the M64 copy layout");
    static_assert(T::kGmemElemsPerLoad == 8 && sizeof(InputT) == 2,
                  "one-pass copy offsets are encoded in 16-byte atoms");
    constexpr unsigned int kInvalidAtom = ~0u;
    const int lane = tidx & 31;
    const int warp = tidx >> 5;
    const int pair = lane >> 3;
    const bool owns_pass1 = (lane & 4) != 0;
    const int token_id = warp * 4 + pair + (owns_pass1 ? 32 : 0);
    const int t_idx = (token_id < tile_valid) ? pre_token_idx[0] : -1;
    // Prefill indices are token IDs and may include positive out-of-range
    // sentinels. Each KV token is mapped to a one-token page by the host.
    const bool meta_valid = (t_idx >= 0) &&
        (!T::kIsPrefill || t_idx < params.num_blocks);
    if constexpr (S == 0) {
        // Each physical token is represented by four lanes. Interleaving makes
        // pass0's consecutive tokens land at ballot bits 0/8/16/24 and pass1's
        // at 4/12/20/28; write the accepted packed-valid word layout directly.
        unsigned int valid_votes = __ballot_sync(0xffffffff, meta_valid);
        if (lane == 0) {
            if constexpr (T::kHasExtraKTile) {
                smem_valid_indices(vi_buf, warp) = valid_votes;
            } else {
                unsigned int packed0 =
                    ((valid_votes >> 0)  & 1u) |
                    (((valid_votes >> 8)  & 1u) << 1) |
                    (((valid_votes >> 16) & 1u) << 2) |
                    (((valid_votes >> 24) & 1u) << 3);
                unsigned int packed1 =
                    ((valid_votes >> 4)  & 1u) |
                    (((valid_votes >> 12) & 1u) << 1) |
                    (((valid_votes >> 20) & 1u) << 2) |
                    (((valid_votes >> 28) & 1u) << 3);
                smem_valid_indices(vi_buf, warp) = packed0;
                smem_valid_indices(vi_buf, 8 + warp) = packed1;
            }
        }
    }

    unsigned int meta_atom = kInvalidAtom;
    if (meta_valid) {
        if constexpr (T::kIsPrefill) {
            // One-token pages: preserve the real padded row stride without
            // the decode page split or its packed-page runtime branch.
            meta_atom = static_cast<unsigned int>(
                static_cast<size_t>(t_idx) * k_stride_elems / T::kGmemElemsPerLoad);
        } else {
            const bool linear_pages = (page_block_size > 0)
                & ((static_cast<unsigned>(page_block_size)
                    & (static_cast<unsigned>(page_block_size) - 1u)) == 0)
                & (k_stride_elems == static_cast<size_t>(page_block_size)
                    * (kBytesPerToken / sizeof(InputT)));
            if (linear_pages) {
                // Preserve the original uint32 atom narrowing, with no page split.
                meta_atom = static_cast<unsigned int>(t_idx)
                    * (kBytesPerToken / 16);
            } else {
                int page_idx    = t_idx >> __builtin_ctz(page_block_size);
                int off_in_page = t_idx & ((page_block_size & (-page_block_size)) - 1);
                size_t row_elem_offset =
                    static_cast<size_t>(page_idx) * k_stride_elems
                    + static_cast<size_t>(off_in_page) * (kBytesPerToken / sizeof(InputT));
                meta_atom = static_cast<unsigned int>(row_elem_offset / T::kGmemElemsPerLoad);
            }
        }
    }

    const unsigned int peer_atom =
        __shfl_xor_sync(0xffffffff, meta_atom, 4);
    const unsigned int atom0 = owns_pass1 ? peer_atom : meta_atom;
    const unsigned int atom1 = owns_pass1 ? meta_atom : peer_atom;
    is_valid[0] = (atom0 != kInvalidAtom);
    is_valid[1] = (atom1 != kInvalidAtom);
    const size_t lane_atom = static_cast<size_t>(lane & 7);
    token_ptr[0] = k_base_ptr
        + (is_valid[0] ? (static_cast<size_t>(atom0) + lane_atom)
                       : lane_atom) * T::kGmemElemsPerLoad;
    token_ptr[1] = k_base_ptr
        + (is_valid[1] ? (static_cast<size_t>(atom1) + lane_atom)
                       : lane_atom) * T::kGmemElemsPerLoad;

    if constexpr (PREFETCH_K && !T::kIsPrefill) {
        // Keep the uniform 64-bit base plus vector 32-bit offset used by
        // the address pipeline. Non-bulk L2 hints preserve lane addresses
        // without changing the real async-copy request geometry.
        // The range check only skips the hint; actual copies stay full-width.
        constexpr unsigned kAtomLimit = 0x0fffffffu
            - T::kNumKTiles * T::kGmemThrPerTok;
        CUTLASS_PRAGMA_UNROLL
        for (int pass = 0; pass < T::kGmemPasses; ++pass) {
            const unsigned atom = pass == 0 ? atom0 : atom1;
            if (atom <= kAtomLimit) {
                CUTLASS_PRAGMA_UNROLL
                for (int tile = 0; tile < T::kNumKTiles; ++tile) {
                    const unsigned byte_offset =
                        (atom + static_cast<unsigned>(lane_atom)
                         + tile * T::kGmemThrPerTok) * 16u;
                    __ppu_prefetch_nonebulk_L2(
                        reinterpret_cast<char*>(k_base_ptr)
                        + static_cast<size_t>(byte_offset));
                }
            }
        }
    }
}

// Runtime version: for prolog (use_extra as runtime param)
template<int S, typename T, typename TensorVI>
__forceinline__ __device__ void compute_K_addr_bf16_dynamic(
    const Flash_fwd_mla_params &params,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    int tidx,
    int ori_block_max,
    TensorVI &smem_valid_indices,
    int vi_buf,
    int *pre_token_idx,
    int extra_seqlen_k,
    typename T::InputT* (&token_ptr)[T::kGmemPasses],
    bool (&is_valid)[T::kGmemPasses],
    bool use_extra
) {
    if (use_extra) {
        compute_K_addr_bf16<S, true, T>(params, batch_idx, block_idx_kv, seqlen_k, tidx, ori_block_max,
            smem_valid_indices, vi_buf, pre_token_idx, extra_seqlen_k, token_ptr, is_valid);
    } else {
        compute_K_addr_bf16<S, false, T>(params, batch_idx, block_idx_kv, seqlen_k, tidx, ori_block_max,
            smem_valid_indices, vi_buf, pre_token_idx, extra_seqlen_k, token_ptr, is_valid);
    }
}

template <int START, int END, int P,
          typename TiledCopy,
          typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__forceinline__ __device__ void hs64_copy_k_tiles_pass_lookahead_impl(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV,
    Tensor<Engine1, Layout1> &sKV,
    decltype(gKV(_, _0{}, Int<START>{})) g_cur)
{
    if constexpr (START + 1 < END) {
        auto g_next = gKV(_, _0{}, Int<START + 1>{});
        auto g_next_ptr = cute::raw_pointer_cast(g_next.data());
        unsigned long long g_next_addr =
            reinterpret_cast<unsigned long long>(g_next_ptr);
        unsigned int g_next_lo = static_cast<unsigned int>(g_next_addr);
        unsigned int g_next_hi = static_cast<unsigned int>(g_next_addr >> 32);

        // Materialize the next source address before issuing the current copy.
        // Split operands avoid an extra consecutive 64-bit register pair. Copy
        // order stays sequential within each pass, preserving 128 B requests.
        asm volatile("" :: "r"(g_next_lo), "r"(g_next_hi));
        cute::copy(tiled_copy, g_cur, sKV(_, Int<P>{}, Int<START>{}));
        hs64_copy_k_tiles_pass_lookahead_impl<START + 1, END, P>(
            tiled_copy, gKV, sKV, g_next);
    } else {
        cute::copy(tiled_copy, g_cur, sKV(_, Int<P>{}, Int<START>{}));
    }
}

template <int START, int END, int P,
          typename TiledCopy,
          typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__forceinline__ __device__ void hs64_copy_k_tiles_pass(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV,
    Tensor<Engine1, Layout1> &sKV)
{
    auto g_cur = gKV(_, _0{}, Int<START>{});
    hs64_copy_k_tiles_pass_lookahead_impl<START, END, P>(
        tiled_copy, gKV, sKV, g_cur);
}


// D512's lane/pass/tile/bank fields occupy disjoint address bits.
template<int TILE, int P, typename T, typename TiledCopy, typename EG, typename LG,
         typename EL, typename LL>
__forceinline__ __device__ void hs64_copy_even_full_or_pass(
    TiledCopy tiled_copy, Tensor<EG, LG> const &gKV, Tensor<EL, LL> &sLow,
    decltype(gKV(_, _0{}, Int<TILE>{})) g_cur, uint32_t high_bank_mask, int tidx)
{
    static_assert(!T::kHasExtraKTile && T::kNumKTiles == 8);
    using Plan = typename T::SharedMemoryPlan;
    static_assert(offsetof(Plan, smem_sK) + 4 * 8192 == 0x18000);
    static_assert(offsetof(Plan, smem_even_high) == 0x38000);
    if constexpr (TILE + 1 < 8) {
        auto g_next = gKV(_, _0{}, Int<TILE + 1>{});
        unsigned long long addr = reinterpret_cast<unsigned long long>(
            cute::raw_pointer_cast(g_next.data()));
        unsigned int lo = static_cast<unsigned int>(addr);
        unsigned int hi = static_cast<unsigned int>(addr >> 32);
        asm volatile("" :: "r"(lo), "r"(hi));
        if constexpr (TILE < 4) {
            cute::copy(tiled_copy, g_cur, sLow(_, Int<P>{}, Int<TILE>{}));
        } else {
            auto src = recast<cute::uint128_t>(g_cur);
            CUTE_STATIC_ASSERT_V(size(src) == Int<1>{});
            const uint32_t tid = static_cast<uint32_t>(hs64_store_thread<T>(tidx)) & 255u;
            const uint32_t lane_offset = typename T::SmemLayoutKHigh4Store{}(
                tid / 8 + (T::kArch == 80 ? P * 32 : 0), (tid % 8) * 8) * sizeof(typename T::InputT);
            const uint32_t dst = lane_offset | 0x18000u | (T::kArch == 80 ? 0u : P * 0x1000u) |
                                 ((TILE - 4) * 0x2000u) | high_bank_mask;
            PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                src[0], *static_cast<cute::uint128_t*>(__cvta_shared_to_generic(dst)),
                tiled_copy.pred);
        }
        hs64_copy_even_full_or_pass<TILE + 1, P, T>(
            tiled_copy, gKV, sLow, g_next, high_bank_mask, tidx);
    } else {
        auto src = recast<cute::uint128_t>(g_cur);
        CUTE_STATIC_ASSERT_V(size(src) == Int<1>{});
        const uint32_t tid = static_cast<uint32_t>(hs64_store_thread<T>(tidx)) & 255u;
        const uint32_t lane_offset = typename T::SmemLayoutKHigh4Store{}(
            tid / 8 + (T::kArch == 80 ? P * 32 : 0), (tid % 8) * 8) * sizeof(typename T::InputT);
        const uint32_t dst = lane_offset | 0x18000u | (T::kArch == 80 ? 0u : P * 0x1000u) |
                             ((TILE - 4) * 0x2000u) | high_bank_mask;
        PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
            src[0], *static_cast<cute::uint128_t*>(__cvta_shared_to_generic(dst)),
            tiled_copy.pred);
    }
}

// Hold four independent copy operands together without changing the stage.
template<int TILE, int P, typename T, typename EL, typename LL>
__forceinline__ __device__ uint32_t hs64_even_bundle_dst(
    Tensor<EL, LL> &sLow, uint32_t high_base, int tidx) {
    if constexpr (TILE < 4 || TILE == 8) {
        // Only the merged even loader uses this helper: low/tail always
        // target buf0. Encode its cube prefix independently of the lane.
        using Plan = typename T::SharedMemoryPlan;
        static_assert(T::kHasExtraKTile && T::kNumKTiles == 9);
        static_assert(offsetof(Plan, smem_sK) == 0x12000);
        const uint32_t tid = static_cast<uint32_t>(hs64_store_thread<T>(tidx)) & 255u;
        const uint32_t lane_offset = typename T::SmemLayoutKHigh4Store{}(
            tid / 8 + (T::kArch == 80 ? P * 32 : 0), (tid % 8) * 8) * sizeof(typename T::InputT);
        constexpr uint32_t prefix = offsetof(Plan, smem_sK)
            + TILE * 0x2000u + (T::kArch == 80 ? 0u : P * 0x1000u);
        return prefix | lane_offset;
    } else {
        const uint32_t tid = static_cast<uint32_t>(hs64_store_thread<T>(tidx)) & 255u;
        const uint32_t lane_offset = typename T::SmemLayoutKHigh4Store{}(
            tid / 8 + (T::kArch == 80 ? P * 32 : 0), (tid % 8) * 8) * sizeof(typename T::InputT);
        return (high_base + (TILE - 4) * 0x2000u + (T::kArch == 80 ? 0u : P * 0x1000u)) | lane_offset;
    }
}

template<int TILE, int P, typename T, typename TiledCopy, typename EG, typename LG,
         typename EL, typename LL>
__forceinline__ __device__ void hs64_copy_even_full_prefix_pass(
    TiledCopy tiled_copy, Tensor<EG, LG> const &gKV, Tensor<EL, LL> &sLow,
    decltype(gKV(_, _0{}, Int<TILE>{})) g_cur, uint32_t high_base, int tidx)
{
    static_assert(T::kHasExtraKTile && T::kNumKTiles == 9);
    static_assert(TILE == 0 || TILE == 4 || TILE == 8);
    if constexpr (TILE < 8) {
        auto src0 = recast<cute::uint128_t>(g_cur);
        auto src1 = recast<cute::uint128_t>(gKV(_, _0{}, Int<TILE + 1>{}));
        auto src2 = recast<cute::uint128_t>(gKV(_, _0{}, Int<TILE + 2>{}));
        auto src3 = recast<cute::uint128_t>(gKV(_, _0{}, Int<TILE + 3>{}));
        CUTE_STATIC_ASSERT_V(size(src0) == Int<1>{});
        CUTE_STATIC_ASSERT_V(size(src1) == Int<1>{});
        CUTE_STATIC_ASSERT_V(size(src2) == Int<1>{});
        CUTE_STATIC_ASSERT_V(size(src3) == Int<1>{});
        const uint32_t dst0 = hs64_even_bundle_dst<TILE, P, T>(sLow, high_base, tidx);
        const uint32_t dst1 = hs64_even_bundle_dst<TILE + 1, P, T>(sLow, high_base, tidx);
        const uint32_t dst2 = hs64_even_bundle_dst<TILE + 2, P, T>(sLow, high_base, tidx);
        const uint32_t dst3 = hs64_even_bundle_dst<TILE + 3, P, T>(sLow, high_base, tidx);
        auto g_next = gKV(_, _0{}, Int<TILE + 4>{});
        unsigned long long next_addr = reinterpret_cast<unsigned long long>(
            cute::raw_pointer_cast(g_next.data()));
        const unsigned lo = static_cast<unsigned>(next_addr);
        const unsigned hi = static_cast<unsigned>(next_addr >> 32);
        asm volatile("" :: "r"(lo), "r"(hi));
        const int zfill = tiled_copy.pred ? 0 : 16;
        __ppu_pipeline_memcpy_async_zfill(
            __cvta_shared_to_generic(dst0), &src0[0], 16, zfill, 1);
        __ppu_pipeline_memcpy_async_zfill(
            __cvta_shared_to_generic(dst1), &src1[0], 16, zfill, 1);
        __ppu_pipeline_memcpy_async_zfill(
            __cvta_shared_to_generic(dst2), &src2[0], 16, zfill, 1);
        __ppu_pipeline_memcpy_async_zfill(
            __cvta_shared_to_generic(dst3), &src3[0], 16, zfill, 1);
        hs64_copy_even_full_prefix_pass<TILE + 4, P, T>(
            tiled_copy, gKV, sLow, g_next, high_base, tidx);
    } else {
        auto src = recast<cute::uint128_t>(g_cur);
        CUTE_STATIC_ASSERT_V(size(src) == Int<1>{});
        const uint32_t dst = hs64_even_bundle_dst<8, P, T>(sLow, high_base, tidx);
        __ppu_pipeline_memcpy_async_zfill(
            __cvta_shared_to_generic(dst), &src[0], 16, tiled_copy.pred ? 0 : 16, 1);
    }
}

// Bundle the four local high-K copy operands (tiles 4..7).
template<int P, typename TiledCopy, typename EG, typename LG,
         typename ES, typename LS>
__forceinline__ __device__ void hs64_copy_high_bundle_pass(
    TiledCopy tiled_copy, Tensor<EG, LG> const &gKV, Tensor<ES, LS> &sKV)
{
    auto src0 = recast<cute::uint128_t>(gKV(_, _0{}, Int<4>{}));
    auto src1 = recast<cute::uint128_t>(gKV(_, _0{}, Int<5>{}));
    auto src2 = recast<cute::uint128_t>(gKV(_, _0{}, Int<6>{}));
    auto src3 = recast<cute::uint128_t>(gKV(_, _0{}, Int<7>{}));
    auto dst0 = recast<cute::uint128_t>(sKV(_, Int<P>{}, Int<4>{}));
    auto dst1 = recast<cute::uint128_t>(sKV(_, Int<P>{}, Int<5>{}));
    auto dst2 = recast<cute::uint128_t>(sKV(_, Int<P>{}, Int<6>{}));
    auto dst3 = recast<cute::uint128_t>(sKV(_, Int<P>{}, Int<7>{}));
    CUTE_STATIC_ASSERT_V(size(src0) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size(src1) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size(src2) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size(src3) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size(dst0) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size(dst1) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size(dst2) == Int<1>{});
    CUTE_STATIC_ASSERT_V(size(dst3) == Int<1>{});
    auto g_next = gKV(_, _0{}, Int<8>{});
    const unsigned long long next_addr = reinterpret_cast<unsigned long long>(
        cute::raw_pointer_cast(g_next.data()));
    asm volatile("" :: "r"(static_cast<unsigned>(next_addr)),
                       "r"(static_cast<unsigned>(next_addr >> 32)));
    const int zfill = tiled_copy.pred ? 0 : 16;
    __ppu_pipeline_memcpy_async_zfill(&dst0[0], &src0[0], 16, zfill, 1);
    __ppu_pipeline_memcpy_async_zfill(&dst1[0], &src1[0], 16, zfill, 1);
    __ppu_pipeline_memcpy_async_zfill(&dst2[0], &src2[0], 16, zfill, 1);
    __ppu_pipeline_memcpy_async_zfill(&dst3[0], &src3[0], 16, zfill, 1);
    cute::copy(tiled_copy, g_next, sKV(_, Int<P>{}, Int<8>{}));
}

template<bool UseExtra>
__forceinline__ __device__ const int* hs64_query_indices(
    const Flash_fwd_mla_params &params)
{
    const int* indices = UseExtra ? params.extra_indices_ptr : params.indices_ptr;
    // grid.x is the query; its grid.y head tiles share the same sparse indices.
    const int row_stride = UseExtra
        ? params.extra_indices_row_stride : params.indices_row_stride;
    return indices + static_cast<int64_t>(blockIdx.x) * row_stride;
}

// Phase 2: Issue cp.async from token_ptr and prefetch the next token index.
// PREFETCH_USE_EXTRA selects the next block's index base.
template<int S, int E, bool PREFETCH_USE_EXTRA, typename T, bool DO_PREFETCH = true,
         bool LOCAL_COPY_GROUP = false, bool MERGE_EVEN_HIGH = false,
         bool ODD_LOW_ONLY = false,
         typename TensorSK, typename TensorHigh = TensorSK>
__forceinline__ __device__ void issue_K_load_bf16(
    const Flash_fwd_mla_params &params,
    TensorSK &sK_buf,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    __mbarrier_t *barriers_K,
    int tidx,
    int ori_block_max,
    typename T::InputT* (&token_ptr)[T::kGmemPasses],
    bool (&is_valid)[T::kGmemPasses],
    int *pre_token_idx,
    int next_block_idx,
    int end_block_idx,
    int extra_seqlen_k,
    TensorHigh *even_high = nullptr
) {
    using InputT = typename T::InputT;
    constexpr int kBlockN = T::kBlockN;

    // Eight lanes cover one token's contiguous 128 B run. Each physical copy
    // wave owns 32 tokens, while the shared-memory partition covers all 64.
    using GmemLayoutAtomK = Layout<Shape<Int<T::kGmemTokPerPass>, Int<T::kGmemThrPerTok>>,
                                   Stride<Int<T::kGmemThrPerTok>, _1>>;
    using GmemTiledCopyKNoAiu = decltype(make_tiled_copy(
        Copy_Atom<std::conditional_t<T::kHasExtraKTile,
            HS64_CP_ASYNC_NO_PREF128_ZFILL,
            PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>>, InputT>{},
        GmemLayoutAtomK{},
        Layout<Shape<_1, Int<T::kGmemElemsPerLoad>>>{}));
    GmemTiledCopyKNoAiu gmem_tiled_copy_K;

    // Dummy k_base_ptr — actual address comes from token_ptr[p] (set per pass)
    InputT* k_base_ptr = reinterpret_cast<InputT*>(params.k_ptr);
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tKsK = [&]() {
        if constexpr (T::kArch == 80) {
            auto store_view = make_tensor(sK_buf.data(), typename T::SmemLayoutKStore{});
            return gmem_tiled_copy_K.get_thread_slice(hs64_store_thread<T>(tidx)).partition_D(store_view);
        } else {
            return gmem_thr_copy_K.partition_D(sK_buf);
        }
    }();
    Tensor gK_tok = make_tensor(make_gmem_ptr(k_base_ptr),
        Shape<Int<T::kGmemTokPerPass>, Int<T::kHeadDim>>{},
        make_stride(params.k_row_stride, _1{}));
    // Keep each pass sequential so adjacent 128 B tiles of one sparse row stay
    // contiguous in the request stream.
    GmemTiledCopyKNoAiu cp0 = gmem_tiled_copy_K;
    cp0.pred = is_valid[0];
    Tensor tKgK0 = gmem_thr_copy_K.partition_S(gK_tok);
    tKgK0.data() = token_ptr[0];
    if constexpr (MERGE_EVEN_HIGH) {
        static_assert(T::kUseEvenHighBank && LOCAL_COPY_GROUP && S == 0 && E == 4);
        auto g_cur = tKgK0(_, _0{}, Int<0>{});
        if constexpr (T::kHasExtraKTile) {
            const uint32_t high_base = cast_smem_ptr_to_uint(
                cute::raw_pointer_cast(even_high->data()));
            hs64_copy_even_full_prefix_pass<0, 0, T>(
                cp0, tKgK0, tKsK, g_cur, high_base, tidx);
        } else {
            const uint32_t high_bank_mask = cast_smem_ptr_to_uint(
                cute::raw_pointer_cast(even_high->data())) & 0x20000u;
            hs64_copy_even_full_or_pass<0, 0, T>(
                cp0, tKgK0, tKsK, g_cur, high_bank_mask, tidx);
        }
    } else {
        if constexpr (T::kHasExtraKTile && LOCAL_COPY_GROUP && S == 4 && E == 9) {
            hs64_copy_high_bundle_pass<0>(cp0, tKgK0, tKsK);
        } else {
            hs64_copy_k_tiles_pass<S, E, 0>(cp0, tKgK0, tKsK);
        }
        if constexpr (T::kHasExtraKTile && LOCAL_COPY_GROUP && S == 0 && !ODD_LOW_ONLY) {
            hs64_copy_k_tiles_pass<8, 9, 0>(cp0, tKgK0, tKsK);
        }
    }
    if constexpr (T::kGmemPasses > 1) {
        static_assert(T::kGmemPasses == 2, "issue_K_load_bf16 handles 1 or 2 gather passes");
        GmemTiledCopyKNoAiu cp1 = gmem_tiled_copy_K;
        cp1.pred = is_valid[1];
        Tensor tKgK1 = gmem_thr_copy_K.partition_S(gK_tok);
        tKgK1.data() = token_ptr[1];
        if constexpr (MERGE_EVEN_HIGH) {
            auto g_cur = tKgK1(_, _0{}, Int<0>{});
            if constexpr (T::kHasExtraKTile) {
                const uint32_t high_base = cast_smem_ptr_to_uint(
                    cute::raw_pointer_cast(even_high->data()));
                hs64_copy_even_full_prefix_pass<0, 1, T>(
                    cp1, tKgK1, tKsK, g_cur, high_base, tidx);
            } else {
                const uint32_t high_bank_mask = cast_smem_ptr_to_uint(
                    cute::raw_pointer_cast(even_high->data())) & 0x20000u;
                hs64_copy_even_full_or_pass<0, 1, T>(
                    cp1, tKgK1, tKsK, g_cur, high_bank_mask, tidx);
            }
        } else {
            if constexpr (T::kHasExtraKTile && LOCAL_COPY_GROUP && S == 4 && E == 9) {
                hs64_copy_high_bundle_pass<1>(cp1, tKgK1, tKsK);
            } else {
                hs64_copy_k_tiles_pass<S, E, 1>(cp1, tKgK1, tKsK);
            }
            if constexpr (T::kHasExtraKTile && LOCAL_COPY_GROUP && S == 0 && !ODD_LOW_ONLY) {
                hs64_copy_k_tiles_pass<8, 9, 1>(cp1, tKgK1, tKsK);
            }
        }
    }
    // One completion after both copy waves. WG-local K1-high/K0-low use
    // the async group; cross-WG publications retain their shared mbarrier.
    if constexpr (LOCAL_COPY_GROUP) {
        static_assert(!ODD_LOW_ONLY || (T::kHasExtraKTile && S == 0 && E == 4));
        static_assert((S == 0 && E == T::kStage1KTiles) ||
                      (S == T::kStage1KTiles &&
                       (E == T::kNumKTiles || (T::kHasExtraKTile && E == 8))));
        __pipeline_commit();
    } else {
        cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
    }

    if constexpr (DO_PREFETCH) {
        if (next_block_idx < end_block_idx) {
            int next_eff_block;
            const int *next_idx_base;
            if constexpr (PREFETCH_USE_EXTRA) {
                next_eff_block = next_block_idx - ori_block_max;
                next_idx_base = hs64_query_indices<true>(params) + static_cast<int64_t>(batch_idx) * params.extra_indices_batch_stride + next_eff_block * kBlockN;
            } else {
                next_eff_block = next_block_idx;
                next_idx_base = hs64_query_indices<false>(params) + static_cast<int64_t>(batch_idx) * params.indices_batch_stride + next_eff_block * kBlockN;
            }
            // Interleave copy-wave 0/1 token owners within each 8-lane pair.
            const int lane = tidx & 31;
            const int warp = tidx >> 5;
            const int pair = lane >> 3;
            const int pass = (lane >> 2) & 1;
            const int meta_token = warp * 4 + pair + pass * 32;
            if constexpr (T::kGuardIndices) {
                // Scheduler padding must not extend the physical index read.
                const int index_count = PREFETCH_USE_EXTRA ? params.extra_topk : params.topk;
                pre_token_idx[0] = next_eff_block * kBlockN + meta_token < index_count
                    ? __ldg(next_idx_base + meta_token) : -1;
            } else {
                pre_token_idx[0] = __ldg(next_idx_base + meta_token);
            }
        } else if constexpr (T::kGuardIndices) {
            // Address lookahead may cross main/extra after this partition ends.
            // It must not reinterpret the previous cache's stale token index.
            pre_token_idx[0] = -1;
        }
    }
}

// Preserve the next-index prefetch at the high-K readiness point.
// The actual even-K copy and completion are owned by WG0.
template <bool PREFETCH_USE_EXTRA, typename T>
__forceinline__ __device__ void hs64_prefetch_next_indices(
    const Flash_fwd_mla_params &params,
    int batch_idx, int tidx, int ori_block_max,
    int *pre_token_idx, int next_block_idx, int end_block_idx)
{
    constexpr int kBlockN = T::kBlockN;
    if (next_block_idx < end_block_idx) {
        int next_eff_block;
        const int *next_idx_base;
        if constexpr (PREFETCH_USE_EXTRA) {
            next_eff_block = next_block_idx - ori_block_max;
            next_idx_base = hs64_query_indices<true>(params) +
                static_cast<int64_t>(batch_idx) *
                    params.extra_indices_batch_stride +
                next_eff_block * kBlockN;
        } else {
            next_eff_block = next_block_idx;
            next_idx_base = hs64_query_indices<false>(params) +
                static_cast<int64_t>(batch_idx) *
                    params.indices_batch_stride +
                next_eff_block * kBlockN;
        }
        const int lane = tidx & 31;
        const int warp = tidx >> 5;
        const int pair = lane >> 3;
        const int pass = (lane >> 2) & 1;
        const int meta_token = warp * 4 + pair + pass * 32;
        if constexpr (T::kGuardIndices) {
            const int index_count = PREFETCH_USE_EXTRA ? params.extra_topk : params.topk;
            pre_token_idx[0] = next_eff_block * kBlockN + meta_token < index_count
                ? __ldg(next_idx_base + meta_token) : -1;
        } else {
            pre_token_idx[0] = __ldg(next_idx_base + meta_token);
        }
    } else if constexpr (T::kGuardIndices) {
        pre_token_idx[0] = -1;
    }
}

template<int S, int E, bool ALLOW_EXTRA, typename T,
         bool DO_PREFETCH = true, bool LOCAL_COPY_GROUP = false,
         typename TensorSK, typename TensorVI>
__forceinline__ __device__ void load_K_tiles_bf16(
    const Flash_fwd_mla_params &params,
    TensorSK &sK_buf,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    __mbarrier_t *barriers_K,
    int tidx,
    int ori_block_max,
    TensorVI &smem_valid_indices,
    int vi_buf,
    int *pre_token_idx,
    int next_block_idx,
    int end_block_idx,
    int extra_seqlen_k
) {
    typename T::InputT* token_ptr[T::kGmemPasses];
    bool is_valid[T::kGmemPasses];
    if constexpr (ALLOW_EXTRA) {
        // Dynamic wrapper for the general image: compute the current cache
        // address and select the next metadata prefetch cache.
        bool use_extra =
            (ori_block_max >= 0) && (block_idx_kv >= ori_block_max);
        bool prefetch_use_extra =
            (ori_block_max >= 0) && (next_block_idx >= ori_block_max);
        compute_K_addr_bf16_dynamic<S, T>(
            params, batch_idx, block_idx_kv, seqlen_k, tidx,
            ori_block_max, smem_valid_indices, vi_buf, pre_token_idx,
            extra_seqlen_k, token_ptr, is_valid, use_extra);

        if (prefetch_use_extra) {
            issue_K_load_bf16<S, E, true, T, DO_PREFETCH, LOCAL_COPY_GROUP>(
                params, sK_buf, batch_idx, block_idx_kv, seqlen_k,
                barriers_K, tidx, ori_block_max, token_ptr, is_valid,
                pre_token_idx, next_block_idx, end_block_idx,
                extra_seqlen_k);
        } else {
            issue_K_load_bf16<S, E, false, T, DO_PREFETCH, LOCAL_COPY_GROUP>(
                params, sK_buf, batch_idx, block_idx_kv, seqlen_k,
                barriers_K, tidx, ori_block_max, token_ptr, is_valid,
                pre_token_idx, next_block_idx, end_block_idx,
                extra_seqlen_k);
        }
    } else {
        compute_K_addr_bf16<S, false, T>(
            params, batch_idx, block_idx_kv, seqlen_k, tidx,
            ori_block_max, smem_valid_indices, vi_buf, pre_token_idx,
            extra_seqlen_k, token_ptr, is_valid);
        issue_K_load_bf16<S, E, false, T, DO_PREFETCH, LOCAL_COPY_GROUP>(
            params, sK_buf, batch_idx, block_idx_kv, seqlen_k, barriers_K,
            tidx, ori_block_max, token_ptr, is_valid, pre_token_idx,
            next_block_idx, end_block_idx, extra_seqlen_k);
    }
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
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<typename T::InputT*>(params.q_ptr)
                                            + batch_idx * params.q_batch_stride),
                            make_shape(params.seqlen_q, params.h_h_k_ratio, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(make_mix_tensor_like(mQ(_, k_head_idx, _)), Shape<Int<T::kBlockM>, Int<T::kHeadDim>>{},
                            make_coord(m_block_idx, 0));  // (kBlockM, kHeadDim)
    typename T::GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);

    if constexpr (T::kArch == 80) {
        gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, params.seqlen_q, params.q_row_stride,
                                       T::kBlockM, T::kBlockKSmem, 0};
    } else {
        gmem_tiled_copy_Q.desc_.init(nullptr, params.seqlen_q, params.d, params.q_row_stride);
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
    Tensor<Engine0, Layout0> &sV_direct)
{
    // Use SmemLayoutVtDirect (swizzled composition) to read V transposed
    // from the independent V buffer. This produces a ComposedLayout that
    // make_mix_tensor_like can handle (rank 2, rank0=1, rank1=1).
    Tensor sVt = make_tensor(sV_direct.data(), (typename T::SmemLayoutVtDirect){});
    return flat_divide(sVt, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
}

// The sQ tensor starts at SharedMemoryPlan byte zero (checked below).
// Preserve D576's raw-Q bank; D512 uses its separately allocated CUBE bank.
template <typename T, typename Engine, typename Layout>
__forceinline__ __device__ auto hs64_even_high_scratch(
    Tensor<Engine, Layout> &sQ)
{
    if constexpr (T::kHasExtraKTile) {
        return local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                          Coord<_0, Int<4>>{}).data();
    } else {
        using Plan = typename T::SharedMemoryPlan;
        static_assert(offsetof(Plan, smem_sQ) == 0);
        static_assert(offsetof(Plan, smem_even_high) == 229376);
        static_assert(sizeof(Plan) == 262144);
        return sQ.data() + offsetof(Plan, smem_even_high) / sizeof(typename T::InputT);
    }
}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool NEXT_EXTRA,
    typename Engine1, typename Layout1,
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
    typename EngineQ6, typename LayoutQ6,
    typename EngineQ4, typename LayoutQ4,
    typename EngineQ5, typename LayoutQ5,
    typename EngineQMid6, typename LayoutQMid6,
    typename EngineQlow0, typename LayoutQlow0,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void wg0_subroutine(
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine3, Layout3> &cur_sK0,
    Tensor<Engine4, Layout4> &cur_sK1,
    Tensor<Engine5, Layout5> &nxt_sK0,
    Tensor<Engine6, Layout6> &sP0,
    Tensor<Engine7, Layout7> &sP1,
    Tensor<Engine8, Layout8> &sM,
    Tensor<Engine9, Layout9> &sScale0,
    Tensor<Engine10, Layout10> &sScale1,
    Tensor<Engine11, Layout11> &rQ8,
    Tensor<EngineQ6, LayoutQ6> &rQ6,
    Tensor<EngineQ4, LayoutQ4> &rQ4,
    Tensor<EngineQ5, LayoutQ5> &rQ5,
    Tensor<EngineQMid6, LayoutQMid6> &rQmid6,
    Tensor<EngineQlow0, LayoutQlow0> &rQlow0,
    Tensor<Engine12, Layout12> &rP0,
    Tensor<Engine13, Layout13> &rO0,
    float rL[2],
    __mbarrier_t barriers_K0[9],
    __mbarrier_t barriers_K1[9],
    Hs64MbarPhase<T> &cur_phase_K0,
    uint32_t &even_high_in_scratch,
    const Flash_fwd_mla_params &params,
    int seqlen_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup,
    int wg_idx,
    int batch_idx,
    int ori_block_max,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    // Pre-fetch pipeline: token index for current block (in/out)
    int *pre_token_idx,
    int *pre_token_idx_b,
    int extra_seqlen_k,
    typename T::InputT* (&precomp_ptr0)[T::kGmemPasses],
    typename T::InputT* (&precomp_ptr1)[T::kGmemPasses],
    bool (&precomp_valid0)[T::kGmemPasses],
    bool (&precomp_valid1)[T::kGmemPasses],
    float* smem_cross_n_reduction
) {
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx+2;
    int nxt_block1 = block_idx+3;

    // Derive V directly from K buffers. cur_sK0 holds even block (local),
    // cur_sK1 holds odd block (remote). V = first 512 dims of each K buffer.
    // Cross-WG smem_valid_indices: 4 bufs, WG0 uses 0/1, WG1 uses 2/3 (alternating)
    int vi_softmax_wg0 = (block_idx / 2) % 2;       // 0 or 1
    auto sV_local = make_tensor(cur_sK0.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV0L = get_half_V<T, 0>(sV_local);
    auto sV_remote = make_tensor(cur_sK1.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV1L = get_half_V<T, 0>(sV_remote);

    // Calc P0 = softmax(P0) and signal sScale0Ready before K load
    // (WG1 is waiting for sScale0Ready — arriving earlier lets WG1 start sooner)
    // rPb must carry the TiledMma's own C-fragment layout: (8,1) gives 16
    // elements per thread, (4,2) only 8. Hard-coding the (8,1) shape leaves the
    // upper half uninitialized and mis-maps retile_S in save_rP*_to_sP.
    Tensor rPb = make_tensor<typename T::InputT>(
        partition_fragment_C(typename T::TiledMma{}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{}).layout());
    Wg0SoftmaxSums wg0_softmax_sums;
    if constexpr (T::kHasExtraKTile) {
        wg0_softmax_sums = wg0_bunch_0_uniform_geometry<T, IS_BLK0_LAST || IS_BLK1_LAST>(
            rPb, rP0, rO0, sScale0, sM, rL, params.scale_softmax_log2,
            start_token_idx, idx_in_warpgroup, wg_idx,
            smem_valid_indices, vi_softmax_wg0, smem_cross_n_reduction);
    } else {
        wg0_softmax_sums = wg0_bunch_0<T, IS_BLK0_LAST || IS_BLK1_LAST>(rPb, rP0, rO0, sScale0, sM, rL, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg0, smem_cross_n_reduction);
    }


    auto nxt_sK1 = cur_sK1;
    Wg0ScaleFactors early_scale_factors;
    // Use the exact-ID NamedBarrier overload: cross-cut exchanges use IDs 7..14
    // and must not collide with the user overload's reserved-ID offset.
    // IDs 1..4 are reserved for these CTA-wide rendezvous.
    // M128 keeps the non-blocking producer arrival.
    if constexpr (T::kIsCrossCut) {
        hs64_shared_exchange_sync<T>(1, T::NUM_THREADS);  // sScale0Ready
    } else {
        cutlass::arch::NamedBarrier::arrive(
            T::NUM_THREADS, static_cast<cutlass::arch::ReservedNamedBarriers>(1));
    }

    // For M64, the CTA-wide bar.sync publishes sScale0/sM to WG1.

    // Issue rO0 += rPb @ sV0L
    wg0_scale0_rO0<T>(
        rO0, sScale0, idx_in_warpgroup, &wg0_softmax_sums, rL);

    if constexpr (T::kIsCrossCut) {
        // (4,2) layout: each N-warp only has 16/32 P columns, localP broken.
        // Save rPb to sP0 temporarily, per-WG barrier, then remoteP from SMEM.
        save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);
        { int _cbar = 6 + (int)(threadIdx.x >> 8); hs64_shared_exchange_sync<T>(_cbar, 256); }
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0L, rO0, idx_in_warpgroup, wg_idx);
    } else {
        warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, idx_in_warpgroup, wg_idx);
    }

    // sScale1Ready also signals sP1 is ready (WG1 saves sP1 before arriving)
    { hs64_shared_exchange_sync<T>(2, T::NUM_THREADS); }  // sScale1Ready

    if constexpr (!IS_BLK0_LAST && T::kUsePv2x4) {
        scale_rO_pv_one<T>(
            rO0, sScale1, idx_in_warpgroup, &early_scale_factors);
    }

    auto p_store_addresses = [&]() {
        if constexpr (T::kIsPrefill && T::kHasExtraKTile) {
            return hs64_prepare_p_store<T>(sP0, idx_in_warpgroup);
        } else {
            return cute::array<uint32_t, 1>{};
        }
    }();

    // [2-buffer] Reload tiles 0-3 only after localP consumes
    // sV0L=cur_sK0 and sScale1Ready releases the buffer.
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {

        if constexpr (T::kUseEvenHighBank) {
            // The inactive even-high bank is free before bar2. Load it in
            // WG0 now; the existing group wait/bar6 publishes both K0 halves.
            auto scratch_base = hs64_even_high_scratch<T>(sQ);
            auto buf0_tile4 = local_tile(
                cur_sK0, Shape<Int<T::kBlockN>, _64>{}, Coord<_0, Int<4>>{});
            auto next_high_base = even_high_in_scratch
                ? buf0_tile4.data() : scratch_base;
            Tensor next_sK0_high4 = make_tensor(
                next_high_base, (typename T::SmemLayoutKHigh4){});
            issue_K_load_bf16<0, T::kStage1KTiles, NEXT_EXTRA, T, true, true, true>(
                params, cur_sK0, batch_idx, nxt_block0, seqlen_k, &barriers_K0[0],
                idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
                pre_token_idx, nxt_block0 + 2, end_block_idx, extra_seqlen_k,
                &next_sK0_high4);
        } else {
            issue_K_load_bf16<0, T::kStage1KTiles, NEXT_EXTRA, T, true, true>(
                params, cur_sK0, batch_idx, nxt_block0, seqlen_k, &barriers_K0[0],
                idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
                pre_token_idx, nxt_block0 + 2, end_block_idx, extra_seqlen_k);
        }
    }

    // For M64, bar.sync(2) publishes WG1's sScale1/sM update to WG0.

    Wg0ScaleFactors wg0_scale_factors = wg0_scale_rP0<T>(
        sScale1, rP0, rPb, idx_in_warpgroup,
        T::kUsePv2x4 && !IS_BLK0_LAST ? &early_scale_factors : nullptr);
    if constexpr (T::kIsPrefill && T::kHasExtraKTile) {
        hs64_save_prepared_p<T>(rPb, p_store_addresses, idx_in_warpgroup);
    } else {
        save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);
    }

    // Sync with WG1: ensures sP0 not overwritten before WG1 reads it.
    if constexpr (T::kIsCrossCut) {
        hs64_shared_exchange_sync<T>(3, T::NUM_THREADS);  // sP0Ready
    } else {
        cutlass::arch::NamedBarrier::arrive(
            T::NUM_THREADS, static_cast<cutlass::arch::ReservedNamedBarriers>(3));  // M128 non-blocking producer
    }

    // bar(3) proves WG1 has finished reading buf1[256, 512). Reuse WG0's
    // existing odd-block row base to issue that half before bar(5).
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST
                  && (!T::kIsCrossCut || !T::kUseQkWeave)
    ) {
        constexpr int kNumTiles = T::kNumKTiles;
        issue_K_load_bf16<T::kStage1KTiles, kNumTiles, NEXT_EXTRA, T, false>(
            params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[1],
            idx_in_warpgroup, ori_block_max, precomp_ptr1, precomp_valid1,
            pre_token_idx_b, nxt_block1 + 2, end_block_idx, extra_seqlen_k);
    }

    // Fill WG0's existing bar(5) wait window with register rescaling, while
    // keeping remote PV after the blocking rendezvous to preserve WG staggering.
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (T::kUsePv2x4) {
            rL[0] *= wg0_scale_factors.row0;
            rL[1] *= wg0_scale_factors.row1;
        } else {
            wg0_rescale_rO0_with_factors<T>(rO0, wg0_scale_factors, rL);
        }
    }

    // [2-buffer] buffer-free barrier (mirrors ref rO1sP0sV0RIssued).
    if constexpr (!T::kIsCrossCut || !T::kUseQkWeave || T::kKeepQkWeaveBar5) {
        cutlass::arch::NamedBarrier::sync(
            T::NUM_THREADS, static_cast<cutlass::arch::ReservedNamedBarriers>(5));
    }

    // sP1 ready via sScale1Ready. remote PV can proceed after sP0Ready sync.
    if constexpr (!IS_BLK0_LAST) {
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, idx_in_warpgroup, wg_idx);
        if constexpr (T::kHasExtraKTile && !IS_BLK1_LAST) {
            // Publish old buf1-low reader completion, not K0 copy completion.
            // WG1 waits for all 256 readers before overwriting this low half.
            __ppu_barrier_arrive(4, T::NUM_THREADS, 15u);
        }
        if constexpr (T::kIsCrossCut) {
            // The following low-half reload writes the same buf1 image that
            // remote PV just read. Drain all WG0 readers before any warp starts
            // the asynchronous overwrite. The same rendezvous publishes the
            // earlier K0-low copy to every WG0 QK reader, without a new barrier.
            if constexpr (!IS_BLK1_LAST) {
                __pipeline_wait_prior(0);
            }
            cutlass::arch::NamedBarrier::sync(
                256, static_cast<cutlass::arch::ReservedNamedBarriers>(6));
        }
    }

    // Keep the original pre-QK reload for all non-target paths.
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST &&
                  (!T::kIsCrossCut || !T::kUseQkWeave)) {

        issue_K_load_bf16<0, T::kStage1KTiles, NEXT_EXTRA, T>(params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[0],
            idx_in_warpgroup, ori_block_max, precomp_ptr1, precomp_valid1,
            pre_token_idx_b, nxt_block1 + 2, end_block_idx, extra_seqlen_k);

    }

    // WG0 QK computes scores for next_block0 (always exists when !BLK0 && !BLK1)
    // Split-buf GEMM reads nxt_sK0 (tiles 0-3) + nxt_sK1 (tiles 4-8).
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP0);
        if constexpr (T::kIsCrossCut && T::kUseQkWeave) {
            if constexpr (!T::kHasExtraKTile) {
                issue_K_load_bf16<0, T::kStage1KTiles, NEXT_EXTRA, T>(params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[0],
                    idx_in_warpgroup, ori_block_max, precomp_ptr1, precomp_valid1,
                    pre_token_idx_b, nxt_block1 + 2, end_block_idx, extra_seqlen_k);
            }
            // K0-low readiness was published by the preceding WG0 bar6.
            warpgroup_cooperative_qkt_gemm<T, 5>(sQ, cur_sK0, cur_sK0, rP0, rQ8, rQ6, rQ4, rQlow0, barriers_K0, cur_phase_K0, idx_in_warpgroup);
        } else {
            warpgroup_cooperative_qkt_gemm<T, 0>(sQ, cur_sK0, cur_sK0, rP0, rQ8, rQ6, rQ4, rQlow0, barriers_K0, cur_phase_K0, idx_in_warpgroup);
        }
    }

    // After QK: precompute addr for NEXT iteration's K loads
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {

        int next_nxt_block0 = block_idx + 4;
        compute_K_addr_bf16<0, NEXT_EXTRA, T>(
            params, batch_idx, next_nxt_block0, seqlen_k,
            idx_in_warpgroup, ori_block_max, smem_valid_indices, vi_softmax_wg0,
            pre_token_idx, extra_seqlen_k, precomp_ptr0, precomp_valid0);
        if constexpr (!T::kHasExtraKTile) {
            int next_nxt_block1 = block_idx + 5;
            int next_vi_preload_wg1 = vi_softmax_wg0 + 2;
            compute_K_addr_bf16<0, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block1, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, next_vi_preload_wg1,
                pre_token_idx_b, extra_seqlen_k, precomp_ptr1, precomp_valid1);
        }

    }

    // Issue P0 = Q @ K0^T (next_block0 always exists)
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (T::kUseEvenHighBank) {
            auto scratch_base = hs64_even_high_scratch<T>(sQ);
            auto buf0_tile4 = local_tile(
                cur_sK0, Shape<Int<T::kBlockN>, _64>{}, Coord<_0, Int<4>>{});
            auto next_high_base = even_high_in_scratch
                ? buf0_tile4.data() : scratch_base;
            Tensor next_sK0_high4 = make_tensor(
                next_high_base, (typename T::SmemLayoutKHigh4){});
            // The earlier WG0 group wait/bar6 already completed this bank.
            auto wait_k0_high = []() {};
            if constexpr (T::kHasExtraKTile) {
                warpgroup_cooperative_qkt_gemm_high4_tail<T, true>(
                    sQ, next_sK0_high4, cur_sK0, rP0,
                    rQ4, rQ5, rQmid6, rQ6,
                    cur_phase_K0, idx_in_warpgroup, wait_k0_high);
            } else {
                warpgroup_cooperative_qkt_gemm_high4_tail<T, true>(
                    sQ, next_sK0_high4, cur_sK0, rP0,
                    rQ4, rQ5, rQ6, rQ8,
                    cur_phase_K0, idx_in_warpgroup, wait_k0_high);
            }
            even_high_in_scratch ^= 1u;
        } else {
            warpgroup_cooperative_qkt_gemm<T, 2>(
                sQ, cur_sK0, cur_sK0, rP0, rQ8, rQ6, rQ4, rQlow0,
                barriers_K0, cur_phase_K0, idx_in_warpgroup);
        }
    }

    // [BlockM=64] CTA-wide barrier at the K-block iteration boundary: wg0's
    // next-iteration PV-remote reads a buffer wg1 loaded in this iteration
    // (sV1L = cur_sK1) without a per-buffer mbarrier wait on barriers_K1.

    // [2-buffer] no rotation: buf0=even, buf1=odd fixed

}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool NEXT_EXTRA,
    typename Engine1, typename Layout1,
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
    typename EngineQ6, typename LayoutQ6,
    typename EngineQ4, typename LayoutQ4,
    typename EngineQ5, typename LayoutQ5,
    typename EngineQMid6, typename LayoutQMid6,
    typename EngineQlow0, typename LayoutQlow0,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void wg1_subroutine(
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine3, Layout3> &cur_sK1,
    Tensor<Engine4, Layout4> &cur_sK0,
    Tensor<Engine5, Layout5> &nxt_sK1,
    Tensor<Engine6, Layout6> &sP0,
    Tensor<Engine7, Layout7> &sP1,
    Tensor<Engine8, Layout8> &sM,
    Tensor<Engine9, Layout9> &sScale0,
    Tensor<Engine10, Layout10> &sScale1,
    Tensor<Engine11, Layout11> &rQ8,
    Tensor<EngineQ6, LayoutQ6> &rQ6,
    Tensor<EngineQ4, LayoutQ4> &rQ4,
    Tensor<EngineQ5, LayoutQ5> &rQ5,
    Tensor<EngineQMid6, LayoutQMid6> &rQmid6,
    Tensor<EngineQlow0, LayoutQlow0> &rQlow0,
    Tensor<Engine12, Layout12> &rP1,
    Tensor<Engine13, Layout13> &rO1,
    float rL[2],
    __mbarrier_t barriers_K0[9],
    __mbarrier_t barriers_K1[9],
    Hs64MbarPhase<T> &cur_phase_K1,
    uint32_t &even_high_in_scratch,
    const Flash_fwd_mla_params &params,
    int seqlen_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup,
    int wg_idx,
    int batch_idx,
    int ori_block_max,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    // Pre-fetch pipeline: token index for current block (in/out)
    int *pre_token_idx,
    int *pre_token_idx_b,
    int extra_seqlen_k,
    typename T::InputT* (&precomp_ptr0)[T::kGmemPasses],
    typename T::InputT* (&precomp_ptr1)[T::kGmemPasses],
    bool (&precomp_valid0)[T::kGmemPasses],
    bool (&precomp_valid1)[T::kGmemPasses],
    float* smem_cross_n_reduction
) {
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx+2;
    int nxt_block1 = block_idx+3;

    auto nxt_sK0 = cur_sK1;
    // Derive V directly from K buffers. In wg1_subroutine, due to
    // LAUNCH_WG1_SUBROUTINE arg-swap: cur_sK0 = odd block (local for wg1),
    // cur_sK1 = even block (remote from wg0). V = first 512 dims of each K buffer.
    // Cross-WG smem_valid_indices: 4 bufs, WG0 uses 0/1, WG1 uses 2/3 (alternating)
    int vi_softmax_wg1 = 2 + (block_idx / 2) % 2;    // 2 or 3
    auto sV_local = make_tensor(cur_sK0.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV0R = get_half_V<T, 1>(sV_local);
    auto sV_remote = make_tensor(cur_sK1.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV1R = get_half_V<T, 1>(sV_remote);


    // [2-buffer] Keep the nxt_block1 reload below the local remoteP that
    // consumes sV0R=cur_sK0.

    // Pre-compute cur_max before barrier because it does not depend on
    // sM/sScale0. The cross-cut path includes its cross-N exchange.
    float r_cur_max[2];
    if constexpr (T::kIsCrossCut) {
        wg1_bunch_0_pre_crosscut<T, IS_BLK0_LAST>(
            r_cur_max, rP1, params.scale_softmax_log2, idx_in_warpgroup,
            smem_valid_indices, vi_softmax_wg1, smem_cross_n_reduction);
    } else {
        wg1_bunch_0_pre<T, IS_BLK0_LAST>(r_cur_max, rP1, params.scale_softmax_log2, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg1);
    }

    // Wait for WG0's sScale0/sM update after precomputing cur_max.
    { hs64_shared_exchange_sync<T>(1, T::NUM_THREADS); }  // sScale0Ready

    if constexpr (T::kIsCrossCut) {
        wg1_bunch_0_pre_crosscut_finish<T>(
            r_cur_max, params.scale_softmax_log2, idx_in_warpgroup,
            smem_cross_n_reduction);
    }

    // Same C-fragment layout requirement as rPb (see wg0 above).
    Tensor rP1b = make_tensor<typename T::InputT>(
        partition_fragment_C(typename T::TiledMma{}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{}).layout());
    Wg1ScaleCache wg1_scale_cache =
        wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, false>(rP1b, sScale1, rO1, sM, rL, sScale0, rP1, params.scale_softmax_log2, start_token_idx + T::kBlockN, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg1, smem_cross_n_reduction, r_cur_max);

    // Save sP1 early (before sScale1Ready) so WG0 can read it without extra barrier
    if constexpr (!IS_BLK0_LAST) {
        save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
    }
    if constexpr (T::kIsCrossCut) {
        hs64_shared_exchange_sync<T>(2, T::NUM_THREADS);  // sScale1Ready
    } else {
        cutlass::arch::NamedBarrier::arrive(
            T::NUM_THREADS, static_cast<cutlass::arch::ReservedNamedBarriers>(2));  // M128 non-blocking producer
    }

    // Issue rO1 += rP1b @ sV1R
    if constexpr (T::kUsePv2x4) {
        scale_rO_pv_product_with_cache<T>(
            rO1, sScale0, sScale1, wg1_scale_cache, rL, idx_in_warpgroup);
    }
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (T::kIsCrossCut) {
            // (4,2) layout: each N-warp only has 16/32 P columns, localP broken.
            warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV0R, rO1, idx_in_warpgroup, wg_idx);
        } else {
            warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV0R, rO1, idx_in_warpgroup, wg_idx);
        }
    }

    // Sync with WG0: ensures WG0 doesn't overwrite sP0 before we read it.
    { hs64_shared_exchange_sync<T>(3, T::NUM_THREADS); }  // sP0Ready

    if constexpr (T::kIsCrossCut && T::kUseQkWeave &&
                  !IS_BLK0_LAST && !IS_BLK1_LAST && !T::kHasExtraKTile) {
        // bar3 already drains WG1's old local buf1-high reads. Reload the
        // disjoint high half here, avoiding a pre-bar async-copy fence.
        constexpr int kNumTiles = T::kNumKTiles;
        issue_K_load_bf16<T::kStage1KTiles, kNumTiles, NEXT_EXTRA, T, true, true>(
            params, cur_sK0, batch_idx, nxt_block1, seqlen_k,
            &barriers_K1[1], idx_in_warpgroup, ori_block_max,
            precomp_ptr0, precomp_valid0, pre_token_idx,
            nxt_block1 + 2, end_block_idx, extra_seqlen_k);
    }

    // Physical buffer ownership (wg1 arguments are swapped):
    //   buf0 = wg0 LOCAL PV (sV0L) + wg1 REMOTE PV (sV1R)
    //   buf1 = wg0 REMOTE PV (sV1L) + wg1 LOCAL PV (sV0R)

    // D576 alternates the even block's high V half between buf0 and the raw
    // sQ4..sQ7 cubes. The producer below always targets the opposite bank, so
    // no warp can overwrite another warp's current remote-PV source.
    if constexpr (T::kUseEvenHighBank) {
        auto scratch_base = hs64_even_high_scratch<T>(sQ);
        auto buf0_tile4 = local_tile(
            cur_sK1, Shape<Int<T::kBlockN>, _64>{}, Coord<_0, Int<4>>{});
        auto current_high_base = even_high_in_scratch
            ? scratch_base : buf0_tile4.data();
        Tensor current_sV1R = make_tensor(
            current_high_base, (typename T::SmemLayoutVtHigh4){});
        if constexpr (T::kHasExtraKTile && !IS_BLK0_LAST && !IS_BLK1_LAST) {
            auto issue_high_after_first_p = [&]() {
                constexpr int kNumTiles = T::kNumKTiles;
                issue_K_load_bf16<T::kStage1KTiles, kNumTiles, NEXT_EXTRA, T, true, true>(
                    params, cur_sK0, batch_idx, nxt_block1, seqlen_k,
                    &barriers_K1[1], idx_in_warpgroup, ori_block_max,
                    precomp_ptr0, precomp_valid0, pre_token_idx,
                    nxt_block1 + 2, end_block_idx, extra_seqlen_k);
            };
            warpgroup_cooperative_pv_gemm_remoteP<T, true>(
                sP0, current_sV1R, rO1, idx_in_warpgroup, wg_idx,
                issue_high_after_first_p);
        } else {
            warpgroup_cooperative_pv_gemm_remoteP<T>(
                sP0, current_sV1R, rO1, idx_in_warpgroup, wg_idx);
        }
    } else {
        // D512 keeps the original fixed buf0 high-half source.
        warpgroup_cooperative_pv_gemm_remoteP<T>(
            sP0, sV1R, rO1, idx_in_warpgroup, wg_idx);
    }

    if constexpr (T::kHasExtraKTile && !IS_BLK0_LAST && !IS_BLK1_LAST) {
        // Both WGs have issued their remote PV independently. Receive only
        // the old odd-low reader release, then cover low-copy latency with
        // high QK, whose separate local group was issued before remote PV.
        hs64_shared_exchange_sync<T>(4, T::NUM_THREADS);
        issue_K_load_bf16<0, T::kStage1KTiles, NEXT_EXTRA, T,
                         false, true, false, true>(
            params, cur_sK0, batch_idx, nxt_block1, seqlen_k,
            &barriers_K1[0], idx_in_warpgroup, ori_block_max,
            precomp_ptr0, precomp_valid0, pre_token_idx,
            nxt_block1 + 2, end_block_idx, extra_seqlen_k);
    }

    if constexpr (T::kIsCrossCut && T::kUseQkWeave &&
                  !IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP1);
        if constexpr (T::kUseEvenHighBank) {
            auto cur_k1_tile4 = local_tile(
                cur_sK0, Shape<Int<T::kBlockN>, _64>{}, Coord<_0, Int<4>>{});
            Tensor cur_sK1_high4 = make_tensor(
                cur_k1_tile4.data(), (typename T::SmemLayoutKHigh4){});
            auto phase6_wait_and_issue_k0 = [&]() {
                if constexpr (T::kHasExtraKTile) {
                    __pipeline_wait_prior(1);
                    // Do not flush the still-pending low VMEM group here.
                    __ppu_barrier_sync(15, 256, 15u);
                } else {
                    hs64_wait_local_k1_high();
                }
                // Preserve the original next-index prefetch and its timing.
                // K0-high copy/completion is now owned by WG0 after bar2.
                hs64_prefetch_next_indices<NEXT_EXTRA, T>(
                    params, batch_idx, idx_in_warpgroup, ori_block_max,
                    pre_token_idx_b, nxt_block0 + 2, end_block_idx);
            };
            if constexpr (T::kHasExtraKTile) {
                warpgroup_cooperative_qkt_gemm_high4_tail<T, false>(
                    sQ, cur_sK1_high4, cur_sK0, rP1,
                    rQ4, rQ5, rQmid6, rQ6,
                    cur_phase_K1, idx_in_warpgroup,
                    phase6_wait_and_issue_k0);
            } else {
                warpgroup_cooperative_qkt_gemm_high4_tail<T, false>(
                    sQ, cur_sK1_high4, cur_sK0, rP1,
                    rQ4, rQ5, rQ6, rQ8,
                    cur_phase_K1, idx_in_warpgroup,
                    phase6_wait_and_issue_k0);
            }
            even_high_in_scratch ^= 1u;
        } else {
            auto phase6_wait_and_issue_k0 = [&]() {
                hs64_wait_local_k1_high();
                constexpr int kNumTiles = T::kNumKTiles;
                issue_K_load_bf16<
                    T::kStage1KTiles, kNumTiles, NEXT_EXTRA, T>(
                    params, nxt_sK0, batch_idx, nxt_block0, seqlen_k,
                    &barriers_K0[1], idx_in_warpgroup, ori_block_max,
                    precomp_ptr1, precomp_valid1, pre_token_idx_b,
                    nxt_block0 + 2, end_block_idx, extra_seqlen_k);
            };
            warpgroup_cooperative_qkt_gemm<T, 6>(
                sQ, cur_sK0, cur_sK0, rP1, rQ8, rQ6, rQ4, rQlow0,
                barriers_K1, cur_phase_K1, idx_in_warpgroup,
                phase6_wait_and_issue_k0);
        }
    }

    // [2-buffer] buffer-free barrier (mirrors ref rO1sP0sV0RIssued): only after
    // BOTH warpgroups' remote PV is issued is any buffer free for reload.
    if constexpr (!T::kIsCrossCut || !T::kUseQkWeave || T::kKeepQkWeaveBar5) {
        cutlass::arch::NamedBarrier::sync(
            T::NUM_THREADS, static_cast<cutlass::arch::ReservedNamedBarriers>(5));
    }

    // [2-buffer] wg1 loads tiles 4-8 for nxt_block0 -> nxt_sK0 (= buf0)
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST &&
                  (!T::kIsCrossCut || !T::kUseQkWeave)) {
        constexpr int kNumTiles = T::kNumKTiles;

        issue_K_load_bf16<T::kStage1KTiles, kNumTiles, NEXT_EXTRA, T>(params, nxt_sK0, batch_idx, nxt_block0, seqlen_k, &barriers_K0[1],
            idx_in_warpgroup, ori_block_max, precomp_ptr1, precomp_valid1,
            pre_token_idx_b, nxt_block0 + 2, end_block_idx, extra_seqlen_k);

    }

    if constexpr (T::kIsCrossCut && T::kUseQkWeave &&
                  !IS_BLK0_LAST && !IS_BLK1_LAST) {
        warpgroup_cooperative_qkt_gemm<T, 4, false, T::kHasExtraKTile>(
            sQ, cur_sK0, cur_sK0, rP1, rQ8, rQ6, rQ4, rQlow0,
            barriers_K1, cur_phase_K1, idx_in_warpgroup);
    }


    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST
                  && (!T::kIsCrossCut || !T::kUseQkWeave)
    ) {
        cute::clear(rP1);
        warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK0, cur_sK0, rP1, rQ8, rQ6, rQ4, rQlow0, barriers_K1, cur_phase_K1, idx_in_warpgroup);
    }

    // Precompute the next addresses after QK<1>, once the current loads have
    // consumed the previous iteration's precomp_ptr.
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {

        if constexpr (T::kIsCrossCut && T::kUseQkWeave) {
            int next_nxt_block1 = block_idx + 5;
            if constexpr (T::kHasExtraKTile) {
                // This old validity slot has been consumed by this iteration's
                // softmax. The next odd epoch uses the other slot first.
                compute_K_addr_bf16<0, NEXT_EXTRA, T, true>(
                    params, batch_idx, next_nxt_block1, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, vi_softmax_wg1,
                    pre_token_idx, extra_seqlen_k, precomp_ptr0, precomp_valid0);
            } else {
                compute_K_addr_bf16<4, NEXT_EXTRA, T, true>(
                    params, batch_idx, next_nxt_block1, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    pre_token_idx, extra_seqlen_k, precomp_ptr0, precomp_valid0);
            }
        }
        int next_nxt_block0 = block_idx + 4;
        compute_K_addr_bf16<4, NEXT_EXTRA, T, true>(
            params, batch_idx, next_nxt_block0, seqlen_k,
            idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
            pre_token_idx_b, extra_seqlen_k, precomp_ptr1, precomp_valid1);

    }

    // [BlockM=64] Mirrors wg0's K-block boundary barrier: wg1's next-iteration
    // PV-remote must see the K data wg0 loaded in this iteration (sV1R).

    // [2-buffer] no rotation: buf0=even, buf1=odd fixed

}

// Both APIs express masking through their sparse indices. Prefill assigns
// one query per CTA; decode uses the split-KV scheduler described below.
template<typename T, bool ALLOW_EXTRA>
__forceinline__ __device__ void hs64_attention(
    const Flash_fwd_mla_params &params, float *max_logits_ptr = nullptr) {
    // grid shape: [
    // 	seqlen_q_ori,
    // 	ngroups / BLOCK_SIZE_M (decode requires one KV head),
    // 	num_sm_parts
    // ]
    // An "sm part" handles all BLOCK_SIZE_M q_heads of one m_block, under one
    // kv head, for the [start_block_idx, end_block_idx) segment of one request.
    // is_no_split means this request is exclusively ours, so write straight to
    // o_ptr / softmax_lse_ptr; otherwise write to the *accum buffers at split
    // idx (n_split_idx + num_splits_ptr[batch_idx]).

    const int m_block_idx = T::kIsPrefill ? 0 : blockIdx.x * gridDim.y + blockIdx.y;
    const int k_head_idx = 0;
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
    // Use a swizzled K layout compatible with TSM_LD_SWZL and SmemLayoutVDirect.
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_sK.data()), (typename T::SmemLayoutKDirect){});
    // Each score tile has a dedicated CUBE; no Q/P shared-memory aliasing.
    typename T::InputT *sP0_base = plan.smem_sP.data();
    typename T::InputT *sP1_base =
        sP0_base + T::BLOCK_SIZE_M * T::kBlockN;
    Tensor sP0 = make_tensor(make_smem_ptr(sP0_base), (typename T::SmemLayoutP0){});
    Tensor sP1 = make_tensor(make_smem_ptr(sP1_base), (typename T::SmemLayoutP0){});
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL_reduction_wksp = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<T::BLOCK_SIZE_M + 128>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    // Cross-cut (4,2) cross-N-warp reduction scratch (max/sum partials exchange).
    float* smem_cross_n_reduction = plan.smem_cross_n_reduction.data();
    // Valid indices mask: 4 alternating buffers, four token flags per word.
    // Cross-WG: WG0 uses bufs 0/1, WG1 uses bufs 2/3 (alternating preload/softmax)
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(plan.smem_valid_indices.data()),
        Shape<_4, Int<T::kValidWords>>{}, Stride<Int<T::kValidWords>, _1>{});
    char *sO_addr = (char *)plan.smem_sQ.data(); // Overlap with sK0 and sK1

    __mbarrier_t *barrier_Q = &(plan.barrier_Q);
    __mbarrier_t *barriers_K0 = plan.barriers_K0;
    __mbarrier_t *barriers_K1 = plan.barriers_K1;

    static_assert(T::SharedMemoryPlan::kNumKBarriers == 2,
                  "HS64 expects exactly 2 K sub-stage barriers");
    if (threadIdx.x == 0) {
        __mbarrier_init(barrier_Q, 32);
        CUTLASS_PRAGMA_UNROLL
        // Initialize two sub-stage barriers per block:
        //   barriers_Kx[0] → tiles 0-3 (dims 0-255) readiness
        //   barriers_Kx[1] → tiles 4-8 (dims 256-575) readiness
        // Count = 256: each barrier is loaded by exactly ONE warpgroup, so the
        // consumer reads data it wrote itself and there is no double-arrive.
        for (int i = 0; i < 2; ++i) {
            __mbarrier_init(&barriers_K0[i], 256);
            __mbarrier_init(&barriers_K1[i], 256);
        }
    }
    __syncthreads();  // ensure all threads see initialized mbarriers
    Hs64MbarPhase<T> cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;

    int *tile_scheduler_metadata_ptr = nullptr;
    int begin_idx, begin_seqlen, end_idx, end_seqlen;
    if constexpr (T::kIsPrefill) {
        begin_idx = end_idx = blockIdx.x;
        begin_seqlen = end_seqlen = 0;
    } else {
        tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
        // Keep this load ordered with the programmatic launch dependency.
        int4 tile_scheduler_metadata = *(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
        begin_idx = tile_scheduler_metadata.x;
        begin_seqlen = tile_scheduler_metadata.y;
        end_idx = tile_scheduler_metadata.z;
        end_seqlen = tile_scheduler_metadata.w;
    }

    if (begin_idx >= params.b)
        return;
    int begin_n_split_idx = 0;
    if constexpr (!T::kIsPrefill)
        begin_n_split_idx = *(tile_scheduler_metadata_ptr + 4);

    // Copy the first Q
    launch_q_copy<T>(params, begin_idx, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);

#pragma unroll 1
#pragma clang loop licm(disable)
    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
        constexpr int kBlockN = T::kBlockN;
        const int n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
        int seqlen_k;
        if (params.topk_len_ptr) {
            seqlen_k = __ldg(params.topk_len_ptr + batch_idx);
        } else {
            seqlen_k = params.topk;
        }
        int seqlen_kpad = max(seqlen_k, 1);
        // Always round main cache up to kBlockN*2 (128) so that:
        //   - the main/extra cache boundary is on an even block, keeping
        //     the mainloop's main/extra switch simple;
        //   - it matches the metadata kernel's block_size_n=128 tiling, so
        //     is_no_split (start==0 && end==total/kBlockN) stays accurate
        //     without an extra __ldg on num_splits_ptr.
        seqlen_kpad = cute::round_up(seqlen_kpad, kBlockN * 2);
        int extra_seqlen_k = 0;
        int ori_block_max = -1;
        if constexpr (ALLOW_EXTRA) {
            if (params.extra_topk >= 0) {
                if (params.extra_topk_len_ptr) {
                    extra_seqlen_k = __ldg(params.extra_topk_len_ptr + batch_idx);
                } else {
                    extra_seqlen_k = params.extra_topk;
                }
                ori_block_max = cute::ceil_div(seqlen_kpad, kBlockN);
            }
        }
        const int total_k = seqlen_kpad + cute::round_up(extra_seqlen_k, kBlockN * 2);
        const int start_block_idx = batch_idx == begin_idx ? begin_seqlen / kBlockN : 0;
        int end_block_idx = batch_idx == end_idx
            ? cute::round_up(end_seqlen, kBlockN * 2) / kBlockN
            : total_k / kBlockN;
        if constexpr (T::kIsPrefill) end_block_idx = total_k / kBlockN;
        const bool is_no_split = T::kIsPrefill ||
            (start_block_idx == 0 && end_block_idx == (total_k / kBlockN));


        // Each physical copy wave has one prefetch slot. XOR4 uses slot zero for
        // its interleaved one-pass metadata and leaves slot one unused.
        // pre_token_idx0: for block0/nxt_block0 (even block)
        // pre_token_idx1: for block1/nxt_block1 (odd block)
        // Prolog initializes them, and each load call updates its own idx.
        int pre_token_idx0[T::kGmemPasses];   // for block0/nxt_block0 (even block)
        int pre_token_idx1[T::kGmemPasses];   // for block1/nxt_block1 (odd block)
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < T::kGmemPasses; ++p) {
            pre_token_idx0[p] = -1;
            pre_token_idx1[p] = -1;
        }

        // Explicit __ldg pre-fetch for the FIRST round (block 0 for wg0, block 1 for wg1).
        // Issue this __ldg before the initial K load so tensor setup and
        // cp.async can hide its latency.

        // Prefetch this thread's first-round token metadata.
        auto prefetch_tok = [&](int _blk, int (&_out)[T::kGmemPasses]) {
            if (_blk < 0 || _blk >= end_block_idx) {
                CUTLASS_PRAGMA_UNROLL
                for (int p = 0; p < T::kGmemPasses; ++p) _out[p] = -1;
                return;
            }
            int _eff;
            int _index_count;
            const int *_base;
            if constexpr (!ALLOW_EXTRA) {
                _eff = _blk;
                _index_count = params.topk;
                _base = hs64_query_indices<false>(params)
                    + static_cast<int64_t>(batch_idx) *
                        params.indices_batch_stride
                    + _eff * T::kBlockN;
            } else {
                bool _use_extra =
                    (ori_block_max >= 0) && (_blk >= ori_block_max);
                _eff = _use_extra ? (_blk - ori_block_max) : _blk;
                _index_count = _use_extra ? params.extra_topk : params.topk;
                _base = _use_extra
                    ? (hs64_query_indices<true>(params)
                       + static_cast<int64_t>(batch_idx) *
                           params.extra_indices_batch_stride
                       + _eff * T::kBlockN)
                    : (hs64_query_indices<false>(params)
                       + static_cast<int64_t>(batch_idx) *
                           params.indices_batch_stride
                       + _eff * T::kBlockN);
            }
            static_assert(T::kBlockN == 64 && T::kGmemPasses == 2,
                          "WI_V2 XOR4 one-pass metadata requires M64");
            const int _lane = idx_in_warpgroup & 31;
            const int _warp = idx_in_warpgroup >> 5;
            const int _pair = _lane >> 3;
            const int _pass = (_lane >> 2) & 1;
            if constexpr (T::kGuardIndices) {
                const int _token = _warp * 4 + _pair + _pass * 32;
                _out[0] = _eff * kBlockN + _token < _index_count
                    ? __ldg(_base + _token) : -1;
            } else {
                _out[0] = __ldg(_base + _warp * 4 + _pair + _pass * 32);
            }
            _out[1] = -1;
        };
        if (warpgroup_idx == 0) {
            // WG0: prolog loads block0 (uses idx0), main loop first loads nxt_block0=block2, nxt_block1=block3
            // idx0: prefetch block0 (prolog), will be updated to block2 by prolog load
            // idx1: prefetch block3 (first main loop nxt_block1)
            prefetch_tok(start_block_idx, pre_token_idx0);
            prefetch_tok(start_block_idx + 3, pre_token_idx1);
        } else {
            // WG1: prolog loads block1 (uses idx1), main loop first loads nxt_block1=block3, nxt_block0=block2
            // idx1: prefetch block1 (prolog), will be updated to block3 by prolog load
            // idx0: prefetch block2 (first main loop nxt_block0)
            prefetch_tok(start_block_idx + 1, pre_token_idx1);
            prefetch_tok(start_block_idx + 2, pre_token_idx0);
        }


        // Sparse decode treats
        // every kv token in [0, seqlen_k) as candidate; mask is driven by
        // smem_valid_indices in wg{0,1}_bunch_0.

        Tensor cur_sK0 = sK(_, _, 0);
        Tensor cur_sK1 = sK(_, _, 1);
        Tensor nxt_sK0 = sK(_, _, 0);

        // Cross-WG prolog: each block's K split across two sK buffers.
        // Block 0: tiles 0-3 → buf 0 (cur_sK0), tiles 4-8 → buf 1 (cur_sK1)
        // Block 1: tiles 0-3 → buf 1 (cur_sK1), tiles 4-8 → buf 0 (cur_sK0)
        // smem_valid_indices uses 4 bufs: WG0 bufs 0/1, WG1 bufs 2/3 (alternating)
        // Prolog blocks become "current" in first iteration, so use softmax bufs.
        int prolog_vi_wg0 = (start_block_idx / 2) % 2;       // 0 or 1
        int prolog_vi_wg1 = 2 + (start_block_idx / 2) % 2;    // 2 or 3
        constexpr int kNumTiles = T::kNumKTiles;
        {
            if (warpgroup_idx == 0) {
                if (start_block_idx < end_block_idx) {
                    // wg0 loads block 0: stage 1 (tiles 0-3 → buf 0) + stage 2 (tiles 4-8 → buf 1)

                    auto vi_wg0 = smem_valid_indices(prolog_vi_wg0, _);

                    if constexpr (!ALLOW_EXTRA) {
                        load_K_tiles_bf16<
                            T::kStage1KTiles, 8, false, T, false, T::kUseEvenHighBank>(
                            params, cur_sK0, batch_idx, start_block_idx,
                            seqlen_k, &barriers_K0[1], idx_in_warpgroup,
                            ori_block_max, smem_valid_indices, prolog_vi_wg0,
                            pre_token_idx0, start_block_idx + 2,
                            end_block_idx, extra_seqlen_k);
                        load_K_tiles_bf16<
                            0, T::kStage1KTiles, false, T, true, true>(
                            params, cur_sK0, batch_idx, start_block_idx,
                            seqlen_k, &barriers_K0[0], idx_in_warpgroup,
                            ori_block_max, smem_valid_indices, prolog_vi_wg0,
                            pre_token_idx0, start_block_idx + 2,
                            end_block_idx, extra_seqlen_k);
                    } else {
                        load_K_tiles_bf16<
                            T::kStage1KTiles, 8, true, T, false, T::kUseEvenHighBank>(
                            params, cur_sK0, batch_idx, start_block_idx,
                            seqlen_k, &barriers_K0[1], idx_in_warpgroup,
                            ori_block_max, smem_valid_indices, prolog_vi_wg0,
                            pre_token_idx0, start_block_idx + 2,
                            end_block_idx, extra_seqlen_k);
                        load_K_tiles_bf16<0, T::kStage1KTiles, true, T, true, true>(
                            params, cur_sK0, batch_idx, start_block_idx,
                            seqlen_k, &barriers_K0[0], idx_in_warpgroup,
                            ori_block_max, smem_valid_indices, prolog_vi_wg0,
                            pre_token_idx0, start_block_idx + 2,
                            end_block_idx, extra_seqlen_k);
                    }

                }
            } else {
                if (start_block_idx+1 < end_block_idx) {
                    // wg1 loads block 1: stage 1 (tiles 0-3 → buf 1) + stage 2 (tiles 4-8 → buf 0)

                    auto vi_wg1 = smem_valid_indices(prolog_vi_wg1, _);

                    if constexpr (!ALLOW_EXTRA) {
                        load_K_tiles_bf16<
                            T::kStage1KTiles, kNumTiles, false, T, false, true>(
                            params, cur_sK1, batch_idx, start_block_idx + 1,
                            seqlen_k, &barriers_K1[1], idx_in_warpgroup,
                            ori_block_max, smem_valid_indices, prolog_vi_wg1,
                            pre_token_idx1, start_block_idx + 3,
                            end_block_idx, extra_seqlen_k);
                        load_K_tiles_bf16<
                            0, T::kStage1KTiles, false, T>(
                            params, cur_sK1, batch_idx, start_block_idx + 1,
                            seqlen_k, &barriers_K1[0], idx_in_warpgroup,
                            ori_block_max, smem_valid_indices, prolog_vi_wg1,
                            pre_token_idx1, start_block_idx + 3,
                            end_block_idx, extra_seqlen_k);
                    } else {
                        load_K_tiles_bf16<
                            T::kStage1KTiles, kNumTiles, true, T, false, true>(
                            params, cur_sK1, batch_idx, start_block_idx + 1,
                            seqlen_k, &barriers_K1[1], idx_in_warpgroup,
                            ori_block_max, smem_valid_indices, prolog_vi_wg1,
                            pre_token_idx1, start_block_idx + 3,
                            end_block_idx, extra_seqlen_k);
                        load_K_tiles_bf16<0, T::kStage1KTiles, true, T>(
                            params, cur_sK1, batch_idx, start_block_idx + 1,
                            seqlen_k, &barriers_K1[0], idx_in_warpgroup,
                            ori_block_max, smem_valid_indices, prolog_vi_wg1,
                            pre_token_idx1, start_block_idx + 3,
                            end_block_idx, extra_seqlen_k);
                    }

                }
            }
        }

        // Initialize the CTA-owned softmax max state before the existing prolog
        // barrier so both warpgroups observe it before entering the mainloop.
        if (threadIdx.x < size(sM)) {
            sM[threadIdx.x] = MAX_INIT_VAL_SM;
        }

        // CTA-wide barrier after prolog loads and sM initialization.
        // Each warpgroup's prolog K load uses a warpgroup-internal sync: the
        // prolog K load only did a 256-thread bar.sync, which
        // covers intra-wg visibility. The remote PV GEMM below reads V from the
        // OTHER wg's K buffer, and on PPU only barrier 0 makes that visible
        // (__threadfence_block is a no-op), so skipping it races.
        __syncthreads();

        Tensor rO = partition_fragment_C((typename T::TiledMmaPV){}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV / 2>>{});	// ((2, 2, 32), 1, 1)
        float rL[2];
        rL[0] = rL[1] = 0.0f;

        // Clear buffers
        cute::fill(rO, 0.);
        while(!cutlass::arch::test_wait(barrier_Q, cur_phase_Q, 1)) {
            kernel_sleep_ns();
        }
        if constexpr (T::kHasExtraKTile) {
            cur_phase_Q = (cur_phase_Q + 1) & 1;
        } else {
            cur_phase_Q ^= 1u;
        }


        // rQ8 holds the cached last Q tile. It must carry the TiledMma's
        // own A-fragment layout -- (8,1) gives 32 elements per thread, (4,2) only
        // 16 -- so allocate it from partition_fragment_A rather than hard-coding
        // the (8,1) shape.
        //
        // D512 caches tile7 here. D576 deliberately leaves Q8 in sQ because
        // its raw Q4..Q7 cubes are the alternate even-high K/V bank.
        typename T::TiledMma tiled_mma_rQ8;
        auto thr_mma_rQ8 = tiled_mma_rQ8.get_thread_slice(idx_in_warpgroup % T::kMmaThreads);
        auto make_rQ8 = [&]() {
            if constexpr (T::kCacheLastQTile) {
                return thr_mma_rQ8.partition_fragment_A(
                    local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                               Coord<_0, Int<T::kCachedQTile>>{}));
            } else {
                return thr_mma_rQ8.partition_fragment_A(
                    local_tile(sQ, Shape<Int<T::kBlockM>, _64>{}, Coord<_0, _0>{}));
            }
        };
        Tensor rQ8 = make_rQ8();
        // Load once per batch and reuse it across every QK in both warpgroups.
        if constexpr (T::kCacheLastQTile) {
            retrieve_rP_from_sP<T>(rQ8,
                local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                           Coord<_0, Int<T::kCachedQTile>>{}),
                idx_in_warpgroup);
        } else {
            cute::clear(rQ8);
        }

        // The active d512 path also keeps tile6 resident. The placeholder tensor
        // is never read on other paths, so it should compile away there.
        Tensor rQ6 = thr_mma_rQ8.partition_fragment_A(
            local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                       Coord<_0, Int<T::kCachedPrevQTile>>{}));
        if constexpr (T::kCachePrevQTile) {
            retrieve_rP_from_sP<T>(rQ6,
                local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                           Coord<_0, Int<T::kCachedPrevQTile>>{}),
                idx_in_warpgroup);
        }

        // Tile4 is the first high-half Q tile consumed after its K wait.
        Tensor rQ4 = thr_mma_rQ8.partition_fragment_A(
            local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                       Coord<_0, Int<T::kCachedFirstHighQTile>>{}));
        if constexpr (T::kCacheFirstHighQTile) {
            retrieve_rP_from_sP<T>(rQ4,
                local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                           Coord<_0, Int<T::kCachedFirstHighQTile>>{}),
                idx_in_warpgroup);
        }

        // D576 preserves all Q operands whose raw cubes become the alternate
        // high bank. These tensors compile away for D512, which retains its
        // existing transient Q5/shared-Q6 path.
        Tensor rQ5 = thr_mma_rQ8.partition_fragment_A(
            local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                       Coord<_0, Int<5>>{}));
        Tensor rQmid6 = thr_mma_rQ8.partition_fragment_A(
            local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                       Coord<_0, Int<6>>{}));
        if constexpr (T::kCacheD576MiddleQTiles) {
            retrieve_rP_from_sP<T>(
                rQ5,
                local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                           Coord<_0, Int<5>>{}),
                idx_in_warpgroup);
            retrieve_rP_from_sP<T>(
                rQmid6,
                local_tile(sQ, Shape<Int<T::kBlockM>, _64>{},
                           Coord<_0, Int<6>>{}),
                idx_in_warpgroup);
        }

        // SM80 reuses Q0 across KV blocks rather than issuing four TSM loads
        // in every low-half QK. Other architectures eliminate this fragment.
        Tensor rQlow0 = thr_mma_rQ8.partition_fragment_A(
            local_tile(sQ, Shape<Int<T::kBlockM>, _64>{}, Coord<_0, _0>{}));
        if constexpr (T::kArch == 80) {
            retrieve_rP_from_sP<T>(rQlow0,
                local_tile(sQ, Shape<Int<T::kBlockM>, _64>{}, Coord<_0, _0>{}),
                idx_in_warpgroup);
        }

        if (warpgroup_idx == 0) {
            // Warpgroup 0
            Tensor rP0 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
            const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

            // Issue P0 = Q @ K0^T, wait
            // Guard on block range, not seqlen_k.  When
            // seqlen_k=0 but extra blocks exist, block 0 is a valid extra
            // block that must participate in QK GEMM.
            if (start_block_idx < end_block_idx) {

                cute::clear(rP0);

                warpgroup_cooperative_qkt_gemm<T, 1, false, true>(sQ, cur_sK0, cur_sK0, rP0, rQ8, rQ6, rQ4, rQlow0, barriers_K0, cur_phase_K0, idx_in_warpgroup);

            }

            // Precomputed K addresses for SIMT/TC overlap
            InputT* precomp_ptr0[T::kGmemPasses] = {};
            InputT* precomp_ptr1[T::kGmemPasses] = {};
            bool precomp_valid0[T::kGmemPasses] = {};
            bool precomp_valid1[T::kGmemPasses] = {};

            // Precompute initial addresses for first iteration

            int first_vi_preload_wg0 = 1 - (start_block_idx / 2) % 2;
            int first_vi_preload_wg1 = 3 - (start_block_idx / 2) % 2;
            if constexpr (!ALLOW_EXTRA) {
                compute_K_addr_bf16<0, false, T>(
                    params, batch_idx, start_block_idx + 2, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices,
                    first_vi_preload_wg0, pre_token_idx0, extra_seqlen_k,
                    precomp_ptr0, precomp_valid0);
                compute_K_addr_bf16<0, false, T>(
                    params, batch_idx, start_block_idx + 3, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices,
                    first_vi_preload_wg1, pre_token_idx1, extra_seqlen_k,
                    precomp_ptr1, precomp_valid1);
            } else {
                bool init_load_extra =
                    (ori_block_max >= 0) &&
                    (start_block_idx + 2 >= ori_block_max);
                compute_K_addr_bf16_dynamic<0, T>(
                    params, batch_idx, start_block_idx + 2, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices,
                    first_vi_preload_wg0, pre_token_idx0, extra_seqlen_k,
                    precomp_ptr0, precomp_valid0, init_load_extra);
                compute_K_addr_bf16_dynamic<0, T>(
                    params, batch_idx, start_block_idx + 3, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices,
                    first_vi_preload_wg1, pre_token_idx1, extra_seqlen_k,
                    precomp_ptr1, precomp_valid1, init_load_extra);
            }

            // The prolog even block always resides in buf0. Toggle only after a
            // normal iteration has produced and consumed a real next-even high.
            uint32_t even_high_in_scratch = 0;

            #define LAUNCH_WG0_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA)    \
                wg0_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA>(                \
                sQ, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, rQ6, rQ4, rQ5, rQmid6, rQlow0, \
                rP0, rO, rL,                                     \
                barriers_K0, barriers_K1, cur_phase_K0, even_high_in_scratch, params, \
                seqlen_k, block_idx, end_block_idx, idx_in_warpgroup, wg_idx, \
                batch_idx, ori_block_max, smem_valid_indices,                         \
                pre_token_idx0,                                \
                pre_token_idx1,                                \
                extra_seqlen_k,                                                      \
                precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1, smem_cross_n_reduction);          \

            int block_idx = start_block_idx;
            if constexpr (!ALLOW_EXTRA) {
                #pragma unroll 1
                for (; block_idx < end_block_idx - 2; block_idx += 2) {
                    LAUNCH_WG0_SUBROUTINE(false, false, false);
                }
            } else {
                // 3-phase loop: NEXT_EXTRA controls prefetch/compute cache.
                int p1_end = min(
                    ori_block_max >= 0 ? ori_block_max - 4
                                       : end_block_idx - 2,
                    end_block_idx - 2);
                #pragma unroll 1
                for (; block_idx < p1_end; block_idx += 2) {
                    LAUNCH_WG0_SUBROUTINE(false, false, false);
                }
                #pragma unroll 1
                for (; block_idx < end_block_idx - 2; block_idx += 2) {
                    LAUNCH_WG0_SUBROUTINE(false, false, true);
                }
            }
            // Phase 3: last 2 blocks (no K load/prefetch/compute_addr)
            LAUNCH_WG0_SUBROUTINE(false, true, false);
        } else {
            // Warpgroup 1
            Tensor rP1 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
            const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

            if (start_block_idx+1 < end_block_idx) {

                warpgroup_cooperative_qkt_gemm<T, 1, true>(sQ, cur_sK1, cur_sK1, rP1, rQ8, rQ6, rQ4, rQlow0, barriers_K1, cur_phase_K1, idx_in_warpgroup);

            } else {
                // When wg1 has no initial K block, rP1 must be zero-
                // initialized.  Otherwise wg1_subroutine runs softmax on garbage
                // registers, producing corrupt rL that contaminates the final output
                // via the cross-warpgroup rL reduction.
                cute::clear(rP1);
            }

            // Precomputed K addresses for SIMT/TC overlap (WG1)
            InputT* precomp_ptr0_wg1[T::kGmemPasses] = {};
            InputT* precomp_ptr1_wg1[T::kGmemPasses] = {};
            bool precomp_valid0_wg1[T::kGmemPasses] = {};
            bool precomp_valid1_wg1[T::kGmemPasses] = {};

            // Precompute initial addresses for first iteration

            if constexpr (!ALLOW_EXTRA) {
                compute_K_addr_bf16<4, false, T>(
                    params, batch_idx, start_block_idx + 3, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    pre_token_idx1, extra_seqlen_k, precomp_ptr0_wg1,
                    precomp_valid0_wg1);
                compute_K_addr_bf16<4, false, T>(
                    params, batch_idx, start_block_idx + 2, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    pre_token_idx0, extra_seqlen_k, precomp_ptr1_wg1,
                    precomp_valid1_wg1);
            } else {
                bool init_load_extra =
                    (ori_block_max >= 0) &&
                    (start_block_idx + 3 >= ori_block_max);
                compute_K_addr_bf16_dynamic<4, T>(
                    params, batch_idx, start_block_idx + 3, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    pre_token_idx1, extra_seqlen_k, precomp_ptr0_wg1,
                    precomp_valid0_wg1, init_load_extra);
                compute_K_addr_bf16_dynamic<4, T>(
                    params, batch_idx, start_block_idx + 2, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    pre_token_idx0, extra_seqlen_k, precomp_ptr1_wg1,
                    precomp_valid1_wg1, init_load_extra);
            }

            // Kept independently in each warpgroup; identical loop control and
            // normal-iteration-only toggles keep both uniform states aligned.
            uint32_t even_high_in_scratch = 0;

            #define LAUNCH_WG1_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA)  \
                wg1_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA>(          \
                sQ, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, rQ6, rQ4, rQ5, rQmid6, rQlow0, \
                rP1, rO, rL,                                     \
                barriers_K0, barriers_K1, cur_phase_K1, even_high_in_scratch, params, \
                seqlen_k, block_idx, end_block_idx, idx_in_warpgroup, wg_idx, \
                batch_idx, ori_block_max, smem_valid_indices,                         \
                pre_token_idx1,                                \
                pre_token_idx0,                                \
                extra_seqlen_k,                                                      \
                precomp_ptr0_wg1, precomp_ptr1_wg1, precomp_valid0_wg1, precomp_valid1_wg1, smem_cross_n_reduction);  \

            int block_idx = start_block_idx;
            if constexpr (!ALLOW_EXTRA) {
                #pragma unroll 1
                for (; block_idx < end_block_idx - 2; block_idx += 2) {
                    LAUNCH_WG1_SUBROUTINE(false, false, false);
                }
            } else {
                // 3-phase loop: same phase boundaries as WG0.
                int p1_end = min(
                    ori_block_max >= 0 ? ori_block_max - 4
                                       : end_block_idx - 2,
                    end_block_idx - 2);
                #pragma unroll 1
                for (; block_idx < p1_end; block_idx += 2) {
                    LAUNCH_WG1_SUBROUTINE(false, false, false);
                }
                #pragma unroll 1
                for (; block_idx < end_block_idx - 2; block_idx += 2) {
                    LAUNCH_WG1_SUBROUTINE(false, false, true);
                }
            }
            // Phase 3: last 2 blocks
            LAUNCH_WG1_SUBROUTINE(false, true, false);
        }

        // Reduce rL across threads within the same warp
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

        int my_row = get_AorC_row_idx<T::kAtomLayoutM>(0, idx_in_warpgroup);

        // Cross-N rL reduction: (4,2) splits the N columns across two N-warps,
        // so each row's rL is the sum of two partials.
        if constexpr (T::kIsCrossCut) {
            int warp_n_idx = (idx_in_warpgroup / 32) / T::kAtomLayoutM;
            int wg_idx_cn = (int)(threadIdx.x >> 8);
            if constexpr (T::kUsePv2x4) {
                // Each N-warp publishes to its own slot. One pair barrier is
                // sufficient before both sides read and add the partner value.
                int own_slot = (wg_idx_cn * 2 + warp_n_idx) * T::kBlockM + my_row;
                if (idx_in_warpgroup % 4 == 0) {
                    smem_cross_n_reduction[own_slot] = rL[0];
                    smem_cross_n_reduction[own_slot + 8] = rL[1];
                }
                {
                    int _wm = (idx_in_warpgroup / 32) % T::kAtomLayoutM;
                    int _bar_id = 7 + (int)(threadIdx.x >> 8) * 4 + _wm;
                    cutlass::arch::NamedBarrier::sync(
                        64, static_cast<cutlass::arch::ReservedNamedBarriers>(_bar_id));
                }
                if (idx_in_warpgroup % 4 == 0) {
                    int peer_slot = (wg_idx_cn * 2 + (1 - warp_n_idx)) * T::kBlockM + my_row;
                    rL[0] += smem_cross_n_reduction[peer_slot];
                    rL[1] += smem_cross_n_reduction[peer_slot + 8];
                }
            } else
            {
                // Generic path: two addends accumulate into one slot.
                if (idx_in_warpgroup % 4 == 0 && warp_n_idx == 0) {
                    smem_cross_n_reduction[wg_idx_cn * T::kBlockM + my_row] = 0.f;
                    smem_cross_n_reduction[wg_idx_cn * T::kBlockM + my_row + 8] = 0.f;
                }
                {
                    int _wm = (idx_in_warpgroup / 32) % T::kAtomLayoutM;
                    int _bar_id = 7 + (int)(threadIdx.x >> 8) * 4 + _wm;
                    cutlass::arch::NamedBarrier::sync(
                        64, static_cast<cutlass::arch::ReservedNamedBarriers>(_bar_id));
                }
                if (idx_in_warpgroup % 4 == 0) {
                    atomicAdd(&smem_cross_n_reduction[wg_idx_cn * T::kBlockM + my_row], rL[0]);
                    atomicAdd(&smem_cross_n_reduction[wg_idx_cn * T::kBlockM + my_row + 8], rL[1]);
                }
                {
                    int _wm = (idx_in_warpgroup / 32) % T::kAtomLayoutM;
                    int _bar_id = 7 + (int)(threadIdx.x >> 8) * 4 + _wm;
                    cutlass::arch::NamedBarrier::sync(
                        64, static_cast<cutlass::arch::ReservedNamedBarriers>(_bar_id));
                }
                if (idx_in_warpgroup % 4 == 0) {
                    rL[0] = smem_cross_n_reduction[wg_idx_cn * T::kBlockM + my_row];
                    rL[1] = smem_cross_n_reduction[wg_idx_cn * T::kBlockM + my_row + 8];
                }
            }
            {
                int lane = threadIdx.x & 31;
                rL[0] = __shfl_sync(0xffffffff, rL[0], (lane / 4) * 4);
                rL[1] = __shfl_sync(0xffffffff, rL[1], (lane / 4) * 4);
            }
        }

        // Reduce rL across warpgroups. Pre-issue sink loads before the CTA sync.
        float pre_sink0 = 0.0f, pre_sink1 = 0.0f;
        float pre_sink_pv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (is_no_split && params.attn_sink_ptr != nullptr) {
            if constexpr (T::kUsePv2x4) {
                CUTLASS_PRAGMA_UNROLL
                for (int mi = 0; mi < 4; ++mi) {
                    int row = get_PV_row_idx<T>(mi, idx_in_warpgroup);
                    int q_head_idx =
                        (m_block_idx * T::BLOCK_SIZE_M + row) % params.ngroups;
                    pre_sink_pv[mi] = __ldg(params.attn_sink_ptr + q_head_idx);
                }
            } else {
                const int row0 = my_row;
                const int row1 = my_row + 8;
                const int q_head_idx_0 = (m_block_idx * T::BLOCK_SIZE_M + row0) % params.ngroups;
                const int q_head_idx_1 = (m_block_idx * T::BLOCK_SIZE_M + row1) % params.ngroups;
                pre_sink0 = __ldg(params.attn_sink_ptr + q_head_idx_0);
                pre_sink1 = __ldg(params.attn_sink_ptr + q_head_idx_1);
            }
        }
        if (idx_in_warpgroup % 4 == 0 && (idx_in_warpgroup / 32) < T::kAtomLayoutM) {
            sL_reduction_wksp[my_row + warpgroup_idx * 128] = rL[0];
            sL_reduction_wksp[my_row + 8 + warpgroup_idx * 128] = rL[1];
        }
        __syncthreads();

        // Reduce rL across warpgroups.
        if constexpr (T::kIsCrossCut && T::kUsePv2x4 && T::kBlockM == 64) {
            if (warpgroup_idx == 0) {
                rL[0] += sL_reduction_wksp[my_row + 128];
                rL[1] += sL_reduction_wksp[my_row + 8 + 128];
                if (is_no_split && idx_in_warpgroup % 4 == 0 &&
                    (idx_in_warpgroup / 32) < T::kAtomLayoutM) {
                    sL_reduction_wksp[T::kBlockM + my_row] = rL[0];
                    sL_reduction_wksp[T::kBlockM + my_row + 8] = rL[1];
                }
            } else {
                int _bar_id = 8 + (int)(threadIdx.x >> 8);
                cutlass::arch::NamedBarrier::sync(
                    256, static_cast<cutlass::arch::ReservedNamedBarriers>(_bar_id));
                rL[0] = sL_reduction_wksp[my_row] + rL[0];
                rL[1] = sL_reduction_wksp[my_row + 8] + rL[1];
            }
        } else
        {
            if (warpgroup_idx == 0) {
                rL[0] += sL_reduction_wksp[my_row + 128];
                rL[1] += sL_reduction_wksp[my_row + 8 + 128];
            } else {
                if (idx_in_warpgroup % 4 == 0 && (idx_in_warpgroup / 32) < T::kAtomLayoutM) {
                    sL_reduction_wksp[my_row] += rL[0];
                    sL_reduction_wksp[my_row + 8] += rL[1];
                }
                {
                    int _bar_id = 8 + (int)(threadIdx.x >> 8);
                    cutlass::arch::NamedBarrier::sync(
                        256, static_cast<cutlass::arch::ReservedNamedBarriers>(_bar_id));
                }
                rL[0] = sL_reduction_wksp[my_row];
                rL[1] = sL_reduction_wksp[my_row + 8];
            }
        }

        float rL_pv[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        bool target_pv_empty[4] = {false, false, false, false};
        if constexpr (T::kUsePv2x4) {
            const int pv_row0 = get_PV_row_idx<T>(0, idx_in_warpgroup);
            const bool owns_low_repeat =
                ((idx_in_warpgroup / 32) % T::kAtomLayoutM) < T::kPvAtomLayoutM;
            if (owns_low_repeat) {
                rL_pv[0] = rL[0];
                rL_pv[1] = rL[1];
                rL_pv[2] = sL_reduction_wksp[pv_row0 + 32]
                          + sL_reduction_wksp[pv_row0 + 32 + 128];
                rL_pv[3] = sL_reduction_wksp[pv_row0 + 40]
                          + sL_reduction_wksp[pv_row0 + 40 + 128];
            } else {
                rL_pv[0] = sL_reduction_wksp[pv_row0]
                          + sL_reduction_wksp[pv_row0 + 128];
                rL_pv[1] = sL_reduction_wksp[pv_row0 + 8]
                          + sL_reduction_wksp[pv_row0 + 8 + 128];
                rL_pv[2] = rL[0];
                rL_pv[3] = rL[1];
            }
        }

        // Prune out when rL is 0.0f or NaN
        bool target_row0_empty = false, target_row1_empty = false;
        if constexpr (T::kUsePv2x4) {
            target_row0_empty = (rL[0] == 0.0f || rL[0] != rL[0]);
            target_row1_empty = (rL[1] == 0.0f || rL[1] != rL[1]);
            rL[0] = target_row0_empty ? 1.0f : rL[0];
            rL[1] = target_row1_empty ? 1.0f : rL[1];
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < 4; ++mi) {
                target_pv_empty[mi] =
                    (rL_pv[mi] == 0.0f || rL_pv[mi] != rL_pv[mi]);
                rL_pv[mi] = target_pv_empty[mi] ? 1.0f : rL_pv[mi];
            }
        } else
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 2; ++i)
                rL[i] = (rL[i] == 0.0f || rL[i] != rL[i]) ? 1.0f : rL[i];
        }

        // The generic cross-WG reduction publishes its final sums through
        // sL_reduction_wksp. Make those writes visible before the split
        // epilogue reads the workspace from arbitrary CTA threads. The D512
        // PV2x4 specialization keeps the final sums in WG0 registers instead.
        if constexpr (T::kBlockM != 64 || !T::kUsePv2x4) {
            __syncthreads();
        }

        // Epilogue
        int num_valid_seq_q = min(params.seqlen_q - m_block_idx * T::BLOCK_SIZE_M, T::BLOCK_SIZE_M);
        if (is_no_split) {
            InputT *o_ptr = (InputT *)params.o_ptr + batch_idx * params.o_batch_stride + m_block_idx * T::BLOCK_SIZE_M * params.o_row_stride + k_head_idx * params.o_head_stride; // (BLOCK_SIZE_M, HEAD_DIM_V) : (params.o_row_stride, 1)
            float *softmax_lse_ptr = (float *)params.softmax_lse_ptr + (batch_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M;                     // (BLOCK_SIZE_M) : (1)

            Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                                                              Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV>>{},
                                                              make_stride(params.o_row_stride, _1{})));
            Tensor gSoftmaxLse = make_tensor(make_gmem_ptr(softmax_lse_ptr), Layout<
                                                                                 Shape<Int<T::BLOCK_SIZE_M>>,
                                                                                 Stride<_1>>{});

            // attn_sink: SM90-style combined scale (don't modify rL, preserve rO/rL error cancellation)
            if (params.attn_sink_ptr != nullptr) {
                if constexpr (T::kUsePv2x4) {
                    Tensor rO_rowcol = make_tensor(
                        rO.data(), hs64_convert_layout_acc_rowcol<T>(rO.layout()));
                    static_assert(decltype(size<0>(rO_rowcol))::value == 4);
                    static_assert(decltype(size<1>(rO_rowcol))::value == 16);
                    CUTLASS_PRAGMA_UNROLL
                    for (int mi = 0; mi < 4; ++mi) {
                        int row = get_PV_row_idx<T>(mi, idx_in_warpgroup);
                        float sink_exp =
                            expf(pre_sink_pv[mi] - sM(row) * (float)M_LN2);
                        float scale = target_pv_empty[mi]
                            ? 0.0f
                            : __fdividef(1.0f, rL_pv[mi] + sink_exp);
                        CUTLASS_PRAGMA_UNROLL
                        for (int ni = 0; ni < 16; ++ni) {
                            rO_rowcol(mi, ni) = target_pv_empty[mi]
                                ? 0.0f
                                : rO_rowcol(mi, ni) * scale;
                        }
                        rL_pv[mi] = 1.0f;
                    }
                } else {
                    const int row0 = my_row;
                    const int row1 = my_row + 8;
                    // Use pre-issued __ldg values (latency hidden by __syncthreads above)
                    const float sink_exp0 = expf(pre_sink0 - sM(row0) * (float)M_LN2);
                    const float sink_exp1 = expf(pre_sink1 - sM(row1) * (float)M_LN2);
                    bool row0_empty, row1_empty;
                    if constexpr (T::kIsCrossCut && T::kUsePv2x4 && T::kBlockM == 64) {
                        row0_empty = target_row0_empty;
                        row1_empty = target_row1_empty;
                    } else
                    {
                        // Read original pre-prune rL to detect empty attention.
                        const float orig_rL0 = sL_reduction_wksp[my_row];
                        const float orig_rL1 = sL_reduction_wksp[my_row + 8];
                        row0_empty = (orig_rL0 == 0.0f || orig_rL0 != orig_rL0);
                        row1_empty = (orig_rL1 == 0.0f || orig_rL1 != orig_rL1);
                    }
                    const float o_scale0 = row0_empty ? 0.0f : __fdividef(1.0f, rL[0] + sink_exp0);
                    const float o_scale1 = row1_empty ? 0.0f : __fdividef(1.0f, rL[1] + sink_exp1);
                    // Apply combined scale to rO (replaces rL division in store_o).
                    // When a row is empty rO may hold NaN, so zero it explicitly.
                    CUTLASS_PRAGMA_UNROLL
                    for (int idx = 0; idx < size(rO); ++idx) {
                        bool is_row0 = (idx % 4 < 2);
                        bool empty = is_row0 ? row0_empty : row1_empty;
                        float scale = is_row0 ? o_scale0 : o_scale1;
                        rO(idx) = empty ? 0.0f : (rO(idx) * scale);
                    }
                    // Set rL to 1.0 so store_o's division becomes a no-op
                    rL[0] = 1.0f;
                    rL[1] = 1.0f;
                }
            }

            // Prune NaN in rO (empty rows can leave NaN behind).
            CUTLASS_PRAGMA_UNROLL
            for (int idx = 0; idx < size(rO); ++idx) {
                if (rO(idx) != rO(idx)) rO(idx) = 0.0f;
            }

            store_o<T, true>(rO, gO, T::kUsePv2x4 ? rL_pv : rL,
                             sO_addr, params, batch_idx, k_head_idx,
                             m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            int i = threadIdx.x;
            if (i < num_valid_seq_q) {
                float cur_L;
                if constexpr (T::kIsCrossCut && T::kUsePv2x4 && T::kBlockM == 64)
                    cur_L = sL_reduction_wksp[T::kBlockM + i];
                else
                    cur_L = sL_reduction_wksp[i];
                float sM_val = sM(i);
                bool sM_nan = ((__float_as_uint(sM_val) & 0x7FFFFFFFu) > 0x7F800000u);
                gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L || sM_nan) ? INFINITY : logf(cur_L) + sM_val / (float)M_LOG2E;
                if constexpr (T::kIsPrefill) {
                    // Like LSE, max_logits excludes attn_sink. sM is in log2
                    // units; the prefill API returns natural-log scores.
                    max_logits_ptr[int64_t(batch_idx) * params.seqlen_q + i] =
                        (cur_L == 0.0f || cur_L != cur_L || sM_nan)
                        ? -INFINITY : sM_val * (float)M_LN2;
                }
            }

            if (batch_idx + 1 <= end_idx) {
                // Skip mbarrier reinit: barrier phases are consistent across batches.
                // store_o stages through sO_addr, which aliases sQ. Ensure every
                // thread has finished its SMEM-to-register read before the next
                // batch's Q copy overwrites that storage.
                __syncthreads();
                launch_q_copy<T>(params, batch_idx + 1, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);
            } else {
                // Allow the next kernel (the combine kernel) to launch
                // The next kernel MUST be the combine kernel
                // PPU stream serialization orders the following combine launch.
            }
        } else {
            // Don't use __ldg because of PDL and instruction reordering
            int split_idx = params.num_splits_ptr[batch_idx] + n_split_idx;

            float *oaccum_ptr = (float *)params.oaccum_ptr + ((split_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M) * T::kHeadDimV;    // (BLOCK_SIZE_M, HEAD_DIM_V) : (HEAD_DIM_V, 1)
            Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), Layout<
                                                                        Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV>>,
                                                                        Stride<Int<T::kHeadDimV>, _1>>{});

            if constexpr (T::kIsCrossCut && T::kUsePv2x4 && T::kBlockM == 64) {
                float *softmax_lseaccum_ptr = (float *)params.softmax_lseaccum_ptr + (split_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M; // (BLOCK_SIZE_M) : (1)
                Tensor gSoftmaxLseAccum = make_tensor(make_gmem_ptr(softmax_lseaccum_ptr), Layout<
                                                                                               Shape<Int<T::BLOCK_SIZE_M>>,
                                                                                               Stride<_1>>{});
                if (warpgroup_idx == 0 && (idx_in_warpgroup / 32) < T::kAtomLayoutM &&
                    idx_in_warpgroup % 2 == 0) {
                    bool second_row = (idx_in_warpgroup & 2) != 0;
                    int row = my_row + (second_row ? 8 : 0);
                    if (row < num_valid_seq_q) {
                        float cur_L = second_row ? rL[1] : rL[0];
                        bool row_empty = second_row ? target_row1_empty : target_row0_empty;
                        float sM_val = sM(row);
                        bool sM_nan = ((__float_as_uint(sM_val) & 0x7FFFFFFFu) > 0x7F800000u);
                        gSoftmaxLseAccum(row) = (row_empty || sM_nan) ? -INFINITY : log2f(cur_L) + sM_val;
                    }
                }
            } else
            {
                float *softmax_lseaccum_ptr = (float *)params.softmax_lseaccum_ptr + (split_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M; // (BLOCK_SIZE_M) : (1)
                Tensor gSoftmaxLseAccum = make_tensor(make_gmem_ptr(softmax_lseaccum_ptr), Layout<
                                                                                               Shape<Int<T::BLOCK_SIZE_M>>,
                                                                                               Stride<_1>>{});
                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL_reduction_wksp[i];
                    float sM_val = sM(i);
                    bool sM_nan = ((__float_as_uint(sM_val) & 0x7FFFFFFFu) > 0x7F800000u);
                    gSoftmaxLseAccum(i) = (cur_L == 0.0f || cur_L != cur_L || sM_nan) ? -INFINITY : log2f(cur_L) + sM_val;
                }
            }

            // Prune NaN in rO (empty rows can leave NaN behind).
            CUTLASS_PRAGMA_UNROLL
            for (int idx = 0; idx < size(rO); ++idx) {
                if (rO(idx) != rO(idx)) rO(idx) = 0.0f;
            }

            store_o<T, false>(rO, gOAccum, T::kUsePv2x4 ? rL_pv : rL,
                              sO_addr, params, batch_idx, k_head_idx,
                              m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            if (batch_idx + 1 <= end_idx) {
                // Keep mbarrier reinit for split path (store_o<T,false> uses sO_addr)
                if (threadIdx.x == 0) {
                    __mbarrier_init(barrier_Q, 32);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        __mbarrier_init(&barriers_K0[i], 256);
                        __mbarrier_init(&barriers_K1[i], 256);
                    }
                }
                __syncthreads();
                cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;
                launch_q_copy<T>(params, batch_idx + 1, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);
            } else {
                // Allow the next kernel (the combine kernel) to launch
                // PPU stream serialization orders the following combine launch.
            }
        }
        if (batch_idx != end_idx)
            __syncthreads();
    }
}



template<typename T, bool ALLOW_EXTRA>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_sparse_decode_wg_kernel_hs64(__grid_constant__ const Flash_fwd_mla_params params) {
#if ACOMPUTE_VERSION == 10000
    if constexpr (T::kArch == 80)
#else
    if constexpr (T::kArch == 89)
#endif
    hs64_attention<T, ALLOW_EXTRA>(params);
}

template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_sparse_prefill_fwd_hs64(__grid_constant__ const SparsePrefillParams params) {
#if ACOMPUTE_VERSION == 10000
    if constexpr (T::kArch == 80) {
#else
    if constexpr (T::kArch == 89) {
#endif
    // Specialize geometry here, where inlining can fold the fixed M64 and
    // one-token-page layout into Q copy, KV addressing and the epilogue.
    Flash_fwd_params p{};
    p.b = params.s_q;
    p.q_orig = 1;
    p.seqlen_q = p.h_q = p.ngroups = T::kBlockM;
    p.h = p.h_h_k_ratio = 1;
    p.d = T::kHeadDim;
    p.d_v = T::kHeadDimV;
    p.scale_softmax = params.sm_scale;
    p.scale_softmax_log2 = params.sm_scale_div_log2;
    p.q_ptr = params.q;
    p.k_ptr = params.kv;
    p.o_ptr = params.out;
    p.softmax_lse_ptr = params.lse;
    p.indices_ptr = params.indices;
    p.topk_len_ptr = params.topk_length;
    p.attn_sink_ptr = params.attn_sink;
    p.topk = params.topk;
    p.extra_topk = -1;
    p.page_block_size = 1;
    p.num_blocks = p.seqlen_k = params.s_kv;
    p.q_batch_stride = params.stride_q_s_q;
    p.q_row_stride = params.stride_q_h_q;
    p.k_batch_stride = params.stride_kv_s_kv;
    p.k_row_stride = T::kHeadDim;
    p.o_batch_stride = T::kBlockM * T::kHeadDimV;
    p.o_row_stride = T::kHeadDimV;
    p.indices_batch_stride = params.stride_indices_s_q;
    hs64_attention<T, false>(p, static_cast<float *>(params.max_logits));
    }
}

// Reuse the no-extra images for full tiles; all other inputs retain guards.
template<int HeadDim, bool AllowExtra = true, int Arch = 89>
static void run_flash_sparse_decode_wg_kernel_hs64_hdim(
    Flash_fwd_params &params,
    hggcStream_t stream)
{
    static_assert(HeadDim == 512 || HeadDim == 576);
    using T = Hs64Traits<HeadDim, /*GuardIndices=*/AllowExtra, Arch>;
    static_assert(T::kBlockM == 64 && T::kBlockN == 64 && T::kHeadDim == HeadDim);
    static_assert(T::kUsePv2x4,
                  "HS64 instances use the shared PV 2x4 layout");

    auto mla_kernel = &flash_sparse_decode_wg_kernel_hs64<T, AllowExtra>;
    constexpr size_t smem_size =
        std::max(sizeof(typename T::SharedMemoryPlan),
                 sizeof(typename T::SharedMemoryOutPut));

    hggcFuncSetAttribute(
        mla_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    const dim3 grid(params.q_orig, params.ngroups / T::kBlockM, params.num_sm_parts);

    int ctas_per_sm = 0;
    const hggcError occupancy_status =
        hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, mla_kernel, T::NUM_THREADS, smem_size);
    (void)occupancy_status;

    char *show_log = std::getenv("show_log");
    if (show_log && std::isdigit(static_cast<unsigned char>(*show_log))) {
        static bool tag_printed = false;
        if (!tag_printed) {
            tag_printed = true;
            printf("[WG_HS64_TAG] %s blockM=64 d=%d no_extra=%d index_guard=%d\n",
                   kHs64BuildTag, HeadDim, int(!AllowExtra), int(T::kGuardIndices));
        }

        hggcFuncAttributes attr;
        hggcFuncGetAttributes(&attr, mla_kernel);
        int sm_count = get_num_sm(get_current_device());
        if (sm_count == 64) {
            sm_count = 20;
        }
        printf("[sparse_decode_wg_hs64]: smem_size=%d, CTAs per SM=%d\n",
               int(smem_size), ctas_per_sm);
        printf("blockM:%d, blockN:%d, threads:%d, block_size:%d\n",
               T::kBlockM, T::kBlockN, T::NUM_THREADS,
               params.page_block_size);
        printf("grid_n[%d, %d, %d]\n",
               int(grid.x), int(grid.y), int(grid.z));
        printf("vreg:%d, stack:%d, sm:%d, occupancy:%0.3f, Arch:%d\n",
               int(attr.numRegs), int(attr.localSizeBytes), sm_count,
               float(grid.x * grid.y * grid.z) /
                   float(sm_count * ctas_per_sm), Arch);
    }

    hggcLaunchAttribute kernel_attributes[1];
    kernel_attributes[0].id =
        hggcLaunchAttributeProgrammaticStreamSerialization;
    kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    hggcLaunchConfig_t kernel_config = {
        grid,
        dim3(T::NUM_THREADS, 1, 1),
        smem_size,
        stream,
        kernel_attributes,
        1
    };
    hggcLaunchKernelEx(&kernel_config, mla_kernel, params);
    CHECK_CUDA_KERNEL_LAUNCH();

    ::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(params, stream);
}

template<int Arch>
static void run_flash_sparse_decode_wg_kernel_hs64_arch(
    Flash_fwd_params &params,
    hggcStream_t stream)
{
    FLASH_ASSERT(params.ngroups % 128 == 64);
    // Without per-batch lengths or extra KV, 128-aligned topk makes every
    // scheduled tile physically complete. The existing block bounds suffice;
    // stale lookahead cannot switch caches and is never consumed by QK/PV.
    // Keep tails (including topk=0) and extra transitions on guarded images.
    constexpr int index_quantum = 2 * Hs64Traits<512>::kBlockN;
    const bool use_full_tile_indices =
        params.extra_topk < 0 && params.topk_len_ptr == nullptr &&
        params.topk > 0 && params.topk % index_quantum == 0;
    if (params.d == 512) {
        if (use_full_tile_indices) {
            run_flash_sparse_decode_wg_kernel_hs64_hdim<512, false, Arch>(
                params, stream);
        } else {
            run_flash_sparse_decode_wg_kernel_hs64_hdim<512, true, Arch>(params, stream);
        }
    } else {
        FLASH_ASSERT(params.d == 576);
        if (use_full_tile_indices) {
            run_flash_sparse_decode_wg_kernel_hs64_hdim<576, false, Arch>(params, stream);
        } else {
            run_flash_sparse_decode_wg_kernel_hs64_hdim<576, true, Arch>(params, stream);
        }
    }
}

void run_flash_sparse_decode_wg_kernel_hs64(
    Flash_fwd_params &params, hggcStream_t stream) {
    const auto [major, minor] = get_compute_capability(get_current_device());
    if (major > 8 || (major == 8 && minor >= 9)) {
        run_flash_sparse_decode_wg_kernel_hs64_arch<89>(params, stream);
    } else {
        run_flash_sparse_decode_wg_kernel_hs64_arch<80>(params, stream);
    }
}

template<int HeadDim, int Arch>
void run_flash_sparse_prefill_fwd_hs64(SparsePrefillParams &params) {
    using T = Hs64PrefillTraits<HeadDim, Arch>;
    auto kernel = &flash_sparse_prefill_fwd_hs64<T>;
    constexpr size_t smem_size =
        std::max(sizeof(typename T::SharedMemoryPlan),
                 sizeof(typename T::SharedMemoryOutPut));
    hggcFuncSetAttribute(
        kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<dim3(params.s_q), T::NUM_THREADS, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace flashmla::dsa::hs64
