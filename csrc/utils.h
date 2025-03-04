/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"

#ifdef USE_PPU
#include "acc_vreg_fraga.h"
#endif

#define FA_LOG_FMT(fmt, ...) 
//#define FA_LOG_FMT(fmt, ...) \
//    if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && blockIdx.x == 0) { \
//        printf(fmt, ##__VA_ARGS__); \
//    } 


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n" 
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

#if USE_AIU
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=true, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_per_warp(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    // support AIU on PPU
    if constexpr (is_mix_iterator<typename Engine0::iterator>::value) {
        if constexpr (!Is_even_MN) {
            tiled_copy.desc_.dim_h = max_MN;
            FA_LOG_FMT("flash::copy_per_warp: tiled_copy.desc_.dim_h:%d, tiled_copy.desc_.cube_h:%d\n", tiled_copy.desc_.dim_h, tiled_copy.desc_.cube_h);
        }
        cute::copy(tiled_copy, S, D);
        return;
    }
}

// resolves offset of a slice of a paged kv for AIU load
// assumes that the tensor has already been positioned at the correct head.
template <typename Kernel_traits>
__forceinline__ __device__
int64_t resolve_next_kv_subblock_offset(const int copied_rows, const int n_block_max /*start from 1*/, const int page_block_size, 
                            const int* block_table, const int page_stride, const int row_stride) {
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int64_t global_row_offset = (n_block_max - 1) * kBlockN + copied_rows;
    const int64_t page_offset = global_row_offset % page_block_size;
    const int64_t virtual_page_idx = global_row_offset / page_block_size;
    int64_t offset = ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride);
    FA_LOG_FMT("copied_rows:%d, n_block_max:%d, page_block_size:%d, block_table:%p, page_stride:%d, row_stride:%d, global_row_offset:%d, page_offset:%d, virtual_page_idx:%d, offset:%lld\n",
            copied_rows, n_block_max, page_block_size, block_table, page_stride, row_stride, global_row_offset, page_offset, virtual_page_idx, offset);
    return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride);
}

template <bool Is_even_MN=true, bool Clear_OOB_MN=false, typename Kernel_traits, 
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3, typename Engine4, typename Layout4, typename Engine5, typename Layout5>
__forceinline__ __device__ void copy_kv_aiu(TiledCopy tiled_copy, Tensor<Engine0, Layout0> &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K,
                            const int kHeadDim, const int kRowsPerAiuLoad, Tensor<Engine4, Layout4> const &gKV, Tensor<Engine5, Layout5> const &sKV,
                            const int n_block /* current block idx, count from 1 */, const int *block_table, const int page_block_size, int64_t kv_batch_stride, int64_t kv_row_stride, const int max_MN=0) {
    FA_LOG_FMT("copy_kv_aiu start\n");
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kBlockKSmem = Kernel_traits::kBlockKSmem;
    int copy_total = max_MN == 0 ? kBlockN : std::min(max_MN, kBlockN);
    auto S_data = S.data();
    auto D_data = D.data();
    if (block_table == nullptr) {
        copy<Is_even_MN>(tiled_copy, S, D, identity_MN, predicate_K, max_MN);
        return;
    }

    const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
    /// paged K/V
    if constexpr (Is_even_MN) {
        for (int copied = warp_idx * kRowsPerAiuLoad; copied < copy_total; copied += kNWarps * kRowsPerAiuLoad) {
            FA_LOG_FMT("Clear_OOB_MN:%d, Is_even_MN:%d, copied:%d, copy_total:%d\n", Clear_OOB_MN, Is_even_MN, copied, copy_total);
            /// update pointer
            S.data() = gKV.data() + resolve_next_kv_subblock_offset<Kernel_traits>(copied, n_block, page_block_size, block_table,
                    kv_batch_stride, kv_row_stride);
            D.data() = sKV.data() + copied * kBlockKSmem;
            copy_per_warp<Is_even_MN>(tiled_copy, S, D, identity_MN, predicate_K);
        }
    } else {
        int copied = 0;
        for (; copied < copy_total; copied += kRowsPerAiuLoad) {
            int actual_copy;
            /// update pointer
            S.data()  = gKV.data() + resolve_next_kv_subblock_offset<Kernel_traits>(copied, n_block, page_block_size, block_table,
                    kv_batch_stride, kv_row_stride);
            D.data() = sKV.data() + copied * kBlockKSmem;
            
            int next_copied = copied + kRowsPerAiuLoad;
            if (Clear_OOB_MN && (next_copied > copy_total)) {
                /// to pad unalinged part for this copy
                /// dim_h to mark the actual rows which <= kRowsPerAiuLoad
                actual_copy = copy_total - copied;
                FA_LOG_FMT("pad unalinged part for this copy: acutal to_copy:%d, cube_h:%d, padding rows:%d\n", actual_copy, tiled_copy.desc_.cube_h, tiled_copy.desc_.cube_h - actual_copy);
            } else {
                /// norm copy
                actual_copy = kRowsPerAiuLoad;
            }
            FA_LOG_FMT("Clear_OOB_MN:%d, Is_even_MN:%d, copied:%d, copy_total:%d, to_copy:%d\n", Clear_OOB_MN, Is_even_MN, copied, copy_total, actual_copy);
            copy<Is_even_MN>(tiled_copy, S, D, identity_MN, predicate_K, actual_copy);
        }

        if (copied < kBlockN) {
            /// zero for tail rows
            D.data() = sKV.data() + copied * kBlockKSmem;
            tiled_copy.desc_.cube_h = kBlockN - copied;
            copy<Is_even_MN>(tiled_copy, S, D, identity_MN, predicate_K, 0);
        }
        /// reset cube_h
        tiled_copy.desc_.cube_h = kRowsPerAiuLoad;
    }
    S.data() = S_data;
    D.data() = D_data;
    FA_LOG_FMT("copy_kv_aiu stop\n");
    return;
}

template<bool A_in_regs=false, bool B_in_regs=false, typename Kernel_traits, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_kv_aiu(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    constexpr int OFFSET_MMA_N = Kernel_traits::kRowsPerAiuLoad * Kernel_traits::kBlockKSmem;
    constexpr int OFFSET_TILE = Kernel_traits::kBlockN * Kernel_traits::kBlockKSmem - Kernel_traits::kRowsPerAiuLoad * Kernel_traits::kBlockKSmem;

    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }

    
    FA_LOG_FMT("gemm_kv_aiu:smem_tiled_copy_B, k=0, start\n");
    auto bak_data = tCsB.data();
    auto org_data = tCsB.data();
    if (!B_in_regs) { 
        /// cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); 
        /// tCrB (MMA=1, MMA_N=8, MMA_K=8), iterate on MMA_N
        for (int n= 0; n < size<1>(tCrB); n++) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _0{}, _0{}), tCrB_copy_view(_, n, _0{})); 
            /// advance to next kRowsPerAiuLoad rows
            tCsB.data() = tCsB.data() + OFFSET_MMA_N;
        }
        /// reset to org pointer
        tCsB.data() = org_data;
    }
    FA_LOG_FMT("gemm_kv_aiu:smem_tiled_copy_B, k=0, stop\n");

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            FA_LOG_FMT("gemm_kv_aiu:smem_tiled_copy_B, k=%d, start\n", i+1);
            if (!B_in_regs) { 
                /// cute::copy(smem_tiled_copy_B, tCsB(_, _, i+1), tCrB_copy_view(_, _, i+1)); 
                /// iterate on MMA_N
                for (int n = 0; n < size<1>(tCrB); n++) {
                    cute::copy(smem_tiled_copy_B, tCsB(_, _0{}, i+1), tCrB_copy_view(_, n, i+1)); 
                    tCsB.data() = tCsB.data() + OFFSET_MMA_N;
                }
                if ((i + 2) % (size<2>(tCrA) / (Kernel_traits::kHeadDim / Kernel_traits::kBlockKSmem)) == 0) {
                    /// move to start of next tile of kBlockN * kBlockKSmem and minus the cube_in_stage offset:
                    org_data = org_data + OFFSET_TILE;
                }
                tCsB.data() = org_data;
            }
            FA_LOG_FMT("gemm_kv_aiu:smem_tiled_copy_B, k=%d, stop\n", i+1);
        }
        FA_LOG_FMT("gemm_kv_aiu:tiled_mma, k=%d, start\n", i);
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        FA_LOG_FMT("gemm_kv_aiu:tiled_mma, k=%d, stop\n", i);
    }
    tCsB.data() = bak_data;
}

template<typename Kernel_traits, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs_kv_aiu(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 &tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    constexpr int OFFSET_MMA_N = Kernel_traits::kRowsPerAiuLoad * Kernel_traits::kBlockKSmem;
    constexpr int OFFSET_TILE = Kernel_traits::kBlockN * Kernel_traits::kBlockKSmem;
    int MMA_K_PER_TILE = size<1>(tCrB) / (Kernel_traits::kHeadDim / Kernel_traits::kBlockKSmem);
    
    auto org_data = tCsB.data();
    ///cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));

    /// iterate on MMA_K
    /// tCrB: (MMA, MMA_K, MMA_N)
	for (int k = 0; k < size<1>(tCrB); k++) {
        int slice = k % MMA_K_PER_TILE;
        if (k > 0 && slice == 0) {
            /// advance to next tile of kRowsPerAiuLoad * kBlockKSmem
            tCsB.data() = tCsB.data() + OFFSET_TILE;
        }
        cute::copy(smem_tiled_copy_B, tCsB(_, slice, _0{}), tCrB_copy_view(_, k, _0{})); 
	}

    #pragma unroll
    /// iterate on MMA_N
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            ///cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));

            /// poniter to next kRowsPerAiuLoad rows
            tCsB.data() = org_data + (i + 1) * OFFSET_MMA_N;
            /// iterate on MMA_K
            for (int k = 0; k < size<1>(tCrB); k++) {
                int slice = k % MMA_K_PER_TILE;
                if (k > 0 && slice == 0) {
                    /// advance to next tile of kRowsPerAiuLoad * kBlockKSmem
                    tCsB.data() = tCsB.data() + OFFSET_TILE;
                }
                cute::copy(smem_tiled_copy_B, tCsB(_, slice, _0{}), tCrB_copy_view(_, k, i+1)); 
            }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
    tCsB.data() = org_data;
}
#endif // USE_AIU

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
#ifdef USE_PPU
    // acc is ppu c layout, size0 is 8, MMA_N size is A100 MMA_N/2
    static_assert(decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_4>{}); //((2, 4), MMA_M, MMA_N)
#else
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
#endif
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_PPU
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    // TAOTAO-FIXME:
    // static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    // constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
    constexpr int MMA_N_divisor = 1;
    auto l = logical_divide(acc_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))

    return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
                       get<0, 1>(l),
                       get<1, 1, 1>(l));
};
#else
// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_PPU
template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_acc(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    NumericArrayConverterPPU<To_type, From_type, numel> convert_op;
    // convert_op:: accum(tensor.data());
    // auto frag = convert_op(accum);
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    Tensor out = FLASH_NAMESPACE::convert_type<To_type>(tensor);
    FLASH_NAMESPACE::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// resolves offset of a slice of a paged kv copy from gmem.
// assumes that the tensor has already been positioned at the correct head.
template <typename Kernel_traits>
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset(const int tidx, const int n_block_max, const int page_block_size, 
                            const int* block_table, const int page_stride, const int row_stride) {
    constexpr int kGmemThreadsPerRow = Kernel_traits::kGmemThreadsPerRow;
    constexpr int kGmemRowsPerThread = Kernel_traits::kGmemRowsPerThread;
    constexpr int kGmemElemsPerLoad = Kernel_traits::kGmemElemsPerLoad;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    
    const int64_t col_offset = tidx % kGmemThreadsPerRow * kGmemElemsPerLoad;
    const int64_t block_row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread;
    const int64_t global_row_offset = block_row_offset + (n_block_max - 1) * kBlockN;
    const int64_t page_offset = global_row_offset % page_block_size;
    const int64_t virtual_page_idx = global_row_offset / page_block_size;

    return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride)
        + col_offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Layout reshape function. Given a layout with modes ((v1, v2), m, k), returns (v1, v2, k),         
// where v2 may be a tuple itself, in the case of swizzled smem-backed thread tiles. This ensures
// that paged and non-paged copies result in equivalently shaped, if not necessarily strided, tensors.
template <class Shape, class Stride>
__forceinline__ __device__
auto reshape_thread_tile(Layout<Shape, Stride> l) {
    return make_layout(append(get<0>(l.shape()), get<2>(l.shape())),
                        append(get<0>(l.stride()), get<2>(l.stride())));
}

// reshapes and flattens the thread tile layout. A separate function is needed for the case where
// one of the modes of l is a layout itself and must be flattened, as opposed to keeping it intact
// for the case of swizzled layouts
template <class Shape, class Stride>
__forceinline__ __device__
auto reshape_flatten_thread_tile(Layout<Shape, Stride> l) {
    auto mode_0 = filter(flatten(get<0>(l)));
    return make_layout(append(mode_0.shape(), get<2>(l.shape())),
                        append(mode_0.stride(), get<2>(l.stride())));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//PPU: shared memory not support init by zero, need clear if not align.
#ifdef USE_PPU
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=true, bool Clear_OOB_K=true,
#else
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
#endif
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
// support AIU on PPU
#if USE_AIU
    if constexpr (is_mix_iterator<typename Engine0::iterator>::value) {
        const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
        if (warp_idx == 0) {
            if constexpr (!Is_even_MN) {
                tiled_copy.desc_.dim_h = max_MN;
            }

            cute::copy(tiled_copy, S, D);
        }
        return;
    }
#endif

    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
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
    // TD [2023-04-13]: Strange that the code below can cause race condition.
    // I think it's because the copies are under an if statement.
    // if (Is_even_K) {
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(S); ++m) {
    //         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //             copy(tiled_copy, S(_, m, _), D(_, m, _));
    //         } else if (Clear_OOB_MN) {
    //             clear(D(_, m, _));
    //         }
    //     }
    // } else {  // It's slightly faster in this case if iterate over K first
    //     #pragma unroll
    //     for (int k = 0; k < size<2>(S); ++k) {
    //         if (predicate_K(k)) {
    //             #pragma unroll
    //             for (int m = 0; m < size<1>(S); ++m) {
    //                 if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //                     copy(tiled_copy, S(_, m, k), D(_, m, k));
    //                 } else if (Clear_OOB_MN) {
    //                     clear(D(_, m, k));
    //                 }
    //             }
    //         } else if (Clear_OOB_K) {  // There's no case where !Clear_OOB_K && Clear_OOB_MN
    //             if (Clear_OOB_MN || Is_even_MN) {
    //                 clear(D(_, _, k));
    //             } else {
    //                 #pragma unroll
    //                 for (int m = 0; m < size<1>(S); ++m) {
    //                     if (!(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)) {
    //                         clear(D(_, m, k));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void calculate_dtanh(Tensor<Engine0, Layout0> &src_tensor, Tensor<Engine1, Layout1> &dst_tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(src_tensor); ++i) {
        dst_tensor(i) = (1.f - (src_tensor(i) * src_tensor(i))) * softcap;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
