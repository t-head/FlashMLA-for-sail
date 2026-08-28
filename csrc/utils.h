/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <hggc_fp16.h>
#include <hggc_bf16.h>

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#if defined USE_PPU
#include "acc_vreg_fraga.h"
#include "ppu/cute/tensor_mix.hpp"
#endif
#include <cute/util/debug.hpp>
#include "params.h"
#include "kerutils/common/common.h"
#include "kerutils/host/hardware_info.h"
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_DRIVER_CHECK(expr)                             \
    HGresult _r = (expr);                                   \
    if (_r != HGGC_SUCCESS) {                               \
        const char* _name = nullptr;                        \
        const char* _str = nullptr;                         \
        hgGetErrorName(_r, &_name);                         \
        hgGetErrorString(_r, &_str);                        \
        printf("HG driver error: %s: %s\n",                \
               (_name ? _name : "?"), (_str ? _str : "?")); \
    }


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

////////////////////////////////////////////////////////////////////////////////////////////////////
template<int kPagedLoad, int UpdateSmemSize, bool ifgemm0,
         bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_pagedkv(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }

    if (!B_in_regs) {
        if (ifgemm0) {
            for (int i = 0; i < size<1>(tCrA); ++i) {
                auto tCsB_tile = tCsB(_, i, _);
                Tensor tCsB_temp = make_tensor(tCsB_tile.data(), tCsB_tile.layout());
                const int coord_h = cute::get<1>(tCsB_temp.data().coord_);
                const int paged_idx = coord_h / kPagedLoad;
                cute::get<1>(tCsB_temp.data().coord_) = coord_h % kPagedLoad;
                tCsB_temp.data().ptr_ = tCsB_temp.data().ptr_  + paged_idx * UpdateSmemSize;
                cute::copy(smem_tiled_copy_B, tCsB_temp, tCrB_copy_view(_, i, _));
            }
        } else {
            #pragma unroll
            for (int i = 0; i < size<2>(tCrA); ++i) {
                auto tCsB_tile = tCsB(_, _, i);
                Tensor tCsB_temp = make_tensor(tCsB_tile.data(), tCsB_tile.layout());
                const int coord_h = cute::get<0>(tCsB_temp.data().coord_);
                const int paged_idx = coord_h / kPagedLoad;
                cute::get<0>(tCsB_temp.data().coord_) = coord_h % kPagedLoad;
                tCsB_temp.data().ptr_ = tCsB_temp.data().ptr_ + paged_idx * UpdateSmemSize;
                cute::copy(smem_tiled_copy_B, tCsB_temp, tCrB_copy_view(_, _, i));
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    // CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    // CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
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

template<int start_kidx_A, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_rss(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    // CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < start_kidx_A - 1; ++i) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }

    #pragma unroll
    for (int i = start_kidx_A - 1; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
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
    // static_assert(decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
#if ACOMPUTE_VERSION == 10000
    auto l = logical_divide(acc_layout, Shape<_4>{}); //((2, 4), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
#else
    auto l = logical_divide(acc_layout, Shape<_4>{}); //((4, 2), MMA_M, MMA_N)
    auto midl = logical_divide(l, Shape<Shape<_2>>{}); //(((2, 2), 2), MMA_M, MMA_N)
    return make_layout(
        make_layout(get<0, 0, 1>(midl), get<1>(midl)),
        make_layout(get<0, 0, 0>(midl), make_layout(get<0, 1>(midl), get<2>(midl)))
    );
#endif
#else
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
#endif
};

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_pv_offset(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B, int remain_k_offset) {

    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (cute::get<0>(tCsB(_, _, i).data().coord_) >= remain_k_offset) { // coord_h >= remain_k_offset
            break;
        }
        if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i), tCrA_copy_view(_, _, i)); }
        if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i), tCrB_copy_view(_, _, i)); }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
#if ACOMPUTE_VERSION == 10500
    return acc_layout;
#else
    using X = Underscore;
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    // static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    // constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
    constexpr int MMA_N_divisor = 1;
    auto l = logical_divide(acc_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))

    return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
                       get<0, 1>(l),
                       get<1, 1, 1>(l));
#endif
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
#if defined USE_PPU
template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_acc(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    NumericArrayConverterPPU<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
#endif


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

    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
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
}

template<typename Kernel_traits, bool Split, bool CrossCut = false, bool IsSparse = false, typename AccO, typename Softmax>
__forceinline__ __device__ void store(const Flash_fwd_params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx,
                                      __shared__ char* smem_,  AccO acc_o, Softmax softmax) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kBlockKSmemV = Kernel_traits::kBlockKSmemV; // 64
    constexpr int MMA_ATOM_M = Kernel_traits::USE_MMA_M8 ? 8 : 16;
    constexpr int MMA_ATOM_K_M = Kernel_traits::USE_MMA_M8 ? 1 : 2;
    const int tidx = threadIdx.x;

    // Epilogue
    const int split_offset = __ldg(params.num_splits_ptr + bidb);

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{});
                                                                                                                             // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum is larger than sQ, so we need to syncthreads here
    // TODO: allocate enough smem for sOaccum
    if constexpr (Split) { __syncthreads(); }
    /// move: reg -> tsm
    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const int h_k_idx = m_block % cute::ceil_div(params.ngroups, kBlockM); // s_q = s_q_ori * h_q
    const int s_q_idx = m_block / cute::ceil_div(params.ngroups, kBlockM);
    const int row_base = IsSparse ? h_k_idx * kBlockM + s_q_idx * params.ngroups : m_block * kBlockM;
    const int seqlen_q_max = IsSparse ? params.ngroups - h_k_idx * kBlockM : params.seqlen_q - m_block * kBlockM;

    const index_t row_offset_o = bidb * params.o_batch_stride + bidh * params.o_head_stride + row_base * params.o_row_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + row_base) * params.d_v;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + row_base;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + row_base;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (Split ? row_offset_lseaccum : row_offset_lse)),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    //
    /// move: tsm -> reg -> glboal
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    /// tsm -> reg
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if constexpr (CrossCut) {
        const int warp_id = tidx / 32;
        // const int line_id = tidx % 32;
        // const int warp_id_m = (tidx / 32) % Kernel_traits::AtomLayoutQ;
        const int row_lse_base = warp_id % Kernel_traits::AtomLayoutQ * MMA_ATOM_M + (tidx % 32) / 4;
        const int warp_stride = MMA_ATOM_M * Kernel_traits::AtomLayoutQ;
        if (warp_id < Kernel_traits::kNWarps0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = row_lse_base + (mi / MMA_ATOM_K_M) * warp_stride + (mi % MMA_ATOM_K_M) *8;
                if (row < seqlen_q_max) { gLSEaccum(row) = lse(mi); }
            }
        }

    } else {
#ifdef USE_PPU
#if ACOMPUTE_VERSION == 10000
        static_assert(decltype(size<0>(taccOcO))::value == 4 * MMA_ATOM_K_M);
        // Convert to ((2, 4), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
#else
        Tensor taccOcO_row = taccOcO(make_coord(0, _, 0), _, 0);
#endif
#else
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
#endif
        // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < seqlen_q_max) { gLSEaccum(row) = lse(mi); }
            }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));

    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<false, true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, seqlen_q_max
    );
}

template<typename Kernel_traits>
void printf_show_log(const void* kernel, Flash_fwd_params &params, const size_t smem_size,
                     bool is_causal, bool is_sparse = false, bool is_fp8 = false) {
    char *pEnv_params = std::getenv("show_log");
    int num_m_block;
    if (pEnv_params && isdigit(*pEnv_params)) {
        int value = std::stoi(std::string(pEnv_params));
        if (value > 0) {
            int ctas_per_sm;
            hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
            if (is_sparse) {
                num_m_block = (params.seqlen_q / params.ngroups) * cute::ceil_div(params.ngroups, Kernel_traits::kBlockM);
                printf("[run_flash_sparse_decode_fwd_]: FP8 KVCache:%d\n", is_fp8);
            } else {
                num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
                printf("[run_flash_splitkv_fwd_]:\n");
            }
            printf("smem_size = %d, CTAs per SM = %d, ", int(smem_size), ctas_per_sm);

            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, kernel);
            int sm_count = get_num_sm(get_current_device());
            if (sm_count == 64) sm_count = 20;
            printf("HeadDim:%d, HeadDimV:%d\n",Kernel_traits::kHeadDim, Kernel_traits::kHeadDimV);
            printf("blockM:%d, blockN:%d, threads:%d, params.num_splits:%d, block_size:%d\n",
                    Kernel_traits::kBlockM, Kernel_traits::kBlockN, Kernel_traits::kNThreads, params.num_splits, params.page_block_size);
            printf("CrossCut:%d, USE_MMA_M8:%d, kStages:%d, kBlockNPagedPerAiuLoad:%d\n",
                    Kernel_traits::CrossCut, Kernel_traits::USE_MMA_M8, Kernel_traits::kStages, Kernel_traits::kBlockNPagedPerAiuLoad);
            printf("kNWarps:%d, AtomLayoutQ:%d, AtomLayoutP:%d, kNWarps0:%d\n",
                    Kernel_traits::kNWarps, Kernel_traits::AtomLayoutQ, Kernel_traits::AtomLayoutP, Kernel_traits::kNWarps0);
            printf("Is_Q_in_regs:%d, Share_Q_K_smem:%d\n", Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem);
            printf("seq[%d, %d], grid_n[%d, %d, %d]\n",
                    params.seqlen_q, params.seqlen_k, num_m_block, params.h, params.num_sm_parts);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                    float(num_m_block * params.h * params.num_sm_parts) / float(sm_count * ctas_per_sm));
        }
    }

}

template<typename Kernel_traits>
void printf_prefill_show_log(const void* kernel, SparsePrefillParams &params, const size_t smem_size) {
    char *pEnv_params = std::getenv("show_log");
    int num_m_block;
    if (pEnv_params && isdigit(*pEnv_params)) {
        int value = std::stoi(std::string(pEnv_params));
        if (value > 0) {
            int ctas_per_sm;
            hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);

            num_m_block = params.s_q * cute::ceil_div(params.h_q, Kernel_traits::kBlockM);
            printf("[run_flash_sparse_prefill_fwd_]:\n");
            printf("smem_size = %d, CTAs per SM = %d,", int(smem_size), ctas_per_sm);
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, kernel);
            int sm_count = get_num_sm(get_current_device());
            if (sm_count == 64) sm_count = 20;
            printf("HeadDim:%d, HeadDimV:%d, blockM:%d, blockN:%d\n",
                    Kernel_traits::kHeadDim, Kernel_traits::kHeadDimV, Kernel_traits::kBlockM, Kernel_traits::kBlockN);
            printf("kNThreads:%d, CrossCut:%d, USE_MMA_M8:%d, kStages:%d\n",
                    Kernel_traits::kNThreads, Kernel_traits::CrossCut, Kernel_traits::USE_MMA_M8, Kernel_traits::kStages);
            printf("kNWarps:%d, AtomLayoutQ:%d, AtomLayoutP:%d, kNWarps0:%d\n",
                    Kernel_traits::kNWarps, Kernel_traits::AtomLayoutQ, Kernel_traits::AtomLayoutP, Kernel_traits::kNWarps0);
            printf("Is_Q_in_regs:%d, Share_Q_K_smem:%d, seq[%d, %d], grid[%d]\n",
                    Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem, params.s_q, params.s_kv, num_m_block);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f\n",
                    int(attr.numRegs), int(attr.localSizeBytes), sm_count, float(num_m_block) / float(sm_count * ctas_per_sm));
        }
    }
}

template <bool Is_even_MN=true, typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void aiu_copy_gemm0swzlld(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                Tensor<Engine1, Layout1> &D, const int max_MN = 0){
    CUTE_STATIC_ASSERT_V(size<2>(D) - size<2>(S) == Int<1>{});
    // warp_idx have been set!
    if constexpr (!Is_even_MN) {
        tiled_copy.desc_.dim_h = max_MN;
    }
    #pragma unroll
    for (int k = 0; k < size<2>(S); k++) {
        cute::copy(tiled_copy, S(_, _, k), D(_, _, k));
    }
    cute::copy(tiled_copy, S(_, _, size<2>(S) - 1), D(_, _, size<2>(S)));
}
}  // namespace flash
