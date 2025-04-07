/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

// for performance propose, enable PPU1.0 m16n16k16 mma instruction adaptation.
// if disable this macro, use ppu compiler compatibility m16n8k16 mma version.
// #define USE_PPU

#ifdef USE_PPU
#include "ppu/ppu_include.hpp"
#endif

#if !USE_AIU
#define make_mix_tensor make_tensor
#define make_mix_tensor_like(x)  x
#endif

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
        // MMA_Atom<PPU_16x16x16_F32F16F16F32_TN>,
        // MMA_Atom<PPU_16x16x16_F32BF16BF16F32_TN>
        MMA_Atom<Acompute10000_8x16x16_F32F16F16F32_TN>,
        MMA_Atom<Acompute10000_8x16x16_F32BF16BF16F32_TN>
#elif defined(USE_PPU) && ACOMPUTE_VERSION == 10500
        MMA_Atom<Acompute10500_16x16x16_F32F16F16F32_TN>,
        MMA_Atom<Acompute10500_16x16x16_F32BF16BF16F32_TN>
#else
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
#endif
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    // using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtom = Copy_Atom<SM75_U32x2_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;

#if USE_AIU
    static constexpr int kBlockKSmem = kHeadDim_ % 64 == 0 ? 64 : 32;
#if ACOMPUTE_VERSION == 10000
    using SmemCopyOpQ = Acompute10000_TSM_LD_SWZL<elem_type, kBlockM_, kBlockKSmem, false, false, 1, 2>;
#else
    using SmemCopyOpQ = Acompute10500_TSM_LD_SWZL<elem_type, kBlockM_, kBlockKSmem, false, false, 1>;
#endif
    using SmemCopyAtomQ = Copy_Atom<SmemCopyOpQ, elem_type>;

#if ACOMPUTE_VERSION == 10000
    using SmemCopyOpQt = Acompute10000_TSM_LD_SWZL<elem_type, kBlockM_, kBlockKSmem, false, true>;
#else
    using SmemCopyOpQt = Acompute10500_TSM_LD_SWZL<elem_type, kBlockM_, kBlockKSmem, true, true, 1>;
#endif
    using SmemCopyAtomQt = Copy_Atom<SmemCopyOpQt, elem_type>;

#if ACOMPUTE_VERSION == 10000
    using SmemCopyOpK = Acompute10000_TSM_LD_SWZL<elem_type, kBlockN_, kBlockKSmem, false, false>;
#else
    using SmemCopyOpK = Acompute10500_TSM_LD_SWZL<elem_type, kBlockN_, kBlockKSmem, true, false, 1>;
#endif
    using SmemCopyAtomK = Copy_Atom<SmemCopyOpK, elem_type>;

    // using SmemCopyOpKVt = Acompute10000_TSM_LD_SWZL<elem_type, kBlockN_, kBlockKSmem, false, true>;
    // using SmemCopyAtomKVt = Copy_Atom<SmemCopyOpKVt, elem_type>;

#else
    using SmemCopyAtomQ = SmemCopyAtom;
    using SmemCopyAtomQt = SmemCopyAtomTransposed;
    using SmemCopyAtomK = SmemCopyAtom;
    // using SmemCopyAtomKVt = SmemCopyAtomTransposed
#endif
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
         int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKSmemV = kHeadDimV % 64 == 0 ? 64 : 32;
    // static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int kSwizzleV = kBlockKSmemV == 32 ? 2 : 3;

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    using SmemCopyOpVt = Acompute10000_TSM_LD_SWZL<elem_type, kBlockN_, kBlockKSmemV, false, true>;
#else
    using SmemCopyOpVt = Acompute10500_TSM_LD_SWZL<elem_type, kBlockN_, kBlockKSmemV, true, true, 1>;
#endif
    using SmemCopyAtomVt = Copy_Atom<SmemCopyOpVt, elem_type>;
#else
    using SmemCopyAtomVt = SmemCopyAtomTransposed;
#endif

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
        Tile<Int<8 * kNWarps>, _16, _16>>;
#else
        Tile<Int<16 * kNWarps>, _16, _16>>;
#endif

#if USE_AIU
    using SmemLayoutAtomQ = Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>;
    using SmemLayoutAtomV = Layout<Shape<_8, Int<kBlockKSmemV>>, Stride<Int<kBlockKSmemV>, _1>>;
#else
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
   using SmemLayoutAtomV = decltype(
        composition(Swizzle<kSwizzleV, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmemV>>,
                           Stride<Int<kBlockKSmemV>, _1>>{}));
#endif
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzleV, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmemV>>,
                           Stride<Int<kBlockKSmemV>, _1>>{}));

    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    // static constexpr int kSmemKVSize = (size(SmemLayoutK{}) + size(SmemLayoutV{})) * sizeof(Element);
    static constexpr int kSmemKVSize = (size(SmemLayoutK{}) * 2) * sizeof(Element);
    static constexpr int OSmemSize = size(SmemLayoutO{}) * sizeof(Element);
    static constexpr int OSmemSizeAccum = size(SmemLayoutO{}) * sizeof(ElementAccum);

    static constexpr int kSmemSize = std::max(kSmemQSize + kSmemKVSize, OSmemSize);
    static constexpr int kSmemSizeAccum = std::max(kSmemQSize + kSmemKVSize, OSmemSizeAccum);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static_assert(kHeadDimV % kGmemElemsPerLoad == 0, "kHeadDimV must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static constexpr int kGmemThreadsPerRowV = kBlockKSmemV / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    static_assert(kNThreads % kGmemThreadsPerRowV == 0, "kNThreads must be a multiple of kGmemThreadsPerRowV");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemLayoutAtomV = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowV>, Int<kGmemThreadsPerRowV>>,
                                  Stride<Int<kGmemThreadsPerRowV>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using GmemTiledCopyQK = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyVWoAiu = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtomV{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
#if USE_AIU
    // static_assert(Block_K{} * sizeof(Element) % 32 == 0, "aiu_no_trans: block_k must be multiple of 32B");
    static constexpr int bits_per_aiu_Q = kBlockM * kBlockKSmem * sizeof(Element) * 8;
#if ACOMPUTE_VERSION == 10000
    using Gmem_copy_struct_Q = Acompute10000_AIU_LOAD<cute::C<bits_per_aiu_Q>, Element, false>;
#else
    using Gmem_copy_struct_Q = Acompute10500_AIU_LOAD<cute::C<bits_per_aiu_Q>, Element, false, kBlockM, kBlockKSmem>;
#endif

    static constexpr int bits_per_aiu_K = kBlockN * kBlockKSmem * sizeof(Element) * 8;
#if ACOMPUTE_VERSION == 10000
    using Gmem_copy_struct_K = Acompute10000_AIU_LOAD<cute::C<bits_per_aiu_K>, Element, false>;
#else
    using Gmem_copy_struct_K = Acompute10500_AIU_LOAD<cute::C<bits_per_aiu_K>, Element, false, kBlockN, kBlockKSmem>;
#endif

    static constexpr int bits_per_aiu_V = kBlockN * kBlockKSmemV * sizeof(Element) * 8;
#if ACOMPUTE_VERSION == 10000
    using Gmem_copy_struct_V = Acompute10000_AIU_LOAD<cute::C<bits_per_aiu_V>, Element, false>;
#else
    using Gmem_copy_struct_V = Acompute10500_AIU_LOAD<cute::C<bits_per_aiu_V>, Element, false, kBlockN, kBlockKSmemV>;
#endif

    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_Q, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockM>, Int<kBlockKSmem>>>{}));
    using GmemTiledCopyK = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_K, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockN>, Int<kBlockKSmem>>>{}));
    using GmemTiledCopyV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_V, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockN>, Int<kBlockKSmemV>>>{}));
#else
    using GmemTiledCopyQ = GmemTiledCopyQK;
    using GmemTiledCopyK = GmemTiledCopyQK;
    using GmemTiledCopyV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtomV{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
#endif

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomV{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmemV == 32,
#ifndef USE_PPU
        Layout<Shape <_16, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <_8, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
#else
        Layout<Shape <Int<kNThreads / 8>, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <Int<kNThreads / 16>, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
#endif
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _1>>{}));  // Val layout, 4 vals per store
};