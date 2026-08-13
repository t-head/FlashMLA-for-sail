/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

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

template<bool USE_MMA_M8=true, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {

#if defined(__HGGC_ARCH__) &&  __HGGC_ARCH__ >= 100
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__HGGC_ARCH__) &&  __HGGC_ARCH__ >= 100
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
        // MMA_Atom<PPU_16x16x16_F32F16F16F32_TN>,
        // MMA_Atom<PPU_16x16x16_F32BF16BF16F32_TN>
        // MMA_Atom<Acompute10000_8x16x16_F32F16F16F32_TN>,
        // MMA_Atom<Acompute10000_8x16x16_F32BF16BF16F32_TN>
        std::conditional_t<USE_MMA_M8, MMA_Atom<PPU_8x16x16_F32F16F16F32_TN>,  MMA_Atom<PPU_16x16x16_F32F16F16F32_TN>>,
        std::conditional_t<USE_MMA_M8, MMA_Atom<PPU_8x16x16_F32BF16BF16F32_TN>, MMA_Atom<PPU_16x16x16_F32BF16BF16F32_TN>>
#elif defined(USE_PPU) && ACOMPUTE_VERSION == 10500
        MMA_Atom<PPU0015_16x16x16_F32F16F16F32_TN>,
        MMA_Atom<PPU0015_16x16x16_F32BF16BF16F32_TN>
#else
        MMA_Atom<PPU_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<PPU_16x8x16_F32BF16BF16F32_TN>
#endif
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<PPU_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__HGGC_ARCH__) &&  __HGGC_ARCH__ >= 100
    using SmemCopyAtom = Copy_Atom<PPU_U32x4_LDSM_N, elem_type>;
    // using SmemCopyAtom = Copy_Atom<PPU_U32x2_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<PPU_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
         int kHeadDimV_ = kHeadDim_,
         bool CrossCut_ = false, bool USE_MMA_M8_ = true, int AtomLayoutQ_ = kNWarps_, int AtomLayoutP_ = kNWarps_,
         int kBlockNPagedPerAiuLoad_ = kBlockN_, int kStages_ = 2, int kNWarps0_ = kNWarps_, bool page_pow2_ = false,
         bool CvtGemm0SwzlLd_ = false, typename Base=Flash_kernel_traits<USE_MMA_M8_, elem_type>>
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNWarps0 = kNWarps0_; // for fp8
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kStages = kStages_;
    static constexpr bool kPagePow2 = page_pow2_;

    static constexpr bool USE_MMA_M8 = USE_MMA_M8_;
    static constexpr bool CrossCut = CrossCut_;
    static constexpr int AtomLayoutQ = CrossCut ? AtomLayoutQ_ : kNWarps0;
    static constexpr int AtomLayoutP = CrossCut ? AtomLayoutP_ : kNWarps;
    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_; // CrossCut ? Share_Q_K_smem_ : 1;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_|| Share_Q_K_smem; // CrossCut ? Is_Q_in_regs_|| Share_Q_K_smem : 1;
    static constexpr int KEEP_Q_NUM = 24;  // tuned for EP32 decode (sweep on case1)
    /// end for CrossCut ///

    static constexpr int MMA_ATOM_M = USE_MMA_M8 ? 8 : 16;
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

    static constexpr bool CvtGemm0SwzlLd = CvtGemm0SwzlLd_;
    static constexpr int kBlockNPagedPerAiuLoad = CvtGemm0SwzlLd ? 16: kBlockNPagedPerAiuLoad_;
    static constexpr int kBlockMPagedPerAiuLoad = CvtGemm0SwzlLd ? 8 : kBlockM;
    static constexpr int kHeadDimPadding = CvtGemm0SwzlLd ? (kHeadDim + 127) / 128 * 128 : kHeadDim;

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    using SmemCopyOpQ = std::conditional_t<
            USE_MMA_M8,
            PPU0010_TSM_LD_SWZL<elem_type, kBlockM, kBlockKSmem, false, false, 1, 2>,
            PPU_TSM_LD_SWZL<elem_type, kBlockM, kBlockKSmem, false, false, 1>
        >;
    using SmemCopyOpK = PPU_TSM_LD_SWZL<elem_type, kBlockNPagedPerAiuLoad, kBlockKSmem, true, false, 
            kBlockN / kBlockNPagedPerAiuLoad * kHeadDim / kBlockKSmem>;
    using SmemCopyOpVt = PPU_TSM_LD_SWZL<elem_type, kBlockNPagedPerAiuLoad, kBlockKSmemV, true, true,
            kBlockN/kBlockNPagedPerAiuLoad * kHeadDimPadding / kBlockKSmemV>;
       
#else
    using SmemCopyOpQ = std::conditional_t<
        CvtGemm0SwzlLd,
        PPU0015_TSM_LD_SWZL_CVT<elem_type, kBlockNPagedPerAiuLoad, kBlockKSmem, kBlockNPagedPerAiuLoad, kHeadDimPadding, false, false,
                                kHeadDimPadding/kBlockKSmem, true, -1>,
        PPU_TSM_LD_SWZL<elem_type, kBlockM, kBlockKSmem, false, false, 1>
    >;
    using SmemCopyOpK = std::conditional_t<
        CvtGemm0SwzlLd,
        PPU0015_TSM_LD_SWZL_CVT<elem_type, kBlockNPagedPerAiuLoad, kBlockKSmem, kBlockNPagedPerAiuLoad, kHeadDimPadding, true, false,
                                      kBlockN / kBlockNPagedPerAiuLoad * kHeadDimPadding/kBlockKSmem, true, -1>,
        PPU_TSM_LD_SWZL<elem_type, kBlockNPagedPerAiuLoad, kBlockKSmem, true, false, kBlockN / kBlockNPagedPerAiuLoad * kHeadDim / kBlockKSmem>
    >;
    using SmemCopyOpVt = std::conditional_t<
        CvtGemm0SwzlLd,
        PPU0015_TSM_LD_SWZL_CVT<elem_type, kBlockNPagedPerAiuLoad, kBlockKSmemV, kBlockNPagedPerAiuLoad, kHeadDimPadding, true, true,
                                      kBlockN / kBlockNPagedPerAiuLoad * kHeadDimPadding / kBlockKSmemV, false, -1>,
        PPU_TSM_LD_SWZL<elem_type, kBlockNPagedPerAiuLoad, kBlockKSmemV, true, true,
                        kBlockN/kBlockNPagedPerAiuLoad * kHeadDimPadding / kBlockKSmemV>
    >;
#endif
    using SmemCopyAtomQ = Copy_Atom<SmemCopyOpQ, elem_type>;
    using SmemCopyAtomK = Copy_Atom<SmemCopyOpK, elem_type>;
    using SmemCopyAtomVt = Copy_Atom<SmemCopyOpVt, elem_type>;
#else
    using SmemCopyAtomQ = SmemCopyAtom;
    using SmemCopyAtomK = SmemCopyAtom;
    using SmemCopyAtomVt = SmemCopyAtomTransposed;
#endif
    static_assert((CrossCut && kStages==3) || kStages == 2, "kStages can be 2 or 3 if CrossCut.");
    /// only for CrossCut ///
    static_assert( CrossCut || kNWarps0 == kNWarps, "kNWarps0 must be same as kNWarps if not CrossCut");
    static_assert(!CrossCut || kNWarps0 % AtomLayoutQ == 0, "kNWarps must be a multiple of AtomLayoutQ if CrossCut");
    static_assert(!CrossCut || kNWarps % AtomLayoutP == 0, "kNWarps must be a multiple of AtomLayoutP if CrossCut");
    static_assert(!CrossCut || kBlockM <= kNWarps0 * 32, "kBlockM must be no larger than kNThreads0 if CrossCut");
    // if kBlockM > kNThreads. softmax should be changed.


    /// TiledMmaS only for CrossCut ///
    using TiledMmaS = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutQ>, Int<kNWarps0/AtomLayoutQ>, _1>>,
        Tile<Int<MMA_ATOM_M * AtomLayoutQ>, Int<16 * kNWarps0/AtomLayoutQ>, _16>>;

    /// The second gemm in CrossCut; gemm in !CrossCut ///
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutP>, Int<kNWarps/AtomLayoutP>, _1>>,
        Tile<Int<MMA_ATOM_M * AtomLayoutP>, Int<16 * kNWarps/AtomLayoutP>, _16>>;

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

    using SmemLayoutKstages = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

    using SmemLayoutQPaged = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockMPagedPerAiuLoad>, Int<kHeadDimPadding>,
        Int<kBlockM/kBlockMPagedPerAiuLoad>>{}));

    using SmemLayoutQTest = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDimPadding>>{}));

    using SmemLayoutKPagedstages = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockNPagedPerAiuLoad>, Int<kHeadDimPadding>,
        Int<kBlockN/kBlockNPagedPerAiuLoad>, Int<kStages>>{}));

    using SmemLayoutVstages = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>, Int<kStages>>{}));
    using SmemLayoutVtstage= decltype(
        composition(SmemLayoutVstages{}, make_ordered_layout(
                Shape<Int<kHeadDimV>, Int<kBlockN>, Int<kStages>>{},
                Step<_2, _1, _3>{})));

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

    /// only for CrossCut ///
    static constexpr int kSwizzleP = kBlockN % 64== 0 ? 3 : 2;// optimize
    // static constexpr int kSwizzleP = 3;
    using SmemLayoutAtomP = decltype(
#if ACOMPUTE_VERSION == 10000
        composition(PPU_Swizzle<kSwizzleP, 3, 3>{},
#else
        composition(Swizzle<kSwizzleP, 3, 3>{},
#endif
                    Layout<Shape<Int<kBlockM>, Int<kBlockN>>,
                           Stride<Int<kBlockN>, _1>>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));
    using SmemCopyAtomP = std::conditional_t<
        USE_MMA_M8,
        Copy_Atom<DefaultCopy, elem_type>, // if m8, stack for tsm.ld.matrix
        SmemCopyAtom
    >;
#if USE_AIU
    using SmemCopyAtomP_TLS = Copy_Atom<PPU_TSM_LD_SWZL<elem_type, kBlockM, kBlockN, false, false, 1>, elem_type>;
#endif
    using SmemCopyAtomS = Copy_Atom<DefaultCopy, elem_type>;
    // using SmemCopyAtomS = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    /// end for CrossCut ///

    static constexpr int kSmemQSize = size(SmemLayoutQPaged{}) * sizeof(Element);
    // static constexpr int kSmemKVSize = (size(SmemLayoutK{}) + size(SmemLayoutV{})) * sizeof(Element);
    // static constexpr int kSmemKVSize = (size(SmemLayoutK{}) * kStages) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKPagedstages{}) * sizeof(Element);
    static constexpr int OSmemSize = size(SmemLayoutO{}) * sizeof(Element);
    static constexpr int OSmemSizeAccum = size(SmemLayoutO{}) * sizeof(ElementAccum);

    static constexpr int kSmemSizeQK = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;

    /// only for CrossCut ///
    static constexpr int kSmemPSize = size(SmemLayoutP{}) * (sizeof(Element)); // store & load P
    static constexpr int kSmemSoftmax = kBlockM * sizeof(ElementAccum)  // rescale o
                                      + (kNWarps0==AtomLayoutQ ? 0: kBlockM * sizeof(ElementAccum) * kNWarps/AtomLayoutQ); // reduce between warps
    static constexpr int kSmemCrossCut = CrossCut ? kSmemPSize + kSmemSoftmax : 0;

    static constexpr int kSmemSize = std::max(kSmemSizeQK + kSmemCrossCut, OSmemSize);
    static constexpr int kSmemSizeAccum = std::max(kSmemSizeQK + kSmemCrossCut, OSmemSizeAccum);

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
        PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using GmemTiledCopyQK = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
#if USE_AIU
    // static_assert(Block_K{} * sizeof(Element) % 32 == 0, "aiu_no_trans: block_k must be multiple of 32B");
    static constexpr int bits_per_aiu_Q = kBlockMPagedPerAiuLoad * kBlockKSmem * sizeof(Element) * 8;
    using Gmem_copy_struct_Q = PPU_AIU_LOAD<cute::C<bits_per_aiu_Q>, Element, false, kBlockMPagedPerAiuLoad, kBlockKSmem>;

    static constexpr int bits_per_aiu_K = kBlockNPagedPerAiuLoad * kBlockKSmem * sizeof(Element) * 8;
    using Gmem_copy_struct_K = PPU_AIU_LOAD<cute::C<bits_per_aiu_K>, Element, false, kBlockNPagedPerAiuLoad, kBlockKSmem>;

    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_Q, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockMPagedPerAiuLoad>, Int<kBlockKSmem>>>{}));
    using GmemTiledCopyK = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_K, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockNPagedPerAiuLoad>, Int<kBlockKSmem>>>{}));
#else
    using GmemTiledCopyQ = GmemTiledCopyQK;
    using GmemTiledCopyK = GmemTiledCopyQK;
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