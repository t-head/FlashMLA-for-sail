/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>

#include <hggc_pipeline.h>
#include <hggc_awbarrier.h>

#include "config.h"

#ifdef USE_PPU
#include "ppu/ppu_include.hpp"
#endif

#if !USE_AIU
#define make_mix_tensor make_tensor
#define make_mix_tensor_like(x)  x
#endif

using namespace cute;

#define DSA_SIM_AIU 1
template<typename InputT_, int HEAD_DIM, int Arch>
struct DSA_Traits {
    static constexpr int Arch_value = Arch;
    using InputT = InputT_;
    using Element = InputT_;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr int BLOCK_SIZE_M = Config::BLOCK_SIZE_M;
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE;
    static constexpr int kBlockM = Config::BLOCK_SIZE_M;
    static constexpr int kBlockN = Config::BLOCK_SIZE_N;
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int NUM_TILES = kHeadDim / 64;  // 9 for 576, 8 for 512
    static constexpr int kHeadDimV = Config::HEAD_DIM_V;
    static constexpr float Page_In_BlockN = float(kBlockN) / (float)PAGE_BLOCK_SIZE;

    static constexpr int NUM_THREADS = 512;
    static constexpr int kNWarps = NUM_THREADS / 32;
    static constexpr bool Share_Q_K_smem = true;
    static constexpr bool Is_Q_in_regs = true || Share_Q_K_smem;

    static constexpr int NUM_K_BUFS = 3;

    static_assert(std::is_same_v<InputT, cutlass::bfloat16_t> || std::is_same_v<InputT, cutlass::half_t>);

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<InputT, cutlass::half_t>,
#if ACOMPUTE_VERSION == 10000
        MMA_Atom<PPU_16x16x16_F32F16F16F32_TN>,
        MMA_Atom<PPU_16x16x16_F32BF16BF16F32_TN>
#else
        MMA_Atom<PPU0015_16x16x16_F32F16F16F32_TN>,
        MMA_Atom<PPU0015_16x16x16_F32BF16BF16F32_TN>
#endif
    >;

    static constexpr int kBlockKSmem = 64;
    static constexpr int kSwizzle = 3;

    using SmemCopyOpQ = PPU_TSM_LD_SWZL<InputT, kBlockM, kBlockKSmem, false, false, 1>;
    using SmemCopyAtomQ = Copy_Atom<SmemCopyOpQ, InputT>;

#if DSA_SIM_AIU
    using SmemCopyOpK = PPU_TSM_LD_SWZL<InputT, kBlockN, kBlockKSmem, true, false, 1>;
    using SmemCopyAtomK = Copy_Atom<SmemCopyOpK, InputT>;
    using SmemCopyOpVt = PPU_TSM_LD_SWZL<InputT, kBlockN, kBlockKSmem, true, true, 1>;
    using SmemCopyAtomVt = Copy_Atom<SmemCopyOpVt, InputT>;
#else
    using SmemCopyAtomK = Copy_Atom<PPU_U32x4_LDSM_N, InputT>;
    using SmemCopyAtomVt = Copy_Atom<PPU_U16x8_LDSM_T, InputT>;
#endif

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<8>, Int<1>, _1>>,
        Tile<Int<16 * 8>, Int<16>, _16>>;

using SmemLayoutAtomQ = Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>;
#if DSA_SIM_AIU
#if ACOMPUTE_VERSION == 10000
    using SmemLayoutAtomKSim = decltype(tile_to_shape(
        composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}),
        Shape<Int<kBlockN>, Int<kBlockKSmem>>{}));
#else
    using SmemLayoutAtomKSim = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
#endif
    using SmemLayoutAtomK = Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>;
#else
#if ACOMPUTE_VERSION == 10000
    using SmemLayoutAtomK = decltype(composition(
        PPU_Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
#else
    using SmemLayoutAtomK = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
#endif
#endif

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
#if DSA_SIM_AIU
    using SmemLayoutKSim = decltype(tile_to_shape(
        SmemLayoutAtomKSim{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<NUM_K_BUFS>>{}));
#endif
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<NUM_K_BUFS>>{}));

    using SmemLayoutV_ = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutV = decltype(composition(
        SmemLayoutV_{},
        make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})
    ));

#if !DSA_SIM_AIU
    using SmemLayoutVHf_ = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDimV/2>>{}));
    using SmemLayoutVHf = decltype(composition(
        SmemLayoutVHf_{},
        make_layout(Shape<Int<kHeadDimV/2>, Int<kBlockN>>{}, GenRowMajor{})
    ));
    using SmemLayoutVNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVHf{}));
#endif

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, InputT>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, ElementAccum>;

    using SmemLayoutAtomP0 = decltype(
#if ACOMPUTE_VERSION == 10000
        composition(PPU_Swizzle<2, 3, 3>{},
#else
        composition(Swizzle<2, 3, 3>{},
#endif
        Layout<Shape<Int<kBlockM>, Int<kBlockN>>,
                        Stride<Int<kBlockN>, _1>>{}));

    using SmemLayoutP0 = decltype(tile_to_shape(
        SmemLayoutAtomP0{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));

    using SmemCopyAtomS = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, InputT>;

    using SmemCopyAtomP = Copy_Atom<PPU_U32x4_LDSM_N, InputT>;

    struct SharedMemoryPlan {
        cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> smem_sQ;
        cute::array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK;
        cute::array_aligned<float, kBlockM> smem_sM;
        cute::array_aligned<float, 2*kBlockM> sL_reduction_wksp;
        cute::array_aligned<float, kBlockM> smem_sScale0;
        cute::array_aligned<float, kBlockM> smem_sScale1;
        __mbarrier_t barrier_Q;
        __mbarrier_t barriers_K0[kHeadDim/256];
        __mbarrier_t barriers_K1[kHeadDim/256];
    };

    struct SharedMemoryOutPut {
        cute::array_aligned<ElementAccum, cosize_v<SmemLayoutO>> smem_out;
    };

    static constexpr int bits_per_aiu_Q = kBlockM * kBlockKSmem * sizeof(InputT) * 8;
    using Gmem_copy_struct_Q = PPU_AIU_LOAD<cute::C<bits_per_aiu_Q>, InputT, false, kBlockM, kBlockKSmem>;

    static constexpr int bits_per_aiu_KV = kBlockN * kBlockKSmem * sizeof(InputT) * 8;
    using Gmem_copy_struct_KV = PPU_AIU_LOAD<cute::C<bits_per_aiu_KV>, InputT, false, kBlockN, kBlockKSmem>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(InputT);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(NUM_THREADS % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NUM_THREADS / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_Q, InputT>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockM>, Int<kBlockKSmem>>>{}));

    using GmemTiledCopyKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_KV, InputT>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockN>, Int<kBlockKSmem>>>{}));

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, InputT>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum =
        Layout<Shape <Int<NUM_THREADS / 16>, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>;

    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _1>>{}));  // Val layout, 4 vals per store
};
