/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>

#include <cuda_pipeline.h>
#include <cuda_awbarrier.h>
#include "sparse_quant_prefill.h"



#ifdef USE_PPU
#include "ppu/ppu_include.hpp"
#endif

#if !USE_AIU
#define make_mix_tensor make_tensor
#define make_mix_tensor_like(x)  x
#endif

namespace flashmla::v41prefill {
using namespace cute;

// Match the tested H128 / N32 paired-cube pipeline independently of BF16.
enum NamedBarriers : int {
    sScale0Ready = 5,
    sScale1Ready = 6,
    sP0Ready = 7,
    sMInitialized = 9,
    mGroup0 = 10,
    mGroup1 = 11,
    tma_copy_issued = 12
};

template<bool IsFP4>
struct QuantTraits {
    using InputT_ = cutlass::bfloat16_t;
    static constexpr int HEAD_DIM = 512;
    static constexpr int Arch = 89;
    static constexpr int Arch_value = Arch;
    using InputT = InputT_;
    using Element = InputT_;
    using ElementAccum = float;
    using index_t = int64_t;
    using Params = SparseQuantPrefillParams;
    using KernelParams = SparseQuantPrefillParams;
    static constexpr bool kDecode = false;
    static constexpr bool kMixedFP4 = false;
    static constexpr bool kDecodeExtraFP4AsyncEarly = false;
    // Leaf KV/scale offsets only; query/index/output addresses remain index_t.
    static constexpr bool kCacheOffset32 = false;
    static constexpr bool kDecodeNativeFP4Steady = false;
    static constexpr bool kDecodeFixedExtraFP8Steady = false;
    static constexpr bool kDecodeFixedExtraIndex = false;
    static constexpr bool kDecodePageBase64 = false;
    static constexpr bool kDecodeFP8ScopeRow32 = false;
    static constexpr bool kDecodeFP8WG1EarlyPacket = false;
    static constexpr bool kDecodeFP4WG1RemotePVBeforeFinish = false;
    static constexpr bool kDecodeFP4WG1FirstFinishAfterScale1 = false;
    static constexpr bool kDecodeFP4WG0Scale0SchedBound = false;
    static constexpr bool kDecodeFP8WG1SupplyAfterScale1 = false;
    static constexpr bool kDecodeFP8WG0FinishAfterP0 = false;

    static constexpr int BLOCK_SIZE_M = 128;
    static constexpr int PAGE_BLOCK_SIZE = 64;
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 32;
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int NUM_TILES = kHeadDim / 64;  // 9 for 576, 8 for 512
    static constexpr int kHeadDimV = 512;
    static constexpr float Page_In_BlockN = float(kBlockN) / (float)PAGE_BLOCK_SIZE;

    static constexpr int NUM_THREADS = 512;
    static constexpr int kNWarps = NUM_THREADS / 32;
    static constexpr bool Share_Q_K_smem = true;
    static constexpr bool Is_Q_in_regs = true || Share_Q_K_smem;

    static constexpr int NUM_K_BUFS = 3;
    static constexpr bool kQuantCache = true;
    static constexpr bool kAllValidNoLengths = false;
    static constexpr bool kFP4Cache = IsFP4;
    static constexpr bool kUsePairedCubes = Arch == 89 &&
        std::is_same_v<InputT, cutlass::bfloat16_t>;
    static constexpr int kQRowsPerLoad = kUsePairedCubes ? 8 : kBlockM;
    static constexpr bool kCacheLastQ = !kUsePairedCubes || HEAD_DIM == 576;

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

#if ACOMPUTE_VERSION >= 10500
    using SmemCopyOpQ = std::conditional_t<kUsePairedCubes,
        PPU0015_TSM_LD_SWZL_CVT<InputT, 16, 64, kBlockM, 128, false, false, 1, true, 8>,
        PPU_TSM_LD_SWZL<InputT, kBlockM, kBlockKSmem, false, false, 1>>;
#else
    using SmemCopyOpQ = PPU_TSM_LD_SWZL<InputT, kBlockM, kBlockKSmem, false, false, 1>;
#endif
    using SmemCopyAtomQ = Copy_Atom<SmemCopyOpQ, InputT>;

#if 1
#if ACOMPUTE_VERSION >= 10500
    using SmemCopyOpK = std::conditional_t<kUsePairedCubes,
        PPU0015_TSM_LD_SWZL_CVT<InputT, 16, 64, kBlockN, 128, true, false, 1, true, 8>,
        PPU_TSM_LD_SWZL<InputT, kBlockN, kBlockKSmem, true, false, 1>>;
    using SmemCopyOpVt = std::conditional_t<kUsePairedCubes,
        PPU0015_TSM_LD_SWZL_CVT<InputT, 16, 64, kBlockN, 128, true, true, 1, true, -1>,
        PPU_TSM_LD_SWZL<InputT, kBlockN, kBlockKSmem, true, true, 1>>;
#else
    using SmemCopyOpK = PPU_TSM_LD_SWZL<InputT, kBlockN, kBlockKSmem, true, false, 1>;
    using SmemCopyOpVt = PPU_TSM_LD_SWZL<InputT, kBlockN, kBlockKSmem, true, true, 1>;
#endif
    using SmemCopyAtomK = Copy_Atom<SmemCopyOpK, InputT>;
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
#if 1
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
    using SmemLayoutKSim = decltype(tile_to_shape(
        SmemLayoutAtomKSim{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<NUM_K_BUFS>>{}));
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

    struct SharedMemoryPlanBase {
        cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> smem_sQ;
        cute::array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK;
        cute::array_aligned<float, kBlockM> smem_sM;
        cute::array_aligned<float, 2*kBlockM> sL_reduction_wksp;
        cute::array_aligned<float, kBlockM> smem_sScale0;
        cute::array_aligned<float, kBlockM> smem_sScale1;
        __mbarrier_t barrier_Q;
        __mbarrier_t barriers_K0[kHeadDim/256];
        __mbarrier_t barriers_K1[kHeadDim/256];
        cute::array_aligned<int, 4*kBlockN> smem_valid_indices;
    };

    struct SharedMemoryPlanPaired : SharedMemoryPlanBase {
        // Paired D512 keeps all Q tiles in shared memory; P must not overwrite Q7.
        cute::array_aligned<InputT, 2 * kBlockM * kBlockN> smem_sP;
    };
    using SharedMemoryPlan = std::conditional_t<kUsePairedCubes && HEAD_DIM == 512,
        SharedMemoryPlanPaired, SharedMemoryPlanBase>;

    struct SharedMemoryOutPut {
        cute::array_aligned<ElementAccum, cosize_v<SmemLayoutO>> smem_out;
    };

    static constexpr int bits_per_aiu_Q = kQRowsPerLoad * kBlockKSmem * sizeof(InputT) * 8;
    using Gmem_copy_struct_Q = PPU_AIU_LOAD<cute::C<bits_per_aiu_Q>, InputT, false, kQRowsPerLoad, kBlockKSmem>;

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
                    Layout<Shape <Int<kQRowsPerLoad>, Int<kBlockKSmem>>>{}));

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


} // namespace flashmla::v41prefill
