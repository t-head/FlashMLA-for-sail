#pragma once

// Sparse decode WG traits. The HS64 base traits were copied from validated
// C5.17 commit c6f41064b31305e21a94555a3e688192052b0acb and remain independent
// from the evolving dense Traits so M64 retains its original layouts and copy
// atoms.

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>

#include <hggc_pipeline.h>
#include <hggc_awbarrier.h>

#include "decode/dense/traits.h"

#ifdef USE_PPU
#include "ppu/ppu_include.hpp"
// setup.py defines ACOMPUTE_VERSION=10000 for the host pass, under which
// ppu_include.hpp omits PPU1.5 MMA declarations. HS64 deliberately retains
// the validated PPU1.5 atom, so include its arch and traits explicitly and
// compile that atom's fma body with the PPU1.5 ISA gate enabled. Restore the
// global value immediately afterward so the rest of FlashMLA keeps main's
// original FP8/PPU1.0 preprocessing behavior.
#pragma push_macro("ACOMPUTE_VERSION")
#undef ACOMPUTE_VERSION
#define ACOMPUTE_VERSION 10500
#include "ppu/cute/arch/mma_ppu0015.hpp"
#include "ppu/cute/atom/mma_traits_ppu0015.hpp"
#pragma pop_macro("ACOMPUTE_VERSION")
#endif

#if !USE_AIU
#define make_mix_tensor make_tensor
#define make_mix_tensor_like(x)  x
#endif

using namespace cute;

// =============================================================================
// Traits_v2<InputT>: inherits from splitkv Traits<InputT> and shadows
// SharedMemoryPlan to add FP8-specific K-barrier count. FP8 nope/scales and
// BF16 rope are read directly from global memory into registers and
// dequantized in-thread (see load_and_dequant_sparse_K).
// All other typedefs are inherited 1:1 from splitkv.
// =============================================================================
template<typename InputT_, int HeadDimK = 576, bool IsFP8_ = true, int BlockM_ = 128>
struct Traits_v2 : public Traits<InputT_> {
    using Base = Traits<InputT_>;
    using InputT = typename Base::InputT;

    // Whether the KV cache stores FP8 (with dequant) or BF16 (direct read).
    static constexpr bool IsFP8 = IsFP8_;

    // Shadow Base::kHeadDim with template parameter
    static constexpr int kHeadDim = HeadDimK;

    // Shadow kBlockM for BlockM=64 Cross layout support
    static constexpr int kBlockM = BlockM_;
    static constexpr int BLOCK_SIZE_M = BlockM_;

    // (4,1) for BlockM=64; (8,1) original for BlockM=128
    static constexpr int kAtomLayoutM = (BlockM_ == 64) ? 4 : 8;
    static constexpr int kAtomLayoutN = 1;  // Always 1: all N columns in one warp
    static constexpr bool kIsCrossCut = false;  // No cross-N-warp split needed
    // Number of threads covered by TiledMMA; used for wrapping idx_in_warpgroup
    static constexpr int kMmaThreads = kAtomLayoutM * 32;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kAtomLayoutM>, _1, _1>>,
        Tile<Int<16 * kAtomLayoutM>, _16, _16>>;

    // Shadow SmemCopyOpQ/AtomQ for BlockM_ dimension
    using SmemCopyOpQ = PPU_TSM_LD_SWZL<typename Base::InputT, kBlockM, Base::kBlockKSmem, false, false, 1>;
    using SmemCopyAtomQ = Copy_Atom<SmemCopyOpQ, typename Base::InputT>;

    // Shadow SmemLayoutP0 for BlockM_ dimension
    using SmemLayoutAtomP0 = decltype(
#if ACOMPUTE_VERSION == 10000
        composition(PPU_Swizzle<2, 3, 3>{},
#else
        composition(Swizzle<2, 3, 3>{},
#endif
        Layout<Shape<Int<kBlockM>, Int<Base::kBlockN>>,
                        Stride<Int<Base::kBlockN>, _1>>{}));
    using SmemLayoutP0 = decltype(tile_to_shape(
        SmemLayoutAtomP0{},
        Shape<Int<kBlockM>, Int<Base::kBlockN>>{}));

    // Shadow SmemLayoutO for BlockM_ dimension
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<8>, Int<Base::kBlockKSmem>>,
                           Stride<Int<Base::kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<Base::kHeadDimV>>{}));

    // Half-V output layout for two-pass float32 epilogue (BlockM>=128).
    // Full SmemLayoutO (128×512×4 = 256KB) exceeds the PPU M890P SMEM ceiling;
    // 128×256 (128KB) keeps smem_size within device limit.
    using SmemLayoutO_Half = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<Base::kHeadDimV / 2>>{}));

    // Shadow SharedMemoryOutPut for BlockM_ dimension
    struct SharedMemoryOutPut {
        // For BlockM>=128: half-V float buffer (two-pass store_o).
        // For BlockM<128:  full float buffer (single-pass, fits easily).
        static constexpr int kOutBufElems = (kBlockM >= 128)
            ? cosize_v<SmemLayoutO_Half>   // 128×256 = 32768 floats = 131072 bytes
            : cosize_v<SmemLayoutO>;        // 64×512  = 32768 floats = 131072 bytes
        cute::array_aligned<float, kOutBufElems> smem_out;
    };

    // Shadow GmemTiledCopyQ for BlockM_ dimension
    static constexpr int bits_per_aiu_Q = kBlockM * Base::kBlockKSmem * sizeof(typename Base::InputT) * 8;
    using Gmem_copy_struct_Q = PPU_AIU_LOAD<cute::C<bits_per_aiu_Q>, typename Base::InputT, false, kBlockM, Base::kBlockKSmem>;
    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_Q, typename Base::InputT>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockM>, Int<Base::kBlockKSmem>>>{}));

    // -----------------------------------------------------------------------
    // FP8 KV cache layout constants.
    //
    // Two layouts are dispatched on HeadDimK:
    //
    //   * V3.2 (HeadDimK == 576):  per-token interleaved, 656 bytes/token
    //     [0,   512)  FP8 nope        (512 e4m3 bytes, 1 byte/elem)
    //     [512, 528)  FP32 scales     (4 floats; one per 128 nope dims)
    //     [528, 656)  BF16 rope       (64 elems * 2 bytes)
    //
    //   * MODEL1 (HeadDimK == 512):  block-level segmented FP8 cache.
    //     Per-token contiguous payload (576 bytes):
    //       [0,   448)  FP8 nope      (448 e4m3 bytes)
    //       [448, 576)  BF16 rope     (64 elems * 2 bytes)
    //     Block tail (page_block_size * 8 bytes) of e8m0 scales follows the
    //     per-token payload area:
    //       offset_in_block = page_block_size * 576 + off_in_page * 8;
    //       8 bytes per token = 7 used e8m0 scales (one per 64 nope dims) + 1 pad.
    //     PyTorch shape stride is bytes_per_token = 584 (= 576 + 8) per token.
    // -----------------------------------------------------------------------
    static constexpr bool kModel1Layout         = (HeadDimK == 512);

    // -----------------------------------------------------------------------
    // KV cache layout constants -- conditioned on IsFP8.
    //
    // When IsFP8=true (existing FP8 path):
    //   Token is split into nope(FP8) + scales + rope(BF16), dequant required.
    //
    // When IsFP8=false (BF16 direct-read path):
    //   Token is HeadDimK contiguous BF16 values, no dequant/scales.
    //   kBytesPerToken = HeadDimK * 2 (all BF16).
    // -----------------------------------------------------------------------
    static constexpr int kFp8NopeBytesPerToken  = IsFP8
        ? (kModel1Layout ? 448 : 512)
        : 0;  // BF16 path: no FP8 nope segment
    static constexpr int kFp8ScaleBytesPerToken = IsFP8
        ? (kModel1Layout ? 8 : 16)
        : 0;  // BF16 path: no scales
    static constexpr int kRopeElems             = 64;                          // 64 BF16 rope elems in both layouts
    static constexpr int kBf16RopeBytesPerToken = IsFP8 ? (kRopeElems * 2) : 0; // BF16 path: rope is part of contiguous token
    static constexpr bool kHasRope              = IsFP8;  // BF16 path: no separate rope segment
    // Whether there is an EXTRA tile 8 beyond the first 8 tiles in the QK GEMM.
    // HeadDimK=576 (both FP8 V3.2 and BF16): tiles 0-7 cover dims [0,512), tile 8
    // covers dims [512,576) — needed for full dot-product. MODEL1 (512): only 8
    // tiles, no extra tile needed.
    static constexpr bool kHasExtraRopeTile      = !kModel1Layout;
    // Per-token contiguous-payload stride.
    // BF16 path: HeadDimK * sizeof(BF16) = HeadDimK * 2 bytes per token.
    static constexpr int kBytesPerToken = IsFP8
        ? (kModel1Layout
            ? (kFp8NopeBytesPerToken + kRopeElems * 2)                                      // 576 (MODEL1 FP8)
            : (kFp8NopeBytesPerToken + kFp8ScaleBytesPerToken + kRopeElems * 2))            // 656 (V3.2 FP8)
        : (HeadDimK * 2);                                                                   // 1152 (BF16, 576*2)
    // Byte offset of BF16 rope inside the per-token payload (FP8 paths only).
    static constexpr int kRopeOffsetBytes = IsFP8
        ? (kModel1Layout
            ? kFp8NopeBytesPerToken                                                          // 448 (rope right after nope)
            : (kFp8NopeBytesPerToken + kFp8ScaleBytesPerToken))                              // 528 (after nope + per-token scales)
        : 0;  // BF16 path: no separate rope offset
    // Number of nope elements per FP8 scale (one e8m0/fp32 entry covers a
    // contiguous tile of nope dims).  V3.2: 128 dims/scale (4 fp32 scales);
    // MODEL1: 64 dims/scale (7 e8m0 scales + 1 pad byte).
    static constexpr int kScaleTileSize = IsFP8 ? (kModel1Layout ? 64 : 128) : 1;  // BF16: unused, avoid div-by-zero
    static constexpr int kNumScaleTiles = IsFP8 ? (kFp8NopeBytesPerToken / kScaleTileSize) : 0;

    // Shadow SmemLayoutQ to use the correct HeadDimK and kBlockM
    using SmemLayoutQ = decltype(tile_to_shape(
        typename Base::SmemLayoutAtom{},
        Shape<Int<kBlockM>, Int<HeadDimK>>{}));

    // Shadow SmemLayoutK to use the correct HeadDimK
    using SmemLayoutK = decltype(tile_to_shape(
        typename Base::SmemLayoutAtom{},
        Shape<Int<Base::kBlockN>, Int<HeadDimK>, Int<Base::NUM_K_BUFS>>{}));

    // Independent V buffer layout -- mirrors dequant.h's approach.
    // SmemLayoutAtomV uses Swizzle<3,3,3> so that scalar writes are compatible
    // with TSM_LD_SWZL hardware reads. The V buffer physically overlays sK buf 2
    // (unused for topk=32), so no extra SMEM is needed.
    static constexpr int kBlockKSmem_v = 64;
    static constexpr int kSwizzle_v    = 3;
    using SmemLayoutAtomV = decltype(composition(Swizzle<kSwizzle_v, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem_v>>, Stride<Int<kBlockKSmem_v>, _1>>{}));
    using SmemLayoutVDirect = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<Base::kBlockN>, Int<Base::kHeadDimV>>{}));  // (Base::kBlockN, Base::kHeadDimV)
    using SmemLayoutVtDirect = decltype(composition(
        SmemLayoutVDirect{},
        make_layout(Shape<Int<Base::kHeadDimV>, Int<Base::kBlockN>>{}, GenRowMajor{})));  // transposed V view

    // K scalar-store compatible layout -- same pattern as SmemLayoutAtomV.
    // Scalar stores address through Swizzle<kSwizzle_v,3,3> so that TSM_LD_SWZL reads
    // see correctly formatted data (same fix applied to V).
    using SmemLayoutAtomK_Direct = decltype(composition(Swizzle<kSwizzle_v, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem_v>>, Stride<Int<kBlockKSmem_v>, _1>>{}));
    using SmemLayoutKDirect = decltype(tile_to_shape(
        SmemLayoutAtomK_Direct{},
        Shape<Int<Base::kBlockN>, Int<HeadDimK>, Int<Base::NUM_K_BUFS>>{}));

    struct SharedMemoryPlan {
        cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> smem_sQ;
        cute::array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK;
        cute::array_aligned<float, kBlockM>     smem_sM;
        cute::array_aligned<float, kBlockM + 128> sL_reduction_wksp;  // max index = my_row_max + 8 + 128
        cute::array_aligned<float, kBlockM>     smem_sScale0;
        cute::array_aligned<float, kBlockM>     smem_sScale1;

        // MODEL1 (d_qk=512): sQ has exactly 8 tiles (HeadDimK/64) — no spare
        // tile 8 to overlap sP0/sP1. Dedicate space for 2 × SmemLayoutP0.
        // V3.2 (d_qk=576): sP0/sP1 overlap with the consumed sQ tile 8 (rope).
        static constexpr int kSPModel1Elems = kModel1Layout
            ? (2 * kBlockM * Base::kBlockN) : 1;
        cute::array_aligned<InputT, kSPModel1Elems> smem_sP_model1;
        // Valid indices mask: (4 buffers, kBlockN tokens per block)
        // Cross-WG: each WG uses 2 buffers (alternating preload/softmax)
        //   WG0: bufs 0/1, WG1: bufs 2/3
        cute::array_aligned<int, 4 * Base::kBlockN> smem_valid_indices;
        static constexpr int kNumKBarriers = 2;  // Two sub-stages: tiles 0-3 and tiles 4-7/8
        __mbarrier_t barrier_Q;
        __mbarrier_t barriers_K0[kNumKBarriers];
        __mbarrier_t barriers_K1[kNumKBarriers];
    };
};

namespace flashmla::dsa::hs64 {

using namespace cute;

struct Hs64BaseTraits {
    using InputT = cutlass::bfloat16_t;
    using ElementAccum = float;

    static constexpr int kBlockN = Config::BLOCK_SIZE_N;
    static constexpr int kHeadDimV = Config::HEAD_DIM_V;

    static constexpr int NUM_THREADS = 512;

    using MMA_Atom_Arch = MMA_Atom<PPU0015_16x16x16_F32BF16BF16F32_TN>;

    static constexpr int kBlockKSmem = 64;

    using SmemLayoutAtom = Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>;

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, InputT>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, ElementAccum>;

    using SmemCopyAtomS = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, InputT>;

    using SmemCopyAtomP = Copy_Atom<PPU_U32x4_LDSM_N, InputT>;

    static constexpr int bits_per_aiu_KV = kBlockN * kBlockKSmem * sizeof(InputT) * 8;
    using Gmem_copy_struct_KV = PPU_AIU_LOAD<cute::C<bits_per_aiu_KV>, InputT, false, kBlockN, kBlockKSmem>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(InputT);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(NUM_THREADS % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NUM_THREADS / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

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

// The private HS64 instance is BF16/M64-only. Other data types and block sizes
// remain on the existing generic kernels.
template<int HeadDimK, bool GuardIndices = true>
struct Hs64Traits : public Hs64BaseTraits {
    static_assert(HeadDimK == 512 || HeadDimK == 576);

    using Base = Hs64BaseTraits;
    using InputT = typename Base::InputT;

    // Unchecked prefetch is reserved for the host-selected full-tile path.
    static constexpr bool kGuardIndices = GuardIndices;
    static constexpr bool kIsPrefill = false;
    static constexpr int kHeadDim = HeadDimK;
    static constexpr int kBlockM = 64;
    static constexpr int BLOCK_SIZE_M = kBlockM;

    // The (4,2) cross-cut splits kBlockN columns across two N-warps, so their
    // per-row softmax max/sum must be merged (see wg*_bunch_0).
    static constexpr int kAtomLayoutM = 4;
    static constexpr int kAtomLayoutN = 2;
    static constexpr bool kIsCrossCut = true;
    // Keep QK on (4,2), but give PV twice as many N-warps. PV always has the
    // same P[64x64] x V[64x512] shape, independent of HeadDimK, so both HS64
    // instances use the deterministic sliced-load path.
    static constexpr bool kUsePv2x4 = true;
    // Both HS64 head dimensions use the split low/high QK weave. D576 carries
    // one extra high tile, but shares the same buffer hand-off schedule.
    static constexpr bool kUseQkWeave = true;
    // The per-warpgroup reader drains establish explicit ownership of all four
    // V-buffer halves, so the extra CTA-wide rendezvous is redundant.
    static constexpr bool kKeepQkWeaveBar5 = false;
    static constexpr int kPvAtomLayoutM = kUsePv2x4 ? 2 : kAtomLayoutM;
    static constexpr int kPvAtomLayoutN = 8 / kPvAtomLayoutM;
    // Number of threads covered by TiledMMA; used for wrapping idx_in_warpgroup
    static constexpr int kMmaThreads = kAtomLayoutM * kAtomLayoutN * 32;

    static constexpr int NUM_K_BUFS = 2;
    static constexpr int kBlockN = 64;

    // Eight lanes copy 16 B each, giving one contiguous 128 B request per sparse
    // token. A 256-thread warpgroup therefore covers 32 tokens per copy wave;
    // kBlockN=64 uses two waves while kBlockN=32 uses one.
    static constexpr int kGmemElemsPerLoad = 8;                       // uint128 / bf16
    static constexpr int kGmemThrPerTok    = 8;
    static constexpr int kGmemTokPerPass   = (Base::NUM_THREADS / 2) / kGmemThrPerTok;
    static constexpr int kGmemPasses       = kBlockN / kGmemTokPerPass;
    static constexpr int kGmemElemsPerTile = kGmemThrPerTok * kGmemElemsPerLoad;
    static constexpr int kNumKTiles        = HeadDimK / kGmemElemsPerTile;
    static constexpr int kValidTokensPerWord = 4;
    static constexpr int kValidWords = (kBlockN + kValidTokensPerWord - 1) / kValidTokensPerWord;
    static constexpr int kStage1KTiles     = 256 / kGmemElemsPerTile;  // dims [0,256)
    static_assert(kGmemTokPerPass * kGmemPasses == kBlockN,
                  "kBlockN must be a whole number of gather passes");
    static_assert(kNumKTiles * kGmemElemsPerTile == HeadDimK,
                  "kGmemElemsPerTile must divide HeadDimK");
    static_assert(kStage1KTiles * kGmemElemsPerTile == 256,
                  "kGmemElemsPerTile must divide the 256-dim stage boundary");

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kAtomLayoutM>, Int<kAtomLayoutN>, _1>>,
        Tile<Int<16 * kAtomLayoutM>, Int<16 * kAtomLayoutN>, _16>>;

    using TiledMmaPV = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kPvAtomLayoutM>, Int<kPvAtomLayoutN>, _1>>,
        Tile<Int<16 * kPvAtomLayoutM>, Int<16 * kPvAtomLayoutN>, _16>>;

    // M64 Q copy atom.
    using SmemCopyOpQ = PPU_TSM_LD_SWZL<typename Base::InputT, kBlockM, Base::kBlockKSmem, false, false, 1>;
    using SmemCopyAtomQ = Copy_Atom<SmemCopyOpQ, typename Base::InputT>;

    // Shadow SmemCopyOpK/Vt on kBlockN. The TSM_LD_SWZL descriptor's row count
    // sets the CUBE stride the hardware uses to step between HeadDim tiles, so
    // it must match sK's actual row count. Base's 32 makes tile t land at
    // t*(32*64) instead of t*(kBlockN*64) -- with kBlockN=64 the rope tile 8
    // reads tile 4's nope data instead (probe: QK score 576 vs 8928).
    using SmemCopyOpK = PPU_TSM_LD_SWZL<typename Base::InputT, kBlockN, Base::kBlockKSmem, true, false, 1>;
    using SmemCopyAtomK = Copy_Atom<SmemCopyOpK, typename Base::InputT>;
    using SmemCopyOpVt = PPU_TSM_LD_SWZL<typename Base::InputT, kBlockN, Base::kBlockKSmem, true, true, 1>;
    using SmemCopyAtomVt = Copy_Atom<SmemCopyOpVt, typename Base::InputT>;

    // The shared-P layout is specific to M64N64. A 64-element BF16 row needs
    // all three swizzle bits to match TSM_LD_SWZL, including m-bit-2 -> k-bit-5.
    static_assert(kBlockM == 64 && kBlockN == 64, "HS64 requires M64N64 tiles");
    static constexpr int kSwizzleP0 = 3;
    using SmemLayoutAtomP0 = decltype(
        composition(Swizzle<kSwizzleP0, 3, 3>{},
        Layout<Shape<Int<kBlockM>, Int<kBlockN>>,
                        Stride<Int<kBlockN>, _1>>{}));
    using SmemLayoutP0 = decltype(tile_to_shape(
        SmemLayoutAtomP0{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));

    // Shadow SmemCopyAtomP for the (4,2) cross-cut P read: Base's
    // PPU_U32x4_LDSM_N partition mis-maps the N-warp half of sP under (4,2)
    // (probe: K-tile 1 read back all zeros, PV lost 3/4). The verified non-WI
    // cross-cut kernel (flash_sparse_fwd_kernel.h, 10500 branch) reads sP with
    // PPU_TSM_LD_SWZL + mix tensor + N-warp-folded thread slice; match it.
    // (8,1) keeps Base's LDSM_N + bare sP, which is what M128 was validated on.
    using SmemCopyAtomP = std::conditional_t<
        kIsCrossCut,
        Copy_Atom<PPU_TSM_LD_SWZL<typename Base::InputT, kBlockM, kBlockN, false, false, 1>,
                  typename Base::InputT>,
        typename Base::SmemCopyAtomP>;

    // M64 output layout.
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<8>, Int<Base::kBlockKSmem>>,
                           Stride<Int<Base::kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<Base::kHeadDimV>>{}));

    // Half-V output layout for two-pass float32 epilogue (BlockM>=128).
    // Full SmemLayoutO (128×512×4 = 256KB) exceeds the PPU M890P SMEM ceiling;
    // 128×256 (128KB) keeps smem_size within device limit.
    using SmemLayoutO_Half = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<Base::kHeadDimV / 2>>{}));

    // M64 output staging.
    struct SharedMemoryOutPut {
        // For BlockM>=128: half-V float buffer (two-pass store_o).
        // For BlockM<128:  full float buffer (single-pass, fits easily).
        static constexpr int kOutBufElems = (kBlockM >= 128)
            ? cosize_v<SmemLayoutO_Half>   // 128×256 = 32768 floats = 131072 bytes
            : cosize_v<SmemLayoutO>;        // 64×512  = 32768 floats = 131072 bytes
        cute::array_aligned<float, kOutBufElems> smem_out;
    };

    // M64 Q global-memory copy.
    static constexpr int bits_per_aiu_Q = kBlockM * Base::kBlockKSmem * sizeof(typename Base::InputT) * 8;
    using Gmem_copy_struct_Q = PPU_AIU_LOAD<cute::C<bits_per_aiu_Q>, typename Base::InputT, false, kBlockM, Base::kBlockKSmem>;
    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_Q, typename Base::InputT>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockM>, Int<Base::kBlockKSmem>>>{}));

    // D576 has a ninth QK tile covering dims [512,576); D512 has eight tiles.
    static constexpr bool kHasExtraKTile = HeadDimK == 576;
    // D512 keeps its last Q tile in the existing rQ8 fragment. D576 leaves Q8
    // in shared memory and instead keeps Q4..Q7 resident: their four contiguous
    // raw cubes form the alternate high-K/V bank used by the no-bar4 pipeline.
    static constexpr bool kUseEvenHighBank = true;
    static constexpr bool kCacheLastQTile = !kHasExtraKTile;
    static constexpr int kCachedQTile = kCacheLastQTile ? (kNumKTiles - 1) : 0;
    // Both HS64 instances reuse Q many times. Keep the adjacent high tile in
    // registers as well: tile 6 for D512 and tile 7 for D576.
    static constexpr bool kCachePrevQTile = true;
    static constexpr int kCachedPrevQTile = kCachePrevQTile ? (kNumKTiles - 2) : 0;
    // Tile 4 is the first high-half Q consumer after the K mbarrier wait. Keep it
    // resident alongside tiles 6 and 7 on the active d512 cross-cut path.
    static constexpr bool kCacheFirstHighQTile = true;
    static constexpr int kCachedFirstHighQTile = kCacheFirstHighQTile ? 4 : 0;
    static constexpr bool kCacheD576MiddleQTiles = kHasExtraKTile;
    static constexpr int kBytesPerToken = HeadDimK * sizeof(InputT);

    // Shadow SmemLayoutQ to use the correct HeadDimK and kBlockM
    using SmemLayoutQ = decltype(tile_to_shape(
        typename Base::SmemLayoutAtom{},
        Shape<Int<kBlockM>, Int<HeadDimK>>{}));

    // Shadow SmemLayoutK to use the correct HeadDimK
    using SmemLayoutK = decltype(tile_to_shape(
        typename Base::SmemLayoutAtom{},
        Shape<Int<kBlockN>, Int<HeadDimK>, Int<NUM_K_BUFS>>{}));

    // V view layout over each current K buffer.
    static constexpr int kBlockKSmem_v = 64;
    static constexpr int kSwizzle_v    = 3;
    using SmemLayoutAtomV = decltype(composition(Swizzle<kSwizzle_v, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem_v>>, Stride<Int<kBlockKSmem_v>, _1>>{}));
    using SmemLayoutVDirect = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<Base::kHeadDimV>>{}));  // (kBlockN, Base::kHeadDimV)
    using SmemLayoutVtDirect = decltype(composition(
        SmemLayoutVDirect{},
        make_layout(Shape<Int<Base::kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));  // transposed V view

    // K layout matching the TSM_LD_SWZL read pattern.
    using SmemLayoutAtomK_Direct = decltype(composition(Swizzle<kSwizzle_v, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem_v>>, Stride<Int<kBlockKSmem_v>, _1>>{}));
    using SmemLayoutKDirect = decltype(tile_to_shape(
        SmemLayoutAtomK_Direct{},
        Shape<Int<kBlockN>, Int<HeadDimK>, Int<NUM_K_BUFS>>{}));

    // Four contiguous 64-dim tiles. For D576 the same layout is rebound either
    // to buf0 tile4 or to the raw sQ4 base, allowing the even high half to
    // ping-pong without changing the QK/PV operand layouts.
    static constexpr int kHigh4Dim = 4 * kBlockKSmem_v;
    using SmemLayoutKHigh4 = decltype(tile_to_shape(
        SmemLayoutAtomK_Direct{},
        Shape<Int<kBlockN>, Int<kHigh4Dim>>{}));
    using SmemLayoutVtHigh4 = decltype(composition(
        SmemLayoutKHigh4{},
        make_layout(Shape<Int<kHigh4Dim>, Int<kBlockN>>{}, GenRowMajor{})));

    struct SharedMemoryPlan {
        cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> smem_sQ;
        cute::array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK;
        cute::array_aligned<float, kBlockM>     smem_sM;
        cute::array_aligned<float, kBlockM + 128> sL_reduction_wksp;  // max index = my_row_max + 8 + 128
        cute::array_aligned<float, kBlockM>     smem_sScale0;
        cute::array_aligned<float, kBlockM>     smem_sScale1;

        // Cross-cut (4,2) only: per-(WG,N-warp) partial softmax max/sum staging,
        // used to merge the two N-warps' partials. Layout: [WG*2 + n_warp][row].
        // 1 element placeholder on the non-cross-cut (8,1) path.
        cute::array_aligned<float, kIsCrossCut ? 4 * kBlockM : 1> smem_cross_n_reduction;

        // Two independent sP buffers are required by the cross-cut remote-P
        // path. In particular, d_qk=576 must not reuse sQ tile 8: both
        // warpgroups read that Q tile into registers, and a leading warpgroup
        // could otherwise overwrite it through sP0 before its peer has read it.
        // Each base remains CUBE-aligned for TSM_LD_SWZL addressing.
        static constexpr size_t kSPElems = size_t{2} * kBlockM * kBlockN;
        static constexpr size_t kSPAlign = size_t{kBlockM} * kBlockN * sizeof(InputT);
        cute::array_aligned<InputT, kSPElems, kSPAlign> smem_sP;
        // Four alternating valid-mask buffers: WG0 uses 0/1 and WG1 uses 2/3.
        // D512 packs four token-valid bits per word. D576 retains each warp's
        // raw ballot and decodes both copy waves from eight words per buffer.
        cute::array_aligned<unsigned int, 4 * kValidWords> smem_valid_indices;
        static constexpr int kNumKBarriers = 2;  // Two sub-stages: tiles 0-3 and tiles 4-7/8
        __mbarrier_t barrier_Q;
        __mbarrier_t barriers_K0[kNumKBarriers];
        __mbarrier_t barriers_K1[kNumKBarriers];

        // D512 cannot reuse raw Q4..Q7: Q5 remains live in shared memory.
        // The loader statically checks offset 229376 and 8192-byte alignment.
        // D576 keeps the existing cached-Q bank; its placeholder fits padding.
        struct EmptyEvenHighBank {};
        using EvenHighBank = std::conditional_t<
            kHasExtraKTile, EmptyEvenHighBank,
            cute::array_aligned<InputT, cosize_v<SmemLayoutKHigh4>, kSPAlign>>;
        EvenHighBank smem_even_high;
    };
};

// Prefill assigns one complete sparse query to each CTA, without split-KV.
template<int HeadDimK>
struct Hs64PrefillTraits : public Hs64Traits<HeadDimK, false> {
    static constexpr bool kIsPrefill = true;
};

}  // namespace flashmla::dsa::hs64
