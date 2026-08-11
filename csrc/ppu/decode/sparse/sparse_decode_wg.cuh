// =============================================================================
// sparse_decode_wg.cuh
//
// Sparse FP8 decode warp-interleave (T2) variant -- forked verbatim from
// csrc/ppu/decode/dense/splitkv_mla_kernel.cuh. The double-warpgroup ping-pong skeleton
// (wg0_subroutine / wg1_subroutine / softmax / O store / mbarrier handshake)
// is reused as-is. The only sparse-specific divergences live in:
//   * Q / K-load layer  (replaces launch_kv_tiles_copy AIU LOAD with cp.async
//                        + indices indirection + per-warpgroup local FP8→BF16
//                        dequant)
//   * sparse mask via GMEM re-read of indices (applied to rP0 / rP1 before softmax)
// extra-K and attn_sink are supported (extra-K load path and attn_sink
// epilogue below); only the causal path is removed -- not supported, gated at
// host dispatch.
// =============================================================================

#pragma once

#include <cute/tensor.hpp>
#include <climits>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cute/util/debug.hpp>

#include "params.h"
// Reuse splitkv config / traits verbatim (kBlockM=128, kBlockN=32,
// NUM_K_BUFS=3, etc). Sparse-specific SMEM extensions are added in a follow-up
// stage; for the skeleton fork we use the splitkv layout 1:1.
#include "decode/dense/traits.h"
#include "acc_vreg_fraga.h"
#include "ppuxx/decode/combine/combine.h"

#include "kerutils/common/common.h"
#include "kerutils/host/hardware_info.h"
#include "kerutils/device/ppu/dequant.cuh"

#include <hggc_ad.h>

using namespace cute;
using cutlass::arch::NamedBarrier;

// e8m0 scale dtype alias (matches kerutils dequant.cuh)
using fp8_e8m0 = __hg_fp8_e8m0;

// ---------------------------------------------------------------------------
// Localized from the patch's flash_splitkv/splitkv_dsa.h: async K tile copy
// helpers for WG kernels, verbatim except for the rename. They are placed in
// namespace flash and drop the dsa_ prefix to avoid ODR clashes with the
// global launch_kv_tiles_dsa_wg* templates in
// prefill/sparse/sparse_prefill_wg.cuh.
// Async K tile copy from GMEM to SMEM for WG kernels (prefill & decode).
// Iterates over head_dim tiles [START, END), issues cp.async per tile,
// then arrives the mbarrier.
// ---------------------------------------------------------------------------
namespace flash {

template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_wg_copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K), partitioned
    Tensor<Engine1, Layout1> &sKV)       // (BLOCK_N, HEAD_DIM_K), swizzled, partitioned
{
    Tensor cur_gKV = gKV(_, _, Int<START_HEAD_DIM_TILE_IDX>{});
    Tensor cur_sKV = sKV(_, _, Int<START_HEAD_DIM_TILE_IDX>{});
    cute::copy(tiled_copy, cur_gKV, cur_sKV);

    if constexpr (START_HEAD_DIM_TILE_IDX + 1 < END_HEAD_DIM_TILE_IDX)
    {
        launch_kv_tiles_wg_copy<START_HEAD_DIM_TILE_IDX + 1, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV);
    }
}

template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_wg(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV,
    Tensor<Engine1, Layout1> &sKV,
    __mbarrier_t *barriers_K)
{
    launch_kv_tiles_wg_copy<START_HEAD_DIM_TILE_IDX, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV);
    cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
}

} // namespace flash

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
        Shape<Int<Base::kBlockN>, Int<Base::kHeadDimV>>{}));  // (32, 512)
    using SmemLayoutVtDirect = decltype(composition(
        SmemLayoutVDirect{},
        make_layout(Shape<Int<Base::kHeadDimV>, Int<Base::kBlockN>>{}, GenRowMajor{})));  // (512, 32) transposed view

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

// Here we use MAX_INIT_VAL_SM to initialize sM, and MAX_INIT_VAL for masking
// The reason is that, we need to calculate new_max = max(sM(row_idx), cur_max*scale_softmax_log2)
// so we must guarantee that MAX_INIT_VAL*scale_softmax_log2 < MAX_INIT_VAL_SM
static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

template <int AtomLayoutM = 8>
__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup)
{
    // In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
    // You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
    int row_idx = ((idx_in_warpgroup / 32) % AtomLayoutM) * 16 + local_row_idx * 8 + (idx_in_warpgroup % 32 / 4);
    return row_idx;
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
    Tensor cur_gKV = gKV(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
    Tensor cur_sKV = sKV(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
    cute::copy(tiled_copy, cur_gKV, cur_sKV);

    if constexpr (START_HEAD_DIM_TILE_IDX + 1 < END_HEAD_DIM_TILE_IDX) {
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
    for (int i = 0; i < size<2>(tCrA); ++i) {
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), tCrC);
    }
}

__forceinline__ __device__ void kernel_sleep_ns()
{
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
    // Wrap thread index for BlockM=64 (4-warp TiledMMA, 8-warp WG)
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    const int cute_warp = (warp_idx % T::kAtomLayoutM) * 32;
    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename T::SmemCopyAtomK{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(cute_warp);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(cute_warp);

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
                    rP, cute_idx); \
        } else if constexpr(TILE_IDX < 4) { \
            qkt_gemm_one_tile_sQ(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, \
                    smem_thr_copy_Q, smem_thr_copy_K, \
                    sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), \
                    sKV0_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV0_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, cute_idx); \
        } else  { \
            qkt_gemm_one_tile_sQ(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, \
                    smem_thr_copy_Q, smem_thr_copy_K, \
                    sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), \
                    sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, cute_idx); \
        }

    if constexpr (PHASE_IDX == 0) {
        // In PHASE-0, warpgroup 0 calculates Q K^T for the first 4 tiles
        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
    } else if constexpr (PHASE_IDX == 1) {
        // In PHASE-1, warpgroup 1 calculates Q K^T for all the 9 tiles
        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        if constexpr (T::kHasExtraRopeTile) { QKT_GEMM_ONE_TILE(8); }

        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
            kernel_sleep_ns();
        };
        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
        cur_phase = (cur_phase + 1) & 1;
    } else {
        // In PHASE-2, warpgroup 0 calculates Q K^T for the last 5 tiles
        static_assert(PHASE_IDX == 2);

        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        if constexpr (T::kHasExtraRopeTile) { QKT_GEMM_ONE_TILE(8); }
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
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);

    Tensor rK = thr_mma.partition_fragment_B(sKV); // (MMA, 1, 576/16=36)
    auto smem_tiled_copy_K = make_tiled_copy_B(typename T::SmemCopyAtomK{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(cute_idx);
    Tensor rK_copy_view = smem_thr_copy_K.retile_D(rK);
    auto tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV));
    CUTE_STATIC_ASSERT_V(size<1>(tSsK) == size<1>(rK_copy_view)); // M
    cute::copy(smem_tiled_copy_K, tSsK, rK_copy_view);

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
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    const int cute_warp = (warp_idx % T::kAtomLayoutM) * 32;
    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);

    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(cute_warp);
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
    typename T::TiledMma tiled_mma;
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    const int cute_warp = (warp_idx % T::kAtomLayoutM) * 32;
    auto smem_tiled_copy_P = make_tiled_copy_A(typename T::SmemCopyAtomP{}, tiled_mma);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(cute_idx);
    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(cute_warp);

    auto tSsP = smem_thr_copy_P.partition_S(sP);
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));

    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);
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
    typename Engine4, typename Layout4,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ auto wg0_bunch_0(
    // Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,	// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,	// ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0,	// (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,	// (BLOCK_SIZE_M)
    float rL[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf)
{
    // Preload smem_valid_indices into registers to hide SMEM latency
    // from the __shfl_xor_sync critical path.
    // ACOMPUTE_VERSION==10000: each thread needs 8 values (2 groups of 4)
    int r_valid[8];
    {
        int lane4 = idx_in_warpgroup % 4;
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            int base = (k * 16 + lane4) % T::kBlockN;
            r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
            r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
            r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
            r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
        }
    }

    // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
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
        // TOCTOU fix: read sM once to avoid race with concurrent warpgroup.
        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);

        __syncwarp(); // Make sure all reads have finished before updating sM

        // For kAtomLayoutM < 8, warps kAtomLayoutM..7 alias the same SMEM rows
        // as warps 0..(kAtomLayoutM-1). Only primary warps write sM/sScale to
        // avoid intra-WG write-write races.
        if (idx_in_warpgroup % 4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
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

    auto rPb_ret = convert_acc<typename T::InputT>(rP0);
    return rPb_ret;
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
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ auto wg1_bunch_0(
    // Tensor<Engine0, Layout0> &rP1b,	// ((2, 2, 8), 1, 1)
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
    float r_cur_max_in[2] = nullptr)  // if non-null, skip mask+max (pre-computed)
{
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);

        float cur_max;
        if (r_cur_max_in) {
            cur_max = r_cur_max_in[local_row_idx];
        } else {
            // Preload smem_valid_indices into registers
            int r_valid[8];
            {
                int lane4 = idx_in_warpgroup % 4;
                #pragma unroll
                for (int k = 0; k < 2; k++) {
                    int base = (k * 16 + lane4) % T::kBlockN;
                    r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
                    r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
                    r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
                    r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
                }
            }
            // Mask, and get row-wise max
            cur_max = MAX_INIT_VAL;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8) {
                if constexpr (IS_BLK0_LAST) {
                    // wg0's block was the last; wg1 has no real block.
                    rP1(i) = rP1(i + 1) = rP1(i + 2) = rP1(i + 3) = MAX_INIT_VAL;
                } else {
                    int k_base = ((i / 8) % 2) * 4;
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
        // Only primary warps write sM/sScale (prevents aliased warp double-write).
        if (idx_in_warpgroup % 4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
            sM(row_idx) = new_max;
            sScale1(row_idx) = scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        if constexpr (!IS_BLK0_LAST) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8) {
                rP1(i) = exp2f(rP1(i) * scale_softmax_log2 - new_max);
                rP1(i + 1) = exp2f(rP1(i + 1) * scale_softmax_log2 - new_max);
                rP1(i + 2) = exp2f(rP1(i + 2) * scale_softmax_log2 - new_max);
                rP1(i + 3) = exp2f(rP1(i + 3) * scale_softmax_log2 - new_max);
                cur_sum += (rP1(i) + rP1(i + 1) + rP1(i + 2) + rP1(i + 3));
            }
        }

        float cur_scale_for_o1 = scale_for_old * sScale0(row_idx);

        // Update rL
        rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
    }

    auto rP1b_ret = convert_acc<typename T::InputT>(rP1);
    return rP1b_ret;
}
#else
template <
    typename T,
    bool DO_OOB_FILLING,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ void wg0_bunch_0(
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
    int valid_indices_buf)
{
    // Preload smem_valid_indices into registers to hide SMEM latency
    // from the __shfl_xor_sync critical path.
    // #else path: each thread needs 8 values (4 groups of 2)
    int r_valid[8];
    {
        int lane4 = idx_in_warpgroup % 4;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int base = (k * 8 + lane4 * 2) % T::kBlockN;
            r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
            r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
        }
    }

     // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);

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
        if (idx_in_warpgroup%4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
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
    float r_cur_max_in[2] = nullptr)  // if non-null, skip mask+max (pre-computed)
{
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);

        float cur_max;
        if (r_cur_max_in) {
            cur_max = r_cur_max_in[local_row_idx];
        } else {
            // Preload smem_valid_indices into registers
            int r_valid[8];
            {
                int lane4 = idx_in_warpgroup % 4;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int base = (k * 8 + lane4 * 2) % T::kBlockN;
                    r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
                    r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
                }
            }
            // Mask, and get row-wise max
            cur_max = MAX_INIT_VAL;
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
            cur_max *= scale_softmax_log2;
        }

        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);
        __syncwarp();
        // Only primary warps write sM/sScale (prevents aliased warp double-write).
        if (idx_in_warpgroup%4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
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

        float cur_scale_for_o1 = scale_for_old * sScale0(row_idx);

        // Update rL
        rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
    }
}
#endif

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
#if ACOMPUTE_VERSION == 10000
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        int base = (k * 16 + lane4) % T::kBlockN;
        r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
        r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
        r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
        r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8) {
            if constexpr (IS_BLK0_LAST) {
                rP1(i) = rP1(i + 1) = rP1(i + 2) = rP1(i + 3) = MAX_INIT_VAL;
            } else {
                int k_base = ((i / 8) % 2) * 4;
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
#else
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int base = (k * 8 + lane4 * 2) % T::kBlockN;
        r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
        r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
    }
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
#endif
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
    typename T::TiledMma tiled_mma;
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    auto r2s_copy = make_tiled_copy_A(typename T::SmemCopyAtomS{}, tiled_mma);
    ThrCopy thr_copy = r2s_copy.get_slice(cute_idx);
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
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    auto r2s_copy = make_tiled_copy_C(typename T::SmemCopyAtomS{}, tiled_mma);
    ThrCopy thr_copy = r2s_copy.get_slice(cute_idx);
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
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    const int warp_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);
    const int cute_warp = (warp_idx % T::kAtomLayoutM) * 32;

    auto thr_mma = tiled_mma.get_thread_slice(cute_idx);
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(cute_warp);
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
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
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
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
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
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
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

    if constexpr (!IS_NO_SPLIT && T::kBlockM >= 128) {
        // Two-pass float32 output for BlockM>=128 to fit PPU M890P SMEM limit.
        using SmemLayoutO_Half = typename T::SmemLayoutO_Half;

        Tensor sHalfBuf = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(sO_addr)),
            SmemLayoutO_Half{});  // (kBlockM, kHeadDimV/2) = 128x256 floats

        typename T::TiledMma tiled_mma;
        const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
        auto r2s_tiled_copy = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
        ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(cute_idx);
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

        typename T::TiledMma tiled_mma;
        const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
        auto r2s_tiled_copy = make_tiled_copy_C(
            SmemTiledCopyO{}, tiled_mma);

        ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(cute_idx);
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

// Phase 1: Compute token_ptr from pre_token_idx + write smem_valid_indices.
// Does NOT issue cp.async, does NOT write to sK buffer.
// Can be placed anywhere (fills SIMT gaps during TC execution).
// Compile-time version: for mainloop (USE_EXTRA as template param)
template<int S, bool USE_EXTRA, typename T, typename TensorVI>
__forceinline__ __device__ typename T::InputT* compute_K_addr_bf16(
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
    bool &is_valid
) {
    using Base   = typename T::Base;
    using InputT = typename T::InputT;
    constexpr int kBlockN        = Base::kBlockN;
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

    int token_id = tidx / 8;
    int t_idx = (token_id < tile_valid) ? *pre_token_idx : -1;
    is_valid = (t_idx >= 0);
    if constexpr (S == 0) {
        if (tidx % 8 == 0) {
            smem_valid_indices(vi_buf, token_id) = is_valid;
        }
    }

    // k_stride_elems set above via if constexpr

    InputT* token_ptr = k_base_ptr + (tidx % 8) * 8;
    if (is_valid) {
        int page_idx    = t_idx >> __builtin_ctz(page_block_size); // t_idx / page_block_size;
        int off_in_page = t_idx & ((page_block_size & (-page_block_size)) - 1); // t_idx - page_idx * page_block_size;
        token_ptr = k_base_ptr
            + page_idx * k_stride_elems
            + off_in_page * (kBytesPerToken / sizeof(InputT))
            + (tidx % 8) * 8;
    }
    return token_ptr;
}

// Runtime version: for prolog (use_extra as runtime param)
template<int S, typename T, typename TensorVI>
__forceinline__ __device__ typename T::InputT* compute_K_addr_bf16_dynamic(
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
    bool &is_valid,
    bool use_extra
) {
    if (use_extra) {
        return compute_K_addr_bf16<S, true, T>(params, batch_idx, block_idx_kv, seqlen_k, tidx, ori_block_max,
            smem_valid_indices, vi_buf, pre_token_idx, extra_seqlen_k, is_valid);
    } else {
        return compute_K_addr_bf16<S, false, T>(params, batch_idx, block_idx_kv, seqlen_k, tidx, ori_block_max,
            smem_valid_indices, vi_buf, pre_token_idx, extra_seqlen_k, is_valid);
    }
}

// Phase 2: Issue cp.async using precomputed token_ptr + prefetch next token index.
// LOAD_USE_EXTRA removed: k_base_ptr is dummy (overwritten by token_ptr), only PREFETCH_USE_EXTRA matters
template<int S, int E, bool PREFETCH_USE_EXTRA, typename T, bool DO_PREFETCH = true, typename TensorSK>
__forceinline__ __device__ void issue_K_load_bf16(
    const Flash_fwd_mla_params &params,
    TensorSK &sK_buf,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    __mbarrier_t *barriers_K,
    int tidx,
    int ori_block_max,
    typename T::InputT* token_ptr,
    bool is_valid,
    int *pre_token_idx,
    int next_block_idx,
    int end_block_idx,
    int extra_seqlen_k
) {
    using Base   = typename T::Base;
    using InputT = typename T::InputT;
    constexpr int kBlockN          = Base::kBlockN;
    constexpr int kThreadsPerWg    = Base::NUM_THREADS / 2;

    using KVCacheGmem = flash::KVCacheGmemBf16<InputT, kBlockN, kThreadsPerWg, T::kHeadDim>;
    using GmemTiledCopyKNoAiu = typename KVCacheGmem::GmemTiledCopy;
    GmemTiledCopyKNoAiu gmem_tiled_copy_K;

    // Dummy k_base_ptr — actual address comes from token_ptr (set below)
    InputT* k_base_ptr = reinterpret_cast<InputT*>(params.k_ptr);
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK_buf);
    Tensor gK_tok = make_tensor(make_gmem_ptr(k_base_ptr),
        Shape<Int<kBlockN>, Int<T::kHeadDim>>{},
        make_stride(params.k_row_stride, _1{}));
    Tensor tKgK_tok = gmem_thr_copy_K.partition_S(gK_tok);
    tKgK_tok.data() = token_ptr;

    gmem_tiled_copy_K.pred = is_valid;
    flash::launch_kv_tiles_wg<S, E>(gmem_tiled_copy_K, tKgK_tok, tKsK, barriers_K);

    if constexpr (DO_PREFETCH) {
        if (next_block_idx < end_block_idx) {
            int next_eff_block;
            const int *next_idx_base;
            if constexpr (PREFETCH_USE_EXTRA) {
                next_eff_block = next_block_idx - ori_block_max;
                next_idx_base = params.extra_indices_ptr + static_cast<int64_t>(batch_idx) * params.extra_indices_batch_stride + next_eff_block * kBlockN;
            } else {
                next_eff_block = next_block_idx;
                next_idx_base = params.indices_ptr + static_cast<int64_t>(batch_idx) * params.indices_batch_stride + next_eff_block * kBlockN;
            }
            int token_id = tidx / 8;
            *pre_token_idx = __ldg(next_idx_base + token_id);
        }
    }
}

template<int S, int E, typename T, bool DO_PREFETCH = true, typename TensorSK, typename TensorVI>
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
    // Dynamic wrapper for prolog: compute addr + dispatch issue_K_load via template
    bool use_extra = (ori_block_max >= 0) && (block_idx_kv >= ori_block_max);
    bool prefetch_use_extra = (ori_block_max >= 0) && (next_block_idx >= ori_block_max);
    bool is_valid;
    typename T::InputT* token_ptr = compute_K_addr_bf16_dynamic<S, T>(
        params, batch_idx, block_idx_kv, seqlen_k, tidx, ori_block_max,
        smem_valid_indices, vi_buf, pre_token_idx, extra_seqlen_k, is_valid, use_extra);
    
    if (prefetch_use_extra) {
        issue_K_load_bf16<S, E, true, T, DO_PREFETCH>(
            params, sK_buf, batch_idx, block_idx_kv, seqlen_k, barriers_K,
            tidx, ori_block_max, token_ptr, is_valid,
            pre_token_idx, next_block_idx, end_block_idx, extra_seqlen_k);
    } else {
        issue_K_load_bf16<S, E, false, T, DO_PREFETCH>(
            params, sK_buf, batch_idx, block_idx_kv, seqlen_k, barriers_K,
            tidx, ori_block_max, token_ptr, is_valid,
            pre_token_idx, next_block_idx, end_block_idx, extra_seqlen_k);
    }
}

// ============================================================================
// load_and_dequant_sparse_K_staged: staged K load with per-stage dim range.
//
// Splits the FP8 dequant into two stages:
//   Stage S=0, E=4: dims [0,   256)  -> nope only, barrier 0
//   Stage S=4, E=9: dims [256, 576)  -> nope [256,512) + rope [0,64), barrier 1
//
//   * Processes only [kDimStart, kDimEnd) instead of full [0, 576)
//   * smem_valid_indices written only when S == 0 (using vi_buf_idx)
//   * Arrives ONLY barriers_K[S/4] instead of both barriers
//   * No sV_direct parameter (V read handled elsewhere)
// ============================================================================
template<int S, int E, typename T, typename TensorSK, typename TensorVI>
__forceinline__ __device__ void load_and_dequant_sparse_K_staged(
    const Flash_fwd_mla_params &params,
    TensorSK &sK,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    int buf_idx,
    __mbarrier_t *barriers_K,
    int tidx,
    int ori_block_max,
    TensorVI &smem_valid_indices,
    int vi_buf_idx
) {
    using Base   = typename T::Base;
    using InputT = typename T::InputT;       // cutlass::bfloat16_t
    constexpr int kBlockN          = Base::kBlockN;                  // 32
    constexpr int kFp8NopeBytes    = T::kFp8NopeBytesPerToken;       // 512 (V3.2) or 448 (MODEL1)
    constexpr int kFp8ScaleBytes   = T::kFp8ScaleBytesPerToken;      // 16  (V3.2) or 8   (MODEL1)
    constexpr int kBytesPerToken   = T::kBytesPerToken;              // 656 (V3.2) or 576 (MODEL1)
    constexpr int kRopeOffset      = T::kRopeOffsetBytes;            // 528 (V3.2) or 448 (MODEL1)
    constexpr int kTileSize        = T::kScaleTileSize;              // 128 (V3.2) or 64  (MODEL1)
    constexpr int kRopeElems       = T::kRopeElems;                  // 64
    constexpr int kThreadsPerWg    = Base::NUM_THREADS / 2;          // 256
    (void)kRopeElems;  // unused in staged path; kRopeRange replaces it

    // ---- Stage dim range constants ----
    constexpr int kDimStart     = S * T::kBlockKSmem;      // 0 or 256
    constexpr int kDimEnd       = E * T::kBlockKSmem;      // 256 or 576
    constexpr int kNopeDimStart = kDimStart < kFp8NopeBytes ? kDimStart : kFp8NopeBytes;
    constexpr int kNopeDimEnd   = kDimEnd   < kFp8NopeBytes ? kDimEnd   : kFp8NopeBytes;
    constexpr int kRopeDimStart = kDimStart > kFp8NopeBytes ? (kDimStart - kFp8NopeBytes) : 0;
    constexpr int kRopeDimEnd   = kDimEnd   > kFp8NopeBytes ? (kDimEnd   - kFp8NopeBytes) : 0;
    constexpr int kNopeRange    = kNopeDimEnd - kNopeDimStart;
    constexpr int kRopeRange    = kRopeDimEnd - kRopeDimStart;

    // ---- Determine if this block is in extra cache ----
    const bool use_extra = (ori_block_max >= 0) && (block_idx_kv >= ori_block_max);
    const int effective_block = use_extra ? (block_idx_kv - ori_block_max) : block_idx_kv;

    const int *indices_base = use_extra
        ? (params.extra_indices_ptr
           + static_cast<int64_t>(batch_idx) * params.extra_indices_batch_stride
           + effective_block * kBlockN)
        : (params.indices_ptr
           + static_cast<int64_t>(batch_idx) * params.indices_batch_stride
           + effective_block * kBlockN);
    const int page_block_size = use_extra ? params.extra_page_block_size : params.page_block_size;
    constexpr size_t kStrideElemSize = T::IsFP8 ? 1 : sizeof(InputT);
    const size_t k_batch_stride = (use_extra
        ? static_cast<size_t>(params.extra_k_batch_stride)
        : static_cast<size_t>(params.k_batch_stride)) * kStrideElemSize;
    const uint8_t *k_base = reinterpret_cast<const uint8_t*>(
        use_extra ? params.extra_k_ptr : params.k_ptr);

    // ---- Tile-local valid-token count ----
    int valid_len;
    if (use_extra) {
        valid_len = params.extra_topk_len_ptr
            ? __ldg(params.extra_topk_len_ptr + batch_idx)
            : params.extra_topk;
    } else {
        valid_len = params.topk_len_ptr
            ? __ldg(params.topk_len_ptr + batch_idx)
            : params.topk;
    }
    if (!use_extra && seqlen_k >= 0 && seqlen_k < valid_len) {
        valid_len = seqlen_k;
    }
    int tile_valid = valid_len - effective_block * kBlockN;
    if (tile_valid < 0) tile_valid = 0;
    if (tile_valid > kBlockN) tile_valid = kBlockN;

    // ---- 0. Valid indices (only for stage 0) ----
    if constexpr (S == 0) {
        if (tidx < kBlockN) {
            int t_idx = (tidx < tile_valid) ? __ldg(indices_base + tidx) : -1;
            smem_valid_indices(vi_buf_idx, tidx) = (t_idx >= 0);
        }
    }

    // ---- 1. Nope dequant: [kNopeDimStart, kNopeDimEnd) ----
    if constexpr (kNopeRange > 0) {
        constexpr int kTotalNopeElems     = kBlockN * kNopeRange;
        constexpr int kNopeElemsPerThread = kTotalNopeElems / kThreadsPerWg;
        static_assert(kTotalNopeElems % kThreadsPerWg == 0,
                      "staged nope: total elems must be divisible by warpgroup threads");
        #pragma unroll
        for (int i = 0; i < kNopeElemsPerThread; ++i) {
            int elem_id       = tidx + i * kThreadsPerWg;
            int token_id      = elem_id / kNopeRange;
            int elem_in_range = elem_id - token_id * kNopeRange;
            int elem_in_tok   = elem_in_range + kNopeDimStart;

            InputT bf16_val = InputT(0.0f);
            if (token_id < tile_valid) {
                int t_idx = __ldg(indices_base + token_id);
                if (t_idx >= 0) {
                    int page_idx    = t_idx / page_block_size;
                    int off_in_page = t_idx - page_idx * page_block_size;
                    const uint8_t *gmem_tok = k_base
                        + page_idx * k_batch_stride
                        + off_in_page * kBytesPerToken;

                    // FP8 nope value
                    uint8_t fp8_raw = __ldg(gmem_tok + elem_in_tok);
                    __hg_fp8_e4m3 fp8_val = *reinterpret_cast<const __hg_fp8_e4m3*>(&fp8_raw);

                    float scale_val;
                    if constexpr (T::kModel1Layout) {
                        // MODEL1: e8m0 scales in block-level tail segment.
                        int tile_idx = elem_in_tok / kTileSize;
                        const uint8_t *scale_ptr = k_base
                            + page_idx * k_batch_stride
                            + page_block_size * kBytesPerToken
                            + off_in_page * kFp8ScaleBytes
                            + tile_idx;
                        uint8_t scale_raw = __ldg(scale_ptr);
                        fp8_e8m0 scale_e8m0 = *reinterpret_cast<const fp8_e8m0*>(&scale_raw);
                        scale_val = float(scale_e8m0);
                    } else {
                        // V3.2: per-token FP32 scales (one per 128 nope dims).
                        int tile_idx = elem_in_tok / kTileSize;
                        scale_val = *reinterpret_cast<const float*>(
                            gmem_tok + kFp8NopeBytes + tile_idx * sizeof(float));
                    }

                    float result = float(fp8_val) * scale_val;
                    bf16_val = InputT(result);
                }
            }
            sK(token_id, elem_in_tok, buf_idx) = bf16_val;
        }
    }

    // ---- 2. Rope copy: [kRopeDimStart, kRopeDimEnd) ----
    if constexpr (kRopeRange > 0 && T::kHasRope) {
        constexpr int kTotalRopeElems      = kBlockN * kRopeRange;
        constexpr int kRopeElemsPerThread = kTotalRopeElems / kThreadsPerWg;
        static_assert(kTotalRopeElems % kThreadsPerWg == 0,
                      "staged rope: total elems must be divisible by warpgroup threads");
        #pragma unroll
        for (int j = 0; j < kRopeElemsPerThread; ++j) {
            int elem_id  = tidx + j * kThreadsPerWg;
            int token_id = elem_id / kRopeRange;
            int rope_idx = elem_id - token_id * kRopeRange;

            InputT rope_val = InputT(0.0f);
            if (token_id < tile_valid) {
                int t_idx = __ldg(indices_base + token_id);
                if (t_idx >= 0) {
                    int page_idx    = t_idx / page_block_size;
                    int off_in_page = t_idx - page_idx * page_block_size;
                    const uint8_t *gmem_tok = k_base
                        + page_idx * k_batch_stride
                        + off_in_page * kBytesPerToken;

                    // BF16 rope value
                    rope_val = *reinterpret_cast<const InputT*>(
                        gmem_tok + kRopeOffset + rope_idx * sizeof(InputT));
                }
            }
            sK(token_id, kFp8NopeBytes + rope_idx, buf_idx) = rope_val;
        }
    }

    // ---- 3. Warpgroup-level sync ----
    {
        int _bar_id = 6 + (int)(tidx >> 8);
        asm volatile("ppu.bar.sync %0, %1;" :: "r"(_bar_id), "r"(kThreadsPerWg));
    }
    __threadfence_block();

    // ---- 4. Arrive only one barrier (key difference from full function) ----
    constexpr int kBarrierIdx = S / 4;  // 0 or 1
    cutlass::arch::cpasync_barrier_arrive_noinc(&barriers_K[kBarrierIdx]);
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
    Tensor<Engine0, Layout0> &sV_direct)
{
    // Use SmemLayoutVtDirect (swizzled composition) to read V transposed
    // from the independent V buffer. This produces a ComposedLayout that
    // make_mix_tensor_like can handle (rank 2, rank0=1, rank1=1).
    Tensor sVt = make_tensor(sV_direct.data(), (typename T::SmemLayoutVtDirect){});
    return flat_divide(sVt, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
}

template <
    typename T,
    bool IS_R,
    typename Engine0, typename Layout0>
__forceinline__ __device__ auto get_half_V2(
    int block_idx,
    Tensor<Engine0, Layout0> sV_direct)
{
    // Same as get_half_V, for alternate calling convention.
    Tensor sVt = make_tensor(sV_direct.data(), (typename T::SmemLayoutVtDirect){});
    return flat_divide(sVt, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
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
    return long(__ldg(block_table_ptr + block_table_idx) * params.k_batch_stride + block_table_offset * params.k_row_stride);
}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool NEXT_EXTRA,
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
    typename EngineSV, typename LayoutSV,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void wg0_subroutine(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> &tKgK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> sK,
    Tensor<EngineSV, LayoutSV> &sV_direct,
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
    __mbarrier_t barriers_K0[9],
    __mbarrier_t barriers_K1[9],
    bool &cur_phase_K0,
    const Flash_fwd_mla_params &params,
    int* block_table_ptr,
    int seqlen_k,
    int total_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup,
    int wg_idx,
    int &kv_idx,
    // [FP8-WI fix3] sparse FP8 path: only batch_idx is needed -- staging
    // SMEM is gone (register-direct dequant in load_and_dequant_sparse_K).
    int batch_idx,
    int ori_block_max,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    // Pre-fetch pipeline: token index for current block (in/out)
    int *pre_token_idx,
    int *pre_token_idx_b,
    int extra_seqlen_k,
    typename T::InputT* &precomp_ptr0,
    typename T::InputT* &precomp_ptr1,
    bool &precomp_valid0,
    bool &precomp_valid1
) {
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx+2;
    int nxt_block1 = block_idx+3;

    // [Task #68] Derive V directly from K buffers. cur_sK0 holds even block (local),
    // cur_sK1 holds odd block (remote). V = first 512 dims of each K buffer.
    // Cross-WG smem_valid_indices: 4 bufs, WG0 uses 0/1, WG1 uses 2/3 (alternating)
    int vi_softmax_wg0 = (block_idx / 2) % 2;       // 0 or 1
    auto sV_local = make_tensor(cur_sK0.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV0L = get_half_V<T, 0>(sV_local);
    auto sV_remote = make_tensor(cur_sK1.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV1L = get_half_V<T, 0>(sV_remote);

    auto nxt_sK1 = cur_sK0;
    // Calc P0 = softmax(P0) and signal sScale0Ready before K load
    // (WG1 is waiting for sScale0Ready — arriving earlier lets WG1 start sooner)
#if ACOMPUTE_VERSION == 10000
    Tensor rPb = wg0_bunch_0< T, IS_BLK0_LAST || IS_BLK1_LAST > (rP0, rO0, sScale0, sM, rL, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg0);
#else
    Tensor rPb = make_tensor<typename T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
    wg0_bunch_0< T, IS_BLK0_LAST || IS_BLK1_LAST > (rPb, rP0, rO0, sScale0, sM, rL, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg0);
#endif

    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale0Ready);

    // [BlockM=64 only] CTA-wide barrier for sScale0/sM cross-WG visibility.
    // On PPU M890, named barriers do NOT guarantee cross-wg SMEM visibility.
    // wg0 wrote sScale0(row) and sM(row) -- wg1 reads them after its matching
    // arrive_and_wait(sScale0Ready).  Paired with wg1's __syncthreads after
    // its arrive_and_wait(sScale0Ready).
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // Cross-WG: wg0 loads tiles 0-3 for nxt_block0 (async cp.async, doesn't block)
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (!T::IsFP8) {
            issue_K_load_bf16<0, 4, NEXT_EXTRA, T>(params, nxt_sK0, batch_idx, nxt_block0, seqlen_k, &barriers_K0[0],
                idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
                pre_token_idx, nxt_block0 + 2, end_block_idx, extra_seqlen_k);
        } else {
            int vi_preload_wg0 = 1 - vi_softmax_wg0;         // 1 or 0
            load_and_dequant_sparse_K_staged<0, 4, T>(
                params, sK, batch_idx, nxt_block0, seqlen_k,
                /*buf_idx=*/(kv_idx + 2) % 3, &barriers_K0[0],
                idx_in_warpgroup, ori_block_max, smem_valid_indices, vi_preload_wg0);
        }
    }

    // Issue rO0 += rPb @ sV0L
    wg0_scale0_rO0<T>(rO0, sScale0, idx_in_warpgroup);

    if constexpr (T::kIsCrossCut) {
        // (4,2) layout: each N-warp only has 16/32 P columns, localP broken.
        // Save rPb to sP0 temporarily, per-WG barrier, then remoteP from SMEM.
        save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);
        { int _cbar = 6 + (int)(threadIdx.x >> 8); asm volatile("ppu.bar.sync %0, %1;" :: "r"(_cbar), "r"(256)); }
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0L, rO0, idx_in_warpgroup, wg_idx);
    } else {
        warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, idx_in_warpgroup, wg_idx);
    }

    // sScale1Ready also signals sP1 is ready (WG1 saves sP1 before arriving)
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    // [BlockM=64 only] CTA-wide barrier for sScale1/sM cross-WG visibility.
    // wg1 wrote sScale1(row) and sM(row) in wg1_bunch_0; wg0 reads sScale1
    // below in wg0_scale_rP0.  Paired with wg1's __syncthreads after its
    // arrive(sScale1Ready).
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // Cross-WG: wg0 loads tiles 0-3 for nxt_block1
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (!T::IsFP8) {
            issue_K_load_bf16<0, 4, NEXT_EXTRA, T>(params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[0],
                idx_in_warpgroup, ori_block_max, precomp_ptr1, precomp_valid1,
                pre_token_idx_b, nxt_block1 + 2, end_block_idx, extra_seqlen_k);
        } else {
            int vi_preload_wg1 = 3 - vi_softmax_wg0;    // 3 or 2 (BF16 nxt_block1 preload; FP8 skips nxt_block1)
            load_and_dequant_sparse_K_staged<0, 4, T>(
                params, sK, batch_idx, nxt_block1, seqlen_k,
                /*buf_idx=*/kv_idx, &barriers_K1[0],
                idx_in_warpgroup, ori_block_max, smem_valid_indices, vi_preload_wg1);
        }
    }

    wg0_scale_rP0<T>(sScale1, rP0, rPb, idx_in_warpgroup);
    save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);

    // Sync with WG1: ensures sP0 not overwritten before WG1 reads it.
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sP0Ready);

    // [BlockM=64 only] CTA-wide barrier for cross-wg SMEM visibility.
    // On PPU, named barriers do NOT guarantee SMEM visibility across wgs.
    // Only barrier 0 (__syncthreads) reliably flushes SMEM writes for
    // cross-wg reads.  Ensures wg0 can see V data written by wg1 (remote PV)
    // and wg1 can see V data written by wg0.  Paired with wg1's matching
    // __syncthreads after its arrive_and_wait(sP0Ready).
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // sP1 ready via sScale1Ready. remote PV can proceed after sP0Ready sync.
    if constexpr (!IS_BLK0_LAST) {
        wg0_rescale_rO0<T>(rO0, sScale1, rL, idx_in_warpgroup);
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, idx_in_warpgroup, wg_idx);
    }

    // WG0 QK computes scores for next_block0 (always exists when !BLK0 && !BLK1)
    // Split-buf GEMM reads nxt_sK0 (tiles 0-3) + nxt_sK1 (tiles 4-8).
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP0);
        warpgroup_cooperative_qkt_gemm<T, 0>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
    }

    // After QK: precompute addr for NEXT iteration's K loads
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (!T::IsFP8) {
            int next_nxt_block0 = block_idx + 4;
            precomp_ptr0 = compute_K_addr_bf16<0, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block0, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, vi_softmax_wg0,
                pre_token_idx, extra_seqlen_k, precomp_valid0);
            int next_nxt_block1 = block_idx + 5;
            int next_vi_preload_wg1 = vi_softmax_wg0 + 2;
            precomp_ptr1 = compute_K_addr_bf16<0, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block1, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, next_vi_preload_wg1,
                pre_token_idx_b, extra_seqlen_k, precomp_valid1);
        }
    }

    // Issue P0 = Q @ K0^T (next_block0 always exists)
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        warpgroup_cooperative_qkt_gemm<T, 2>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
    }

    // [BlockM=64 only] CTA-wide barrier at K-block iteration boundary.
    // Without this sync, wg0's PV-remote in the NEXT iteration reads a buffer
    // loaded by wg1 in THIS iteration (sV1L = cur_sK1), but wg0 has no
    // per-buffer mbarrier wait on barriers_K1.  bar.sync 0 forces all writes
    // to be visible to the other warpgroup before the next iteration begins.
    // Paired with wg1's matching __syncthreads at the same logical location.
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        kv_idx = (kv_idx + 2) % 3;
        cur_sK0 = sK(_, _, kv_idx);
        cur_sK1 = sK(_, _, (kv_idx + 1) % 3);
        nxt_sK0 = sK(_, _, (kv_idx + 2) % 3);
    }
}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool NEXT_EXTRA,
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
    typename EngineSV, typename LayoutSV,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void wg1_subroutine(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> &tKgK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> sK,
    Tensor<EngineSV, LayoutSV> &sV_direct,
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
    __mbarrier_t barriers_K0[9],
    __mbarrier_t barriers_K1[9],
    bool &cur_phase_K1,
    const Flash_fwd_mla_params &params,
    int* block_table_ptr,
    int seqlen_k,
    int total_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup,
    int wg_idx,
    int &kv_idx,
    // [FP8-WI fix3] sparse FP8 path: only batch_idx is needed -- staging
    // SMEM is gone (register-direct dequant in load_and_dequant_sparse_K).
    int batch_idx,
    int ori_block_max,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    // Pre-fetch pipeline: token index for current block (in/out)
    int *pre_token_idx,
    int *pre_token_idx_b,
    int extra_seqlen_k,
    typename T::InputT* &precomp_ptr0,
    typename T::InputT* &precomp_ptr1,
    bool &precomp_valid0,
    bool &precomp_valid1
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

    // Cross-WG: wg1 loads tiles 4-8 for nxt_block1 at the very beginning.
    // Does NOT write smem_valid_indices — wg0 writes it for the same block.
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
        if constexpr (!T::IsFP8) {
            issue_K_load_bf16<4, kNumTiles, NEXT_EXTRA, T>(params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[1],
                idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
                pre_token_idx, nxt_block1 + 2, end_block_idx, extra_seqlen_k);
        } else {
            load_and_dequant_sparse_K_staged<4, kNumTiles, T>(
                params, sK, batch_idx, nxt_block1, seqlen_k,
                /*buf_idx=*/(kv_idx + 2) % 3, &barriers_K1[0],
                idx_in_warpgroup, ori_block_max, smem_valid_indices, 0);
        }
    }

    // Pre-compute cur_max before barrier (doesn't depend on sM or sScale0)
    float r_cur_max[2];
    wg1_bunch_0_pre<T, IS_BLK0_LAST>(r_cur_max, rP1, params.scale_softmax_log2, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg1);

    // Wait for sScale0 from WG0 (delayed: cur_max already computed)
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);

    // [BlockM=64 only] CTA-wide barrier for sScale0/sM cross-WG visibility.
    // Pairs with wg0's __syncthreads after its arrive(sScale0Ready).
    // After this point wg1 can safely read sScale0(row) and sM(row) written by wg0.
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

#if ACOMPUTE_VERSION == 10000
    Tensor rP1b = wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, false>(sScale1, rO1, sM, rL, sScale0, rP1, params.scale_softmax_log2, start_token_idx+T::kBlockN, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg1, r_cur_max);
#else
    Tensor rP1b = make_tensor<typename T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
    wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, false>(rP1b, sScale1, rO1, sM, rL, sScale0, rP1, params.scale_softmax_log2, start_token_idx+T::kBlockN, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg1, r_cur_max);
#endif
    
    // Save sP1 early (before sScale1Ready) so WG0 can read it without extra barrier
    if constexpr (!IS_BLK0_LAST) {
        save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
    }
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    // [BlockM=64 only] CTA-wide barrier for sScale1/sM cross-WG visibility.
    // Pairs with wg0's __syncthreads after its arrive_and_wait(sScale1Ready).
    // wg1 wrote sScale1(row) and sM(row) in wg1_bunch_0 above; this ensures
    // wg0 sees the updated values when it reads them.
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // Issue rO1 += rP1b @ sV1R
    wg1_scale0_rO1<T>(rO1, sScale0, sScale1, idx_in_warpgroup);
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (T::kIsCrossCut) {
            // (4,2) layout: each N-warp only has 16/32 P columns, localP broken.
            // sP1 already saved above, use per-WG barrier then remoteP from SMEM.
            { int _cbar = 6 + (int)(threadIdx.x >> 8); asm volatile("ppu.bar.sync %0, %1;" :: "r"(_cbar), "r"(256)); }
            warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1R, rO1, idx_in_warpgroup, wg_idx);
        } else {
            warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV1R, rO1, idx_in_warpgroup, wg_idx);
        }
    }

    // Sync with WG0: ensures WG0 doesn't overwrite sP0 before we read it.
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sP0Ready);

    // [BlockM=64 only] CTA-wide barrier for cross-wg SMEM visibility.
    // Matches the __syncthreads() in wg0_subroutine after its arrive(sP0Ready).
    // Also implicitly publishes wg1's sP1 store to wg0 for remote PV GEMM.
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // Cross-WG: wg1 loads tiles 4-8 for nxt_block0
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
        if constexpr (!T::IsFP8) {
            issue_K_load_bf16<4, kNumTiles, NEXT_EXTRA, T>(params, nxt_sK0, batch_idx, nxt_block0, seqlen_k, &barriers_K0[1],
                idx_in_warpgroup, ori_block_max, precomp_ptr1, precomp_valid1,
                pre_token_idx_b, nxt_block0 + 2, end_block_idx, extra_seqlen_k);
        } else {
            load_and_dequant_sparse_K_staged<4, kNumTiles, T>(
                params, sK, batch_idx, nxt_block0, seqlen_k,
                /*buf_idx=*/kv_idx, &barriers_K0[0],
                idx_in_warpgroup, ori_block_max, smem_valid_indices, 0);
        }
    }

    // After QK: precompute addr for NEXT iteration's K loads
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (!T::IsFP8) {
            int next_nxt_block1 = block_idx + 5;
            precomp_ptr0 = compute_K_addr_bf16<4, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block1, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                pre_token_idx, extra_seqlen_k, precomp_valid0);
            int next_nxt_block0 = block_idx + 4;
            precomp_ptr1 = compute_K_addr_bf16<4, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block0, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                pre_token_idx_b, extra_seqlen_k, precomp_valid1);
        }
    }

    // Remote PV GEMM: reads sP0 (ready via sP0Ready) and sV0R
    warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx);

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP1);
        warpgroup_cooperative_qkt_gemm<T, 1>(sQ, nxt_sK0, nxt_sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
    }

    // [BlockM=64 only] CTA-wide barrier at K-block iteration boundary.
    // Mirrors the __syncthreads() at the end of wg0_subroutine.  Required so
    // wg1's PV-remote in the NEXT iteration sees the K data wg0 loaded in
    // THIS iteration (sV1R = cur_sK1 maps to wg0's previously loaded buf).
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        kv_idx = (kv_idx + 2) % 3;
        cur_sK1 = sK(_, _, kv_idx);
        cur_sK0 = sK(_, _, (kv_idx + 1) % 3);
        nxt_sK1 = sK(_, _, (kv_idx + 2) % 3);
    }
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


// [SPARSE-WI Stage B] Is_causal template parameter dropped: sparse decode
// expresses causal masking via indices, not via traditional causal mask path.
template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_sparse_decode_wg_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
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
    // Use swizzled layout so scalar stores in load_and_dequant_sparse_K
    // are TSM_LD_SWZL-compatible (same pattern as sV_direct using SmemLayoutVDirect).
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_sK.data()), (typename T::SmemLayoutKDirect){});
    // [FP8-WI fix10] For MODEL1, sQ only has 8 tiles (512/64) -- no spare tile 8.
    // Use dedicated smem_sP_model1; for 576-dim (FP8 V3.2 or BF16), reuse sQ tile 8 (consumed into rQ8).
    typename T::InputT *sP_base;
    if constexpr (T::kHasExtraRopeTile) {
        // 576-dim: sQ has 9 tiles (576/64), tile 8 already consumed into rQ8.
        sP_base = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, _0{}, _8{}).data().get();
    } else {
        // MODEL1: sQ has exactly 8 tiles (512/64), no spare tile 8; use dedicated buffer.
        sP_base = plan.smem_sP_model1.data();
    }
    Tensor sP0 = make_tensor(make_smem_ptr(sP_base), (typename T::SmemLayoutP0){});
    Tensor sP1 = make_tensor(sP0.data() + sP0.size(), (typename T::SmemLayoutP0){});
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL_reduction_wksp = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<T::BLOCK_SIZE_M + 128>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    // Valid indices mask: (4 buffers, kBlockN tokens)
    // Cross-WG: WG0 uses bufs 0/1, WG1 uses bufs 2/3 (alternating preload/softmax)
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(plan.smem_valid_indices.data()),
        Shape<_4, Int<T::kBlockN>>{}, Stride<Int<T::kBlockN>, _1>{});
    // char* sO_addr = (char*)plan.smem_sK0.data();	// Overlap with sK0 and sK1
    char *sO_addr = (char *)plan.smem_sQ.data(); // Overlap with sK0 and sK1

    // // Define TMA stuffs
    __mbarrier_t *barrier_Q = &(plan.barrier_Q);
    __mbarrier_t *barriers_K0 = plan.barriers_K0;
    __mbarrier_t *barriers_K1 = plan.barriers_K1;

    // // Initialize TMA barriers
    static_assert(T::SharedMemoryPlan::kNumKBarriers == 2, "FP8 WI kernel expects exactly 2 K sub-stage barriers");
    if (threadIdx.x == 0) {
        __mbarrier_init(barrier_Q, 32);
        CUTLASS_PRAGMA_UNROLL
        // Initialize two sub-stage barriers per block:
        //   barriers_Kx[0] → tiles 0-3 (dims 0-255) readiness
        //   barriers_Kx[1] → tiles 4-8 (dims 256-575) readiness
        // [Task 225 precision fix] Count = 256: each barrier is loaded by
        // exactly ONE warpgroup (wg0 loads for K0, wg1 loads for K1).
        // This avoids cross-wg SMEM visibility issues (each consumer reads
        // data it wrote itself) AND avoids over-arrival (no double-arrive).
        for (int i = 0; i < 2; ++i) {
            __mbarrier_init(&barriers_K0[i], 256);
            __mbarrier_init(&barriers_K1[i], 256);
        }
    }
    __syncthreads();  // ensure all threads see initialized mbarriers
    bool cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;

    // // Programmatic Dependent Launch: Wait for the previous kernel to finish
    // cudaGridDependencySynchronize();

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    // We don't use __ldg here, otherwise NVCC (ptxas, in particular) will do instruction reorder and place __ldg (LDG.E.128.CONSTANT in SASS) in front of cudaGridDependencySynchronize() (ACQBULK in SASS), leading to data race.
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
        // Always round main cache up to kBlockN*2 (64) so that:
        //   - the main/extra cache boundary is on an even block, keeping
        //     the mainloop's main/extra switch simple;
        //   - it matches the metadata kernel's block_size_n=64 tiling, so
        //     is_no_split (start==0 && end==total/kBlockN) stays accurate
        //     without an extra __ldg on num_splits_ptr.
        seqlen_kpad = cute::round_up(seqlen_kpad, kBlockN * 2);
        int extra_seqlen_k = 0;
        int ori_block_max = -1;
        if (params.extra_topk >= 0) {
            if (params.extra_topk_len_ptr) {
                extra_seqlen_k = __ldg(params.extra_topk_len_ptr + batch_idx);
            } else {
                extra_seqlen_k = params.extra_topk;
            }
            ori_block_max = cute::ceil_div(seqlen_kpad, kBlockN);
        }
        const int total_k = seqlen_kpad + cute::round_up(extra_seqlen_k, kBlockN * 2);
        const int start_block_idx = batch_idx == begin_idx ? begin_seqlen / kBlockN : 0;
        int end_block_idx = batch_idx == end_idx
            ? cute::round_up(end_seqlen, kBlockN * 2) / kBlockN
            : total_k / kBlockN;
        const bool is_no_split = start_block_idx == 0 && end_block_idx == (total_k / kBlockN);

        // Pre-fetch pipeline: each WG has TWO pre_token_idx values.
        // pre_token_idx0: for block0/nxt_block0 (even block)
        // pre_token_idx1: for block1/nxt_block1 (odd block)
        // Prolog initializes them, and each load call updates its own idx.
        int pre_token_idx0 = -1;   // for block0/nxt_block0 (even block)
        int pre_token_idx1 = -1;   // for block1/nxt_block1 (odd block)

        // Explicit __ldg pre-fetch for the FIRST round (block 0 for wg0, block 1 for wg1).
        // This __ldg is issued before load_and_dequant_sparse_K, so its latency
        // is hidden by the subsequent tensor setup and cp.async inside the function.
        if constexpr (!T::IsFP8) {
            int _token_id = idx_in_warpgroup / 8;
            // Helper lambda to prefetch token index for a given block
            auto prefetch_tok = [&](int _blk) -> int {
                if (_blk < 0 || _blk >= end_block_idx) return -1;
                bool _use_extra = (ori_block_max >= 0) && (_blk >= ori_block_max);
                int _eff = _use_extra ? (_blk - ori_block_max) : _blk;
                const int *_base = _use_extra
                    ? (params.extra_indices_ptr + static_cast<int64_t>(batch_idx) * params.extra_indices_batch_stride + _eff * T::kBlockN)
                    : (params.indices_ptr + static_cast<int64_t>(batch_idx) * params.indices_batch_stride + _eff * T::kBlockN);
                return __ldg(_base + _token_id);
            };
            if (warpgroup_idx == 0) {
                // WG0: prolog loads block0 (uses idx0), main loop first loads nxt_block0=block2, nxt_block1=block3
                // idx0: prefetch block0 (prolog), will be updated to block2 by prolog load
                // idx1: prefetch block3 (first main loop nxt_block1)
                pre_token_idx0 = prefetch_tok(start_block_idx);
                pre_token_idx1 = prefetch_tok(start_block_idx + 3);
            } else {
                // WG1: prolog loads block1 (uses idx1), main loop first loads nxt_block1=block3, nxt_block0=block2
                // idx1: prefetch block1 (prolog), will be updated to block3 by prolog load
                // idx0: prefetch block2 (first main loop nxt_block0)
                pre_token_idx1 = prefetch_tok(start_block_idx + 1);
                pre_token_idx0 = prefetch_tok(start_block_idx + 2);
            }
        }

        // [SPARSE-WI Stage B] causal branch removed -- sparse path always treats
        // every kv token in [0, seqlen_k) as candidate; mask is driven by
        // smem_valid_indices in wg{0,1}_bunch_0.

        // [FP8-WI fix6] block_table is nullptr for sparse decode;
        // gK/tKgK/tKsK0/tKsK1/gmem_tiled_copy_K are unused in BF16 path.
        // Create dummy values only for FP8 path API compatibility.
        int* block_table_ptr = nullptr;
        typename T::GmemTiledCopyKV gmem_tiled_copy_K{};
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);
        Tensor gK_dummy = make_tensor(make_gmem_ptr(reinterpret_cast<InputT*>(params.k_ptr)),
                        Shape<Int<kBlockN>, Int<T::kHeadDim>>{},
                        make_stride(params.k_row_stride, _1{}));
        Tensor tKgK = gmem_thr_copy_K.partition_S(make_mix_tensor_like(gK_dummy));

        Tensor cur_sK0 = sK(_, _, 0);
        Tensor cur_sK1 = sK(_, _, 1);
        Tensor nxt_sK0 = sK(_, _, 2);

        // sV_direct: needed by FP8 load_and_dequant_sparse_K; BF16 path uses get_half_V from sK directly.
        Tensor sV_direct = make_tensor(make_smem_ptr(reinterpret_cast<InputT*>(&sK(0, 0, 0))),
                                        (typename T::SmemLayoutVDirect){});
        // [SPARSE-WI Stage C-step2b-2 / FP8-WI fix3] FP8 sparse cutover:
        // load entire blocks via register-direct global -> dequant -> sK
        // pipeline (no SMEM staging).  Each block's ALL 576 dims are written
        // to a SINGLE sK buf slot.  The GEMM consumer reads tiles 0-8 from
        // the same buf, and V GEMM also reads dims 0-511 from one buf.
        // Mapping:
        //   block 0 -> all 576 dims in buf 0
        //   block 1 -> all 576 dims in buf 1

        // [FP8-WI fix9] Zero-fill removed: sK is naturally covered by K load.
        // load_K_tiles_bf16 writes zeros for invalid tiles, and 3-buffer rotation
        // ensures every read buffer was loaded first.
        // {
        //     using InputT_local = typename T::InputT;
        //     InputT_local* raw_sK = reinterpret_cast<InputT_local*>(&sK(0, 0, 0));
        //     constexpr int sK_total_elems = cute::cosize_v<typename T::SmemLayoutK>;
        //     for (int i = threadIdx.x; i < sK_total_elems; i += T::NUM_THREADS) {
        //         raw_sK[i] = InputT_local(0.0f);
        //     }
        //     __syncthreads();
        // }

        // Cross-WG prolog: each block's K split across two sK buffers.
        // Block 0: tiles 0-3 → buf 0 (cur_sK0), tiles 4-8 → buf 1 (cur_sK1)
        // Block 1: tiles 0-3 → buf 1 (cur_sK1), tiles 4-8 → buf 0 (cur_sK0)
        // smem_valid_indices uses 4 bufs: WG0 bufs 0/1, WG1 bufs 2/3 (alternating)
        // Prolog blocks become "current" in first iteration, so use softmax bufs.
        int prolog_vi_wg0 = (start_block_idx / 2) % 2;       // 0 or 1
        int prolog_vi_wg1 = 2 + (start_block_idx / 2) % 2;    // 2 or 3
        constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
        {
            if (warpgroup_idx == 0) {
                if (start_block_idx < end_block_idx) {
                    // wg0 loads block 0: stage 1 (tiles 0-3 → buf 0) + stage 2 (tiles 4-8 → buf 1)
                    if constexpr (!T::IsFP8) {
                        auto vi_wg0 = smem_valid_indices(prolog_vi_wg0, _);
                        load_K_tiles_bf16<4, kNumTiles, T, false>(params, cur_sK1, batch_idx, start_block_idx, seqlen_k, &barriers_K0[1],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg0, 
                            &pre_token_idx0, start_block_idx + 2, end_block_idx, extra_seqlen_k);
                        load_K_tiles_bf16<0, 4, T>(params, cur_sK0, batch_idx, start_block_idx, seqlen_k, &barriers_K0[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg0, 
                            &pre_token_idx0, start_block_idx + 2, end_block_idx, extra_seqlen_k);
                    } else {
                        load_and_dequant_sparse_K_staged<4, kNumTiles, T>(
                            params, sK, batch_idx, start_block_idx, seqlen_k,
                            /*buf_idx=*/1, &barriers_K0[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg0);
                        load_and_dequant_sparse_K_staged<0, 4, T>(
                            params, sK, batch_idx, start_block_idx, seqlen_k,
                            /*buf_idx=*/0, &barriers_K0[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg0);
                    }
                }
            } else {
                if (start_block_idx+1 < end_block_idx) {
                    // wg1 loads block 1: stage 1 (tiles 0-3 → buf 1) + stage 2 (tiles 4-8 → buf 0)
                    if constexpr (!T::IsFP8) {
                        auto vi_wg1 = smem_valid_indices(prolog_vi_wg1, _);
                        load_K_tiles_bf16<4, kNumTiles, T, false>(params, cur_sK0, batch_idx, start_block_idx + 1, seqlen_k, &barriers_K1[1],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg1, 
                            &pre_token_idx1, start_block_idx + 3, end_block_idx, extra_seqlen_k);
                        load_K_tiles_bf16<0, 4, T>(params, cur_sK1, batch_idx, start_block_idx + 1, seqlen_k, &barriers_K1[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg1, 
                            &pre_token_idx1, start_block_idx + 3, end_block_idx, extra_seqlen_k);
                    } else {
                        load_and_dequant_sparse_K_staged<4, kNumTiles, T>(
                            params, sK, batch_idx, start_block_idx + 1, seqlen_k,
                            /*buf_idx=*/0, &barriers_K1[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg1);
                        load_and_dequant_sparse_K_staged<0, 4, T>(
                            params, sK, batch_idx, start_block_idx + 1, seqlen_k,
                            /*buf_idx=*/1, &barriers_K1[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg1);
                    }
                }
            }
        }

        // CTA-wide barrier after prolog loads.
        // Each warpgroup's load_and_dequant_sparse_K uses a warpgroup-internal
        // bar.sync (256 threads) which only ensures intra-wg SMEM visibility.
        // The remote PV GEMM later reads V from the OTHER wg's K buffer.
        // Without this __syncthreads(), cross-wg SMEM visibility is NOT
        // guaranteed on PPU (__threadfence_block() is a no-op).  This causes
        // non-deterministic precision collapse for d_qk=512 + h_q=128 where
        // tight timing (fewer QKT tiles) + all rows active exposes the race.
        __syncthreads();

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
        if constexpr (T::kHasExtraRopeTile && !T::kIsCrossCut) {
            retrieve_rP_from_sP<T>(rQ8, local_tile(sQ, Shape<Int<T::kBlockM>, _64>{}, Coord<_0, _8>{}), idx_in_warpgroup);
        } else {
            cute::clear(rQ8);
        }

        if (warpgroup_idx == 0) {
            // Warpgroup 0
            // Tensor rP0 = make_tensor<float>((typename T::rP0Layout){});
            Tensor rP0 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
            const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

            // NOTE We don't use the pipelined version of Q K^T here since it leads
            // to a slow-down (or even register spilling, thanks to the great NVCC)
            // Issue P0 = Q @ K0^T, wait
            // Guard on block range, not seqlen_k.  When
            // seqlen_k=0 but extra blocks exist, block 0 is a valid extra
            // block that must participate in QK GEMM.
            if (start_block_idx < end_block_idx) {

                cute::clear(rP0);
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK0, cur_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
            }

            int idx = 0;

            // Precomputed K addresses for SIMT/TC overlap
            InputT* precomp_ptr0 = nullptr;
            InputT* precomp_ptr1 = nullptr;
            bool precomp_valid0 = false;
            bool precomp_valid1 = false;

            // Precompute initial addresses for first iteration
            if constexpr (!T::IsFP8) {
                int first_vi_preload_wg0 = 1 - (start_block_idx / 2) % 2;
                int first_vi_preload_wg1 = 3 - (start_block_idx / 2) % 2;
                bool init_load_extra = (ori_block_max >= 0) && (start_block_idx + 2 >= ori_block_max);
                precomp_ptr0 = compute_K_addr_bf16_dynamic<0, T>(
                    params, batch_idx, start_block_idx + 2, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, first_vi_preload_wg0,
                    &pre_token_idx0, extra_seqlen_k, precomp_valid0, init_load_extra);
                precomp_ptr1 = compute_K_addr_bf16_dynamic<0, T>(
                    params, batch_idx, start_block_idx + 3, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, first_vi_preload_wg1,
                    &pre_token_idx1, extra_seqlen_k, precomp_valid1, init_load_extra);
            }

            #define LAUNCH_WG0_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA)    \
                wg0_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA>(                \
                gmem_tiled_copy_K, tKgK, sQ, sK, sV_direct, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
                rP0, rO, rL,                                     \
                barriers_K0, barriers_K1, cur_phase_K0, params,                       \
                block_table_ptr, seqlen_k, total_k, block_idx, end_block_idx, idx_in_warpgroup, wg_idx, idx, \
                batch_idx, ori_block_max, smem_valid_indices,                         \
                T::IsFP8 ? nullptr : &pre_token_idx0,                                \
                T::IsFP8 ? nullptr : &pre_token_idx1,                                \
                extra_seqlen_k,                                                      \
                precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1);          \

            int block_idx = start_block_idx;
            // 3-phase loop: NEXT_EXTRA controls prefetch/compute_addr cache selection
            // Phase 1: prefetch/compute for primary cache (NEXT_EXTRA=false)
            int p1_end = min(ori_block_max >= 0 ? ori_block_max - 4 : end_block_idx - 2, end_block_idx - 2);
            #pragma unroll 1
            for (; block_idx < p1_end; block_idx += 2) {
                LAUNCH_WG0_SUBROUTINE(false, false, false);
            }
            // Phase 2: prefetch/compute for extra cache (NEXT_EXTRA=true)
            #pragma unroll 1
            for (; block_idx < end_block_idx-2; block_idx += 2) {
                LAUNCH_WG0_SUBROUTINE(false, false, true);
            }
            // Phase 3: last 2 blocks (no K load/prefetch/compute_addr)
            LAUNCH_WG0_SUBROUTINE(false, true, false);
        } else {
            // // Warpgroup 1
            // Tensor rP1 = make_tensor<float>((typename T::rP0Layout){});
            Tensor rP1 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
            const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

            if (start_block_idx+1 < end_block_idx) {
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK1, cur_sK0, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
            } else {
                // [FP8-WI fix9] When wg1 has no initial K block, rP1 must be zero-
                // initialized.  Otherwise wg1_subroutine runs softmax on garbage
                // registers, producing corrupt rL that contaminates the final output
                // via the cross-warpgroup rL reduction.
                cute::clear(rP1);
            }

            int idx = 0;

            // Precomputed K addresses for SIMT/TC overlap (WG1)
            InputT* precomp_ptr0_wg1 = nullptr;
            InputT* precomp_ptr1_wg1 = nullptr;
            bool precomp_valid0_wg1 = false;
            bool precomp_valid1_wg1 = false;

            // Precompute initial addresses for first iteration
            if constexpr (!T::IsFP8) {
                bool init_load_extra = (ori_block_max >= 0) && (start_block_idx + 3 >= ori_block_max);
                precomp_ptr0_wg1 = compute_K_addr_bf16_dynamic<4, T>(
                    params, batch_idx, start_block_idx + 3, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    &pre_token_idx1, extra_seqlen_k, precomp_valid0_wg1, init_load_extra);
                precomp_ptr1_wg1 = compute_K_addr_bf16_dynamic<4, T>(
                    params, batch_idx, start_block_idx + 2, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    &pre_token_idx0, extra_seqlen_k, precomp_valid1_wg1, init_load_extra);
            }

            #define LAUNCH_WG1_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA)  \
                wg1_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA>(          \
                gmem_tiled_copy_K, tKgK, sQ, sK, sV_direct, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
                rP1, rO, rL,                                     \
                barriers_K0, barriers_K1, cur_phase_K1, params,                       \
                block_table_ptr, seqlen_k, total_k, block_idx, end_block_idx, idx_in_warpgroup, wg_idx, idx, \
                batch_idx, ori_block_max, smem_valid_indices,                         \
                T::IsFP8 ? nullptr : &pre_token_idx1,                                \
                T::IsFP8 ? nullptr : &pre_token_idx0,                                \
                extra_seqlen_k,                                                      \
                precomp_ptr0_wg1, precomp_ptr1_wg1, precomp_valid0_wg1, precomp_valid1_wg1);  \

            int block_idx = start_block_idx;
            // 3-phase loop: same phase boundaries as WG0
            // Phase 1: prefetch/compute for primary cache (NEXT_EXTRA=false)
            int p1_end = min(ori_block_max >= 0 ? ori_block_max - 4 : end_block_idx - 2, end_block_idx - 2);
            #pragma unroll 1
            for (; block_idx < p1_end; block_idx += 2) {
                LAUNCH_WG1_SUBROUTINE(false, false, false);
            }
            // Phase 2: prefetch/compute for extra cache (NEXT_EXTRA=true)
            #pragma unroll 1
            for (; block_idx < end_block_idx-2; block_idx += 2) {
                LAUNCH_WG1_SUBROUTINE(false, false, true);
            }
            // Phase 3: last 2 blocks
            LAUNCH_WG1_SUBROUTINE(false, true, false);
        }

        // Reduce rL across threads within the same warp
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

        // Reduce rL across warpgroups
        int my_row = get_AorC_row_idx<T::kAtomLayoutM>(0, idx_in_warpgroup);
        // Pre-issue attn_sink __ldg before syncthreads to hide GMEM latency
        float pre_sink0 = 0.0f, pre_sink1 = 0.0f;
        if (is_no_split) {
            if (params.attn_sink_ptr != nullptr) {
                const int row0 = my_row;
                const int row1 = my_row + 8;
                const int q_head_idx_0 = (m_block_idx * T::BLOCK_SIZE_M + row0) % params.ngroups;
                const int q_head_idx_1 = (m_block_idx * T::BLOCK_SIZE_M + row1) % params.ngroups;
                pre_sink0 = __ldg(params.attn_sink_ptr + q_head_idx_0);
                pre_sink1 = __ldg(params.attn_sink_ptr + q_head_idx_1);
            }
        }
        if (idx_in_warpgroup % 4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
            sL_reduction_wksp[my_row + warpgroup_idx * 128] = rL[0];
            sL_reduction_wksp[my_row + 8 + warpgroup_idx * 128] = rL[1];
        }
        __syncthreads();

        if constexpr (T::kBlockM == 64) {
            // BlockM=64: Merge WG0 + WG1 partial rL sums.
            // Only primary warps (idx < kMmaThreads) do the +=;
            // after __syncthreads(), ALL threads re-read the merged rL.
            if (warpgroup_idx == 1
                    && idx_in_warpgroup % 4 == 0
                    && idx_in_warpgroup < T::kMmaThreads) {
                sL_reduction_wksp[my_row] += rL[0];
                sL_reduction_wksp[my_row + 8] += rL[1];
            }
            __syncthreads();
            rL[0] = sL_reduction_wksp[my_row];
            rL[1] = sL_reduction_wksp[my_row + 8];
        } else {
            // BlockM=128: optimized rL cross-WG reduction.
            // WG0 directly reads WG1's partial from register; WG1 uses __syncwarp.
            if (warpgroup_idx == 0) {
                rL[0] += sL_reduction_wksp[my_row + 128];
                rL[1] += sL_reduction_wksp[my_row + 8 + 128];
            } else {
                if (idx_in_warpgroup % 4 == 0) {
                    sL_reduction_wksp[my_row] += rL[0];
                    sL_reduction_wksp[my_row + 8] += rL[1];
                }
                __syncwarp();
                rL[0] = sL_reduction_wksp[my_row];
                rL[1] = sL_reduction_wksp[my_row + 8];
            }
        }

        // Prune out when rL is 0.0f or NaN
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i)
            rL[i] = (rL[i] == 0.0f || rL[i] != rL[i]) ? 1.0f : rL[i];

        // [Task #105] CTA-wide barrier for sM visibility before attn_sink.
        // For BlockM=128, the rL reduction above doesn't include a CTA-wide
        // barrier (WG0 uses register add, WG1 uses __syncwarp).  This
        // __syncthreads() guarantees sM coherence for the attn_sink epilogue.
        // For BlockM=64, the second __syncthreads() in the rL reduction already
        // serves this purpose, so we skip it here.
        if constexpr (T::kBlockM != 64) {
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
                const int row0 = my_row;
                const int row1 = my_row + 8;
                // Use pre-issued __ldg values (latency hidden by __syncthreads above)
                const float sink_exp0 = expf(pre_sink0 - sM(row0) * (float)M_LN2);
                const float sink_exp1 = expf(pre_sink1 - sM(row1) * (float)M_LN2);
                if constexpr (T::kBlockM == 64) {
                    // BlockM=64: Read original pre-prune rL to detect empty attention (all tokens masked).
                    const float orig_rL0 = sL_reduction_wksp[my_row];
                    const float orig_rL1 = sL_reduction_wksp[my_row + 8];
                    const bool row0_empty = (orig_rL0 == 0.0f || orig_rL0 != orig_rL0);
                    const bool row1_empty = (orig_rL1 == 0.0f || orig_rL1 != orig_rL1);
                    const float o_scale0 = row0_empty ? 0.0f : __fdividef(1.0f, rL[0] + sink_exp0);
                    const float o_scale1 = row1_empty ? 0.0f : __fdividef(1.0f, rL[1] + sink_exp1);
                    CUTLASS_PRAGMA_UNROLL
                    for (int idx = 0; idx < size(rO); ++idx) {
#if ACOMPUTE_VERSION == 10000
                        bool is_row0 = ((idx / 4) % 2 == 0);
#else
                        bool is_row0 = (idx % 4 < 2);
#endif
                        bool empty = is_row0 ? row0_empty : row1_empty;
                        float scale = is_row0 ? o_scale0 : o_scale1;
                        rO(idx) = empty ? 0.0f : (rO(idx) * scale);
                    }
                } else {
                    // BlockM=128: optimized path (rL already pruned, 0->1.0)
                    const float o_scale0 = rL[0] == 0.0f ? 0.0f : __fdividef(1.0f, rL[0] + sink_exp0);
                    const float o_scale1 = rL[1] == 0.0f ? 0.0f : __fdividef(1.0f, rL[1] + sink_exp1);
                    CUTLASS_PRAGMA_UNROLL
                    for (int idx = 0; idx < size(rO); ++idx) {
#if ACOMPUTE_VERSION == 10000
                        bool is_row0 = ((idx / 4) % 2 == 0);
#else
                        bool is_row0 = (idx % 4 < 2);
#endif
                        float scale = is_row0 ? o_scale0 : o_scale1;
                        rO(idx) *= scale;
                    }
                }
                // Set rL to 1.0 so store_o's division becomes a no-op
                rL[0] = 1.0f;
                rL[1] = 1.0f;
            }

            if constexpr (T::kBlockM == 64) {
                CUTLASS_PRAGMA_UNROLL
                for (int idx = 0; idx < size(rO); ++idx) {
                    if (rO(idx) != rO(idx)) rO(idx) = 0.0f;
                }
            }

            store_o<T, true>(rO, gO, rL, sO_addr, params, batch_idx, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            int i = threadIdx.x;
            if (i < num_valid_seq_q) {
                float cur_L = sL_reduction_wksp[i];
                float sM_val = sM(i);
                if constexpr (T::kBlockM == 64) {
                    bool sM_nan = ((__float_as_uint(sM_val) & 0x7FFFFFFFu) > 0x7F800000u);
                    gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L || sM_nan) ? INFINITY : logf(cur_L) + sM_val / (float)M_LOG2E;
                } else {
                    gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L) ? INFINITY : logf(cur_L) + sM_val / (float)M_LOG2E;
                }
            }

            if (batch_idx + 1 <= end_idx) {
                // Skip mbarrier reinit: barrier phases are consistent across batches.
                // store_o<T, true> writes directly to GMEM, no sO_addr SMEM race.
                launch_q_copy<T>(params, batch_idx + 1, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);
            } else {
                // Allow the next kernel (the combine kernel) to launch
                // The next kernel MUST be the combine kernel
            }
        } else {
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
            if (i < num_valid_seq_q) {
                float cur_L = sL_reduction_wksp[i];
                float sM_val = sM(i);
                if constexpr (T::kBlockM == 64) {
                    bool sM_nan = ((__float_as_uint(sM_val) & 0x7FFFFFFFu) > 0x7F800000u);
                    gSoftmaxLseAccum(i) = (cur_L == 0.0f || cur_L != cur_L || sM_nan) ? -INFINITY : log2f(cur_L) + sM_val;
                } else {
                    gSoftmaxLseAccum(i) = (cur_L == 0.0f || cur_L != cur_L) ? -INFINITY : log2f(cur_L) + sM_val;
                }
            }

            if constexpr (T::kBlockM == 64) {
                CUTLASS_PRAGMA_UNROLL
                for (int idx = 0; idx < size(rO); ++idx) {
                    if (rO(idx) != rO(idx)) rO(idx) = 0.0f;
                }
            }

            store_o<T, false>(rO, gOAccum, rL, sO_addr, params, batch_idx, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

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
            }
        }
        if (batch_idx != end_idx)
            __syncthreads();
    }
}

template <typename InputT, int Arch, bool IsFP8, int BlockM>
void run_flash_sparse_decode_wg_kernel(Flash_fwd_mla_params &params, hggcStream_t stream)
{
    // [SPARSE-WI Stage B] BOOL_SWITCH on is_causal removed -- only one
    // instantiation; sparse path ignores params.is_causal.
    if (params.d == 576) {
        // [SPARSE-WI Stage D-step1] Switch to Traits_v2 -- adds fp8 nope/scales
        // and indices SMEM regions; everything else inherits 1:1 from splitkv.
        using T = Traits_v2<InputT, 576, IsFP8, BlockM>;

        auto mla_kernel = &flash_sparse_decode_wg_kernel<T>;
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
            // [SPARSE-WI Stage B] Is_causal removed; sparse decode is mask-by-indices.
            printf("Is_causal:%d (sparse: ignored)\n", int(params.is_causal));
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

        CHECK_CUDA_KERNEL_LAUNCH();

        run_flash_mla_combine_kernel<InputT>(params, stream);
    } else {
        // Headdim=512
        using T = Traits_v2<InputT, 512, IsFP8, BlockM>;

        auto mla_kernel = &flash_sparse_decode_wg_kernel<T>;
        constexpr size_t smem_size = std::max(sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));

        hggcFuncSetAttribute(mla_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        const int num_m_block = cute::ceil_div(params.seqlen_q, T::kBlockM);

        int ctas_per_sm;
        hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, mla_kernel, T::NUM_THREADS, smem_size);

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            printf("[splitkv_mla (d=512)]:\n");
            printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, mla_kernel);
            int sm_count = get_num_sm(get_current_device());
            if (sm_count == 64) sm_count = 20;

            printf("blockM:%d, blockN:%d, threads:%d, block_size:%d\n",
                    T::kBlockM, T::kBlockN, T::NUM_THREADS, params.page_block_size);
            printf("Is_causal:%d (sparse: ignored)\n", int(params.is_causal));
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

        CHECK_CUDA_KERNEL_LAUNCH();

        run_flash_mla_combine_kernel<InputT>(params, stream);
    }
}

