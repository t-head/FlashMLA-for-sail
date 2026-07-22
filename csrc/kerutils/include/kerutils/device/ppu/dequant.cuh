/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/
#pragma once

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>
#include <cutlass/arch/memory.h>

#include "utils.h"

namespace flash {

using namespace cute;
using fp8_e8m0 = __hg_fp8_e8m0;
using cutlass::arch::NamedBarrier;

template<typename T, int Size>
__device__ __forceinline__ void load_128b_from_gmem(const void* src_ptr, T* dst_ptr) {
    static_assert(sizeof(T) * Size == 128/8);
    int4* ptr = reinterpret_cast<int4*>(dst_ptr);
    asm volatile("ppu.ld.global.nc.LLC::128B.v4.s32 {%0, %1, %2, %3}, [%4];" \
            : "=r"((*ptr).x), "=r"((*ptr).y), "=r"((*ptr).z), "=r"((*ptr).w) \
            : "l"(src_ptr)); \

}

template<typename fp8T, typename bf16T, int Size>
__device__ __forceinline__ void cvt_fp8_bf16(const fp8T* src_ptr, bf16T* dst_ptr, const float &scale) {
    //dequant//
    bf16T scale_bf162 = (bf16T)(scale);
    #pragma unroll
    for (int m = 0; m < Size; m++) {
        dst_ptr[m] = (bf16T)(src_ptr[m])*scale_bf162;
    }

    // TODO: acceleration with bf162/__hg_fp8x4_e4m3/float4.
    // static_assert(Size % 4== 0);
    // // static_assert(sizeof(fp8T) == 8);
    // __ppu_bfloat162 scale_bf162 = __float2bfloat162_rn(scale);
    // const __hg_fp8x4_e4m3* src_fp82 =  reinterpret_cast<const __hg_fp8x4_e4m3*>(src_ptr);
    // __ppu_bfloat162* dst_bf162 =  reinterpret_cast<__ppu_bfloat162*>(dst_ptr);
}


struct fp8x8 {
    __hg_fp8x4_e4m3 lo;
    __hg_fp8x4_e4m3 hi;
};

struct bf16x8 {
    __ppu_bfloat162 a01;
    __ppu_bfloat162 a23;
    __ppu_bfloat162 a45;
    __ppu_bfloat162 a67;
};

template<typename Tensor0, typename Tensor1>
__device__ __forceinline__ void fp8_to_bf16(Tensor0 src, Tensor1 dst, float scale) {
#if ACOMPUTE_VERSION == 10000
    Tensor src_32b = recast<uint32_t>(src);
    Tensor dst_64b = recast<uint2>(dst);
    Tensor dst_bf16x2 = recast<__ppu_bfloat162>(dst);
    __ppu_bfloat162 scale_bf16x2 = __float2bfloat162_rn(scale);
    __ppu_bfloat162 dob = __float2bfloat162_rn(2.0);
    __ppu_bfloat162 eps = __float2bfloat162_rn(-0.015625);
    #pragma unroll
    for (int m = 0; m < size(src_32b); ++m) {
        uint32_t m0 = src_32b[m] & 0x00070007;
        uint32_t m1 = src_32b[m] & 0x07000700;
        dst_64b[m].x = m0 << 4;
        dst_64b[m].y = m1 >> 4;
        uint32_t e0 = (src_32b[m] & 0x00780078) << 4;
        uint32_t e1 = (src_32b[m] & 0x78007800) >> 4;
        dst_64b[m].x |= e0 + 0x3C003C00U;
        dst_64b[m].y |= e1 + 0x3C003C00U;
        __ppu_bfloat162 y0 = __hfma2(dst_bf16x2[m*2 + 0], dob, eps);
        __ppu_bfloat162 y1 = __hfma2(dst_bf16x2[m*2 + 1], dob, eps);
        dst_bf16x2[m*2 + 0] = __hmin2(y0, dst_bf16x2[m*2 + 0]);
        dst_bf16x2[m*2 + 1] = __hmin2(y1, dst_bf16x2[m*2 + 1]);
        dst_bf16x2[m*2 + 0] = __hmul2(dst_bf16x2[m*2 + 0], scale_bf16x2);
        dst_bf16x2[m*2 + 1] = __hmul2(dst_bf16x2[m*2 + 1], scale_bf16x2);
        uint32_t s0 = src_32b[m] & 0x00800080;
        uint32_t s1 = src_32b[m] & 0x80008000;
        dst_64b[m].x |= s0 << 8;
        dst_64b[m].y |= s1;
        uint32_t tmp = dst_64b[m].x;
        dst_64b[m].x = __byte_perm(tmp, dst_64b[m].y, 0x5410);
        dst_64b[m].y = __byte_perm(tmp, dst_64b[m].y, 0x7632);
    }
#else
    bf16x8 &result = recast<bf16x8>(dst)[0];
    fp8x8  &inputs = recast<fp8x8>(src)[0];
    __ppu_bfloat162 scale_bf162 = __float2bfloat162_rn(scale);

    #define DEQUANT_FP8x4(OUTPUT_BF16_LO, OUTPUT_BF16_HI, FP8x4) \
    { \
        float4 fp32x4 = (float4)(FP8x4); \
        OUTPUT_BF16_LO = __float22bfloat162_rn({fp32x4.x, fp32x4.y})*scale_bf162; \
        OUTPUT_BF16_HI = __float22bfloat162_rn({fp32x4.z, fp32x4.w})*scale_bf162; \
    }

    DEQUANT_FP8x4(result.a01, result.a23, inputs.lo);
    DEQUANT_FP8x4(result.a45, result.a67, inputs.hi);
#endif
}



template <typename ElementKVCache, int kBlockN, int kNThreads, int kHeadDim_ = 576, int kHeadDimV_ = 512>
struct KVCacheGmemBf16 {
    using index_t = int64_t;
    static constexpr int kBlockKSmem = 64;
    static constexpr int kSwizzle = 3;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementKVCache);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

    using SmemLayoutAtomK = decltype(composition(
#if ACOMPUTE_VERSION == 10000
        PPU_Swizzle<kSwizzle, 3, 3>{},
#else
        Swizzle<kSwizzle, 3, 3>{},
#endif
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{},
        make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>,
        Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;
    using Gmem_copy_struct = PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
    // using Gmem_copy_struct = PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, ElementKVCache>{},
        GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
};

template <typename ElementKVCache, int kBlockN, int kNThreads, int kHeadDim_ = 576, int kHeadDimV_ = 512>
struct KVCacheGmemBf16SimAIU {
    using index_t = int64_t;
    static constexpr int kBlockKSmem = 64;
    static constexpr int kSwizzle = 3;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementKVCache);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

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

    using SmemLayoutKSim = decltype(tile_to_shape(SmemLayoutAtomKSim{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{},
        make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemCopyOpK = PPU_TSM_LD_SWZL<ElementKVCache, kBlockN, kBlockKSmem, true, false, kHeadDim / kBlockKSmem>;
    using SmemCopyAtomK = Copy_Atom<SmemCopyOpK, ElementKVCache>;
    using SmemCopyOpV = PPU_TSM_LD_SWZL<ElementKVCache, kBlockN, kBlockKSmem, true, true, kHeadDimV / kBlockKSmem>;
    using SmemCopyAtomV = Copy_Atom<SmemCopyOpV, ElementKVCache>;

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>,
        Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;
    using Gmem_copy_struct = PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, ElementKVCache>{},
        GmemLayoutAtom{}, Layout<Shape<_1, _8>, Stride<_8, _1>>{}));
};

template <int kBlockN, int kNThreads>
struct KVCacheGmemFP8 {
    using ElementKVCache = cutlass::float_e4m3_t;
    using Element = cutlass::bfloat16_t;
    using index_t = int64_t;
    static constexpr int kBlockKSmemNope = 128; // fp8
    static constexpr int kBlockKSmem = 64;
    static constexpr int kSwizzle = 3;
    static constexpr int kHeadDim = 576;
    static constexpr int BytesPerToken = 656;
    static constexpr int kHeadDimV = 512;
    static constexpr int kHeadDimRope = 64;

    //no_aiu smem layout
    using SmemLayoutAtomK = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomV = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{},
        make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    // gmem -> register
    using Gmem_copy_struct = PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

    // fp8
    static constexpr int kGmemElemsPerLoadNope = sizeof(cute::uint128_t) / sizeof(ElementKVCache); // 128/8=16
    static constexpr int kGmemThreadsPerRowNope = kBlockKSmemNope / kGmemElemsPerLoadNope; // 128/16=8
    using TensorKNope = decltype(make_tensor(make_gmem_ptr(static_cast<ElementKVCache*>(nullptr)),
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}, Stride<Int<BytesPerToken>, _1>{}));
    using SmemLayoutKNope = decltype(tile_to_shape(SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    // fp8 gmem
    using GmemLayoutAtomNope = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowNope>,
        Int<kGmemThreadsPerRowNope>>, Stride<Int<kGmemThreadsPerRowNope>, _1>>; // 512/8=64, 8
    using GmemTiledCopyNope = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, ElementKVCache>{},
        GmemLayoutAtomNope{}, Layout<Shape<_1, Int<kGmemElemsPerLoadNope>>>{}));
    using TensortKgKNope = decltype(
        GmemTiledCopyNope{}.get_thread_slice(int(0)).partition_S(TensorKNope{}));

    using SmemTiledCopyNope = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
        GmemLayoutAtomNope{}, Layout<Shape<_1, Int<kGmemElemsPerLoadNope>>>{}));

    // bf16
    static constexpr int kGmemElemsPerLoadRope = sizeof(cute::uint128_t) / sizeof(Element); // 128/16=8
    static constexpr int kGmemThreadsPerRowRope = kBlockKSmem / kGmemElemsPerLoadRope; // 128/16=8
    using TensorKRope = decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
        Shape<Int<kBlockN>, Int<kHeadDimRope>>{}, Stride<Int<BytesPerToken>, _1>{}));
    using SmemLayoutKRope = decltype(tile_to_shape(SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDimRope>>{}));

    // bf16 gmem
    using GmemLayoutAtomRope = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowRope>,
        Int<kGmemThreadsPerRowRope>>, Stride<Int<kGmemThreadsPerRowRope>, _1>>; // 512/8=64, 8
    using GmemTiledCopyRope = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
        GmemLayoutAtomRope{}, Layout<Shape<_1, Int<kGmemElemsPerLoadRope>>>{}));
    using TensortKgKRope = decltype(
        GmemTiledCopyRope{}.get_thread_slice(int(0)).partition_S(TensorKRope{}));

    TensortKgKNope tKgK_nope;
    TensortKgKRope tKgK_rope;
    SmemTiledCopyNope smem_tiled_copy_K_nope;
    GmemTiledCopyRope gmem_tiled_copy_K_rope;

    const int tidx;
    int* gIndices_ptr;
    ElementKVCache* gK_ptr;
    const int block_size;
    const index_t batch_stride;
    const index_t row_stride;

    CUTLASS_DEVICE
    KVCacheGmemFP8(int tidx, int* gIndices_ptr, ElementKVCache* gK_ptr,
        const int block_size, const index_t batch_stride, const index_t row_stride)
        : tidx(tidx), block_size(block_size), batch_stride(batch_stride), row_stride(row_stride),
         gIndices_ptr(gIndices_ptr), gK_ptr(gK_ptr)
    {

        TensorKNope gK_nope = make_tensor(make_gmem_ptr(static_cast<ElementKVCache*>(nullptr)),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}, Stride<Int<BytesPerToken>, _1>{});
        tKgK_nope = GmemTiledCopyNope{}.get_thread_slice(tidx).partition_S(gK_nope);
        TensorKRope gK_rope = make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
            Shape<Int<kBlockN>, Int<kHeadDimRope>>{}, Stride<Int<BytesPerToken>, _1>{});
        tKgK_rope = gmem_tiled_copy_K_rope.get_thread_slice(tidx).partition_S(gK_rope);
    };

    template <bool Clear_OOB_K=true, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void
    load_from_gmem(int n_block, int kv_store_num, Tensor0 &smem_valid_indices, Tensor1 &sK) {

        const int load_col_idx = tidx/8;
        #if ACOMPUTE_VERSION ==10000
        const int col_in_indices = (load_col_idx % 4) * 16 + (load_col_idx / 16) * 4 + (load_col_idx % 16) / 4;
        #else
        const int col_in_indices = load_col_idx;
        #endif

        int row_offset_indices = load_col_idx + n_block * kBlockN;
        int token_index = __ldg(gIndices_ptr + row_offset_indices);
        bool is_token_valid = token_index >= 0;
        smem_valid_indices(kv_store_num%2, col_in_indices) = is_token_valid;
        int block_index = token_index/block_size;
        int rel_idx_in_block = (token_index+block_size) % block_size;

        ElementKVCache *gK_now = gK_ptr + block_index*batch_stride
                                + rel_idx_in_block*row_stride;
                                // + (tidx%8)*16;

        Element* gK_rope = (Element*)(gK_now+kHeadDimV+4*sizeof(float)); // + (lane_idx/8)*8;

        // nope: gmem(fp8) -> reg(fp8) -> (dequant) -> reg(bf16) -> smem(bf16)
        // rope: gmem(bf16) -> smem(bf16)
        Tensor sK_nope = make_tensor(sK.data(), SmemLayoutKNope{});
        Tensor sK_rope = make_tensor(sK.data()+ size(sK_nope), SmemLayoutKRope{});
        float scales[4] = {0.f, 0.f, 0.f, 0.f};
        if (is_token_valid) {
            load_128b_from_gmem<float, 4>((float*)(gK_now+512), scales);
        }

        // nope: gmem(fp8) -> reg(fp8) -> (dequant) -> reg(bf16)
        tKgK_nope.data() = gK_now + (tidx%8)*16; //((_16,_1),_1,_4):((1,_0),_0,128)
        Tensor tKrK = make_tensor<ElementKVCache>(shape(tKgK_nope)); //((_16,_1),_1,_4):((_1,_0),_0,_16):
        Tensor rK = make_tensor<Element>(shape(tKrK));
        #pragma unroll
        for (int m = 0; m < size<1>(tKrK); ++m) {
            #pragma unroll
            for (int k = 0; k < size<2>(tKrK); ++k) {
                if (is_token_valid) {
                    load_128b_from_gmem<ElementKVCache, kGmemElemsPerLoadNope>(
                        &tKgK_nope(0, m, k), &tKrK(0, m, k));
                    cvt_fp8_bf16<ElementKVCache, Element, kGmemElemsPerLoadNope>(
                        &tKrK(0, m, k), &rK(0, m, k), scales[k]);
                }
            }
        }

        // rope: gmem(bf16) -> smem(bf16)
        Tensor tKsK_rope = gmem_tiled_copy_K_rope.get_thread_slice(tidx).partition_D(sK_rope);
        tKgK_rope.data() = gK_rope + (tidx%8)*8;
        if (is_token_valid) {
            cute::copy(gmem_tiled_copy_K_rope, tKgK_rope, tKsK_rope);
        } else if (Clear_OOB_K) {
            cute::clear(tKsK_rope);
        }

        // nope: reg(bf16) -> smem(bf16)
        Tensor tKsK_nope = smem_tiled_copy_K_nope.get_thread_slice(tidx).partition_D(sK_nope);
        if (is_token_valid) {
            cute::copy(smem_tiled_copy_K_nope, rK, tKsK_nope);
        } else if (Clear_OOB_K) {
            cute::clear(tKsK_nope);
        }

    };

};

template<int kBlockN, int kNThreads, int kHeadDim_>
struct KVCacheG2SFP8 {
    using ElementKVCache = cutlass::float_e4m3_t;
    using Element = cutlass::bfloat16_t;
    using index_t = int64_t;
    static constexpr int kBlockKSmem = 64;
    static constexpr int kBlockKSmemFp8 = 128;
    static constexpr int kSwizzle = 3;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int BytesPerToken = kHeadDim_ == 576 ? 656 : 576;
    static constexpr int kHeadDimV = 512;
    static constexpr int kHeadDimRope = 64;
    static constexpr int kHeadDimNope = kHeadDim - kHeadDimRope;
    static constexpr int kGroupNope = 128;
    static constexpr int NUM_SCALES = 8;


    // smem layout for MLA
    using SmemLayoutAtomK = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomV = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{},
        make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    // smem for G2S
    using SmemLayoutAtomFP8 = decltype(composition(Swizzle<3, 4, 3>{},
        Layout<Shape<_8, Int<128>>, Stride<Int<128>, _1>>{}));
    using SmemLayoutKNopeFP8 = decltype(tile_to_shape(SmemLayoutAtomFP8{}, Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutKNopeBF16 = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutKRope = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDimRope>>{}));

#if ACOMPUTE_VERSION != 10000
    using SmemCopyOpK = PPU_TSM_LD_SWZL<Element, kBlockN, kBlockKSmem, true, false, kHeadDim / kBlockKSmem>;
    using SmemCopyAtomK = Copy_Atom<SmemCopyOpK, Element>;
    using SmemCopyOpV = PPU_TSM_LD_SWZL<Element, kBlockN, kBlockKSmem, true, true, kHeadDimV / kBlockKSmem>;
    using SmemCopyAtomV = Copy_Atom<SmemCopyOpV, Element>;
#endif

    using SmemLayoutKNopeFP8Np = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDimNope>>{}));

    // G2S
    static constexpr int kGmemElemsPerLoadNope = sizeof(cute::uint128_t) / sizeof(ElementKVCache);
    static constexpr int kGmemThreadsPerRowNope = kGroupNope / kGmemElemsPerLoadNope;
    static constexpr int kRowsPerGmemNope = kNThreads / kGmemThreadsPerRowNope;
    static constexpr int kColPerGmemNope = kBlockN / kRowsPerGmemNope;
    static constexpr int kGmemElemsPerLoadRope = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemThreadsPerRowRope = kHeadDimRope / kGmemElemsPerLoadRope;

    using Gmem_copy_struct_nope = PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
    using Gmem_copy_struct_rope = PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
    using GmemTiledCopyNope = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_nope, ElementKVCache>{},
                        Layout<Shape<Int<kNThreads / kGmemThreadsPerRowNope>, Int<kGmemThreadsPerRowNope>>,
                               Stride<Int<kGmemThreadsPerRowNope>, _1>>{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadNope>>, Stride<Int<kGmemElemsPerLoadNope>, _1>>{}));
    using GmemTiledCopyRope = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_rope, Element>{},
                        Layout<Shape<Int<kNThreads / kGmemThreadsPerRowRope>, Int<kGmemThreadsPerRowRope>>,
                               Stride<Int<kGmemThreadsPerRowRope>, _1>>{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadRope>>, Stride<Int<kGmemElemsPerLoadRope>, _1>>{}));

    using TensorKNope = decltype(make_tensor(make_gmem_ptr(static_cast<ElementKVCache*>(nullptr)),
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}, Stride<Int<BytesPerToken>, _1>{}));
    using TensorKRope = decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
        Shape<Int<kBlockN>, Int<kHeadDimRope>>{}, Stride<Int<BytesPerToken>, _1>{}));
    using TensortKgKNope = decltype(
        GmemTiledCopyNope{}.get_thread_slice(int(0)).partition_S(TensorKNope{}));
    using TensortKgKRope = decltype(
        GmemTiledCopyRope{}.get_thread_slice(int(0)).partition_S(TensorKRope{}));


    // CVT: S2R & R2S
    static constexpr int kSmemElemsPerLoadNope = sizeof(cute::uint64_t) / sizeof(ElementKVCache);
    static constexpr int kSmemThreadsPerRowNope = 8;
    using S2RTileCopyNope = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopy, ElementKVCache>{},
                        Layout<Shape<Int<kNThreads / kSmemThreadsPerRowNope>, Int<kSmemThreadsPerRowNope>>,
                               Stride<Int<kSmemThreadsPerRowNope>, _1>>{},
                        Layout<Shape<_1, Int<kSmemElemsPerLoadNope>>, Stride<Int<kSmemElemsPerLoadNope>, _1>>{}));
    // Scale: load from gmem
    static constexpr int kSmemElemsPerStoreNope = sizeof(cute::uint128_t) / sizeof(Element);
    using R2STileCopyNope = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopy, Element>{},
                        Layout<Shape<Int<kNThreads / kSmemThreadsPerRowNope>, Int<kSmemThreadsPerRowNope>>,
                               Stride<Int<kSmemThreadsPerRowNope>, _1>>{},
                        Layout<Shape<_1, Int<kSmemElemsPerStoreNope>>, Stride<Int<kSmemElemsPerStoreNope>, _1>>{}));

    using FragScale = std::conditional_t<
        kHeadDim == 576,
        decltype(make_fragment_like<float>(Layout<Shape<_4>>{})),
        decltype(make_fragment_like<int64_t>(Layout<Shape<_1>>{}))
    >;

    TensortKgKNope tKgK_nope;
    TensortKgKRope tKgK_rope;
    GmemTiledCopyNope gmem_tiled_copy_K_nope;
    GmemTiledCopyRope gmem_tiled_copy_K_rope;
    S2RTileCopyNope s2r_tile_copy_K_nope;
    R2STileCopyNope r2s_tile_copy_K_nope;

    const int tidx;
    ElementKVCache* gK_nope = nullptr;
    Element* gK_rope = nullptr;

    CUTLASS_DEVICE
    KVCacheG2SFP8(int tidx)
        : tidx(tidx) {
        tKgK_nope = gmem_tiled_copy_K_nope.get_thread_slice(tidx).partition_S(TensorKNope{});
        tKgK_rope = gmem_tiled_copy_K_rope.get_thread_slice(tidx).partition_S(TensorKRope{});
    };

    template <typename Tensor0>
    __forceinline__ __device__ void
    load_g2s_async(ElementKVCache* gK_ptr, Tensor0& sK, int blk_idx, int rel_idx_in_blk,
            const index_t stride_b, const index_t stride_r, bool is_token_valid) {

        auto addr_nope = make_smem_ptr(reinterpret_cast<ElementKVCache*>(sK.data().get()));
        Tensor sK_nope = make_tensor(addr_nope, SmemLayoutKNopeFP8{});
        Tensor sK_rope = make_tensor(sK.data()+ size(SmemLayoutKNopeFP8Np{}), SmemLayoutKRope{});
        Tensor tKsK_nope = gmem_tiled_copy_K_nope.get_thread_slice(tidx).partition_D(sK_nope);
        Tensor tKsK_rope = gmem_tiled_copy_K_rope.get_thread_slice(tidx).partition_D(sK_rope);

        if constexpr (kHeadDim == 512) {
            gK_nope = gK_ptr + blk_idx * stride_b + rel_idx_in_blk * BytesPerToken;
            gK_rope = (Element*)(gK_nope + kHeadDimNope);
        } else {
            gK_nope = gK_ptr + blk_idx * stride_b + rel_idx_in_blk * stride_r;
            gK_rope = (Element*)(gK_nope + kHeadDimV + 4 * sizeof(float));
        }

        tKgK_nope.data() = gK_nope + (tidx % kGmemThreadsPerRowNope) * kGmemElemsPerLoadNope;
        gmem_tiled_copy_K_nope.pred = is_token_valid;
        cute::copy(gmem_tiled_copy_K_nope, tKgK_nope(_, 0, _), tKsK_nope(_, 0, _));

        tKgK_rope.data() = gK_rope + (tidx % kGmemThreadsPerRowRope) * kGmemElemsPerLoadRope;
        gmem_tiled_copy_K_rope.pred = is_token_valid;
        cute::copy(gmem_tiled_copy_K_rope, tKgK_rope(_, 0, _), tKsK_rope(_, 0, _));

        cute::cp_async_fence();
    }

    template <typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void
    cvt_fp8_store(Tensor0& Tsr_org, Tensor1& Tsr_bf16, Tensor2& scale) {
        auto addr_fp8 = make_smem_ptr(reinterpret_cast<ElementKVCache*>(Tsr_org.data().get()));
        Tensor sK_fp8 = make_tensor(addr_fp8, SmemLayoutKNopeFP8{});
        Tensor tKsK_fp8 = s2r_tile_copy_K_nope.get_thread_slice(tidx).partition_D(sK_fp8);
        Tensor tKrK_fp8 = make_fragment_like(tKsK_fp8);

        Tensor sK_bf16 = make_tensor(Tsr_bf16.data(), SmemLayoutKNopeBF16{});
        Tensor tKsK_bf16 = r2s_tile_copy_K_nope.get_thread_slice(tidx).partition_D(sK_bf16);
        Tensor tKrK_bf16 = make_fragment_like(tKsK_bf16);

        if constexpr (kHeadDim == 512) {
            fp8_e8m0* fp8_scale = (fp8_e8m0*)(scale.data());

            #pragma unroll
            for (int k = size<2>(tKrK_fp8) - 2; k >= 0; --k) {
                // last 64 elements is rope
                cute::copy(s2r_tile_copy_K_nope, tKsK_fp8(_, _, k), tKrK_fp8(_, _, k));
                fp8_to_bf16(tKrK_fp8(_, _, k), tKrK_bf16(_, _, k), float(fp8_scale[k]));
                cute::copy(r2s_tile_copy_K_nope, tKrK_bf16(_, _, k), tKsK_bf16(_, _, k));
            }
        } else {
            #pragma unroll
            for (int k = size<2>(tKrK_fp8) - 1; k >= 0; --k) {
                cute::copy(s2r_tile_copy_K_nope, tKsK_fp8(_, _, k), tKrK_fp8(_, _, k));
                fp8_to_bf16(tKrK_fp8(_, _, k), tKrK_bf16(_, _, k), scale[k/2]);
                cute::copy(r2s_tile_copy_K_nope, tKrK_bf16(_, _, k), tKsK_bf16(_, _, k));
            }
        }
    }
};

} // namespace flash
