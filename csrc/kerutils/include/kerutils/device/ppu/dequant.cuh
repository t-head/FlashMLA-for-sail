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
    auto dstv = recast<bf16x8>(dst);
    auto srcv = recast<fp8x8>(src);
    __ppu_bfloat162 scale_bf162 = __float2bfloat162_rn(scale);

    #define DEQUANT_FP8x4(OUTPUT_BF16_LO, OUTPUT_BF16_HI, FP8x4) \
    { \
        float4 fp32x4 = (float4)(FP8x4); \
        OUTPUT_BF16_LO = __float22bfloat162_rn({fp32x4.x, fp32x4.y})*scale_bf162; \
        OUTPUT_BF16_HI = __float22bfloat162_rn({fp32x4.z, fp32x4.w})*scale_bf162; \
    }

    #pragma unroll
    for (int _di = 0; _di < size(srcv); ++_di) {
        bf16x8 &result = dstv[_di];
        fp8x8  &inputs = srcv[_di];
        DEQUANT_FP8x4(result.a01, result.a23, inputs.lo);
        DEQUANT_FP8x4(result.a45, result.a67, inputs.hi);
    }
#endif
}



static constexpr int kFp4LutReplicas = 32;
static constexpr int kFp4LutEntries = 256 * kFp4LutReplicas;

__device__ __forceinline__ uint16_t fp4_e2m1_to_bf16_bits(uint8_t code) {
    uint16_t magnitude = code & 0x7;
    uint16_t value = magnitude == 0 ? 0
        : (magnitude == 1 ? 0x3f00 : uint16_t(0x3f00 + (magnitude << 6)));
    return value | uint16_t((code & 0x8) << 12);
}

__device__ __forceinline__ uint32_t fp4_pair_to_bf16x2_bits(uint8_t pair) {
    return uint32_t(fp4_e2m1_to_bf16_bits(pair & 0xF))
        | (uint32_t(fp4_e2m1_to_bf16_bits(pair >> 4)) << 16);
}

__device__ __forceinline__ uint4 fp4_word_to_bf16x2_bits(uint32_t word) {
    const uint32_t high_selectors = ((word >> 1) & 0x77777777u) | (word & 0x11111111u);
    const uint32_t low_selectors0 = word & 0x7777u;
    const uint32_t low_selectors1 = (word >> 16) & 0x7777u;
    const uint32_t low0 = __byte_perm(0xc0800000u, 0xc0804000u, low_selectors0);
    const uint32_t high0 = __byte_perm(0x40403f00u, 0xc0c0bf80u, high_selectors);
    const uint32_t low1 = __byte_perm(0xc0800000u, 0xc0804000u, low_selectors1);
    const uint32_t high1 = __byte_perm(0x40403f00u, 0xc0c0bf80u, high_selectors >> 16);
    return uint4{__byte_perm(low0, high0, 0x5140), __byte_perm(low0, high0, 0x7362),
                 __byte_perm(low1, high1, 0x5140), __byte_perm(low1, high1, 0x7362)};
}

template <bool RegisterDecode = false, typename Tensor0>
__device__ __forceinline__ void fp4x16_to_bf16(
        uint64_t packed, Tensor0 dst, float scale, const uint32_t* lut) {
    auto dst_bf16x2 = recast<__ppu_bfloat162>(dst);
    __ppu_bfloat162 scale_bf16x2 = __float2bfloat162_rn(scale);
    if constexpr (RegisterDecode) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint4 values = fp4_word_to_bf16x2_bits(uint32_t(packed >> (half * 32)));
            dst_bf16x2[half * 4 + 0] = __hmul2(*reinterpret_cast<const __ppu_bfloat162*>(&values.x), scale_bf16x2);
            dst_bf16x2[half * 4 + 1] = __hmul2(*reinterpret_cast<const __ppu_bfloat162*>(&values.y), scale_bf16x2);
            dst_bf16x2[half * 4 + 2] = __hmul2(*reinterpret_cast<const __ppu_bfloat162*>(&values.z), scale_bf16x2);
            dst_bf16x2[half * 4 + 3] = __hmul2(*reinterpret_cast<const __ppu_bfloat162*>(&values.w), scale_bf16x2);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t values = lut[uint8_t(packed >> (i * 8)) * kFp4LutReplicas];
            dst_bf16x2[i] = __hmul2(*reinterpret_cast<__ppu_bfloat162*>(&values), scale_bf16x2);
        }
    }
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

template<int kBlockN, int kNThreads, int kHeadDim_, bool IsV41FP8_ = false>
struct KVCacheG2SFP8 {
    using ElementKVCache = cutlass::float_e4m3_t;
    using Element = cutlass::bfloat16_t;
    using index_t = int64_t;
    static constexpr int kBlockKSmem = 64;
    static constexpr int kBlockKSmemFp8 = 128;
    static constexpr int kSwizzle = 3;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool IsV41FP8 = IsV41FP8_;
    static_assert(!IsV41FP8 || kHeadDim == 512);
    static constexpr int BytesPerToken = kHeadDim == 576 ? 656 : (IsV41FP8 ? 528 : 576);
    static constexpr int kHeadDimV = 512;
    static constexpr int kHeadDimRope = 64;
    static constexpr int kHeadDimNope = IsV41FP8 ? kHeadDim : kHeadDim - kHeadDimRope;
    static constexpr int kGroupNope = 128;
    static constexpr int NUM_SCALES = kHeadDim == 576 ? 4 : (IsV41FP8 ? 16 : 8);


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

    // V4.1-only 16-elem wide s2r/r2s (8->4 tiles).
    static constexpr int kSmemElemsWideNope = 16;
    using S2RTileCopyNopeWide = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopy, ElementKVCache>{},
                        Layout<Shape<Int<kNThreads / kSmemThreadsPerRowNope>, Int<kSmemThreadsPerRowNope>>,
                               Stride<Int<kSmemThreadsPerRowNope>, _1>>{},
                        Layout<Shape<_1, Int<kSmemElemsWideNope>>, Stride<Int<kSmemElemsWideNope>, _1>>{}));
    using R2STileCopyNopeWide = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopy, Element>{},
                        Layout<Shape<Int<kNThreads / kSmemThreadsPerRowNope>, Int<kSmemThreadsPerRowNope>>,
                               Stride<Int<kSmemThreadsPerRowNope>, _1>>{},
                        Layout<Shape<_1, Int<kSmemElemsWideNope>>, Stride<Int<kSmemElemsWideNope>, _1>>{}));

    using FragScale = std::conditional_t<
        kHeadDim == 576,
        decltype(make_fragment_like<float>(Layout<Shape<_4>>{})),
        std::conditional_t<
            IsV41FP8,
            decltype(make_fragment_like<int64_t>(Layout<Shape<_2>>{})),
            decltype(make_fragment_like<int64_t>(Layout<Shape<_1>>{}))
        >
    >;

    TensortKgKNope tKgK_nope;
    TensortKgKRope tKgK_rope;
    GmemTiledCopyNope gmem_tiled_copy_K_nope;
    GmemTiledCopyRope gmem_tiled_copy_K_rope;
    S2RTileCopyNope s2r_tile_copy_K_nope;
    R2STileCopyNope r2s_tile_copy_K_nope;
    S2RTileCopyNopeWide s2r_wide;
    R2STileCopyNopeWide r2s_wide;

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
            if constexpr (IsV41FP8) {
                gK_nope = gK_ptr + blk_idx * stride_b + index_t(rel_idx_in_blk) * BytesPerToken;
            } else {
                gK_nope = gK_ptr + blk_idx * stride_b + rel_idx_in_blk * BytesPerToken;
                gK_rope = (Element*)(gK_nope + kHeadDimNope);
            }
        } else {
            gK_nope = gK_ptr + blk_idx * stride_b + rel_idx_in_blk * stride_r;
            gK_rope = (Element*)(gK_nope + kHeadDimV + 4 * sizeof(float));
        }

        tKgK_nope.data() = gK_nope + (tidx % kGmemThreadsPerRowNope) * kGmemElemsPerLoadNope;
        gmem_tiled_copy_K_nope.pred = is_token_valid;
        cute::copy(gmem_tiled_copy_K_nope, tKgK_nope(_, 0, _), tKsK_nope(_, 0, _));

        if constexpr (!IsV41FP8) {
            tKgK_rope.data() = gK_rope + (tidx % kGmemThreadsPerRowRope) * kGmemElemsPerLoadRope;
            gmem_tiled_copy_K_rope.pred = is_token_valid;
            cute::copy(gmem_tiled_copy_K_rope, tKgK_rope(_, 0, _), tKsK_rope(_, 0, _));
        }

        cute::cp_async_fence();
    }

    template <bool RegisterV41Scale = true, typename Tensor0, typename Tensor1, typename Tensor2>
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
            constexpr int kNumQuantTiles = IsV41FP8 ? size<2>(tKrK_fp8) : size<2>(tKrK_fp8) - 1;

            if constexpr (IsV41FP8) {
                // 16-elem wide s2r/r2s + load-hoisted dequant.
                // per-32 scale: thread's 16 dims are within one 32-region -> w*4 + (tid%8)/2.
                Tensor tKsK_fp8_w = s2r_wide.get_thread_slice(tidx).partition_D(sK_fp8);
                Tensor tKrK_fp8_w = make_fragment_like(tKsK_fp8_w);
                Tensor tKsK_bf16_w = r2s_wide.get_thread_slice(tidx).partition_D(sK_bf16);
                Tensor tKrK_bf16_w = make_fragment_like(tKsK_bf16_w);
                constexpr int kWideTiles = size<2>(tKrK_fp8_w);
                #pragma unroll
                for (int w = kWideTiles - 1; w >= 0; --w) {
                    cute::copy(s2r_wide, tKsK_fp8_w(_, _, w), tKrK_fp8_w(_, _, w));
                }
                if constexpr (RegisterV41Scale) {
                    cute::for_each(make_seq<kWideTiles>{}, [&](auto tile) {
                        constexpr int w = kWideTiles - 1 - decltype(tile)::value;
                        // Keep scale words in registers instead of indexing their byte address.
                        const uint64_t scale_word = static_cast<uint64_t>(scale(Int<w / 2>{}));
                        const int scale_byte = (w % 2) * 4 + ((tidx % kSmemThreadsPerRowNope) / 2);
                        fp8_e8m0 scale_value;
                        scale_value.__x = static_cast<uint8_t>(scale_word >> (scale_byte * 8));
                        fp8_to_bf16(tKrK_fp8_w(_, _, w), tKrK_bf16_w(_, _, w), float(scale_value));
                        cute::copy(r2s_wide, tKrK_bf16_w(_, _, w), tKsK_bf16_w(_, _, w));
                    });
                } else {
                    fp8_e8m0* fp8_scale = (fp8_e8m0*)(scale.data());
                    #pragma unroll
                    for (int w = kWideTiles - 1; w >= 0; --w) {
                        int scale_idx = w * 4 + ((tidx % kSmemThreadsPerRowNope) / 2);
                        fp8_to_bf16(tKrK_fp8_w(_, _, w), tKrK_bf16_w(_, _, w), float(fp8_scale[scale_idx]));
                        cute::copy(r2s_wide, tKrK_bf16_w(_, _, w), tKsK_bf16_w(_, _, w));
                    }
                }
            } else {
                fp8_e8m0* fp8_scale = (fp8_e8m0*)(scale.data());
                #pragma unroll
                for (int k = kNumQuantTiles - 1; k >= 0; --k) {
                    cute::copy(s2r_tile_copy_K_nope, tKsK_fp8(_, _, k), tKrK_fp8(_, _, k));
                    fp8_to_bf16(tKrK_fp8(_, _, k), tKrK_bf16(_, _, k), float(fp8_scale[k]));
                    cute::copy(r2s_tile_copy_K_nope, tKrK_bf16(_, _, k), tKsK_bf16(_, _, k));
                }
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

template<int kBlockN, int kNThreads, bool RegisterDecode = false>
struct KVCacheG2SFP4 {
    using Base = KVCacheG2SFP8<kBlockN, kNThreads, 512, true>;
    using index_t = int64_t;
    using SmemLayoutKBF16 = typename Base::SmemLayoutKNopeBF16;
    using R2STileCopyWide = typename Base::R2STileCopyNopeWide;

    static constexpr int kDataBytesPerToken = 256;
    static constexpr int kScalesPerToken = 32;
    static constexpr int kThreadsPerRow = 8;
    static constexpr int kWideTiles = 4;
    static constexpr int kLutReplicas = kFp4LutReplicas;
    static constexpr int kLutEntries = kFp4LutEntries;
    static_assert(kNThreads == kBlockN * kThreadsPerRow);

    R2STileCopyWide r2s_wide;
    const int tidx;
    const uint32_t* lut;

    CUTLASS_DEVICE
    KVCacheG2SFP4(int tidx, const uint32_t* lut)
        : tidx(tidx), lut(RegisterDecode ? lut : lut + tidx % kLutReplicas) {}

    // Stage packed data and scales in the sK buffer while the current block runs.
    template <typename Tensor0>
    __forceinline__ __device__ void
    load_g2s_async(const uint8_t* gK_ptr, Tensor0& sK, int blk_idx, int rel_idx_in_blk,
                   index_t stride_b, bool is_token_valid) {
        const uint8_t* block = gK_ptr + blk_idx * stride_b;
        const uint8_t* data_row = block + index_t(rel_idx_in_blk) * (kDataBytesPerToken + kScalesPerToken);
        const uint8_t* scale_row = data_row + kDataBytesPerToken;
        const int lane_in_row = tidx % kThreadsPerRow;
        uint64_t* sdata = reinterpret_cast<uint64_t*>(sK.data().get());
        uint32_t* sscale = reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(sK.data().get()) + kBlockN * kDataBytesPerToken);
        #pragma unroll
        for (int w = 0; w < kWideTiles; ++w) {
            const uint64_t* g = reinterpret_cast<const uint64_t*>(data_row + w * 64 + lane_in_row * 8);
            PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint64_t>::copy(*g, sdata[tidx * kWideTiles + w], is_token_valid);
        }
        const uint32_t* gscale = reinterpret_cast<const uint32_t*>(scale_row + lane_in_row * 4);
        PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint32_t>::copy(*gscale, sscale[tidx], is_token_valid);
        cute::cp_async_fence();
    }

    // Dequantize staged values in place after all threads finish reading them.
    template <typename Tensor0>
    __forceinline__ __device__ void
    cvt_fp4_store(Tensor0& sK) {
        auto sK_bf16 = make_tensor(sK.data(), SmemLayoutKBF16{});
        auto tKsK_bf16 = r2s_wide.get_thread_slice(tidx).partition_D(sK_bf16);
        auto tKrK_bf16 = make_fragment_like(tKsK_bf16);
        const uint64_t* sdata = reinterpret_cast<const uint64_t*>(sK.data().get());
        const uint8_t* sscale = reinterpret_cast<const uint8_t*>(sK.data().get())
            + kBlockN * kDataBytesPerToken;
        const int lane_in_row = tidx % kThreadsPerRow;
        const int token = tidx / kThreadsPerRow;
        uint64_t packed[kWideTiles];
        uint8_t scale_raw[kWideTiles];
        #pragma unroll
        for (int w = 0; w < kWideTiles; ++w) {
            packed[w] = sdata[tidx * kWideTiles + w];
            scale_raw[w] = sscale[token * kScalesPerToken + w * kThreadsPerRow + lane_in_row];
        }
        __syncthreads();  // staging read done before in-place bf16 write clobbers it
        #pragma unroll
        for (int w = 0; w < kWideTiles; ++w) {
            __hg_fp8_e4m3 scale_e4m3 = *reinterpret_cast<const __hg_fp8_e4m3*>(&scale_raw[w]);
            fp4x16_to_bf16<RegisterDecode>(packed[w], tKrK_bf16(_, _, w), float(scale_e4m3), lut);
            cute::copy(r2s_wide, tKrK_bf16(_, _, w), tKsK_bf16(_, _, w));
        }
    }
};

} // namespace flash
