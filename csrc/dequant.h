#pragma once

#include "utils.h"

namespace flash {

using namespace cute;

template<typename T, int Size>
__device__ __forceinline__ void load_128b_from_gmem(const void* src_ptr, T* dst_ptr) {
    static_assert(sizeof(T) * Size== 128/8);
    // int4 ret;
    int4* ptr = reinterpret_cast<int4*>(dst_ptr);
    asm volatile("ld.global.nc.L1::evict_last.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];" \
            : "=r"((*ptr).x), "=r"((*ptr).y), "=r"((*ptr).z), "=r"((*ptr).w) \
            : "l"(src_ptr)); \

}

template<typename fp8T, typename bf16T, int Size>
__device__ __forceinline__ void cvt_fp8_bf16(const fp8T* src_ptr, bf16T* dst_ptr, const float &scale) {
    //dequant//
    bf16T scale_bf162 = (bf16T)(scale);
    //fp8 * float -> bf16 with 16 numbers.
    #pragma unroll
    for (int m = 0; m < Size; m++) {
        dst_ptr[m] = (bf16T)(src_ptr[m])*scale_bf162;
    }
    // TODO: acceleration with bf162/__nv_fp8x4_e4m3/float4.
    // static_assert(Size % 4== 0);
    // // static_assert(sizeof(fp8T) == 8);
    // __nv_bfloat162 scale_bf162 = __float2bfloat162_rn(scale);
    // const __nv_fp8x4_e4m3* src_fp82 =  reinterpret_cast<const __nv_fp8x4_e4m3*>(src_ptr);
    // __nv_bfloat162* dst_bf162 =  reinterpret_cast<__nv_bfloat162*>(dst_ptr);
}


template <int kBlockN, int kNThreads>
struct KVCacheGmemBf16 {
    using ElementKVCache = cutlass::bfloat16_t;
    using index_t = int64_t;
    static constexpr int kBlockKSmem = 64;
    static constexpr int kSwizzle = 3;
    static constexpr int kHeadDim = 576;
    static constexpr int kHeadDimV = 512;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementKVCache);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

    using TensorK = decltype(make_tensor(make_gmem_ptr(static_cast<ElementKVCache*>(nullptr)),
        Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{}));

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

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>,
        Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, ElementKVCache>{},
        GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));

    using TensortKgK = decltype(GmemTiledCopy{}.get_thread_slice(int(0)).partition_S(TensorK{}));

    GmemTiledCopy gmem_tiled_copy_K;
    TensortKgK tKgK;

    const int tidx;
    int* gIndices_ptr;
    ElementKVCache* gK_ptr;
    const int block_size;
    const index_t batch_stride;
    const index_t row_stride;

    CUTLASS_DEVICE
    KVCacheGmemBf16(int tidx, int* gIndices_ptr, ElementKVCache* gK_ptr,
        const int block_size, const index_t batch_stride, const index_t row_stride)
        : tidx(tidx), block_size(block_size), batch_stride(batch_stride), row_stride(row_stride),
          gIndices_ptr(gIndices_ptr), gK_ptr(gK_ptr)
    {

        TensorK gK = make_tensor(make_gmem_ptr(gK_ptr),
            Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
        tKgK = gmem_tiled_copy_K.get_thread_slice(tidx).partition_S(gK);
    };

    template <bool Clear_OOB_K=true, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void
    load_from_gmem(int n_block, int kv_store_num, Tensor0 &smem_valid_indices, Tensor1 &sK) {

        int row_offset_indices = tidx/8 + n_block * kBlockN;
        int token_index = __ldg(gIndices_ptr + row_offset_indices);
        bool is_token_valid = token_index >= 0;
        smem_valid_indices(kv_store_num%2, tidx/8) = is_token_valid;
        int block_index = token_index/block_size;
        int rel_idx_in_block = (token_index+block_size) % block_size;
        // ElementKVCache *gK_now = gK_base + block_index*batch_stride
        //                           + rel_idx_in_block*row_stride
        //                           + (tidx%8)*8;
        Tensor tKsK = gmem_tiled_copy_K.get_thread_slice(tidx).partition_D(sK);
        // Tensor tKgK = gmem_thr_copy_K.partition_S(gK);


        tKgK.data() = gK_ptr + (int64_t) block_index*batch_stride
                              + rel_idx_in_block*row_stride
                              + (tidx%8)*8;
        if (is_token_valid) {
            cute::copy(gmem_tiled_copy_K, tKgK, tKsK);
        } else if (Clear_OOB_K) {
            cute::clear(tKsK);
        }

    };
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
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

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
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}, Stride<Int<BytesPerToken>, _1>{}); // (_64,_512):(656,_1)
        tKgK_nope = GmemTiledCopyNope{}.get_thread_slice(tidx).partition_S(gK_nope); //((_16,_1),_1,_4):((_1,_0),_0,_128)
        TensorKRope gK_rope = make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
            Shape<Int<kBlockN>, Int<kHeadDimRope>>{}, Stride<Int<BytesPerToken>, _1>{}); // (_64,_64):(_656,_1)
        tKgK_rope = gmem_tiled_copy_K_rope.get_thread_slice(tidx).partition_S(gK_rope); //((_8,_1),_1,_1):((_1,_0),_0,_0)
    };

    template <bool Clear_OOB_K=true, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void
    load_from_gmem(int n_block, int kv_store_num, Tensor0 &smem_valid_indices, Tensor1 &sK) {

        int row_offset_indices = tidx/8 + n_block * kBlockN;
        int token_index = __ldg(gIndices_ptr + row_offset_indices);
        bool is_token_valid = token_index >= 0;
        smem_valid_indices(kv_store_num%2, tidx/8) = is_token_valid;
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

}