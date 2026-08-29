#include "mla_combine.h"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

// #include "params.h"

#include "flash.h"
#include "config.h"  // for BLOCK_SIZE_M and HEAD_DIM_V
#include "hardware_info.h"
// #include "flash_fwd_kernel.h"
#include "flash_splitkv/traits.h"


#include <hggc_ad.h>
#include "utils.h"

using namespace cute;

template<typename Kernel_traits, int kMaxSplits>
__global__ void __launch_bounds__(256, 1, 1)
flash_fwd_mla_combine_kernel_small_size(__grid_constant__ const Flash_fwd_params params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = 128;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int hs = params.h * params.seqlen_q;
    const int batch_idx = bidx / hs;
    const int hs_idx = bidx % hs;

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int actual_num_splits = __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    FLASH_DEVICE_ASSERT(actual_num_splits <= kMaxSplits);
    if (actual_num_splits == 1) return;

    __shared__ ElementAccum sLseScale[kMaxSplits];

    const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
    const index_t row_offset_lse = bidx;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                   Shape<Int<kMaxSplits>>{}, make_stride(hs));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
        constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

        float local_lse[kNLsePerThread];
        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            local_lse[i] = split < actual_num_splits ? gLSEaccum(split) : -INFINITY;
        }

        float max_lse = -INFINITY;
        for (int i = 0; i < kNLsePerThread; ++i) max_lse = max(max_lse, local_lse[i]);
        for (int offset = 16; offset >= 1; offset /= 2) max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kNLsePerThread; ++i)
            sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);

        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);


        float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : log2f(sum_lse) + max_lse;
        if (tidx == 0) gLSE(0) = global_lse / (float)M_LOG2E;

        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            if (split < actual_num_splits) sLseScale[split] = exp2f(local_lse[i] - global_lse);
        }
    }
    __syncthreads();

    static_assert(kHeadDimV % kNThreads == 0);
    constexpr int Elements = kHeadDimV / kNThreads;
    const index_t row_offset_oaccum = (split_offset * hs + hs_idx) * kHeadDimV;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kHeadDimV>>{}, Stride<_1>{});
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            Layout<Shape<Int<kNThreads>>>{},
            Layout<Shape<Int<Elements>>>{}));
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    for (int split = 0; split < actual_num_splits; ++split) {
        cute::copy(tOgOaccum, tOrOaccum);
        ElementAccum lse_scale = sLseScale[split];
        for (int i = 0; i < size(tOrO); ++i) {
            tOrO(i) += lse_scale * tOrOaccum(i);
        }
        tOgOaccum.data() = tOgOaccum.data() + hs * kHeadDimV;
    }

    Tensor rO = flash::convert_type<Element>(tOrO);
    const int head_idx = (bidx - batch_idx * hs) / params.seqlen_q;
    const int row = bidx - batch_idx * hs - head_idx * params.seqlen_q;
    auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride + row * params.o_row_stride;
    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + tidx * Elements), Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
    cute::copy(rO, gO);
}

template<typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_mla_combine_kernel(__grid_constant__ const Flash_fwd_params params) {
    // grid_shape: [batch_size, num_q_heads*seqlen_q / BLOCK_SIZE_M]
    // Each CTA gathers the activation of some heads from one batch, do scaling & accumulation, and save the result
    // static_assert(NUM_THREADS/32 == BLOCK_SIZE_M); // The number of warps == block_size_m
    const int batch_idx = blockIdx.x;
    const int m_block_idx = blockIdx.y;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;

    const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
    const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
    const int my_num_splits = end_split_idx - start_split_idx;
    FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);

    if (my_num_splits == 1) {
        return;
    }
    
    const int num_q_seqs = params.seqlen_q * params.h;
    const int num_cur_valid_q_seqs = min(BLOCK_SIZE_M, num_q_seqs - m_block_idx*BLOCK_SIZE_M);

    Tensor gLseAccum = make_tensor(
        make_gmem_ptr((float*)params.softmax_lseaccum_ptr + start_split_idx*num_q_seqs + m_block_idx*BLOCK_SIZE_M),
        Shape<Int<MAX_SPLITS>, Int<BLOCK_SIZE_M>>{},
        make_stride(num_q_seqs, _1{})
    );
    Tensor gLse = make_tensor(
        make_gmem_ptr((float*)params.softmax_lse_ptr + batch_idx*num_q_seqs + m_block_idx*BLOCK_SIZE_M),
        Shape<Int<BLOCK_SIZE_M>>{},
        Stride<_1>{}
    );
    
    extern __shared__ float smem_buf[];
    Tensor sLseScale = make_tensor(
        make_smem_ptr(smem_buf),
        Shape<Int<BLOCK_SIZE_M>, Int<MAX_SPLITS>>{},
        Stride<Int<MAX_SPLITS+1>, _1>{} // +1 to avoid bank conflict
    );

    // Read gLseAccum into sLseScale
    {
        #pragma unroll 4
        for (int elem_idx = threadIdx.x; elem_idx < my_num_splits*BLOCK_SIZE_M; elem_idx += NUM_THREADS) {
            int split_idx = elem_idx / BLOCK_SIZE_M;
            int seq_idx = elem_idx % BLOCK_SIZE_M;
            sLseScale(seq_idx, split_idx) = seq_idx < num_cur_valid_q_seqs ? gLseAccum(split_idx, seq_idx) : -INFINITY;
        }
        __syncthreads();
    }

    if (warp_idx >= num_cur_valid_q_seqs)
        return;

    // Warp #i gathers LseAccum for seq #i
    {
        constexpr int NUM_LSE_PER_THREAD = cute::ceil_div(MAX_SPLITS, 32);

        float local_lse[NUM_LSE_PER_THREAD];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i*32 + lane_idx;
            local_lse[i] = split_idx < my_num_splits ? sLseScale(warp_idx, split_idx) : -INFINITY;
        }

        float max_lse = -INFINITY;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            max_lse = max(max_lse, local_lse[i]);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);

        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : log2f(sum_lse) + max_lse;

        if (lane_idx == 0)
            gLse(warp_idx) = global_lse / (float)M_LOG2E;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i*32 + lane_idx;
            if (split_idx < my_num_splits) sLseScale(warp_idx, split_idx) = exp2f(local_lse[i] - global_lse);
        }
    }

    __syncwarp();

    // Warp #i accumulates activation for seq #i
    {
        const int64_t row_offset_oaccum = (int64_t)(start_split_idx*num_q_seqs+m_block_idx*BLOCK_SIZE_M+warp_idx) * HEAD_DIM_V;
        Tensor gOaccum = make_tensor(
            make_gmem_ptr(reinterpret_cast<float *>(params.oaccum_ptr) + row_offset_oaccum),
            Shape<Int<MAX_SPLITS>, Int<HEAD_DIM_V>>{},
            make_stride(num_q_seqs*HEAD_DIM_V, _1{})
        );

        static_assert(HEAD_DIM_V % 32 == 0);
        constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / 32;
        float result[ELEMS_PER_THREAD];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            result[i] = 0.0f;

        #pragma unroll 2
        for (int split = 0; split < my_num_splits; ++split) {
            float lse_scale = sLseScale(warp_idx, split);
            if (lse_scale != 0.f) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                    result[i] += lse_scale * gOaccum(split, lane_idx + i*32);
                }
            }
        }

        const int q_seq_idx = m_block_idx*BLOCK_SIZE_M + warp_idx;
        const int k_head_idx = q_seq_idx / params.seqlen_q;
        auto o_ptr = reinterpret_cast<ElementT *>(params.o_ptr) + batch_idx*params.o_batch_stride + k_head_idx*params.o_head_stride + (q_seq_idx%params.seqlen_q)*params.o_row_stride;
        Tensor gO = make_tensor(
            make_gmem_ptr(o_ptr),
            Shape<Int<HEAD_DIM_V>>{},
            Stride<_1>{}
        );

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            gO(lane_idx+i*32) = (ElementT)result[i];
    }
}


#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)       \
    [&] {                                                  \
        if (NUM_SPLITS <= 32) {                            \
            constexpr static int NAME = 32;                \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 64) {                     \
            constexpr static int NAME = 64;                \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 96) {                     \
            constexpr static int NAME = 96;                \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 128) {                    \
            constexpr static int NAME = 128;               \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 160) {                    \
            constexpr static int NAME = 160;               \
            return __VA_ARGS__();                          \
        } else {                                           \
            FLASH_ASSERT(false);                           \
        }                                                  \
    }()


template<typename ElementT>
void run_flash_mla_combine_kernel(Flash_fwd_mla_params &params, hggcStream_t stream) {
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, NUM_SPLITS, [&] {
        if (params.b <= 2) {
            // small input use one block per head
            dim3 grid_combine(params.b * params.h * params.seqlen_q);
            auto combine_kernel = &flash_fwd_mla_combine_kernel_small_size<Traits<ElementT>, NUM_SPLITS>;
#ifdef __HGGCCC__
            const void *flash_func = reinterpret_cast<const void*>(combine_kernel);
            HGfunction func = static_cast<HGfunction>(NULL);
            hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&func), flash_func);

            void* kernel_args[] = {&params};
            HGlaunchAttributeAD LaunchAttr = {HGAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
            HGlaunchConfigAD LaunchCfg = {grid_combine.x, grid_combine.y, grid_combine.z, 128, 1, 1, 0, stream, &LaunchAttr, 0};
            // LaunchAttr.value.schedPreference.blocksPerMultiprocessor = 1;//schedule.bits.tb_per_cu;
            // LaunchAttr.value.schedPreference.gridStepX = 2;
            // LaunchAttr.value.schedPreference.gridStepY = 2;
            // LaunchAttr.value.schedPreference.flags = 2;
            CUDA_DRIVER_CHECK(hgLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
            combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
#endif
        } else {
            // bit input use one warp per head
            constexpr int BLOCK_SIZE_M = 8;
            constexpr int NUM_THREADS = BLOCK_SIZE_M*32;
            constexpr size_t smem_size = BLOCK_SIZE_M*(NUM_SPLITS+1)*sizeof(float);
            auto combine_kernel = &flash_fwd_mla_combine_kernel<ElementT, Config::HEAD_DIM_V, BLOCK_SIZE_M, NUM_SPLITS, NUM_THREADS>;
            CHECK_CUDA(hggcFuncSetAttribute(combine_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            // Use hggcLaunchKernelExC to enable PDL (Programmatic Dependent Launch)
            hggcLaunchAttribute attribute[1];
            attribute[0].id = hggcLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            hggcLaunchConfig_t combine_kernel_config = {
                dim3(params.b, cute::ceil_div(params.h*params.seqlen_q, BLOCK_SIZE_M), 1),
                dim3(NUM_THREADS, 1, 1),
                smem_size,
                stream,
                attribute,
                1
            };
            hggcLaunchKernelExC(&combine_kernel_config, (const void*)combine_kernel, (void**)&params);
        }
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template void run_flash_mla_combine_kernel<cutlass::bfloat16_t>(Flash_fwd_mla_params &params, hggcStream_t stream);

#ifndef FLASH_MLA_DISABLE_FP16
template void run_flash_mla_combine_kernel<cutlass::half_t>(Flash_fwd_mla_params &params, hggcStream_t stream);
#endif
