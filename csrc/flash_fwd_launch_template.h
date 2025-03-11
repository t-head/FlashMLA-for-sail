/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/CUDAContext.h>
#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_even_MN, bool Split) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_even_MN, Split>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, true>(params);
}

template<typename Kernel_traits>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    BOOL_SWITCH(params.num_splits > 1, Split, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            constexpr size_t smem_size = Split ? Kernel_traits::kSmemSizeAccum: Kernel_traits::kSmemSize;
            auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, Split>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
#if 0
            int ctas_per_sm;
            cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
            printf("[run_flash_splitkv_fwd_]:\n");
            printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, kernel);
            auto dprops = at::cuda::getCurrentDeviceProperties();

            int sm_count = dprops->multiProcessorCount == 64 ? 20 : dprops->multiProcessorCount;
            printf("blockM:%d, blockN:%d, threads:%d, params.num_splits:%d, block_size:%d\n",
                    Kernel_traits::kBlockM, Kernel_traits::kBlockN, Kernel_traits::kNThreads, params.num_splits, params.page_block_size);
            printf("Is_causal:%d, ngroups:%d\n", Is_causal, params.ngroups);
            printf("seq[%d, %d], grid_n[%d, %d, %d]\n",
                    params.seqlen_q, params.seqlen_k, grid.x, grid.y, grid.z);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                    float(grid.x * grid.y * grid.z) / float(sm_count * ctas_per_sm));
#endif
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });

    if (params.num_splits > 1) {
       // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        constexpr static int kNThreads = 128;

        if (params.num_splits <= 2) {
            flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1><<<grid_combine, kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 4) {
            flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2><<<grid_combine, kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 8) {
            flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3><<<grid_combine, kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 16) {
            flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4><<<grid_combine, kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 32) {
            flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5><<<grid_combine, kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 64) {
            flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6><<<grid_combine, kNThreads, 0, stream>>>(params);
        } else if (params.num_splits <= 128) {
            flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7><<<grid_combine, kNThreads, 0, stream>>>(params);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template<typename T, int Headdim, int Headdim_V>
void run_mha_fwd_splithd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    constexpr static int kBlockN = 16;
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kBlockM / 8, false, false, T, Headdim_V>>(params, stream);
}
