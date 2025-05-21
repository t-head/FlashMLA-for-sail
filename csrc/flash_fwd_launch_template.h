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

template<typename Kernel_traits, bool CrossCut = false>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    //FLASH_ASSERT(params.page_block_size == Kernel_traits::kBlockN);
    //constexpr size_t smem_size = Kernel_traits::kSmemSize;
    constexpr size_t smem_size = Kernel_traits::kSmemSizeAccum;
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, CrossCut>;
        //CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            int value = std::stoi(std::string(pEnv_params));
            if (value > 0) {
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
                printf("CrossCut:%d, USE_MMA_M8:%d\n", Kernel_traits::CrossCut, Kernel_traits::USE_MMA_M8);
                printf("kNWarps:%d, AtomLayoutQ:%d, AtomLayoutP:%d\n", Kernel_traits::kNWarps, Kernel_traits::AtomLayoutQ, Kernel_traits::AtomLayoutP);
                printf("Is_Q_in_regs:%d, Share_Q_K_smem:%d\n", Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem);
                printf("seq[%d, %d], grid_n[%d, %d, %d]\n",
                        params.seqlen_q, params.seqlen_k, num_m_block, params.h, params.num_sm_parts);
                printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                        float(num_m_block * params.h * params.num_sm_parts) / float(sm_count * ctas_per_sm));
            }
        }
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<Kernel_traits, kMaxSplits>;
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, int Headdim, int Headdim_V>
void run_mha_fwd_splithd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    // constexpr static int kBlockM = 64;  // Fixed for all head dimensions

    /// seqlen_q = 128 use CrossCut method
    bool cross_cut = use_cross_cut(params.seqlen_q, params.b);
#if ACOMPUTE_VERSION==10000
    if (cross_cut) {
        // support seqlen_q > 16.
        constexpr static int kBlockN= 32;
        if (params.seqlen_q <= 32) {
            constexpr static int kBlockM = 32;
            constexpr bool USE_MMA_M8 = 1;
            constexpr int kNwarps = 8;
            constexpr int AtomLayoutQ = 4;
            constexpr int AtomLayoutP = 1;
            run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                >, 1/*CrossCut*/>(params, stream);
        } else if (params.seqlen_q <= 64) {
            constexpr static int kBlockM = 64;
            constexpr bool USE_MMA_M8 = 1;
            constexpr int kNwarps = 8;
            constexpr int AtomLayoutQ = 8;
            constexpr int AtomLayoutP = 1;
            run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                >, 1/*CrossCut*/>(params, stream);
        } else if (params.seqlen_q > 64) {
            constexpr static int kBlockM = 128;
            constexpr bool USE_MMA_M8 = 0;
            constexpr int kNwarps = 16;
            constexpr int AtomLayoutQ = 8;
            constexpr int AtomLayoutP = 2;
            run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                >, 1/*CrossCut*/>(params, stream);
        }
        return;
    }
#else
    if (cross_cut) {
        // support seqlen_q > 16.
        constexpr static int kBlockN = 32;
        constexpr bool USE_MMA_M8 = 0;
        if (params.seqlen_q <= 32) {
            constexpr static int kBlockM = 32;
            constexpr int kNwarps = 4;
            constexpr int AtomLayoutQ = 2;
            constexpr int AtomLayoutP = 1;
            run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                >, 1/*CrossCut*/>(params, stream);
        } else if (params.seqlen_q <= 64) {
            constexpr static int kBlockM = 64;
            constexpr int kNwarps = 8;
            constexpr int AtomLayoutQ = 4;
            constexpr int AtomLayoutP = 1;
            run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                >, 1/*CrossCut*/>(params, stream);
        } else if (params.seqlen_q > 64) {
            constexpr static int kBlockM = 128;
            constexpr int kNwarps = 16;
            constexpr int AtomLayoutQ = 8;
            constexpr int AtomLayoutP = 2;
            run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                >, 1/*CrossCut*/>(params, stream);
        }
        return;
    }
#endif

    constexpr static int kBlockN = 16;
#if ACOMPUTE_VERSION==10000
    constexpr bool USE_MMA_M8 = 1;
#else
    constexpr bool USE_MMA_M8 = 0;
#endif
    SEQLENG_SWITCH(params.seqlen_q, [&] {
        constexpr int kNwarps = USE_MMA_M8 ? kBlockM / 8 : kBlockM / 16;
        run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
            Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
            T, Headdim_V, 0/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/>>(params, stream);

    });
}
