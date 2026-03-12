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
#include "flash_sparse_fwd_kernel.h"
#include "flash_splitkv/config.h" // for splitkv kernel
#include "flash_splitkv/splitkv_mla.h"

#ifdef __HGGCCC__
#include "cuda_ad.h"
#include "utils.h"
#endif


template<typename Kernel_traits>
void printf_show_log(const void* kernel, Flash_fwd_params &params, const size_t smem_size,
                     bool is_causal, bool is_sparse = false, bool is_fp8 = false) {
    char *pEnv_params = std::getenv("show_log");
    int num_m_block;
    if (pEnv_params && isdigit(*pEnv_params)) {
        int value = std::stoi(std::string(pEnv_params));
        if (value > 0) {
            int ctas_per_sm;
            cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
            if (is_sparse) {
                num_m_block = (params.seqlen_q / params.ngroups) * cute::ceil_div(params.ngroups, Kernel_traits::kBlockM);
                printf("[run_flash_sparse_decode_fwd_]: FP8 KVCache:%d\n", is_fp8);
            } else {
                num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
                printf("[run_flash_splitkv_fwd_]:\n");
            }
            printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, kernel);
            auto dprops = at::cuda::getCurrentDeviceProperties();
            int sm_count = dprops->multiProcessorCount == 64 ? 20 : dprops->multiProcessorCount;

            printf("blockM:%d, blockN:%d, threads:%d, params.num_splits:%d, block_size:%d\n",
                    Kernel_traits::kBlockM, Kernel_traits::kBlockN, Kernel_traits::kNThreads, params.num_splits, params.page_block_size);
            printf("Is_causal:%d, ngroups:%d\n", is_causal, params.ngroups);
            printf("CrossCut:%d, USE_MMA_M8:%d, kStages:%d\n", Kernel_traits::CrossCut, Kernel_traits::USE_MMA_M8, Kernel_traits::kStages);
            printf("kNWarps:%d, AtomLayoutQ:%d, AtomLayoutP:%d\n", Kernel_traits::kNWarps, Kernel_traits::AtomLayoutQ, Kernel_traits::AtomLayoutP);
            printf("Is_Q_in_regs:%d, Share_Q_K_smem:%d\n", Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem);
            printf("seq[%d, %d], grid_n[%d, %d, %d]\n",
                    params.seqlen_q, params.seqlen_k, num_m_block, params.h, params.num_sm_parts);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                    float(num_m_block * params.h * params.num_sm_parts) / float(sm_count * ctas_per_sm));
        }
    }

}

template<typename Kernel_traits, bool CrossCut = false>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    //FLASH_ASSERT(params.page_block_size == Kernel_traits::kBlockN);
    //constexpr size_t smem_size = Kernel_traits::kSmemSize;
    constexpr size_t smem_size = Kernel_traits::kSmemSizeAccum;
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    // FLASH_ASSERT(params.page_block_size % Kernel_traits::kBlockN == 0);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, CrossCut>;
        //CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        printf_show_log<Kernel_traits>(reinterpret_cast<const void*>(kernel), params, smem_size, Is_causal);
#ifdef __HGGCCC__
        const void *flash_func = reinterpret_cast<const void*>(kernel);
        CUfunction func = static_cast<CUfunction>(NULL);
        cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        CUlaunchAttributeAD LaunchAttr = {CUAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        CUlaunchConfigAD LaunchCfg = {num_m_block, params.h, params.num_sm_parts, Kernel_traits::kNThreads, 1, 1, smem_size, stream, &LaunchAttr, 0};
        // LaunchAttr.value.schedPreference.blocksPerMultiprocessor = 1;//schedule.bits.tb_per_cu;
        // LaunchAttr.value.schedPreference.gridStepX = 2;
        // LaunchAttr.value.schedPreference.gridStepY = 2;
        // LaunchAttr.value.schedPreference.flags = 2;
        CUDA_DRIVER_CHECK(cuLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
#endif
    });
    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<Kernel_traits, kMaxSplits>;
#ifdef __HGGCCC__
        const void *flash_func = reinterpret_cast<const void*>(combine_kernel);
        CUfunction func = static_cast<CUfunction>(NULL);
        cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        CUlaunchAttributeAD LaunchAttr = {CUAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        CUlaunchConfigAD LaunchCfg = {grid_combine.x, grid_combine.y, grid_combine.z, 128, 1, 1, 0, stream, &LaunchAttr, 0};
        // LaunchAttr.value.schedPreference.blocksPerMultiprocessor = 1;//schedule.bits.tb_per_cu;
        // LaunchAttr.value.schedPreference.gridStepX = 2;
        // LaunchAttr.value.schedPreference.gridStepY = 2;
        // LaunchAttr.value.schedPreference.flags = 2;
        CUDA_DRIVER_CHECK(cuLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
#endif
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, int Headdim, int Headdim_V>
void run_mha_fwd_splithd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    // constexpr static int kBlockM = 64;  // Fixed for all head dimensions

    bool cross_cut = use_cross_cut(params.seqlen_q, params.b);

    bool warp_interleave = params.seqlen_q >=128 && params.page_block_size == 64;

    // temp to disable warp interleave for random issue.
    auto dprops = at::cuda::getCurrentDeviceProperties();
    if (std::string(dprops->name).find("610") != std::string::npos)
        warp_interleave = false;

// #if ACOMPUTE_VERSION==10000
    if (!is_sm89_or_newer()) {
        if (cross_cut) {
            // support seqlen_q > 16.
            // constexpr static int kBlockN= 32;
            if (params.seqlen_q <= 32) {
                constexpr static int kBlockM = 32;
                constexpr static int kBlockN= 32;
                constexpr bool USE_MMA_M8 = 1;
                constexpr int kNwarps = 8;
                constexpr int AtomLayoutQ = 4;
                constexpr int AtomLayoutP = 1;
                run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                    Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                    Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                    >, 1/*CrossCut*/>(params, stream);
            } else if (params.seqlen_q <= 48) {
                constexpr static int kBlockM = 48;
                constexpr static int kBlockN= 64;
                constexpr bool USE_MMA_M8 = 0;
                constexpr int kNwarps = 12;
                constexpr int AtomLayoutQ = 3;
                constexpr int AtomLayoutP = 3;
                run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                    Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                    Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                    >, 1/*CrossCut*/>(params, stream);
            } else if (params.seqlen_q <= 64) {
                constexpr static int kBlockM = 64;
                constexpr static int kBlockN= 64;
                constexpr bool USE_MMA_M8 = 0;
                constexpr int kNwarps = 16;
                constexpr int AtomLayoutQ = 4;
                constexpr int AtomLayoutP = 1;
                run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                    Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                    Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                    >, 1/*CrossCut*/>(params, stream);
            } else if (params.seqlen_q > 64) {
                if (warp_interleave) {
                    run_flash_splitkv_mla_kernel<T, 80>(params, stream);
                    return;
                }
                constexpr static int kBlockM = 128;
                constexpr static int kBlockN= 32;
                constexpr bool USE_MMA_M8 = 0;
                constexpr int kNwarps = 16;
                constexpr int AtomLayoutQ = 8;
                constexpr int AtomLayoutP = 4;
                run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                    Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                    Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                    >, 1/*CrossCut*/>(params, stream);
            }
        } else {
            constexpr bool USE_MMA_M8 = 1;
            constexpr static int kBlockN = 16;
            SEQLENG_SWITCH(params.seqlen_q, [&] {
                constexpr int kNwarps = USE_MMA_M8 ? kBlockM / 8 : kBlockM / 16;
                run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                    Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
                    T, Headdim_V, 0/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/>>(params, stream);
            });

        }
    } else {
// #else
        if (warp_interleave  && params.num_blocks > 40) {
            // cta size small than sms do not use warp interleave since memory efficiency not good on 890.
            run_flash_splitkv_mla_kernel<T, 89>(params, stream);
            return;
        }
        FLASH_ASSERT(cross_cut);
        constexpr bool USE_MMA_M8 = 0;
        constexpr static int kBlockN = 64;
        SEQLENG_SWITCH(params.seqlen_q, [&] {
            constexpr int AtomLayoutQ = kBlockM / 16;
            constexpr int kNwarps = AtomLayoutQ * (kBlockN / 16);
            constexpr int kStages = kBlockM <= 16 ? 3 : 2;
            constexpr int AtomLayoutP = 1;
            run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
                T, Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP, kStages
                >, 1/*CrossCut*/>(params, stream);
        });
    }
// #endif
    return;

}

////
template<typename Kernel_traits>
void run_flash_sparse_prefill_fwd(SparsePrefillParams &params) {
    // TODO.
    constexpr size_t smem_size = Kernel_traits::kSmemSize + Kernel_traits::kBlockN * 2 * sizeof(bool);
    const int num_m_block = params.s_q*cute::ceil_div(params.h_q, Kernel_traits::kBlockM);

        auto kernel = &flash::flash_sparse_prefill_fwd_kernel<Kernel_traits>;
        //CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
#if 0 //def __HGGCCC__
       //TODO
        const void *flash_func = reinterpret_cast<const void*>(kernel);
        CUfunction func = static_cast<CUfunction>(NULL);
        cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        CUlaunchAttributeAD LaunchAttr = {CUAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        CUlaunchConfigAD LaunchCfg = {num_m_block, 1, 1, Kernel_traits::kNThreads, 1, 1, smem_size, params.stream, &LaunchAttr, 0};
        CUDA_DRIVER_CHECK(cuLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        kernel<<<dim3(num_m_block, 1, 1), Kernel_traits::kNThreads, smem_size, params.stream>>>(params);
#endif
    CHECK_CUDA_KERNEL_LAUNCH();

}

template<typename T>
void run_sparse_prefill_fwd_dispatch(SparsePrefillParams& params) {
    constexpr int B_H = 64; // kBlockM
    constexpr int B_TOPK = 64;    // kBlockM
    // constexpr int NUM_THREADS = 128*4; // 16*32
    // static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)

    FLASH_ASSERT(params.h_kv == 1);
    FLASH_ASSERT(params.topk % (2*B_TOPK) == 0);   // To save some boundry checkings
    FLASH_ASSERT(params.topk > 0);
    FLASH_ASSERT(params.h_q % B_H == 0);
    run_flash_sparse_prefill_fwd<Flash_fwd_kernel_traits<
        576/*Headdim*/, 64/*kBlockM*/, 64/*kBlockN*/, 16/*kNwarps*/,
        0/*Is_Q_in_regs*/, 0/*Share_Q_K_smem*/, T, 512/*Headdim_V*/,
        1/*CrossCut*/, 0/*USE_MMA_M8*/, 4/*AtomLayoutQ*/, 1/*AtomLayoutP*/
        >>(params);
}

////
template<typename Kernel_traits, bool IsFP8>
void run_flash_sparse_decode_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // TODO.
    constexpr size_t smem_size = Kernel_traits::kSmemSizeAccum + Kernel_traits::kBlockN * 2 * sizeof(bool);
    const int num_m_block = (params.seqlen_q / params.ngroups) * cute::ceil_div(params.ngroups, Kernel_traits::kBlockM);

        auto kernel = &flash::flash_sparse_decode_fwd_kernel<Kernel_traits, IsFP8>;
        printf_show_log<Kernel_traits>(reinterpret_cast<const void*>(kernel), params, smem_size, false, true, IsFP8);
        //CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
#ifdef __HGGCCC__
       //TODO
        const void *flash_func = reinterpret_cast<const void*>(kernel);
        CUfunction func = static_cast<CUfunction>(NULL);
        cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        CUlaunchAttributeAD LaunchAttr = {CUAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        CUlaunchConfigAD LaunchCfg = {num_m_block, params.h,
        params.num_sm_parts, Kernel_traits::kNThreads, 1, 1, smem_size, stream, &LaunchAttr, 0};
        CUDA_DRIVER_CHECK(cuLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
#endif
    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<Kernel_traits, kMaxSplits>;
#ifdef __HGGCCC__
        const void *flash_func = reinterpret_cast<const void*>(combine_kernel);
        CUfunction func = static_cast<CUfunction>(NULL);
        cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        CUlaunchAttributeAD LaunchAttr = {CUAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        CUlaunchConfigAD LaunchCfg = {grid_combine.x, grid_combine.y, grid_combine.z, 128, 1, 1, 0, stream, &LaunchAttr, 0};
        // LaunchAttr.value.schedPreference.blocksPerMultiprocessor = 1;//schedule.bits.tb_per_cu;
        // LaunchAttr.value.schedPreference.gridStepX = 2;
        // LaunchAttr.value.schedPreference.gridStepY = 2;
        // LaunchAttr.value.schedPreference.flags = 2;
        CUDA_DRIVER_CHECK(cuLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
#endif
    });
    CHECK_CUDA_KERNEL_LAUNCH();

}

template<typename T, bool IsFP8>
void run_sparse_decode_fwd_dispatch(Flash_fwd_params& params, cudaStream_t stream) {
    constexpr int TOPK_BLOCK_SIZE = 64;    // kBlockN
    // constexpr int NUM_THREADS = 128*4; // 16*32
    // static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)
    FLASH_ASSERT(params.h == 1);
    FLASH_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);

    constexpr bool USE_MMA_M8 = 0;
    constexpr static int kBlockN = 64;
    // SEQLENG_SWITCH(params.seqlen_q, [&] {
        constexpr int kBlockM = 64;
        constexpr int AtomLayoutQ = kBlockM / 16;
        constexpr int kNwarps = AtomLayoutQ * (kBlockN / 16);
        constexpr int AtomLayoutP = 1;
        run_flash_sparse_decode_fwd<Flash_fwd_kernel_traits<
            576, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
            T, 512, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
            >, IsFP8>(params, stream);
    // });

}