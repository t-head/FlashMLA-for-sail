/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"
#include "flash_sparse_fwd_kernel.h"
#include "flash_splitkv/config.h" // for splitkv kernel
#include "flash_splitkv/splitkv_mla.h"
#include "flash_splitkv/splitkv_dsa.h"

#include <hggc_ad.h>
#include "utils.h"


template<typename Kernel_traits>
void printf_show_log(const void* kernel, Flash_fwd_params &params, const size_t smem_size,
                     bool is_causal, bool is_sparse = false, bool is_fp8 = false) {
    char *pEnv_params = std::getenv("show_log");
    int num_m_block;
    if (pEnv_params && isdigit(*pEnv_params)) {
        int value = std::stoi(std::string(pEnv_params));
        if (value > 0) {
            int ctas_per_sm;
            hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
            if (is_sparse) {
                num_m_block = (params.seqlen_q / params.ngroups) * cute::ceil_div(params.ngroups, Kernel_traits::kBlockM);
                printf("[run_flash_sparse_decode_fwd_]: FP8 KVCache:%d\n", is_fp8);
            } else {
                num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
                printf("[run_flash_splitkv_fwd_]:\n");
            }
            printf("smem_size = %d, CTAs per SM = %d, ", int(smem_size), ctas_per_sm);

            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, kernel);
            int sm_count = get_num_sm(get_current_device());
            if (sm_count == 64) sm_count = 20;
            printf("HeadDim:%d, HeadDimV:%d\n",Kernel_traits::kHeadDim, Kernel_traits::kHeadDimV);
            printf("blockM:%d, blockN:%d, threads:%d, params.num_splits:%d, block_size:%d\n",
                    Kernel_traits::kBlockM, Kernel_traits::kBlockN, Kernel_traits::kNThreads, params.num_splits, params.page_block_size);
            printf("CrossCut:%d, USE_MMA_M8:%d, kStages:%d, kBlockNPagedPerAiuLoad:%d\n",
                    Kernel_traits::CrossCut, Kernel_traits::USE_MMA_M8, Kernel_traits::kStages, Kernel_traits::kBlockNPagedPerAiuLoad);
            printf("kNWarps:%d, AtomLayoutQ:%d, AtomLayoutP:%d, kNWarps0:%d\n",
                    Kernel_traits::kNWarps, Kernel_traits::AtomLayoutQ, Kernel_traits::AtomLayoutP, Kernel_traits::kNWarps0);
            printf("Is_Q_in_regs:%d, Share_Q_K_smem:%d\n", Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem);
            printf("seq[%d, %d], grid_n[%d, %d, %d]\n",
                    params.seqlen_q, params.seqlen_k, num_m_block, params.h, params.num_sm_parts);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                    float(num_m_block * params.h * params.num_sm_parts) / float(sm_count * ctas_per_sm));
        }
    }

}

template<typename Kernel_traits>
void printf_prefill_show_log(const void* kernel, SparsePrefillParams &params, const size_t smem_size) {
    char *pEnv_params = std::getenv("show_log");
    int num_m_block;
    if (pEnv_params && isdigit(*pEnv_params)) {
        int value = std::stoi(std::string(pEnv_params));
        if (value > 0) {
            int ctas_per_sm;
            hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);

            num_m_block = params.s_q * cute::ceil_div(params.h_q, Kernel_traits::kBlockM);
            printf("[run_flash_sparse_prefill_fwd_]:\n");
            printf("smem_size = %d, CTAs per SM = %d,", int(smem_size), ctas_per_sm);
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, kernel);
            int sm_count = get_num_sm(get_current_device());
            if (sm_count == 64) sm_count = 20;
            printf("HeadDim:%d, HeadDimV:%d, blockM:%d, blockN:%d\n",
                    Kernel_traits::kHeadDim, Kernel_traits::kHeadDimV, Kernel_traits::kBlockM, Kernel_traits::kBlockN);
            printf("kNThreads:%d, CrossCut:%d, USE_MMA_M8:%d, kStages:%d\n",
                    Kernel_traits::kNThreads, Kernel_traits::CrossCut, Kernel_traits::USE_MMA_M8, Kernel_traits::kStages);
            printf("kNWarps:%d, AtomLayoutQ:%d, AtomLayoutP:%d, kNWarps0:%d\n",
                    Kernel_traits::kNWarps, Kernel_traits::AtomLayoutQ, Kernel_traits::AtomLayoutP, Kernel_traits::kNWarps0);
            printf("Is_Q_in_regs:%d, Share_Q_K_smem:%d, seq[%d, %d], grid[%d]\n",
                    Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem, params.s_q, params.s_kv, num_m_block);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f\n",
                    int(attr.numRegs), int(attr.localSizeBytes), sm_count, float(num_m_block) / float(sm_count * ctas_per_sm));
        }
    }
}

template<typename Kernel_traits, bool CrossCut = false>
void run_flash_splitkv_fwd(Flash_fwd_params &params, hggcStream_t stream) {
    //constexpr size_t smem_size = Kernel_traits::kSmemSize;
    constexpr size_t smem_size = Kernel_traits::kSmemSizeAccum;
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    // FLASH_ASSERT(params.page_block_size % Kernel_traits::kBlockN == 0);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, CrossCut>;
        if (smem_size >= 48 * 1024) {
            hggcFuncSetAttribute(
                kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        printf_show_log<Kernel_traits>(reinterpret_cast<const void*>(kernel), params, smem_size, Is_causal);
#ifdef __HGGCCC__
        const void *flash_func = reinterpret_cast<const void*>(kernel);
        HGfunction func = static_cast<HGfunction>(NULL);
        hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        HGlaunchAttributeAD LaunchAttr = {HGAD_LAUNCH_ATTRIBUTE_IGNORE};
        HGlaunchConfigAD LaunchCfg = {num_m_block, params.h, params.num_sm_parts, Kernel_traits::kNThreads, 1, 1, smem_size, stream, &LaunchAttr, 0};
        CUDA_DRIVER_CHECK(hgLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
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
        HGfunction func = static_cast<HGfunction>(NULL);
        hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        HGlaunchAttributeAD LaunchAttr = {HGAD_LAUNCH_ATTRIBUTE_IGNORE};
        HGlaunchConfigAD LaunchCfg = {grid_combine.x, grid_combine.y, grid_combine.z, 128, 1, 1, 0, stream, &LaunchAttr, 0};
        CUDA_DRIVER_CHECK(hgLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
#endif
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, int Headdim, int Headdim_V>
void run_mha_fwd_splithd_splitkv_dispatch(Flash_fwd_params &params, hggcStream_t stream) {
    FLASH_ASSERT(params.page_block_size % 16 == 0);
    bool cross_cut = use_cross_cut(params.seqlen_q, params.b);

    // mtp3/5 tp4/8 seq_m is 160 or 192, blockM 256 not good.
    bool warp_interleave = ((params.seqlen_q % 128 == 0) || (params.seqlen_q == 96) || (params.seqlen_q == 80) || (params.seqlen_q > 256)) && params.page_block_size == 64;
    // temp to disable warp interleave for random issue.
#ifdef __HGGCCC__
    // Under hgcc, check compute capability instead of device name
    {
        auto [cap_major, cap_minor] = get_compute_capability(get_current_device());
        // "610" device corresponds to older architecture
        if (cap_major < 8 || (cap_major == 8 && cap_minor < 9))
            warp_interleave = false;
    }
#else
    {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        if (std::string(dprops->name).find("610") != std::string::npos)
            warp_interleave = false;
    }
#endif

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
                BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThankBlockN, [&] {
                    constexpr int kBlockNPagedPerAiuLoad = PageLargerThankBlockN ? kBlockN : 16;
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                        Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                        Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP, kBlockNPagedPerAiuLoad
                        >, 1/*CrossCut*/>(params, stream);
                });
            } else if (params.seqlen_q <= 48) {
                constexpr static int kBlockM = 48;
                constexpr static int kBlockN= 64;
                constexpr bool USE_MMA_M8 = 0;
                constexpr int kNwarps = 12;
                constexpr int AtomLayoutQ = 3;
                constexpr int AtomLayoutP = 3;
                BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThankBlockN, [&] {
                    constexpr int kBlockNPagedPerAiuLoad = PageLargerThankBlockN ? kBlockN : 16;
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                        Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                        Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP, kBlockNPagedPerAiuLoad
                        >, 1/*CrossCut*/>(params, stream);
                });
            } else if (params.seqlen_q <= 64) {
                constexpr static int kBlockM = 64;
                constexpr static int kBlockN= 64;
                constexpr bool USE_MMA_M8 = 0;
                constexpr int kNwarps = 16;
                constexpr int AtomLayoutQ = 4;
                constexpr int AtomLayoutP = 1;
                BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThankBlockN, [&] {
                    constexpr int kBlockNPagedPerAiuLoad = PageLargerThankBlockN ? kBlockN : 16;
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                        Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                        Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP, kBlockNPagedPerAiuLoad
                        >, 1/*CrossCut*/>(params, stream);
                });
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
                BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThankBlockN, [&] {
                    constexpr int kBlockNPagedPerAiuLoad = PageLargerThankBlockN ? kBlockN : 16;
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                        Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/, T,
                        Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP, kBlockNPagedPerAiuLoad
                        >, 1/*CrossCut*/>(params, stream);
                });
            }
        } else {
            constexpr bool USE_MMA_M8 = 1;
            constexpr static int kBlockN = 16;  // == kBlockNPagedPerAiuLoad
            SEQLENG_SWITCH_ALIGN(params.seqlen_q, [&] {
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
            BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThankBlockN, [&] {
                constexpr int kBlockNPagedPerAiuLoad = PageLargerThankBlockN ? kBlockN : 16;
                constexpr int AtomLayoutQ = kBlockM / 16;
                constexpr int kNwarps = AtomLayoutQ * (kBlockN / 16);
                constexpr int kStages = kBlockM <= 16 ? 3 : 2;
                constexpr int AtomLayoutP = kBlockM == 48 ? 3 : 1;
                run_flash_splitkv_fwd<Flash_fwd_kernel_traits<
                    Headdim, kBlockM, kBlockN, kNwarps, USE_MMA_M8/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
                    T, Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP,
                    kBlockNPagedPerAiuLoad, kStages
                    >, 1/*CrossCut*/>(params, stream);
            });
        });
    }
    return;
}

////
template<typename Kernel_traits, bool HAVE_TOPK_LENGTH>
void run_flash_sparse_prefill_fwd(SparsePrefillParams &params) {
    // TODO.
    constexpr size_t smem_size = Kernel_traits::kSmemSize + Kernel_traits::kBlockN * 2 * sizeof(bool);
    const int num_m_block = params.s_q*cute::ceil_div(params.h_q, Kernel_traits::kBlockM);

    auto kernel = &flash::flash_sparse_prefill_fwd_kernel<Kernel_traits, HAVE_TOPK_LENGTH>;
    printf_prefill_show_log<Kernel_traits>(reinterpret_cast<const void*>(kernel), params, smem_size);
    CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    if (smem_size >= 48 * 1024) {
        hggcFuncSetAttribute(
            kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
#ifdef __HGGCCC__
       //TODO
        const void *flash_func = reinterpret_cast<const void*>(kernel);
        HGfunction func = static_cast<HGfunction>(NULL);
        hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        HGlaunchAttributeAD LaunchAttr = {HGAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        HGlaunchConfigAD LaunchCfg = {num_m_block, 1, 1, Kernel_traits::kNThreads, 1, 1, smem_size, params.stream, &LaunchAttr, 0};
        CUDA_DRIVER_CHECK(hgLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        kernel<<<dim3(num_m_block, 1, 1), Kernel_traits::kNThreads, smem_size, params.stream>>>(params);
#endif
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T>
void run_sparse_prefill_fwd_dispatch(SparsePrefillParams& params) {
    // constexpr int B_H = 64; // kBlockM
    constexpr int B_TOPK = 64;    // kBlockM
    // constexpr int NUM_THREADS = 128*4; // 16*32
    // static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)

    FLASH_ASSERT(params.h_kv == 1);
    FLASH_ASSERT(params.topk % (2*B_TOPK) == 0);   // To save some boundry checkings
    FLASH_ASSERT(params.topk > 0);
    // FLASH_ASSERT(params.h_q % B_H == 0);

    bool warp_interleave = ((params.s_q % 128 == 0) || (params.s_q > 256)) && (params.h_q == 128)
        && (params.s_kv >= params.topk);
    if (warp_interleave) {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            if (!is_sm89_or_newer()) {
                run_flash_sparse_prefill_fwd_wg<cutlass::bfloat16_t, 80, HEAD_DIM_QK>(params);
            } else {
                run_flash_sparse_prefill_fwd_wg<cutlass::bfloat16_t, 89, HEAD_DIM_QK>(params);
            }
        });
        return;
    }

    constexpr bool USE_MMA_M8 = 0;
    constexpr static int kBlockN = 64;
    SEQLENG_SWITCH_ALIGN(params.h_q, [&] {
        // constexpr static int kBlockM = 64;
        constexpr int AtomLayoutQ = kBlockM / 16;
        constexpr int AtomLayoutP = 1;
        constexpr int kNwarps0 = AtomLayoutQ * (kBlockN / 16);
        constexpr int kNwarps = 16;
        constexpr bool Is_Q_in_regs = 0; //(kBlockM == 16) ? 1 : 0;

        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            DISPATCH_BOOLEAN_FLAG(params.topk_length != nullptr, HAVE_TOPK_LENGTH, [&]() {
                run_flash_sparse_prefill_fwd<Flash_fwd_kernel_traits<
                    HEAD_DIM_QK/*Headdim*/, kBlockM, kBlockN, kNwarps,
                    Is_Q_in_regs/*Is_Q_in_regs*/, Is_Q_in_regs/*Share_Q_K_smem*/, T, 512/*Headdim_V*/,
                    1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP,
                    kBlockN/*kBlockNPagedPerAiuLoad*/, 2/*kStages*/, kNwarps0
                    >, HAVE_TOPK_LENGTH>(params);
            });
        });
    });
}

////
template<typename Kernel_traits, bool IsFP8>
void run_flash_sparse_decode_fwd(Flash_fwd_params &params, hggcStream_t stream) {
    // TODO.
    constexpr size_t smem_size = Kernel_traits::kSmemSizeAccum + Kernel_traits::kBlockN * 2 * sizeof(int);
    const int num_m_block = (params.seqlen_q / params.ngroups) * cute::ceil_div(params.ngroups, Kernel_traits::kBlockM);

        auto kernel = &flash::flash_sparse_decode_fwd_kernel<Kernel_traits, IsFP8>;
        printf_show_log<Kernel_traits>(reinterpret_cast<const void*>(kernel), params, smem_size, false, true, IsFP8);
        //CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        if (smem_size >= 48 * 1024) {
            hggcFuncSetAttribute(
                kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
#ifdef __HGGCCC__
       //TODO
        const void *flash_func = reinterpret_cast<const void*>(kernel);
        HGfunction func = static_cast<HGfunction>(NULL);
        hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        HGlaunchAttributeAD LaunchAttr = {HGAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        HGlaunchConfigAD LaunchCfg = {num_m_block, params.h,
        params.num_sm_parts, Kernel_traits::kNThreads, 1, 1, smem_size, stream, &LaunchAttr, 0};
        CUDA_DRIVER_CHECK(hgLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
#endif
    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<Kernel_traits, kMaxSplits>;
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
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, bool IsFP8, int Headdim, int Headdim_V>
void run_sparse_decode_fwd_dispatch(Flash_fwd_params& params, hggcStream_t stream) {
    constexpr int TOPK_BLOCK_SIZE = 64;    // kBlockN
    // constexpr int NUM_THREADS = 128*4; // 16*32
    // static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)
    FLASH_ASSERT(params.h == 1);
    FLASH_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);

    if constexpr (IsFP8) {
        constexpr bool USE_MMA_M8 = 0;
        constexpr bool KeepQ = true;
        constexpr static int kBlockN = 64;
        SEQLENG_SWITCH_ALIGN(params.seqlen_q, [&] {
            IS_PAGE_POWER2(params.page_block_size, params.extra_page_block_size, [&] {
                constexpr int AtomLayoutQ = kBlockM / 16;
                constexpr int kNwarps0 = AtomLayoutQ * (kBlockN / 16);
                constexpr int kNwarps = 16;
                constexpr int AtomLayoutP = kBlockM == 64 ? 2 : 1; // to save regs(sum/max in softmax)
                run_flash_sparse_decode_fwd<Flash_fwd_kernel_traits<
                    Headdim, kBlockM, kBlockN, kNwarps, KeepQ/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
                    T, Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP,
                    kBlockN/*kBlockNPagedPerAiuLoad*/, 2/*kStages*/, kNwarps0, kPagePow2
                    >, IsFP8>(params, stream);
            });
        });
    } else {
        constexpr bool USE_MMA_M8 = 0;
        constexpr bool KeepQ = true;
        constexpr static int kBlockN = 64;
        SEQLENG_SWITCH_ALIGN(params.seqlen_q, [&] {
            constexpr int AtomLayoutQ = kBlockM / 16;
            constexpr int kNwarps = AtomLayoutQ * (kBlockN / 16);
            constexpr int AtomLayoutP = kBlockM == 64 ? 2 : 1;
            run_flash_sparse_decode_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, KeepQ/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
                T, Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                >, IsFP8>(params, stream);
        });
    }
}