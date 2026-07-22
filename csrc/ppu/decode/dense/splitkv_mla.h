/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <c10/cuda/CUDAException.h>
#include "kerutils/common/static_switch.h"
#include "params.h"
#include "kerutils/host/host.h"
#include "api/common.h"
#include "kerutils/device/block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "kerutils/device/ppu/softmax.cuh"
#include "kerutils/device/ppu/mask.cuh"
#include "ppuxx/decode/combine/combine.h"

#ifdef __HGGCCC__
#include <hggc_ad.h>
#endif

template<typename InputT, int Arch>
void run_flash_splitkv_mla_kernel(Flash_fwd_mla_params &params, hggcStream_t stream);

template<typename Kernel_traits, bool CrossCut = false>
void run_flash_splitkv_fwd(Flash_fwd_params &params, hggcStream_t stream);

template<typename T, int Headdim, int Headdim_V>
void run_mha_fwd_splithd_splitkv_dispatch(Flash_fwd_params &params, hggcStream_t stream) {
    FLASH_ASSERT(params.page_block_size % 16 == 0);
    bool cross_cut = use_cross_cut(params.seqlen_q, params.b);

    // mtp3/5 tp4/8 seq_m is 160 or 192, blockM 256 not good.
    bool warp_interleave = ((params.seqlen_q % 128 == 0) || (params.seqlen_q == 96) || (params.seqlen_q == 80) || (params.seqlen_q > 256)) && params.page_block_size == 64;
    // temp to disable warp interleave for random issue.
    {
        auto [cap_major, cap_minor] = get_compute_capability(get_current_device());
        if (cap_major < 8 || (cap_major == 8 && cap_minor < 9))
            warp_interleave = false;
    }

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
                BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThanBlockN, [&] {
                    constexpr int kBlockNPagedPerAiuLoad = PageLargerThanBlockN ? kBlockN : 16;
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
                BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThanBlockN, [&] {
                    constexpr int kBlockNPagedPerAiuLoad = PageLargerThanBlockN ? kBlockN : 16;
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
                BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThanBlockN, [&] {
                    constexpr int kBlockNPagedPerAiuLoad = PageLargerThanBlockN ? kBlockN : 16;
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
                BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThanBlockN, [&] {
                    constexpr int kBlockNPagedPerAiuLoad = PageLargerThanBlockN ? kBlockN : 16;
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
            BOOL_SWITCH(params.page_block_size >= kBlockN, PageLargerThanBlockN, [&] {
                constexpr int kBlockNPagedPerAiuLoad = PageLargerThanBlockN ? kBlockN : 16;
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

#include "splitkv_mla_kernel.cuh"
#include "splitkv_mla_fwd.cuh"
