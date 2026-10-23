/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <type_traits>

#include "params.h"
#include "kerutils/host/host.h"

#include "prefill/sparse/sparse_prefill_wg.cuh"
#include "prefill/sparse/sparse_prefill_std.cuh"

namespace flashmla::dsa::hs64 {

// Implemented alongside decode to share the M64N64 attention pipeline.
template<int HeadDim, int Arch>
void run_flash_sparse_prefill_fwd_hs64(SparsePrefillParams &params);

} // namespace flashmla::dsa::hs64

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

    // The double-warpgroup pipeline amortizes its setup on long topk;
    // retain the standard prefill kernel for short sparse rows.
    const bool use_hs64 = params.h_q % 128 == 64 && params.h_kv == 1 &&
        params.topk >= 512;
    const bool warp_interleave = ((params.s_q % 128 == 0) || (params.s_q > 256)) &&
        params.h_q == 128 && params.s_kv >= params.topk;

    if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
        if (use_hs64) {
            DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
                if (is_sm89_or_newer()) {
                    flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<HEAD_DIM_QK, 89>(params);
                } else {
                    flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<HEAD_DIM_QK, 80>(params);
                }
            });
            return;
        }
    }

    if (warp_interleave) {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            if (!is_sm89_or_newer()) {
                run_flash_sparse_prefill_fwd_wg<cutlass::bfloat16_t, 80, HEAD_DIM_QK>(params);
            } else {
                if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
                    if (params.topk == 128 && params.topk_length != nullptr) {
                        using ShortTraits = Flash_fwd_kernel_traits<
                            HEAD_DIM_QK, 64, 32, 8, false, false, T, 512, true, false,
                            4, 1, 32, 2, 8>;
                        run_flash_sparse_prefill_fwd<ShortTraits, true, true>(params);
                        return;
                    }
                }
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
        constexpr int kNwarps = 16;
        constexpr bool Is_Q_in_regs = 0; //(kBlockM == 16) ? 1 : 0;

        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            DISPATCH_BOOLEAN_FLAG(params.topk_length != nullptr, HAVE_TOPK_LENGTH, [&]() {
                // Small query batches retain full QK parallelism to minimize latency.
                DISPATCH_BOOLEAN_FLAG(params.s_q >= 32, REDUCE_QK_WARPS, [&]() {
                    constexpr bool reduce_qk_warps = REDUCE_QK_WARPS && kBlockM == 64 &&
                        std::is_same_v<T, cutlass::bfloat16_t>;
                    if constexpr (HAVE_TOPK_LENGTH && reduce_qk_warps && HEAD_DIM_QK == 512) {
                        if (params.h_q == 64 && params.topk == 128) {
                            const auto *props = at::cuda::getCurrentDeviceProperties();
                            constexpr int kMinCtaWaves = 8;
                            if ((props->major > 8 || (props->major == 8 && props->minor >= 9)) &&
                                params.s_q >= kMinCtaWaves * props->multiProcessorCount) {
                                using ShortTraits = Flash_fwd_kernel_traits<
                                    512, 64, 32, 8, false, false, T, 512, true, false, 4, 1, 32, 2, 8>;
                                run_flash_sparse_prefill_fwd<ShortTraits, true, true>(params);
                                return;
                            }
                        }
                    }
                    constexpr int kNwarps0 = AtomLayoutQ * (reduce_qk_warps ? 2 : kBlockN / 16);
                    run_flash_sparse_prefill_fwd<Flash_fwd_kernel_traits<
                        HEAD_DIM_QK/*Headdim*/, kBlockM, kBlockN, kNwarps,
                        Is_Q_in_regs/*Is_Q_in_regs*/, Is_Q_in_regs/*Share_Q_K_smem*/, T, 512/*Headdim_V*/,
                        1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP,
                        kBlockN/*kBlockNPagedPerAiuLoad*/, 2/*kStages*/, kNwarps0
                        >, HAVE_TOPK_LENGTH>(params);
                });
            });
        });
    });
}
