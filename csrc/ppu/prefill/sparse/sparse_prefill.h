/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include "params.h"
#include "kerutils/host/host.h"

#include "prefill/sparse/sparse_prefill_wg.cuh"
#include "prefill/sparse/sparse_prefill_std.cuh"

template<typename T>
void run_sparse_prefill_fwd_dispatch_impl(SparsePrefillParams& params) {
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

    // TP16 production uses eight query heads per rank.  The generic aligned
    // sequence switch rounds that shape up to a 16-row MMA tile, doing twice
    // the useful Q/O work.  Keep this specialization local to sparse prefill:
    // dense and decode have independent dispatch contracts.
    if (params.h_q == 8) {
        constexpr int kBlockM = 8;
        constexpr int kBlockN = 64;
        constexpr int AtomLayoutQ = 1;
        constexpr int AtomLayoutP = 1;
        constexpr int kNwarps0 = AtomLayoutQ * (kBlockN / 16);
        constexpr int kNwarps = 16;
        // PPU10's M8 MMA path uses the dedicated Q TSM load and must retain Q
        // in registers before the double-buffered KV load reuses its SMEM.
        constexpr bool Is_Q_in_regs = true;
        constexpr bool Share_Q_K_smem = true;
        constexpr bool USE_MMA_M8 = true;

        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            DISPATCH_BOOLEAN_FLAG(params.topk_length != nullptr, HAVE_TOPK_LENGTH, [&]() {
                run_flash_sparse_prefill_fwd<Flash_fwd_kernel_traits<
                    HEAD_DIM_QK/*Headdim*/, kBlockM, kBlockN, kNwarps,
                    Is_Q_in_regs/*Is_Q_in_regs*/, Share_Q_K_smem/*Share_Q_K_smem*/, T, 512/*Headdim_V*/,
                    1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP,
                    kBlockN/*kBlockNPagedPerAiuLoad*/, 2/*kStages*/, kNwarps0
                    >, HAVE_TOPK_LENGTH>(params);
            });
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

template<typename T>
void run_sparse_prefill_fwd_dispatch(SparsePrefillParams& params) {
    run_sparse_prefill_fwd_dispatch_impl<T>(params);
}
