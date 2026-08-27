/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cute/util/debug.hpp>

#include "params.h"
#include "kernel_traits.h"
#include "utils.h"
#include "kerutils/device/ppu/softmax.cuh"
#include "kerutils/device/ppu/mask.cuh"
#include "kerutils/device/ppu/dequant.cuh"

#ifndef DSA_SIM_AIU
#define DSA_SIM_AIU 1
#endif

using namespace cute;

namespace flash {

template<typename Kernel_traits, bool HAVE_TOPK_LENGTH, bool Is_causal = false, bool CrossCut = true>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_sparse_prefill_fwd_kernel(__grid_constant__ const SparsePrefillParams params) {

    static_assert(CrossCut);
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kNWarps0 = Kernel_traits::kNWarps0;
    constexpr int AtomLayoutQ = Kernel_traits::AtomLayoutQ;
    constexpr int AtomLayoutP = Kernel_traits::AtomLayoutP;
    constexpr bool USE_MMA_M8 = Kernel_traits::USE_MMA_M8;
    constexpr int kBlockKSmemV = Kernel_traits::kBlockKSmemV; // 64
    constexpr int MMA_ATOM_K_M = Kernel_traits::USE_MMA_M8 ? 1 : 2;
    constexpr int MMA_ATOM_M = USE_MMA_M8 ? 8 : 16;

#if DSA_SIM_AIU
    using KVCacheGmem = KVCacheGmemBf16SimAIU<Element, kBlockN, Kernel_traits::kNThreads, kHeadDim>;
    using SmemLayoutKSim = typename KVCacheGmem::SmemLayoutKSim;
#else
    using KVCacheGmem = KVCacheGmemBf16<Element, kBlockN, Kernel_traits::kNThreads, kHeadDim>;
#endif
    using SmemLayoutKNoAiu = typename KVCacheGmem::SmemLayoutK;
    using GmemTiledCopyKNoAiu = typename KVCacheGmem::GmemTiledCopy;
    using SmemLayoutVtNoAiu = typename KVCacheGmem::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename KVCacheGmem::SmemLayoutVtransposedNoSwizzle;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    const int m_block = blockIdx.x % cute::ceil_div(params.h_q, kBlockM);
    const index_t s_q_idx = blockIdx.x / cute::ceil_div(params.h_q, kBlockM);
    const int lane_idx = tidx % 32;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    const int load_col_idx = tidx/8; // 0~64
    // col = col_x * 16 + col_y * 4 + col_z; -> col_in_indices = col_z * 16 + col_x * 4 + col_y;
    // (col_x, col_y, col_z) = (col_load / 16, (col_load % 16) / 4, col_load % 4) ->
#if ACOMPUTE_VERSION ==10000
    const int col_in_indices = (lane_idx/8) * 16 + warp_idx;
    // const int col_in_indices = (load_col_idx % 4) * 16 + (load_col_idx / 16) * 4 + (load_col_idx % 16) / 4;
    // const int col_in_indices = load_col_idx;
#else
    const int col_in_indices = load_col_idx;
#endif

    int* gIndices_ptr = params.indices + s_q_idx * params.stride_indices_s_q + load_col_idx;   // [topk]
    int nxt_token_idx = __ldg(gIndices_ptr);

    // const int n_block_max = params.topk / kBlockN;

    const int topk_length = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int n_block_max = HAVE_TOPK_LENGTH ? cute::ceil_div(topk_length, (int)kBlockN) : (int)((unsigned int)params.topk/(unsigned int)kBlockN);

    const index_t row_offset_q = s_q_idx * params.stride_q_s_q + m_block * (kBlockM * params.stride_q_h_q);
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.stride_q_h_q, _1{}));

    // Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q)),
    //                         make_shape(params.h_q, params.d_qk, params.s_q),
    //                         make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q));
    // Tensor gQ = local_tile(make_mix_tensor_like(mQ(_, _, s_q_idx)), Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    // TODO: copy of gK, ignore s_q
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.kv)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.stride_kv_s_kv, _1{})); // may not used

    // smem
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)), typename Kernel_traits::SmemLayoutQ{});

    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), SmemLayoutKNoAiu{});
#if DSA_SIM_AIU
    Tensor sKSim = make_tensor(sK.data(), SmemLayoutKSim{});
#endif
    Tensor sVt = make_tensor(sK.data(), SmemLayoutVtNoAiu{});
    Tensor sVtNoSwizzle = make_tensor(sK.data(), SmemLayoutVtNoSwizzle{}); // only for layout

    Tensor sK_double = make_tensor(sK.data() + size(sK), SmemLayoutKNoAiu{});
#if DSA_SIM_AIU
    Tensor sKSim_double = make_tensor(sK_double.data(), SmemLayoutKSim{});
#endif
    Tensor sVt_double = make_tensor(sK_double.data(), SmemLayoutVtNoAiu{});

    Tensor sP = make_tensor(sK_double.data() + size(sK_double), typename Kernel_traits::SmemLayoutP{});
    Tensor smem_row_scale = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum*>((sP.data() + size(sP)).get())),
        Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor smem_row_via_warp = make_tensor(smem_row_scale.data() + size(smem_row_scale),
        Shape<Int<kBlockM>, Int<kNWarps0/AtomLayoutQ>>{}, Stride<Int<kNWarps0/AtomLayoutQ>, _1>{});
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(
        reinterpret_cast<bool*>((smem_row_via_warp.data() + ((kNWarps0==AtomLayoutQ) ? 0: size(smem_row_via_warp))).get())),
        Shape<_2, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{});


    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;
    GmemTiledCopyKNoAiu gmem_tiled_copy_K;

    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
#if DSA_SIM_AIU && (ACOMPUTE_VERSION ==10000)
    int cross_tid_h = (tidx & 0xFFFFFFF8) >> 3;
    int cross_tid_l = tidx & 0x7;
    int cross_bias = (cross_tid_l / 2 == 1) ? 2 : ((cross_tid_l / 2 == 2) ? 1 : cross_tid_l / 2);
    cross_tid_h = (cross_tid_h & 0xFFFFFFFC) | (((cross_tid_h & 0x3) + cross_bias) & 0x3);
    int sim_cross_tid = (cross_tid_h << 3) | cross_tid_l;
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(sim_cross_tid);
#else
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);
#endif

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_K.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
#if DSA_SIM_AIU
    Tensor tKsK = gmem_thr_copy_K.partition_D(sKSim);
    Tensor tKsK_double = gmem_thr_copy_K.partition_D(sKSim_double);
#else
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
    Tensor tKsK_double = gmem_thr_copy_K.partition_D(sK_double);
#endif

    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMma tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrP  = thr_mma_o.partition_fragment_A(sP);                           // (MMA,MMA_M,MMA_N)
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, kBlockM, params.stride_q_h_q, kBlockM, Kernel_traits::kBlockKSmem, 0};
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, kBlockM, params.d_qk, params.stride_q_h_q);
#endif
    const int tid_thread_slice = warp_idx * 32;
#else
    const int tid_thread_slice = tidx;
#endif

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid_thread_slice);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    copy<false, true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                             params.h_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // KV not use AIU copy
#if DSA_SIM_AIU
    auto smem_tiled_copy_K = make_tiled_copy_B(typename KVCacheGmem::SmemCopyAtomK{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(warp_idx * 32);
    auto tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK));
    auto tSsK_double = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK_double));

    auto smem_tiled_copy_V = make_tiled_copy_B(typename KVCacheGmem::SmemCopyAtomV{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(warp_idx * 32);
    auto tOsVt = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt));
    auto tOsVt_double = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt_double));
#else
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    auto tSsK_double = smem_thr_copy_K.partition_S(sK_double);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_V.partition_S(sVt);
    auto tOsVt_double = smem_thr_copy_V.partition_S(sVt_double);
#endif

    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_Q.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));

    auto smem_tiled_copy_S = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomS{}, tiled_mma_s);
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tidx);
    Tensor tSsS = smem_thr_copy_S.partition_D(sP);

    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomP{}, tiled_mma_o);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tidx);
    Tensor tOsP = smem_thr_copy_P.partition_S(sP);

    int n_block = 0;

    // use kv_block_num to decide number.
    int kv_store_num = 0;
    int kv_load_num = 0;

    // int* gIndices = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;   // [topk]
    Element *gK_base = reinterpret_cast<Element *>(params.kv);

    // row is threadIdx.x/8. error tKgK = gK_base + (threadIdx.x/8) * 576 +  8*(threadIdx.x%8)
    // real row is gIndices[threadIdx.x/8]
    // real tKgK= gK_base + indice_idx * 576 +  72*(threadIdx.x%8)
    int indice_idx = nxt_token_idx;

    bool is_token_valid = indice_idx >= 0 && indice_idx < params.s_kv;

    if constexpr (HAVE_TOPK_LENGTH) {
        is_token_valid &= (load_col_idx < topk_length);
    }

    tKgK.data() = gK_base + indice_idx * (index_t)params.stride_kv_s_kv + (tidx%8)*8;
    gmem_tiled_copy_K.pred = is_token_valid;
    cute::copy(gmem_tiled_copy_K, tKgK, tKsK);

    if (n_block < n_block_max - 1) {
        nxt_token_idx = __ldg(gIndices_ptr + (n_block + 1)* kBlockN);
    }

    smem_valid_indices(kv_store_num%2, col_in_indices) = is_token_valid;

    kv_store_num++;
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::SoftmaxBetweenWarps<USE_MMA_M8, kBlockM, AtomLayoutQ, AtomLayoutP, kNWarps0/AtomLayoutQ, 1/*ForceUseTsm*/> softmax;

    // Q is loop-invariant across n_block; keep the leading QK k-steps of the Q A-operand
    // in registers so that subsequent QK GEMMs reuse them instead of reloading from smem.
    constexpr int kKeepQQkSteps = 24;
    static_assert(kKeepQQkSteps + 1 <= decltype(size<2>(tSrQ))::value, "kKeepQQkSteps exceeds QK k-step count");

    // These are the iterations where we don't need masking on S
    for (int n_block = 0; n_block < n_block_max; ++n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block < n_block_max -1) {
            // int row = load_col_idx + (n_block+1) * kBlockN;
            auto tKsK_current = kv_store_num % 2 == 0 ? tKsK : tKsK_double;
            int indice_idx = nxt_token_idx;
            bool is_token_valid = indice_idx >= 0 && indice_idx < params.s_kv;
            if constexpr (HAVE_TOPK_LENGTH) {
                is_token_valid &= ((load_col_idx + (n_block + 1) * kBlockN) < topk_length);
            }
            tKgK.data() = gK_base + indice_idx * (index_t)params.stride_kv_s_kv + (tidx%8)*8;

            gmem_tiled_copy_K.pred = is_token_valid;

            cute::copy(gmem_tiled_copy_K, tKgK, tKsK_current);

            smem_valid_indices(kv_store_num%2, col_in_indices) = is_token_valid;
            if (n_block < n_block_max - 2) {
                nxt_token_idx = __ldg(gIndices_ptr + (n_block + 2)* kBlockN);
            }
            cute::cp_async_fence();
            kv_store_num++;
        }

        // determine use kv buffer 0 or 1
        auto tSsK_current = kv_load_num % 2 == 0 ? tSsK : tSsK_double;
        auto tOsVt_current = kv_load_num % 2 == 0 ? tOsVt : tOsVt_double;

        if (warp_idx < kNWarps0) {
            if constexpr (!Kernel_traits::Share_Q_K_smem) {
                // Q and K occupy disjoint smem here, so Q is still intact at n_block == 0;
                // stage its leading QK k-steps into registers once and reuse them below.
                if (n_block == 0) {
                    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
                    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
                    #pragma unroll
                    for (int i = 0; i < kKeepQQkSteps; ++i) {
                        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, i), tSrQ_copy_view(_, _, i));
                    }
                }
                flash::gemm_rss<kKeepQQkSteps>(
                    acc_s, tSrQ, tSrK, tSsQ, tSsK_current, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K
                );
            } else {
                // Q/K alias the same smem (Share_Q_K_smem): sQ has already been overwritten by
                // the first K cp_async before the loop, and Q was staged to registers in the
                // prologue, so keep the original QK GEMM path instead of re-reading the stale sQ.
                flash::gemm<Kernel_traits::Is_Q_in_regs>(
                    acc_s, tSrQ, tSrK, tSsQ, tSsK_current, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K
                );
            }

            constexpr int MMA_N_S = kBlockN / decltype(typename Kernel_traits::TiledMmaS{}.template tile_size_mnk<1>())::value;
            flash::apply_indices_mask(acc_s, smem_valid_indices, (warp_idx / AtomLayoutQ) * MMA_N_S, kv_load_num % 2);

            n_block == 0
                ? softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.sm_scale_div_log2)
                : softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.sm_scale_div_log2);

            Tensor rS = flash::convert_type<Element>(acc_s);
            Tensor tSaS = smem_thr_copy_S.retile_S(rS);
            cute::copy(smem_tiled_copy_S, tSaS, tSsS);
        }
        __syncthreads();
        if (n_block > 0) {
            softmax.template softmax_rescale_o(acc_o, smem_row_scale);
        }
        flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt_current,
            tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V,
            smem_thr_copy_P, smem_thr_copy_V);
        kv_load_num++;
    }

    if (warp_idx < kNWarps0) {
        softmax.template normalize_softmax_lse_per_warp<false, /*UseTbSync=*/false>(smem_row_via_warp, smem_row_scale, params.sm_scale, params.attn_sink, params.sm_scale_div_log2, m_block);
    }
    __syncthreads();

    softmax.template softmax_rescale_o(acc_o, smem_row_scale);
    // store
    Tensor lse = make_tensor_like<ElementAccum>(softmax.lse);
    Tensor mlogits = make_tensor_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        lse(mi) = softmax.lse(mi) * float(M_LOG2E);
        mlogits(mi) = softmax.row_max(mi) *  params.sm_scale_div_log2;
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);

    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_o);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = s_q_idx * (params.h_q * kHeadDimV) + m_block * (kBlockM * kHeadDimV);
    const index_t row_offset_lse = s_q_idx * params.h_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.out) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(kHeadDimV, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.lse) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gMLogits = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.max_logits) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)

    const int row_lse_base = warp_idx % AtomLayoutQ * MMA_ATOM_M + (tidx % 32) / 4;
    const int warp_stride = MMA_ATOM_M * AtomLayoutQ;
    if (warp_idx < Kernel_traits::kNWarps0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = row_lse_base + (mi / MMA_ATOM_K_M) * warp_stride + (mi % MMA_ATOM_K_M) *8;
            if (row < params.h_q - m_block * kBlockM) {
                gLSE(row) = lse(mi) * CUDART_LN2_F;
                gMLogits(row) = topk_length == 0 ? -INFINITY : mlogits(mi) * CUDART_LN2_F;
            }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    // if (!Is_even_K) {
    //     #pragma unroll
    //     for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_v; }
    // }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    if constexpr (HAVE_TOPK_LENGTH) {
        // //  If h_q % kBlockM == 0, use the method as follow:
        // copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true, /*Clear_OOB_K=*/true>(
        //     gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, topk_length > 0 ? params.h_q - m_block * kBlockM : 0
        // );
        // If h_q % kBlockM != 0, use the method as follow:
        if (topk_length > 0) {
            copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, params.h_q - m_block * kBlockM
            );
        } else {
            #pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                if (get<0>(tOcO(0, m, 0)) < params.h_q - m_block * kBlockM) {
                    cute::clear(tOgO(_, m, _));
                }
            }
        }
    } else {
        copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, params.h_q - m_block * kBlockM
        );
    }

}

} // namespace flash

#ifdef __HGGCCC__
#include <hggc_ad.h>
#endif

template<typename Kernel_traits, bool HAVE_TOPK_LENGTH>
void run_flash_sparse_prefill_fwd(SparsePrefillParams &params) {
    // TODO.
    constexpr size_t smem_size = Kernel_traits::kSmemSize + Kernel_traits::kBlockN * 2 * sizeof(bool);
    const int num_m_block = params.s_q*cute::ceil_div(params.h_q, Kernel_traits::kBlockM);

    auto kernel = &flash::flash_sparse_prefill_fwd_kernel<Kernel_traits, HAVE_TOPK_LENGTH>;
    flash::printf_prefill_show_log<Kernel_traits>(reinterpret_cast<const void*>(kernel), params, smem_size);
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
