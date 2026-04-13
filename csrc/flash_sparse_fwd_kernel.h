/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dequant.h"
#include "flash_fwd_kernel.h"

#include <cute/util/debug.hpp>

namespace flash {
using namespace cute;

#define DSA_SIM_AIU 1
template<typename Kernel_traits, bool Is_causal = false, bool CrossCut = true>
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
    constexpr int AtomLayoutQ = Kernel_traits::AtomLayoutQ;
    constexpr int AtomLayoutP = Kernel_traits::AtomLayoutP;
    constexpr bool USE_MMA_M8 = Kernel_traits::USE_MMA_M8;
    constexpr int kBlockKSmemV = Kernel_traits::kBlockKSmemV; // 64
    constexpr int MMA_ATOM_K_M = Kernel_traits::USE_MMA_M8 ? 1 : 2;
    constexpr int MMA_ATOM_M = USE_MMA_M8 ? 8 : 16;

#if DSA_SIM_AIU
    using KVCacheGmem = KVCacheGmemBf16SimAIU<Element, kBlockN, Kernel_traits::kNThreads>;
    using SmemLayoutKSim = typename KVCacheGmem::SmemLayoutKSim;
#else
    using KVCacheGmem = KVCacheGmemBf16<Element, kBlockN, Kernel_traits::kNThreads>;
#endif
    using SmemLayoutKNoAiu = typename KVCacheGmem::SmemLayoutK;
    using GmemTiledCopyKNoAiu = typename KVCacheGmem::GmemTiledCopy;
    using SmemLayoutVtNoAiu = typename KVCacheGmem::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename KVCacheGmem::SmemLayoutVtransposedNoSwizzle;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    const int m_block = blockIdx.x % (params.h_q/kBlockM);
    const index_t s_q_idx = blockIdx.x / (params.h_q/kBlockM);
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

    const int n_block_max = params.topk / kBlockN;

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
        Shape<Int<kBlockM>, Int<kNWarps/AtomLayoutQ>>{}, Stride<Int<kNWarps/AtomLayoutQ>, _1>{});
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(
        reinterpret_cast<bool*>((smem_row_via_warp.data() + ((kNWarps==AtomLayoutQ) ? 0: size(smem_row_via_warp))).get())),
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

    flash::copy<true, true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
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

    flash::SoftmaxBetweenWarps<USE_MMA_M8, kBlockM, AtomLayoutQ, AtomLayoutP, kNWarps/AtomLayoutQ> softmax;

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

        flash::gemm<Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK_current, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        constexpr int MMA_N_S = kBlockN / decltype(typename Kernel_traits::TiledMmaS{}.template tile_size_mnk<1>())::value;
        flash::apply_indices_mask(acc_s, smem_valid_indices, (tidx / 32 / AtomLayoutQ) * MMA_N_S, kv_load_num % 2);

        n_block == 0
            ? softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.sm_scale_div_log2)
            : softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.sm_scale_div_log2);

        Tensor rS = flash::convert_type<Element>(acc_s);
        Tensor tSaS = smem_thr_copy_S.retile_S(rS);
        cute::copy(smem_tiled_copy_S, tSaS, tSsS);
        __syncthreads();
        if (n_block > 0) {
            softmax.template softmax_rescale_o(acc_o, smem_row_scale);
        }
        flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt_current,
            tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V,
            smem_thr_copy_P, smem_thr_copy_V);
        kv_load_num++;
    }
    softmax.template normalize_softmax_lse_per_warp<false>(smem_row_via_warp, smem_row_scale, params.sm_scale);
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
    const index_t row_offset_lse = s_q_idx * params.h_q  + m_block * kBlockM;
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

    const int row_lse_base = (tidx / 32) % AtomLayoutQ * MMA_ATOM_M + (tidx % 32) / 4;
    const int warp_stride = MMA_ATOM_M * AtomLayoutQ;
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = row_lse_base + (mi / MMA_ATOM_K_M) * warp_stride + (mi % MMA_ATOM_K_M) *8;
        if (row < params.h_q - m_block * kBlockM) {
            gLSE(row) = lse(mi);
            gMLogits(row) = mlogits(mi);
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
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, params.h_q - m_block * kBlockM
    );

}

#define CVT_OPT 1
template<typename Kernel_traits, typename Params>
__forceinline__ __device__ void compute_attn_fp8_sparse_splitkv(
    const Params &params, const int batch_id, const int bidh, const int m_block,
    const int n_split_idx, const int n_block_min, int n_block_max, const bool NoSplit) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int AtomLayoutQ = Kernel_traits::AtomLayoutQ;
    constexpr int AtomLayoutP = Kernel_traits::AtomLayoutP;
    constexpr bool USE_MMA_M8 = Kernel_traits::USE_MMA_M8;
    constexpr int MMA_ATOM_M = USE_MMA_M8 ? 8 : 16;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int h_k_idx = m_block % cute::ceil_div(params.ngroups, kBlockM); // s_q = s_q_ori * h_q
    const int s_q_idx = m_block / cute::ceil_div(params.ngroups, kBlockM);
    const int row_base = h_k_idx * kBlockM + s_q_idx * params.ngroups;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
#if CVT_OPT
    using KVCacheGmem = KVCacheG2SFP8<kBlockN, Kernel_traits::kNThreads>;
#else
    using KVCacheGmem = KVCacheGmemFP8<kBlockN, Kernel_traits::kNThreads>;
#endif
    using ElementKVCache = typename KVCacheGmem::ElementKVCache;
    using SmemLayoutKNoAiu = typename KVCacheGmem::SmemLayoutK;
    using SmemLayoutVtransposedNoAiu = typename KVCacheGmem::SmemLayoutVtransposed;

    if (row_base >= params.seqlen_q) return;
    // never has n_block_min >= n_block_max in tile scheduler mode
    assert(n_block_min < n_block_max);

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const int row_offset_q = batch_id * params.q_batch_stride + bidh * params.q_head_stride + row_base * params.q_row_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));

    // Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKVCache *>(params.k_ptr)),
    //                         Shape<Int<kBlockN>, Int<kHeadDim>>{},
    //                         make_stride(params.k_row_stride, _1{}));;

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), SmemLayoutKNoAiu{});
    Tensor sVt = make_tensor(sK.data(), SmemLayoutVtransposedNoAiu{});
    Tensor sVtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // double shared memory for k/v cache.
    Tensor sK_double = make_tensor(sK.data() + size(sK), SmemLayoutKNoAiu{});
    Tensor sVt_double = make_tensor(sK_double.data(), SmemLayoutVtransposedNoAiu{});

    Tensor sP = make_tensor(sK_double.data() + size(sK_double), typename Kernel_traits::SmemLayoutP{});

    Tensor smem_row_scale = make_tensor(make_smem_ptr(reinterpret_cast<float *>((sP.data() + size(sP)).get())),
        Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor smem_row_via_warp = make_tensor(smem_row_scale.data() + size(smem_row_scale),
        Shape<Int<kBlockM>, Int<kNWarps/AtomLayoutQ>>{}, Stride<Int<kNWarps/AtomLayoutQ>, _1>{});
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(
        reinterpret_cast<bool*>((smem_row_via_warp.data() + ((kNWarps==AtomLayoutQ) ? 0: size(smem_row_via_warp))).get())),
        Shape<_2, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{});

    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;

    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMma tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrP  = thr_mma_o.partition_fragment_A(sP);                           // (MMA,MMA_M,MMA_N)
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    // Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, kBlockM, params.q_row_stride, kBlockM, Kernel_traits::kBlockKSmem, 0};
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, kBlockM, params.d, params.q_row_stride);
#endif
    const int tid_thread_slice = warp_idx * 32;
#else
    const int tid_thread_slice = tidx;
#endif

    // auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    // auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid_thread_slice);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));

    // PREDICATES
    //
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    flash::copy<true, true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                params.ngroups - h_k_idx * kBlockM);

    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    auto smem_tiled_copy_S = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomS{}, tiled_mma_s);
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tidx);
    Tensor tSsS = smem_thr_copy_S.partition_D(sP);

    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomP{}, tiled_mma_o);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tidx);
    Tensor tOsP = smem_thr_copy_P.partition_S(sP);

    // KV not use AIU copy
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    auto tSsK_double = smem_thr_copy_K.partition_S(sK_double);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_V.partition_S(sVt);
    auto tOsVt_double = smem_thr_copy_V.partition_S(sVt_double);


    int n_block = n_block_min;
    // use kv_block_num to decide number.
    int kv_store_num = 0;
    int kv_load_num = 0;

    int* gIndices = params.indices_ptr + batch_id * params.indices_batch_stride + s_q_idx * params.indices_row_stride; // (topk) : (1)
    ElementKVCache *gK_base = reinterpret_cast<ElementKVCache *>(params.k_ptr)
          + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    KVCacheGmem kvload_gmem(tidx, gIndices, gK_base, params.page_block_size,
        params.k_batch_stride, params.k_row_stride);

#if CVT_OPT
    kvload_gmem.template load_g2s_async(n_block, kv_store_num, smem_valid_indices, sK);
#else
    kvload_gmem.template load_from_gmem(n_block, kv_store_num, smem_valid_indices, sK);
#endif
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

    flash::SoftmaxBetweenWarps<USE_MMA_M8, kBlockM, AtomLayoutQ, AtomLayoutP, kNWarps/AtomLayoutQ> softmax;

    // #pragma unroll
    for (; n_block < n_block_max; ++n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
#if CVT_OPT
        auto sK_cvt = kv_store_num % 2 == 1 ? sK : sK_double;
        auto sK_load = kv_store_num % 2 == 0 ? sK : sK_double;
        auto Scale = kv_store_num % 2 == 1 ? kvload_gmem.tKrScale(_, 0) : kvload_gmem.tKrScale(_, 1);
        if (n_block < n_block_max - 1) {
            kvload_gmem.template load_g2s_async(n_block + 1, kv_store_num, smem_valid_indices, sK_load);
            cute::cp_async_fence();
            kv_store_num++;
            flash::cp_async_wait<1>();
        } else {
            flash::cp_async_wait<0>();
        }

        __syncthreads();
        kvload_gmem.template cvt_fp8_store(sK_cvt, sK_cvt, Scale);
        __syncthreads();
#else
        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block < n_block_max -1) { // doble buffer for next part
            auto sK_current = kv_store_num % 2 == 0 ? sK : sK_double;
            kvload_gmem.template load_from_gmem(n_block+1, kv_store_num, smem_valid_indices, sK_current);
            // if constexpr(IsFP8) {
            //     kvload_gmem.template load_from_gmem(n_block+1, kv_store_num, smem_valid_indices, sK_current);
            // } else {
            //     kvload_gmem.template load_from_gmem(n_block+1, kv_store_num, smem_valid_indices, sK_current);
            // }
            cute::cp_async_fence();
            kv_store_num++;
        }
#endif

        // determine use kv buffer 0 or 1
        auto tSsK_current = kv_load_num % 2 == 0 ? tSsK : tSsK_double;
        auto tOsVt_current = kv_load_num % 2 == 0 ? tOsVt : tOsVt_double;

        flash::gemm<Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK_current, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.z ==0) {
    //     printf("smem_valid_indices, blockIdx.x:%d, blockIdx.z:%d, bidb:%d, n_block:%d, s_q_idx:%d, h_k_idx: %d, kv_load_num:%d\n",
    //     bidb, blockIdx.x, blockIdx.z, n_block, s_q_idx, h_k_idx, kv_load_num);
    //     print_tensor(smem_valid_indices);
    //     printf("\n");
    // }
        constexpr int MMA_N_S = kBlockN / decltype(typename Kernel_traits::TiledMmaS{}.template tile_size_mnk<1>())::value;
        const int warpN_idx = (warp_idx / AtomLayoutQ) * MMA_N_S;
        flash::apply_indices_mask(acc_s, smem_valid_indices, warpN_idx, kv_load_num % 2);

        n_block == n_block_min
            ? softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2)
            : softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2);

        Tensor rS = flash::convert_type<Element>(acc_s);
        Tensor tSaS = smem_thr_copy_S.retile_S(rS);     // ((Atom,AtomNum), MMA_N, MMA_N)

        cute::copy(smem_tiled_copy_S, tSaS, tSsS);
        __syncthreads();

        if (n_block > n_block_min) {
            softmax.template softmax_rescale_o(acc_o, smem_row_scale);
        }
        flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt_current, tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V,
            smem_thr_copy_P, smem_thr_copy_V);
        kv_load_num++;
    }

    if (NoSplit) {
        softmax.template normalize_softmax_lse_per_warp<false>(smem_row_via_warp, smem_row_scale, params.scale_softmax);
    } else {
        softmax.template normalize_softmax_lse_per_warp<true>(smem_row_via_warp, smem_row_scale, params.scale_softmax);
    }

    __syncthreads();

    softmax.template softmax_rescale_o(acc_o, smem_row_scale);
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.z ==0) {
    //     printf("acc_o:");print_tensor(acc_o);printf("\n");
    // }

    // Epilogue
    if (NoSplit) {
        flash::store<Kernel_traits, false, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    } else {
        flash::store<Kernel_traits, true, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }
}

template<typename Kernel_traits, typename Params>
__forceinline__ __device__ void compute_attn_bf16_sparse_splitkv(
    const Params &params, const int batch_id, const int bidh, const int m_block,
    const int n_split_idx, const int n_block_min, int n_block_max, const bool NoSplit) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int AtomLayoutQ = Kernel_traits::AtomLayoutQ;
    constexpr int AtomLayoutP = Kernel_traits::AtomLayoutP;
    constexpr bool USE_MMA_M8 = Kernel_traits::USE_MMA_M8;
    constexpr int MMA_ATOM_M = USE_MMA_M8 ? 8 : 16;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int lane_idx = tidx % 32;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int h_k_idx = m_block % cute::ceil_div(params.ngroups, kBlockM); // s_q = s_q_ori * h_q
    const int s_q_idx = m_block / cute::ceil_div(params.ngroups, kBlockM);
    const int row_base = h_k_idx * kBlockM + s_q_idx * params.ngroups;

#if DSA_SIM_AIU
    using KVCacheGmem = KVCacheGmemBf16SimAIU<Element, kBlockN, Kernel_traits::kNThreads>;
    using SmemLayoutKSim = typename KVCacheGmem::SmemLayoutKSim;
#else
    using KVCacheGmem = KVCacheGmemBf16<Element, kBlockN, Kernel_traits::kNThreads>;
#endif
    using SmemLayoutKNoAiu = typename KVCacheGmem::SmemLayoutK;
    using GmemTiledCopyKNoAiu = typename KVCacheGmem::GmemTiledCopy;
    using SmemLayoutVtNoAiu = typename KVCacheGmem::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename KVCacheGmem::SmemLayoutVtransposedNoSwizzle;
    constexpr int RowsPerGmem = Kernel_traits::kNThreads / KVCacheGmem::kGmemThreadsPerRow;
    static_assert(RowsPerGmem % 16 == 0 && kBlockN % RowsPerGmem == 0);
    // 4warps: 16; 8warps: 32; 16 warps: 64
    constexpr int indices_per_load = kBlockN / RowsPerGmem;
    // kBlockN64: 1; kBlockN32: 2; kBlockN16: 16


    if (row_base >= params.seqlen_q) return;
    // never has n_block_min >= n_block_max in tile scheduler mode
    assert(n_block_min < n_block_max);

    const int load_col_idx = tidx/8; // + v * RowsPerGmem; // tidx/8,  0~64
    // One warps:[0 1 2 3], [5 6 7 8], ...., -> [0 16 32 48], [1 17 33 49]
    // col = col_x * 16 + col_y * 4 + col_z; -> col_in_indices = col_z * 16 + col_x * 4 + col_y;
    // (col_x, col_y, col_z) = (col_load / 16, (col_load % 16) / 4, col_load % 4)
    // -> (warp_idx/4 + v * RowsPerGmem/16, warp_idx%4, lane_idx/8)
    // convert to: col_in_indices = lane_idx/8 * 16 + warp_idx + v * RowsPerGmem/4
#if ACOMPUTE_VERSION ==10000
    // const int col_in_indices = ((load_col_idx)% 8)*8 + (load_col_idx)/8;
    const int col_in_indices = (lane_idx/8) * 16 + warp_idx; // + v * RowsPerGmem/4
    // const int col_in_indices1 = (load_col_idx % 4) * 16 + (load_col_idx / 16) * 4 + (load_col_idx % 16) / 4;
    // const int col_in_indices = load_col_idx;
#else
    const int col_in_indices = load_col_idx; // + v * RowsPerGmem;
#endif
    int* gIndices_ptr = params.indices_ptr + batch_id * params.indices_batch_stride
                      + s_q_idx * params.indices_row_stride + load_col_idx; // (topk) : (1)

    // #define LOAD_TOKEN_INDEX(block_idx) __ldg((gIndices_ptr + (block_idx) * kBlockN))
    // int nxt_token_idx; // = LOAD_TOKEN_INDEX(n_block_min);
    
    int nxt_token_idx1[indices_per_load];
    
    auto token_idx_update = [&](const int block_idx_load){
        // nxt_token_idx = __ldg((gIndices_ptr + (block_idx_load) * kBlockN));
        #pragma unroll
        for (int v = 0; v < indices_per_load; v++) {
            nxt_token_idx1[v] = __ldg((gIndices_ptr
                // + v * (Kernel_traits::kNThreads / KVCacheGmem::kGmemThreadsPerRow)
                + v * RowsPerGmem
                + (block_idx_load) * kBlockN));
        }
    };
    
    token_idx_update(n_block_min);


    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const int row_offset_q = batch_id * params.q_batch_stride + bidh * params.q_head_stride + row_base * params.q_row_stride;
    // q = q.view({batch_size, seqlen_q_ori, 1, ngroups, head_size}).transpose(2, 3)
    //     .reshape({batch_size,      seqlen_q_ori * ngroups, 1,             head_size});
    //               q_batch_stride,  q_row_stride,           q_head_stride, 1

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), SmemLayoutKNoAiu{});
#if DSA_SIM_AIU
    Tensor sKSim = make_tensor(sK.data(), SmemLayoutKSim{});
#endif
    Tensor sVt = make_tensor(sK.data(), SmemLayoutVtNoAiu{});
    Tensor sVtNoSwizzle = make_tensor(sK.data(), SmemLayoutVtNoSwizzle{});

    // double shared memory for k/v cache.
    Tensor sK_double = make_tensor(sK.data() + size(sK), SmemLayoutKNoAiu{});
#if DSA_SIM_AIU
    Tensor sKSim_double = make_tensor(sK_double.data(), SmemLayoutKSim{});
#endif
    Tensor sVt_double = make_tensor(sK_double.data(), SmemLayoutVtNoAiu{});

    Tensor sP = make_tensor(sK_double.data() + size(sK_double), typename Kernel_traits::SmemLayoutP{});

    Tensor smem_row_scale = make_tensor(make_smem_ptr(reinterpret_cast<float *>((sP.data() + size(sP)).get())),
        Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor smem_row_via_warp = make_tensor(smem_row_scale.data() + size(smem_row_scale),
        Shape<Int<kBlockM>, Int<kNWarps/AtomLayoutQ>>{}, Stride<Int<kNWarps/AtomLayoutQ>, _1>{});
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(
        reinterpret_cast<bool*>((smem_row_via_warp.data() + ((kNWarps==AtomLayoutQ) ? 0: size(smem_row_via_warp))).get())),
        Shape<_2, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{});

    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;

    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMma tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrP  = thr_mma_o.partition_fragment_A(sP);                           // (MMA,MMA_M,MMA_N)
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    // Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, kBlockM, params.q_row_stride, kBlockM, Kernel_traits::kBlockKSmem, 0};
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, kBlockM, params.d, params.q_row_stride);
#endif
    const int tid_thread_slice = warp_idx * 32;
#else
    const int tid_thread_slice = tidx;
#endif

    // auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    // auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid_thread_slice);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));

    // PREDICATES
    //
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    flash::copy<true, true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                params.ngroups - h_k_idx * kBlockM);

    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    auto smem_tiled_copy_S = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomS{}, tiled_mma_s);
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tidx);
    Tensor tSsS = smem_thr_copy_S.partition_D(sP);

    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomP{}, tiled_mma_o);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tidx);
    Tensor tOsP = smem_thr_copy_P.partition_S(sP);

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

    GmemTiledCopyKNoAiu gmem_tiled_copy_K;
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
    Tensor tKgK = gmem_thr_copy_K.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
#if DSA_SIM_AIU
    Tensor tKsK = gmem_thr_copy_K.partition_D(sKSim);
    Tensor tKsK_double = gmem_thr_copy_K.partition_D(sKSim_double);
#else
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
    Tensor tKsK_double = gmem_thr_copy_K.partition_D(sK_double);
#endif

    int n_block = n_block_min;
    // use kv_block_num to decide number.
    int kv_store_num = 0;
    int kv_load_num = 0;

    Element *gK_base = reinterpret_cast<Element *>(params.k_ptr)
          + (bidh / params.h_h_k_ratio) * params.k_head_stride;

    static_assert(indices_per_load == size<1>(tKgK));
    auto KV_load = [&](){
        // auto tKsK_current = kv_store_num % 2 == 0 ? tKsK : tKsK_double;
        #pragma unroll
        for (int v = 0; v < indices_per_load; v++) {
            auto tKgK_current = tKgK(_, v, _);
            auto tKsK_current = kv_store_num % 2 == 0 ? tKsK(_, v, _) : tKsK_double(_, v, _);
            int token_index = nxt_token_idx1[v];
            bool is_token_valid = token_index >= 0;
            int block_index = token_index/params.page_block_size;
            int rel_idx_in_block = (token_index+params.page_block_size) % params.page_block_size;
            tKgK_current.data() = gK_base + (int64_t) block_index * params.k_batch_stride
                        + rel_idx_in_block * params.k_row_stride + (tidx%8)*8;
            gmem_tiled_copy_K.pred = is_token_valid;
            cute::copy(gmem_tiled_copy_K, tKgK_current, tKsK_current);
#if ACOMPUTE_VERSION ==10000
            smem_valid_indices(kv_store_num%2, col_in_indices + v * RowsPerGmem / 4) = is_token_valid;                
# else
            smem_valid_indices(kv_store_num%2, col_in_indices + v * RowsPerGmem) = is_token_valid;                
#endif
        }
        cute::cp_async_fence();
        kv_store_num++;
    };

    KV_load();

    if (n_block < n_block_max - 1) {
        // nxt_token_idx = LOAD_TOKEN_INDEX(n_block+1);
        token_idx_update(n_block+1);
    }

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::SoftmaxBetweenWarps<USE_MMA_M8, kBlockM, AtomLayoutQ, AtomLayoutP, kNWarps/AtomLayoutQ> softmax;

    for (; n_block < n_block_max; ++n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block < n_block_max -1) { // doble buffer for next part
            KV_load();
            if (n_block < n_block_max - 2) {
                // nxt_token_idx = LOAD_TOKEN_INDEX(n_block+2);
                token_idx_update(n_block+2);
            }
        }

        // determine use kv buffer 0 or 1
        auto tSsK_current = kv_load_num % 2 == 0 ? tSsK : tSsK_double;
        auto tOsVt_current = kv_load_num % 2 == 0 ? tOsVt : tOsVt_double;

        flash::gemm<Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK_current, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        constexpr int MMA_N_S = kBlockN / decltype(typename Kernel_traits::TiledMmaS{}.template tile_size_mnk<1>())::value;
        const int warpN_idx = (warp_idx / AtomLayoutQ) * MMA_N_S;

        flash::apply_indices_mask(acc_s, smem_valid_indices, warpN_idx, kv_load_num % 2);

        n_block == n_block_min
            ? softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2)
            : softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2);

        Tensor rS = flash::convert_type<Element>(acc_s);
        Tensor tSaS = smem_thr_copy_S.retile_S(rS);     // ((Atom,AtomNum), MMA_N, MMA_N)

        cute::copy(smem_tiled_copy_S, tSaS, tSsS);
        __syncthreads();

        if (n_block > n_block_min) {
            softmax.template softmax_rescale_o(acc_o, smem_row_scale);
        }
        flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt_current, tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V,
            smem_thr_copy_P, smem_thr_copy_V);
        kv_load_num++;
    }

    if (NoSplit) {
        softmax.template normalize_softmax_lse_per_warp<false>(smem_row_via_warp, smem_row_scale, params.scale_softmax);
    } else {
        softmax.template normalize_softmax_lse_per_warp<true>(smem_row_via_warp, smem_row_scale, params.scale_softmax);
    }

    __syncthreads();

    softmax.template softmax_rescale_o(acc_o, smem_row_scale);

    // Epilogue
    if (NoSplit) {
        flash::store<Kernel_traits, false, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    } else {
        flash::store<Kernel_traits, true, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }
}

template<typename Kernel_traits, bool IsFP8 = false>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_sparse_decode_fwd_kernel(__grid_constant__ const Flash_fwd_params params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    static_assert(Kernel_traits::CrossCut);
    static_assert(!Kernel_traits::USE_MMA_M8);

    //extern __shared__ char shared_memory[];
    //auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y; // sched_begin_block_idx*kBlockN
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;
    if (begin_idx >= params.b) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

#pragma unroll 1
// #pragma clang loop licm(disable)
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = params.topk;
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == cute::ceil_div(seqlen_k, kBlockN);
        if (batch_id > begin_idx) {
            __syncthreads();  // Barrier between two tiles.
        }
        if constexpr (IsFP8) {
            compute_attn_fp8_sparse_splitkv<Kernel_traits>(
                params, batch_id, bidh, m_block, n_split_idx,
                n_block_min, n_block_max, NoSplit);

        } else {
            compute_attn_bf16_sparse_splitkv<Kernel_traits>(
                params, batch_id, bidh, m_block, n_split_idx,
                n_block_min, n_block_max, NoSplit);
        }
    }
}

} // namespace flash
