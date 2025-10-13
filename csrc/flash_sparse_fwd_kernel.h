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

#include <cute/util/debug.hpp>

namespace flash {
using namespace cute;

template <typename Tensor0, typename Tensor1>
__forceinline__ __device__ void apply_mask(Tensor0 &tensor_, Tensor1 &smem_valid_indices,  const int col_base, const int buffer) {
    Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
    const int lane_id = threadIdx.x % 32;
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
    const int col_idx_offset = col_base + (lane_id % 4);
#else
    const int col_idx_offset = col_base + (lane_id % 4) * 2;
#endif

    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
        const int col_idx_base = col_idx_offset + nj * 16;
#else
        const int col_idx_base = col_idx_offset + nj * 8;
#endif
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
            const int col_idx = col_idx_base + j * 4;
#else
            const int col_idx = col_idx_base + j;
#endif
            bool is_vaild = smem_valid_indices(buffer, col_idx);
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    if (!is_vaild) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
            }
        }
    }
}

template<typename Kernel_traits, bool Is_causal = false, bool CrossCut = true>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_sparse_fwd_kernel(__grid_constant__ const SparsePrefillParams params) {

    static_assert(CrossCut);
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;

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


    using SmemLayoutAtomKNoAiu = decltype(
        composition(Swizzle<Kernel_traits::kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<Kernel_traits::kBlockKSmem>>,
                           Stride<Int<Kernel_traits::kBlockKSmem>, _1>>{}));
   using SmemLayoutAtomVNoAiu = decltype(
        composition(Swizzle<Kernel_traits::kSwizzleV, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<Kernel_traits::kBlockKSmemV>>,
                           Stride<Int<Kernel_traits::kBlockKSmemV>, _1>>{}));

    using SmemLayoutKNoAiu = decltype(tile_to_shape(
        SmemLayoutAtomKNoAiu{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVNoAiu = decltype(tile_to_shape(
        SmemLayoutAtomVNoAiu{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposedNoAiu = decltype(
        composition(SmemLayoutVNoAiu{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    const int m_block = blockIdx.x % (params.h_q/kBlockM);
    const int s_q_idx = blockIdx.x / (params.h_q/kBlockM);
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    const int n_block_max = params.topk / kBlockN;

    const int row_offset_q = s_q_idx * params.stride_q_s_q + m_block * (kBlockM * params.stride_q_h_q);
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
    Tensor sVt = make_tensor(sK.data(), SmemLayoutVtransposedNoAiu{});
    Tensor sVtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{}); // only for layout

    Tensor sK_double = make_tensor(sK.data() + size(sK), SmemLayoutKNoAiu{});
    Tensor sVt_double = make_tensor(sK_double.data(), SmemLayoutVtransposedNoAiu{});

    Tensor sP = make_tensor(sK_double.data() + size(sK_double), typename Kernel_traits::SmemLayoutP{});
    Tensor smem_row_scale = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum*>((sP.data() + size(sP)).get())),
        Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor smem_row_via_warp = make_tensor(smem_row_scale.data() + size(smem_row_scale),
        Shape<Int<kBlockM>, Int<kNWarps/AtomLayoutQ>>{}, Stride<Int<kNWarps/AtomLayoutQ>, _1>{});
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(
        reinterpret_cast<bool*>((smem_row_via_warp.data() + ((kNWarps==AtomLayoutQ) ? 0: size(smem_row_via_warp))).get())),
        Shape<_2, Int<kBlockM>>{}, Stride<Int<kBlockM>, _1>{});


    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;
    typename Kernel_traits::GmemTiledCopyQK gmem_tiled_copy_K;

    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_K.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
    Tensor tKsK_double = gmem_thr_copy_K.partition_D(sK_double);

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
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    auto tSsK_double = smem_thr_copy_K.partition_S(sK_double);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_V.partition_S(sVt);
    auto tOsVt_double = smem_thr_copy_V.partition_S(sVt_double);

    // Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_Q.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
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

    int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;   // [topk]
    Element *gK_base = reinterpret_cast<Element *>(params.kv);

    // row is threadIdx.x/8. error tKgK = gK_base + (threadIdx.x/8) * 576 +  8*(threadIdx.x%8)
    // real row is gIndices[threadIdx.x/8]
    // real tKgK= gK_base + indice_idx * 576 +  72*(threadIdx.x%8)
    int indice_idx = __ldg(gIndices + tidx/8);
    bool is_token_valid = indice_idx >= 0 && indice_idx < params.s_kv;
    tKgK.data() = gK_base + indice_idx * (int64_t)params.stride_kv_s_kv + (tidx%8)*8;
    #pragma unroll
    for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = is_token_valid; }
    smem_valid_indices(kv_store_num%2, tidx/8) = is_token_valid;

    flash::copy<false/*Is_even_MN*/, false>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK,
                                            params.topk - n_block * kBlockN);
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

    flash::SoftmaxBetweenWarps<USE_MMA_M8, kBlockM, AtomLayoutQ, kNWarps/AtomLayoutQ> softmax;

    // These are the iterations where we don't need masking on S
    for (int n_block = 0; n_block < n_block_max; ++n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block < n_block_max -1) {
            int row = tidx/8 + (n_block+1) * kBlockN;
            int indice_idx = __ldg(gIndices + row);
            bool is_token_valid = indice_idx >= 0 && indice_idx < params.s_kv;
            tKgK.data() = gK_base + indice_idx * (int64_t)params.stride_kv_s_kv + (tidx%8)*8;
            smem_valid_indices(kv_store_num%2, tidx/8) = is_token_valid;
            #pragma unroll
            for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = is_token_valid;}
            // Advance gK

            auto tKsK_current = kv_store_num % 2 == 0 ? tKsK : tKsK_double;
            flash::copy</*Is_even_MN=*/true, false>(gmem_tiled_copy_K, tKgK, tKsK_current, tKcK, tKpK);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
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
        apply_mask(acc_s, smem_valid_indices, (tidx / 32 / AtomLayoutQ) * MMA_N_S * 16, kv_load_num % 2);

        n_block == 0
            ? softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.sm_scale_div_log2)
            : softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.sm_scale_div_log2);

        Tensor rS = flash::convert_type<Element>(acc_s);
        Tensor tSaS = smem_thr_copy_S.retile_S(rS);
        cute::copy(smem_tiled_copy_S, tSaS, tSsS);
        __syncthreads();
        if (n_block > 0) {
            softmax.template softmax_rescale_o<AtomLayoutP>(acc_o, smem_row_scale);
        }
        flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt_current,
            tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V,
            smem_thr_copy_P, smem_thr_copy_V);
        kv_load_num++;
    }
    softmax.template normalize_softmax_lse_per_warp<false>(smem_row_via_warp, smem_row_scale, params.sm_scale);
    __syncthreads();

    softmax.template softmax_rescale_o<AtomLayoutP>(acc_o, smem_row_scale);

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

    const int row_offset_o = s_q_idx * (params.h_q * kHeadDimV) + m_block * (kBlockM * kHeadDimV);
    const int row_offset_lse = s_q_idx * params.h_q  + m_block * kBlockM;
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

} // namespace flash
