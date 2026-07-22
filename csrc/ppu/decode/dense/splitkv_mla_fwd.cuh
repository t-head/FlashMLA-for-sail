/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

using namespace cute;

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, const int m_block,
                                                               const int n_split_idx, const bool have_zero_seqlen_k,
                                                               const int n_block_min, int n_block_max, const bool NoSplit) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int MMA_ATOM_M = Kernel_traits::USE_MMA_M8 ? 8 : 16;
    constexpr int MMA_ATOM_K_M = Kernel_traits::USE_MMA_M8 ? 1 : 2;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    //const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    //const int n_block_min = n_split_idx * n_blocks_per_split;
    //int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q / params.ngroups, kBlockN));
    }
    // if (threadIdx.x == 0) {
    //     printf("block:%d, n_block_max:%d, n_block_min:%d, binfo.actual_seqlen_k:%d, kBlockN:%d, n_blocks_per_split:%d, num_n_splits:%d,\n",
    //         blockIdx.x, n_block_max, n_block_min, binfo.actual_seqlen_k, kBlockN, n_blocks_per_split, num_n_splits);
    // }

    // [deprecated] never has n_block_min >= n_block_max in tile scheduler mode
    // if have_zero_seqlen_k, n_block_max = n_block_min = 0
    if (have_zero_seqlen_k) n_block_max = max(1, n_block_max);
    assert(n_block_min < n_block_max);
    static_assert(Kernel_traits::kBlockNPagedPerAiuLoad == kBlockN);

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = bidb;
    const int *block_table = params.block_table != nullptr ? params.block_table + bidb * params.block_table_batch_stride : nullptr ;
    const int64_t *hllm_block_table = params.block_table == nullptr ? params.hllm_block_table + bidb * params.block_table_batch_stride : nullptr;
    const int block_table_idx = (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset = (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const index_t row_offset_k = block_table != nullptr
        ? __ldg(block_table + block_table_idx) * params.k_batch_stride + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    // const index_t row_offset_v = block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(make_mix_tensor_like(mQ(_, bidh, _)), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(
                                block_table != nullptr
                                    ? reinterpret_cast<Element *>(params.k_ptr)
                                    : reinterpret_cast<Element *>(__ldg(hllm_block_table + block_table_idx))) + row_offset_k,
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));;

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutK{});

    //use k/v shared
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});

    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // double shared memory for k/v cache.
    Tensor sK_double = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutK{});

    Tensor sVt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVt_double = make_tensor(sK_double.data(), typename Kernel_traits::SmemLayoutVtransposed{});

    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;
    typename Kernel_traits::GmemTiledCopyQK gmem_tiled_copy_K;

    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_K.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK);

    // Tensor tKsK_double = gmem_tiled_copy_K.partition_D(sK_double);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    // Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    int aiu_offset_q = 0;
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, binfo.actual_seqlen_q, params.q_row_stride, kBlockM, Kernel_traits::kBlockKSmem, aiu_offset_q};
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, binfo.actual_seqlen_q, params.d, params.q_row_stride);
#endif
    const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
    const int tid_thread_slice = warp_idx * 32;
#else
    const int tid_thread_slice = tidx;
#endif

    // auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    // auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma);
    // FIXME: the use of "tidx" causes v.mov.v2s, but the bugfix causes perf regression.
    // auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid_thread_slice);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));

    // PREDICATES
    //
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Tensor tKVcKV = gmem_tiled_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_Q.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Prologue
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // FIXME: Is_even_MN should be false ?
    flash::copy<true, true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    // load Q from tsm to verg and keep use.
    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    auto tKgK_data = tKgK.data();
    { // use new namespace to create mix tensor with the same name

    //////////////////////// switch to mix tensors start ////////////////////////

    typename Kernel_traits::GmemTiledCopyK gmem_tiled_copy_K;
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_K.partition_S(make_mix_tensor_like(gK));  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
    Tensor tKsK_double = gmem_thr_copy_K.partition_D(sK_double);

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    int aiu_offset_k = 0;
    gmem_tiled_copy_K.desc_ = AiuDesc{nullptr, kBlockN, params.k_row_stride, kBlockN, Kernel_traits::kBlockKSmem, aiu_offset_k};
#else
    gmem_tiled_copy_K.desc_.init(nullptr, kBlockN, params.d, params.k_row_stride);
#endif
    const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
    const int tid_thread_slice = warp_idx * 32;
#else
    const int tid_thread_slice = tidx;
#endif

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomK{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid_thread_slice);
    auto tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK));

    auto tSsK_double = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK_double));

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid_thread_slice);

    auto tOsVt = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt));
    auto tOsVt_double = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt_double));

    int n_block = n_block_max - 1;

    // use kv_block_num to decide number.
    int kv_store_num = 0;
    int kv_load_num = 0;

    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    if (!have_zero_seqlen_k)
    flash::copy<Is_even_MN, true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK,
                                  binfo.actual_seqlen_k - n_block * kBlockN);
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

    flash::Softmax<MMA_ATOM_K_M * size<1>(acc_o)> softmax;
    flash::Mask mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    constexpr int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        // FIXME: bug: copy 0->wait 0->sync->copy 1->compute 0->wait 1->sync->copy 0->compute 1->...
        // bug perf regression in: copy 0->copy 1->wait 0->sync->compute 0->copy 0->wait 1->sync->compute 1 ->..., but
        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block > n_block_min) {
            auto tKsK_current = kv_store_num % 2 == 0 ? tKsK : tKsK_double;
            // Advance gK
            if (block_table == nullptr && hllm_block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =(n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                const index_t table_diff = block_table
                    ? (__ldg(block_table + block_table_idx_next) - __ldg(block_table +block_table_idx_cur)) * params.k_batch_stride
                    : reinterpret_cast<Element *>(__ldg(hllm_block_table + block_table_idx_next)) - reinterpret_cast<Element *>(__ldg(hllm_block_table + block_table_idx_cur));
                tKgK.data() = tKgK.data() + table_diff + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
            }

            flash::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_K, tKgK, tKsK_current, tKcK, tKpK);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
            kv_store_num++;
            // flash::cp_async_wait<1>();
        // } else {
            // flash::cp_async_wait<0>();
        }
        // __syncthreads();


        // determine use kv buffer 0 or 1
        auto tSsK_current = kv_load_num % 2 == 0 ? tSsK : tSsK_double;
        auto tOsVt_current = kv_load_num % 2 == 0 ? tOsVt : tOsVt_double;

        if (!have_zero_seqlen_k)
        flash::gemm<Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK_current, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * MMA_ATOM_M + (tidx % 32) / 4, kNWarps * MMA_ATOM_M, params.ngroups
        );

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2);

#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
        Tensor rP = flash::convert_acc<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(), make_layout(get<0>(tSrQ.layout()), get<1>(acc_s.layout()), get<2>(acc_s.layout())));
#else
        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
#endif
        if (!have_zero_seqlen_k)
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt_current, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        kv_load_num++;

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        // // FIXME: bug: copy 0->wait 0->sync->copy 1->compute 0->wait 1->sync->copy 0->compute 1->...
        // // bug perf regression in: copy 0->copy 1->wait 0->sync->compute 0->copy 0->wait 1->sync->compute 1 ->..., but
        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block > n_block_min) {
            // Advance gK
            auto tKsK_current = kv_store_num % 2 == 0 ? tKsK : tKsK_double;
            if (block_table == nullptr && hllm_block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
                const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
                const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
                const int block_table_offset_next =(n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
                const index_t table_diff = block_table
                    ? (__ldg(block_table + block_table_idx_next) - __ldg(block_table +block_table_idx_cur)) * params.k_batch_stride
                    : reinterpret_cast<Element *>(__ldg(hllm_block_table + block_table_idx_next)) - reinterpret_cast<Element *>(__ldg(hllm_block_table + block_table_idx_cur));
                tKgK.data() = tKgK.data() + table_diff + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
            }

            flash::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_K, tKgK, tKsK_current, tKcK, tKpK);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
            kv_store_num++;
        //     flash::cp_async_wait<1>();
        // } else {
        //     flash::cp_async_wait<0>();
        }
        // __syncthreads();
        // determine use kv buffer 0 or 1
        auto tSsK_current = kv_load_num % 2 == 0 ? tSsK : tSsK_double;
        auto tOsVt_current = kv_load_num % 2 == 0 ? tOsVt : tOsVt_double;

        flash::gemm<Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK_current, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2);
#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000
        Tensor rP = flash::convert_acc<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(), make_layout(get<0>(tSrQ.layout()), get<1>(acc_s.layout()), get<2>(acc_s.layout())));
#else
        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
#endif
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt_current, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        kv_load_num++;
    }

    // Epilogue
    if (NoSplit) {
        store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    } else {
        store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }
    } // new namespace end for the mix tensor
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, typename Params>
__forceinline__ __device__ void compute_attn_cross_cut_splitkv(const Params &params, const int bidb, const int bidh, const int m_block,
                                                               const int n_split_idx, const bool have_zero_seqlen_k,
                                                               const int n_block_min, int n_block_max,  const bool NoSplit) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int AtomLayoutQ = Kernel_traits::AtomLayoutQ;
    constexpr int AtomLayoutP = Kernel_traits::AtomLayoutP;
    constexpr bool USE_MMA_M8 = Kernel_traits::USE_MMA_M8;
    constexpr int MMA_ATOM_M = USE_MMA_M8 ? 8 : 16;
    constexpr int kStages = Kernel_traits::kStages;
    constexpr int kBlockNPagedPerAiuLoad = Kernel_traits::kBlockNPagedPerAiuLoad;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    //const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    //const int n_block_min = n_split_idx * n_blocks_per_split;
    //int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q / params.ngroups, kBlockN));
    }

    // [deprecated] never has n_block_min >= n_block_max in tile scheduler mode
    // if have_zero_seqlen_k, n_block_max = n_block_min = 0
    if (have_zero_seqlen_k) n_block_max = max(1, n_block_max);
    assert(n_block_min < n_block_max);

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = bidb;
    const int *block_table = params.block_table != nullptr ? params.block_table + bidb * params.block_table_batch_stride : nullptr ;
    const int64_t *hllm_block_table = params.block_table == nullptr ? params.hllm_block_table + bidb * params.block_table_batch_stride : nullptr;
    #define GET_BLOCK_INDEX(block_idx, table_idx) \
        ((block_idx) >= n_block_min && (block_table) ? __ldg(block_table + (table_idx)) : 0)

    int n_block = n_block_max - 1;
    // use kv_block_num to decide number.
    int kv_store_num = 0;
    int kv_load_num = 0;
    const int page_block_size = params.page_block_size;
    int block_table_idx = n_block * kBlockN / page_block_size;
    int block_table_offset = n_block * kBlockN - block_table_idx * page_block_size;
    int cur_block_table = GET_BLOCK_INDEX(n_block, block_table_idx);

    index_t row_offset_k = cur_block_table * params.k_batch_stride
                         + block_table_offset * params.k_row_stride
                         + (bidh / params.h_h_k_ratio) * params.k_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(make_mix_tensor_like(mQ(_, bidh, _)), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_mix_tensor(make_gmem_ptr(block_table == nullptr
                                ? reinterpret_cast<Element *>(__ldg(hllm_block_table + block_table_idx))
                                : reinterpret_cast<Element *>(params.k_ptr)) + row_offset_k,
                            Shape<Int<kBlockNPagedPerAiuLoad>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKstages{});
    Tensor sKPagedforCopy = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKPagedstages{});
    Tensor sVt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtstage{});

    // sVtNoSwizzle
    Tensor sVtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor sP = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutP{});

    Tensor smem_row_scale = make_tensor(make_smem_ptr(reinterpret_cast<float *>((sP.data() + size(sP)).get())),
        Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor smem_row_via_warp = make_tensor(smem_row_scale.data() + size(smem_row_scale),
        Shape<Int<kBlockM>, Int<kNWarps/AtomLayoutQ>>{}, Stride<Int<kNWarps/AtomLayoutQ>, _1>{});

    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;
    typename Kernel_traits::GmemTiledCopyK gmem_tiled_copy_K;

    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_K.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_K.partition_D(sKPagedforCopy);

    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK(_, _, _0{}));               // (MMA,MMA_N,MMA_K)

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
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, binfo.actual_seqlen_q, params.q_row_stride, kBlockM, Kernel_traits::kBlockKSmem, 0};
    // gmem_tiled_copy_K.desc_ = AiuDesc{nullptr, kBlockN, params.k_row_stride, kBlockN, Kernel_traits::kBlockKSmem, 0};
    gmem_tiled_copy_K.desc_ = AiuDesc{nullptr, kBlockNPagedPerAiuLoad, params.k_row_stride, kBlockNPagedPerAiuLoad, Kernel_traits::kBlockKSmem, 0};
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, binfo.actual_seqlen_q, params.d, params.q_row_stride);
    // gmem_tiled_copy_K.desc_.init(nullptr, kBlockN, params.d, params.k_row_stride);
    gmem_tiled_copy_K.desc_.init(nullptr, kBlockNPagedPerAiuLoad, params.d, params.k_row_stride);
#endif
    const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
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
    // Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Tensor tKVcKV = gmem_tiled_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_K.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));

    auto smem_tiled_copy_S = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomS{}, tiled_mma_s);
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tidx);
    Tensor tSsS = smem_thr_copy_S.partition_D(sP);

    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomP{}, tiled_mma_o);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tidx);
    Tensor tOsP = smem_thr_copy_P.partition_S(sP);


    // Prologue
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // FIXME: Is_even_MN should be false ?
    flash::copy<true, true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomK{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid_thread_slice);
    auto tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK));

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomVt{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid_thread_slice);
    auto tOsVt = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt));

    int block_table_idx_nxt = block_table_idx;
    int block_table_offset_nxt = 0;
    index_t row_offset_k_nxt = 0;

    auto KV_load_kBlockN = [&](int const n_block_load, auto is_first_iter_type){
        static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
        auto tKsK_current = tKsK(_, _, _, 0, kv_store_num);
        if constexpr (Is_first_iter) {
            if (!have_zero_seqlen_k)
            flash::copy<Is_even_MN, true>(gmem_tiled_copy_K, tKgK, tKsK_current, tKcK, tKpK,
                                          binfo.actual_seqlen_k - n_block_load * kBlockN);
            kv_store_num = kv_store_num < kStages - 1 ? kv_store_num + 1 : 0;
        } else if (n_block_load >= n_block_min) {
            if (block_table) {
                tKgK.data().ptr_ = make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k_nxt);
            } else if (hllm_block_table) {
                tKgK.data().ptr_ = make_gmem_ptr(reinterpret_cast<Element *>(__ldg(hllm_block_table + block_table_idx_nxt)) + row_offset_k_nxt);
            } else {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            }
            flash::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_K, tKgK, tKsK_current, tKcK, tKpK);
            kv_store_num = kv_store_num < kStages -1 ? kv_store_num + 1 : 0;
        }
        cute::cp_async_fence();
    };

    auto KV_load_Paged = [&](int const n_block_load, auto is_first_iter_type){
        static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
        if (n_block_load >= n_block_min) {
            for (int vv = 0; vv < size<3>(tKsK); vv ++) {
                auto tKsK_current = tKsK(_, _, _, vv, kv_store_num);
                const int seqlen_k_begin = n_block_load * kBlockN + vv * kBlockNPagedPerAiuLoad;
                if (Is_first_iter && seqlen_k_begin >= binfo.actual_seqlen_k) {
                    cute::clear(tKsK_current);
                    continue;
                }
                const int block_table_idx_paged = seqlen_k_begin / page_block_size;
                const int block_table_offset_paged = seqlen_k_begin - block_table_idx_paged * page_block_size;
                const index_t row_offset_k_page = block_table_offset_paged * params.k_row_stride
                                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
                if (block_table) {
                    tKgK.data().ptr_ = make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr)
                        + GET_BLOCK_INDEX(n_block_load, block_table_idx_paged) * params.k_batch_stride + row_offset_k_page);
                } else if (hllm_block_table) {
                    tKgK.data().ptr_ = make_gmem_ptr(reinterpret_cast<Element *>(__ldg(hllm_block_table + block_table_idx_paged))
                        + row_offset_k_page);
                } else {
                    tKgK.data().ptr_ = make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k_page);
                }

                if (!have_zero_seqlen_k)
                flash::copy<!Is_first_iter || Is_even_MN, true>(gmem_tiled_copy_K, tKgK,
                    tKsK_current, tKcK, tKpK, binfo.actual_seqlen_k - seqlen_k_begin);
            }
            kv_store_num = kv_store_num < kStages -1 ? kv_store_num + 1 : 0;
        }
        cute::cp_async_fence();
    };

    auto KV_load = [&](int const n_block_load, auto is_first_iter_type){
        if constexpr (kBlockNPagedPerAiuLoad == kBlockN) {
            KV_load_kBlockN(n_block_load, is_first_iter_type);
        } else {
            KV_load_Paged(n_block_load, is_first_iter_type);
        }
    };

    KV_load(n_block, cute::true_type{});

    block_table_idx_nxt = (n_block - 1) * kBlockN / page_block_size;
    block_table_offset_nxt = (n_block - 1) * kBlockN - block_table_idx_nxt * page_block_size;
    int nxt_block_table = GET_BLOCK_INDEX((n_block - 1), block_table_idx_nxt);
    row_offset_k_nxt = nxt_block_table * params.k_batch_stride
                     + block_table_offset_nxt * params.k_row_stride
                     + (bidh / params.h_h_k_ratio) * params.k_head_stride;

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    // #pragma unroll
    // for(int stage = 1; stage < kStages-1; stage ++) {
    if constexpr (kStages == 3) {
        KV_load(n_block - 1, cute::false_type{});
        block_table_idx_nxt = (n_block - 2) * kBlockN / page_block_size;
        nxt_block_table = GET_BLOCK_INDEX((n_block - 2), block_table_idx_nxt);
        block_table_offset_nxt = (n_block - 2) * kBlockN - block_table_idx_nxt * page_block_size;
        row_offset_k_nxt = nxt_block_table * params.k_batch_stride
                         + block_table_offset_nxt * params.k_row_stride
                         + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    }

    clear(acc_o);

    flash::SoftmaxBetweenWarps<USE_MMA_M8, kBlockM, AtomLayoutQ, AtomLayoutP, kNWarps/AtomLayoutQ> softmax;
    flash::Mask mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    constexpr int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        if (masking_step > 0){
            __syncthreads();
        }

        KV_load(n_block - kStages + 1, cute::false_type{});
        flash::cp_async_wait<kStages-1>();
        __syncthreads();

        block_table_idx_nxt = (n_block - kStages) * kBlockN / page_block_size;
        nxt_block_table = GET_BLOCK_INDEX((n_block - kStages), block_table_idx_nxt);
        block_table_offset_nxt = (n_block - kStages) * kBlockN - block_table_idx_nxt * page_block_size;

        if (!have_zero_seqlen_k)
        (kBlockNPagedPerAiuLoad == kBlockN)
            ? flash::gemm<Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, kv_load_num),
                tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K)
            : flash::gemm_pagedkv<kBlockNPagedPerAiuLoad, kBlockNPagedPerAiuLoad*kHeadDim, 1,
                Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, kv_load_num),
                tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);


        // const int warp_id = threadIdx.x  / 32;
        // const int line_id = threadIdx.x % 32;
        // const int warp_id_m = warp_id % AtomLayoutQ;
        // const int warp_id_n = warp_id / AtomLayoutQ;
        // const int warp_id_p = warp_id % AtomLayoutP;
        // MMA_N_S = 1

        constexpr int MMA_N_S = kBlockN / decltype(typename Kernel_traits::TiledMmaS{}.template tile_size_mnk<1>())::value;
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN + (tidx / 32 / AtomLayoutQ) * MMA_N_S * 16,
            m_block * kBlockM + (tidx / 32) % AtomLayoutQ * MMA_ATOM_M + (tidx % 32) / 4,
            AtomLayoutQ * MMA_ATOM_M, params.ngroups
        );

        masking_step == 0
            ? softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/Is_causal || !Is_even_MN>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2)
            : softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/Is_causal || !Is_even_MN>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2);

        Tensor rS = flash::convert_type<Element>(acc_s);
        Tensor tSaS = smem_thr_copy_S.retile_S(rS);     // ((Atom,AtomNum), MMA_N, MMA_N)

        cute::copy(smem_tiled_copy_S, tSaS, tSsS);
        __syncthreads();

        // row_offset_k_nxt move here to avoid Stall Memory Dependency of  s.wait sldcnt(0).
        row_offset_k_nxt = nxt_block_table * params.k_batch_stride
                         + block_table_offset_nxt * params.k_row_stride
                         + (bidh / params.h_h_k_ratio) * params.k_head_stride;

        if (masking_step > 0) {
            softmax.template softmax_rescale_o(acc_o, smem_row_scale);
        }
        if (!have_zero_seqlen_k)
        (kBlockNPagedPerAiuLoad == kBlockN)
            ? flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt(_, _, _, kv_load_num),
                 tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V, smem_thr_copy_P, smem_thr_copy_V)
            : flash::gemm_pagedkv<kBlockNPagedPerAiuLoad, kBlockNPagedPerAiuLoad*kHeadDim, 0>(
                acc_o, tOrP, tOrVt, tOsP, tOsVt(_, _, _, kv_load_num), tiled_mma_o,
                smem_tiled_copy_P, smem_tiled_copy_V, smem_thr_copy_P, smem_thr_copy_V);

        kv_load_num = kv_load_num < kStages -1 ? kv_load_num + 1 : 0;

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        __syncthreads();

        KV_load(n_block - kStages + 1, cute::false_type{});
        flash::cp_async_wait<kStages-1>();
        __syncthreads();

        block_table_idx_nxt = (n_block - kStages) * kBlockN / page_block_size;
        nxt_block_table = GET_BLOCK_INDEX((n_block - kStages), block_table_idx_nxt);
        block_table_offset_nxt = (n_block - kStages) * kBlockN - block_table_idx_nxt * page_block_size;

        (kBlockNPagedPerAiuLoad == kBlockN)
            ? flash::gemm<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, kv_load_num),
                tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K)
            : flash::gemm_pagedkv<kBlockNPagedPerAiuLoad, kBlockNPagedPerAiuLoad*kHeadDim, 1,
                Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, kv_load_num),
                tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        softmax.template softmax_rescale_per_warp</*Is_first=*/false,  /*Check_inf=*/false>(
            acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2);
        Tensor rS = flash::convert_type<Element>(acc_s);
        Tensor tSaS = smem_thr_copy_S.retile_S(rS);
        cute::copy(smem_tiled_copy_S, tSaS, tSsS);
        __syncthreads();

        row_offset_k_nxt = nxt_block_table * params.k_batch_stride
                         + block_table_offset_nxt * params.k_row_stride
                         + (bidh / params.h_h_k_ratio) * params.k_head_stride;

        softmax.template softmax_rescale_o(acc_o, smem_row_scale);
        (kBlockNPagedPerAiuLoad == kBlockN)
            ? flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt(_, _, _, kv_load_num), tiled_mma_o,
                smem_tiled_copy_P, smem_tiled_copy_V, smem_thr_copy_P, smem_thr_copy_V)
            : flash::gemm_pagedkv<kBlockNPagedPerAiuLoad, kBlockNPagedPerAiuLoad*kHeadDim, 0>(
                acc_o, tOrP, tOrVt, tOsP, tOsVt(_, _, _, kv_load_num), tiled_mma_o,
                smem_tiled_copy_P, smem_tiled_copy_V, smem_thr_copy_P, smem_thr_copy_V);

        kv_load_num = kv_load_num < kStages -1 ? kv_load_num + 1 : 0;
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
        store<Kernel_traits, false, true>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    } else {
        store<Kernel_traits, true, true>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }
}

template<typename Kernel_traits, bool Is_causal, bool CrossCut = false>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const Flash_fwd_params params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    static_assert(CrossCut == Kernel_traits::CrossCut);

    //extern __shared__ char shared_memory[];
    //auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;
    if (begin_idx >= params.b) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

#pragma unroll 1
#pragma clang loop licm(disable)
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = __ldg(params.cu_seqlens_k + batch_id);
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == cute::ceil_div(seqlen_k, kBlockN);
        if (batch_id > begin_idx) {
            __syncthreads();  // Barrier between two tiles.
        }
#if ACOMPUTE_VERSION != 10000
    if constexpr (!Kernel_traits::USE_MMA_M8)
#endif
    {
        if constexpr (CrossCut) {
            compute_attn_cross_cut_splitkv<Kernel_traits, Is_causal, false>(
                params, batch_id, bidh, m_block, n_split_idx, seqlen_k == 0,
                n_block_min, n_block_max, NoSplit);
        } else {
            compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, false>(
                params, batch_id, bidh, m_block, n_split_idx, seqlen_k == 0,
                n_block_min, n_block_max, NoSplit);
        }
    }
    }
}

} // namespace flash

template<typename Kernel_traits, bool CrossCut>
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
        flash::printf_show_log<Kernel_traits>(reinterpret_cast<const void*>(kernel), params, smem_size, Is_causal);
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
