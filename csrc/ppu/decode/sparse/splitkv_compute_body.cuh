/******************************************************************************
 * Shared sparse SplitKV compute statements; included inside existing scopes.
 * Intentionally no include guard: exactly one named phase is selected per include.
 * Loader, scale prefetch, barriers, task traversal and normalization stay at the
 * original call sites. QUANT changes only existing first-block control syntax.
 ******************************************************************************/

#if defined(FLASHMLA_SPLITKV_COMPUTE_QK)

auto tSsk_curr = make_tensor(tSsK.data() + buf_idx * size(sK), layout(tSsK));
if constexpr (Kernel_traits::Is_Q_in_regs) {
    flash::gemm_rss<Kernel_traits::KEEP_Q_NUM>(
            acc_s, tSrQ, tSrK, tSsQ, tSsk_curr, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
    );
} else {
    flash::gemm<false>(
            acc_s, tSrQ, tSrK, tSsQ, tSsk_curr, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
    );
}

#elif defined(FLASHMLA_SPLITKV_COMPUTE_MASK_SOFTMAX_P)

constexpr int MMA_N_S = kBlockN / decltype(typename Kernel_traits::TiledMmaS{}.template tile_size_mnk<1>())::value;
const int warpN_idx = (warp_idx / AtomLayoutQ) * MMA_N_S;
flash::apply_indices_mask(acc_s, smem_valid_indices, warpN_idx, buf_idx);

#define FLASHMLA_SPLITKV_COMPUTE_SOFTMAX_FIRST softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2)
#define FLASHMLA_SPLITKV_COMPUTE_SOFTMAX_NEXT softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2)
#if defined(FLASHMLA_SPLITKV_COMPUTE_QUANT)
if constexpr (IS_FIRST_BLOCK) {
    FLASHMLA_SPLITKV_COMPUTE_SOFTMAX_FIRST;
} else {
    FLASHMLA_SPLITKV_COMPUTE_SOFTMAX_NEXT;
}
#else
block_idx == n_block_min
    ? FLASHMLA_SPLITKV_COMPUTE_SOFTMAX_FIRST
    : FLASHMLA_SPLITKV_COMPUTE_SOFTMAX_NEXT;
#endif
#undef FLASHMLA_SPLITKV_COMPUTE_SOFTMAX_FIRST
#undef FLASHMLA_SPLITKV_COMPUTE_SOFTMAX_NEXT

Tensor rS = flash::convert_type<Element>(acc_s);
Tensor tSaS = smem_thr_copy_S.retile_S(rS);     // ((Atom,AtomNum), MMA_N, MMA_N)
cute::copy(smem_tiled_copy_S, tSaS, tSsS);

#elif defined(FLASHMLA_SPLITKV_COMPUTE_RESCALE_PV)

#if defined(FLASHMLA_SPLITKV_COMPUTE_QUANT)
if constexpr (!IS_FIRST_BLOCK) {
#else
if (block_idx > n_block_min) {
#endif
    softmax.template softmax_rescale_o(acc_o, smem_row_scale);
}

auto tOsVt_curr = make_tensor(tOsVt.data() + buf_idx * size(sK), layout(tOsVt));
flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt_curr, tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V,
        smem_thr_copy_P, smem_thr_copy_V);

#elif defined(FLASHMLA_SPLITKV_COMPUTE_FINAL_STORE)

if (NoSplit) {
    flash::store<Kernel_traits, false, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
} else {
    flash::store<Kernel_traits, true, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
}

#else
#error "Select one sparse SplitKV compute phase before including this file"
#endif
