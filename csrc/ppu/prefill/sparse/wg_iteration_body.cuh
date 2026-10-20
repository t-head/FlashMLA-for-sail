// This file is intentionally included within each existing WG template.
// It owns actual compute/publication order, not just a list of phase callbacks.
// Hooks below are cache supply/address work or the necessary softmax ABI adapter.
#if FLASHMLA_WI_CURRENT_WG == 0
FLASHMLA_WI_WG0_MAKE_P()
NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale0Ready);
FLASHMLA_WI_WG0_FIRST_SUPPLY()
wg0_scale0_rO0<T>(rO0, sScale0, idx_in_warpgroup);
#if FLASHMLA_WI_LEGACY_DECODE
if constexpr (T::kIsCrossCut) {
    save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);
    const int barrier_id = 6 + static_cast<int>(threadIdx.x >> 8);
    NamedBarrier::sync(256, static_cast<cutlass::arch::ReservedNamedBarriers>(barrier_id));
    FLASHMLA_WI_PV_REMOTE<T>(sP0, sV0L, rO0, idx_in_warpgroup, wg_idx);
} else {
#endif
    FLASHMLA_WI_PV_LOCAL<T>(rPb, sV0L, rO0, idx_in_warpgroup, wg_idx);
#if FLASHMLA_WI_LEGACY_DECODE
}
#endif
NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale1Ready);
FLASHMLA_WI_WG0_SECOND_SUPPLY_BEFORE_P()
wg0_scale_rP0<T>(sScale1, rP0, rPb, idx_in_warpgroup);
save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);
NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sP0Ready);
FLASHMLA_WI_WG0_AFTER_P()
if constexpr (!IS_BLK0_LAST) {
    wg0_rescale_rO0<T>(rO0, sScale1, rL, idx_in_warpgroup);
    FLASHMLA_WI_PV_REMOTE<T>(sP1, sV1L, rO0, idx_in_warpgroup, wg_idx);
}
if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
    cute::clear(rP0);
    FLASHMLA_WI_QK<T, 0>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
}
FLASHMLA_WI_WG0_PREPARE_NEXT()
if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
    FLASHMLA_WI_QK<T, 2>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
}
#if FLASHMLA_WI_LEGACY_DECODE
if constexpr (T::kBlockM == 64) { __syncthreads(); }
if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
#endif
kv_idx = (kv_idx + 2) % 3;
cur_sK0 = sK(_, _, kv_idx);
cur_sK1 = sK(_, _, (kv_idx + 1) % 3);
nxt_sK0 = sK(_, _, (kv_idx + 2) % 3);
#if FLASHMLA_WI_LEGACY_DECODE
}
#endif
#else
FLASHMLA_WI_WG1_INITIAL_SUPPLY()
#if !FLASHMLA_WI_LEGACY_DECODE
constexpr bool first_finish_after_scale1 = T::kDecodeFP4WG1FirstFinishAfterScale1
    && T::kDecode && T::kFP4Cache && !T::kMixedFP4 && T::kDecodeFixedExtraIndex
    && T::kAllValidNoLengths && !IS_BLK0_LAST && !IS_BLK1_LAST;
#endif
float r_cur_max[2];
#if !FLASHMLA_WI_LEGACY_DECODE
constexpr bool supply_after_scale1 = T::kDecodeFP8WG1SupplyAfterScale1 && T::kDecode
    && !T::kFP4Cache && !T::kMixedFP4 && T::kDecodeFixedExtraIndex
    && T::kAllValidNoLengths && !IS_BLK0_LAST && !IS_BLK1_LAST;
auto early_rP1b = [&]() {
V41QuantRegisterPacket<T> first_register_packet{};
FLASHMLA_WI_WG1_EARLY_PACKET_SUPPLY()
#endif
FLASHMLA_WI_WG1_PRE_CURMAX()
NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);
#if FLASHMLA_WI_LEGACY_DECODE
if constexpr (T::kBlockM == 64) { __syncthreads(); }
#else
// Keep the original first packet scope and early P/scales publication.
auto published_rP1b = [&]() {
    if constexpr (supply_after_scale1) {
        static_assert(T::Arch_value == 89 && T::kDecodeFP8WG1EarlyPacket && T::kUsePairedCubes &&
            T::kBlockM == 128 && T::kBlockN == 32 && T::kHeadDim == 512);
        Tensor rP1b = wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST>(sScale1, rO1, sM, rL,
            sScale0, rP1, params.sm_scale_div_log2, start_token_idx+T::kBlockN, idx_in_warpgroup,
            smem_valid_indices, 2+(block_idx/2)%2, r_cur_max);
        if constexpr (!IS_BLK0_LAST) { save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup); }
        NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);
        __ppu_sched_bound();
        return rP1b;
    } else { return cute::make_tuple(); }
}();
FLASHMLA_WI_WG1_FINISH_FIRST()
return published_rP1b;
}();
FLASHMLA_WI_WG1_SECOND_PACKET()
#endif
#if FLASHMLA_WI_LEGACY_DECODE
FLASHMLA_WI_WG1_MAKE_P()
#else
Tensor rP1b = [&]() {
    if constexpr (supply_after_scale1) { return early_rP1b; }
    else {
        return wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST>(sScale1, rO1, sM, rL,
            sScale0, rP1, params.sm_scale_div_log2, start_token_idx+T::kBlockN, idx_in_warpgroup,
            smem_valid_indices, 2+(block_idx/2)%2, r_cur_max);
    }
}();
if constexpr (!supply_after_scale1) {
#endif
if constexpr (!IS_BLK0_LAST) { save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup); }
NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);
#if !FLASHMLA_WI_LEGACY_DECODE
}
FLASHMLA_WI_WG1_FINISH_FIRST_AFTER_P()
#else
if constexpr (T::kBlockM == 64) { __syncthreads(); }
#endif
wg1_scale0_rO1<T>(rO1, sScale0, sScale1, idx_in_warpgroup);
if constexpr (!IS_BLK0_LAST) {
#if FLASHMLA_WI_LEGACY_DECODE
    if constexpr (T::kIsCrossCut) {
        const int barrier_id = 6 + static_cast<int>(threadIdx.x >> 8);
        NamedBarrier::sync(256, static_cast<cutlass::arch::ReservedNamedBarriers>(barrier_id));
        FLASHMLA_WI_PV_REMOTE<T>(sP1, sV1R, rO1, idx_in_warpgroup, wg_idx);
    } else {
#endif
        FLASHMLA_WI_PV_LOCAL<T>(rP1b, sV1R, rO1, idx_in_warpgroup, wg_idx);
#if FLASHMLA_WI_LEGACY_DECODE
    }
#endif
}
NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sP0Ready);
#if FLASHMLA_WI_LEGACY_DECODE
if constexpr (T::kBlockM == 64) { __syncthreads(); }
#else
constexpr bool remote_pv_before_finish = T::kDecodeFP4WG1RemotePVBeforeFinish && T::kDecode && T::kFP4Cache
    && T::kAllValidNoLengths && !IS_BLK0_LAST && !IS_BLK1_LAST;
if constexpr (remote_pv_before_finish) {
    static_assert(T::kUsePairedCubes && T::kBlockN == 32 && T::kHeadDim == 512);
    FLASHMLA_WI_PV_REMOTE<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx);
}
#if defined(FLASHMLA_V41_FP8_WG1_REMOTE_PV_FIRST_PREFETCH)
auto remote_pv_first = [&]() {
    if constexpr(!T::kFP4Cache && T::kAllValidNoLengths && !IS_BLK0_LAST && !IS_BLK1_LAST) {
        return v41_prefill_remote_pv_first_prefetch<T>(sP0, sV0R, idx_in_warpgroup, wg_idx);
    } else { return cute::make_tuple(); }
}();
#endif
#endif
FLASHMLA_WI_WG1_SECOND_SUPPLY_AND_ADDRESS()
#if !FLASHMLA_WI_LEGACY_DECODE
if constexpr (!remote_pv_before_finish) {
#if defined(FLASHMLA_V41_FP8_WG1_REMOTE_PV_FIRST_PREFETCH)
    if constexpr(!T::kFP4Cache && T::kAllValidNoLengths && !IS_BLK0_LAST && !IS_BLK1_LAST) {
        dsa_warpgroup_cooperative_pv_gemm_remoteP_prefetched_first<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx, remote_pv_first);
    } else
#endif
#endif
    FLASHMLA_WI_PV_REMOTE<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx);
#if !FLASHMLA_WI_LEGACY_DECODE
}
#endif
if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
    cute::clear(rP1);
    FLASHMLA_WI_QK<T, 1>(sQ, nxt_sK0, nxt_sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
}
#if FLASHMLA_WI_LEGACY_DECODE
if constexpr (T::kBlockM == 64) { __syncthreads(); }
if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
#endif
kv_idx = (kv_idx + 2) % 3;
cur_sK1 = sK(_, _, kv_idx);
cur_sK0 = sK(_, _, (kv_idx + 1) % 3);
nxt_sK1 = sK(_, _, (kv_idx + 2) % 3);
#if FLASHMLA_WI_LEGACY_DECODE
}
#endif
#endif
