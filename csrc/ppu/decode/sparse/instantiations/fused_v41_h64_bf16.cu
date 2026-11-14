#include "sparse/fused_v41_hs64.cuh"
#include "decode/sparse/fused_v41_splitkv.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist, bool PerTokenScale>
struct FusedHs64PackedTraits
    : FusedHs64Traits<Norm, std::conditional_t<ExtraFP4,
        flashmla::dsa::hs64::Hs64PackedDecode<MainCapacity, Worklist>,
        flashmla::dsa::hs64::Hs64PackedFp8ExtraWorklist<MainCapacity, Worklist>>,
        ExtraFP4 || (!Norm && MainCapacity == 128 &&
            Worklist != flashmla::dsa::hs64::Hs64Worklist::Batch), false, PerTokenScale> {
    // Flat uses 39 CTAs, so its next query always selects the other RoPE slot.
    static constexpr bool kCachePrefillRope =
        !Norm && ExtraFP4 && MainCapacity == 128 && PerTokenScale &&
        Worklist == flashmla::dsa::hs64::Hs64Worklist::Flat;
    static constexpr bool kFp8LeaderPrefetch =
        !Norm && !ExtraFP4 && MainCapacity == 128 && Worklist == flashmla::dsa::hs64::Hs64Worklist::Flat;
};

}

namespace flashmla::dsa::hs64 {

template<bool Norm, bool ExtraFP4, int MainCapacity, Hs64Worklist Worklist, bool PerTokenScale>
inline constexpr int kPackedMain<fused_norm_rope_attn_rope_cast_fwd::core_attn::FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist, PerTokenScale>> = MainCapacity;

template<bool Norm, bool ExtraFP4, int MainCapacity, Hs64Worklist Worklist, bool PerTokenScale>
inline constexpr Hs64Worklist kWorklist<fused_norm_rope_attn_rope_cast_fwd::core_attn::FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist, PerTokenScale>> = Worklist;

}

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist, bool PerTokenScale>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_decode_hs64_packed_kernel(__grid_constant__ const FusedHs64Params<PerTokenScale> params) {
#if ACOMPUTE_VERSION >= 10500
    using T = FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist, PerTokenScale>;
    if constexpr (T::kFusedPackedFp4) {
        // The packed launcher flattens queries and fixes these capacities.
        FusedHs64Params<PerTokenScale> fixed = params;
        fixed.q_orig = 1;
        fixed.seqlen_q = fixed.ngroups = T::kBlockM;
        fixed.h = 1;
        fixed.topk = MainCapacity;
        fixed.extra_topk = 512;
        flashmla::dsa::hs64::hs64_attention<T, true>(fixed);
    } else {
        flashmla::dsa::hs64::hs64_attention<T, true>(params);
    }
#endif
}

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist, bool PerTokenScale>
static void launch_fused_hs64_packed(const FusedHs64Params<PerTokenScale>& params, hggcStream_t stream) {
    using T = FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist, PerTokenScale>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &fused_norm_rope_attn_rope_cast_decode_hs64_packed_kernel<Norm, ExtraFP4, MainCapacity, Worklist, PerTokenScale>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const int grid_rows = Worklist == flashmla::dsa::hs64::Hs64Worklist::Single ? params.b : 39;
    kernel<<<dim3(1, 1, grid_rows), T::NUM_THREADS, bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool ExtraFP4, bool PerTokenScale>
static FusedHs64Params<PerTokenScale> make_fused_hs64_decode_params(const OutputParams<PerTokenScale>& p) {
    FusedHs64Params<PerTokenScale> params{};
    // Fused decode already flattens [B, Sq] and repeats lengths per query.
    params.b = p.s_q;
    params.q_orig = 1;
    params.seqlen_q = params.h_q = params.ngroups = 64;
    params.h = params.h_h_k_ratio = 1;
    params.d = params.d_v = 512;
    params.scale_softmax = p.sm_scale;
    params.scale_softmax_log2 = p.sm_scale_log2;
    params.q_ptr = const_cast<void*>(p.q);
    params.q_batch_stride = p.stride_q_s_q;
    params.q_row_stride = 512;
    params.k_ptr = const_cast<void*>(p.kv);
    params.k_batch_stride = p.stride_kv_s_kv;
    params.k_row_stride = 528;
    params.page_block_size = p.page_block_size;
    params.num_blocks = p.num_blocks;
    params.indices_ptr = const_cast<int32_t*>(p.indices);
    params.indices_batch_stride = p.stride_indices_s_q;
    params.topk_len_ptr = const_cast<int32_t*>(p.topk_length);
    params.topk = p.topk;
    params.extra_k_ptr = const_cast<void*>(p.extra_kv);
    params.extra_k_batch_stride = p.stride_extra_kv_block;
    params.extra_k_row_stride = ExtraFP4 ? 288 : 528;
    params.extra_page_block_size = p.extra_page_block_size;
    params.extra_num_blocks = p.extra_num_blocks;
    params.extra_indices_ptr = const_cast<int32_t*>(p.extra_indices);
    params.extra_indices_batch_stride = p.stride_extra_indices_s_q;
    params.extra_topk_len_ptr = const_cast<int32_t*>(p.extra_topk_length);
    params.extra_topk = p.extra_topk;
    params.attn_sink_ptr = const_cast<float*>(p.attn_sink);
    params.o_ptr = p.out_fp8;
    // The shared HS64 body forms its BF16 output view; the hook stores FP8.
    params.o_row_stride = 512 / sizeof(cutlass::bfloat16_t);
    params.o_batch_stride = 64 * params.o_row_stride;
    params.softmax_lse_ptr = p.lse;
    params.rms_norm_eps = p.rms_norm_eps;
    params.token_positions = p.token_positions;
    params.cos_sin_cache = p.cos_sin_cache;
    params.out_fp8 = p.out_fp8;
    params.out_sf = p.out_sf;
    params.stride_q_s_q = p.stride_q_s_q;
    params.stride_out_sf_token = p.stride_out_sf_token;
    params.stride_out_sf_group = p.stride_out_sf_group;
    params.stride_out_sf_word = p.stride_out_sf_word;
    return params;
}

// Keep generic KV addressing (kPackedMain == 0); only query ownership differs.
template<bool PerTokenScale>
struct FusedHs64DirectTraits : FusedHs64Traits<false,
    flashmla::dsa::hs64::Hs64Traits<512, true, 89, true, true>, true, false, PerTokenScale> {
    using Base = FusedHs64Traits<false,
        flashmla::dsa::hs64::Hs64Traits<512, true, 89, true, true>, true, false, PerTokenScale>;
    using SharedMemoryPlan = typename Base::SharedMemoryPlan;
    static constexpr bool kFusedDirectQueries = true;
    // Consecutive owned queries alternate the existing two RoPE slots.
    static constexpr bool kCachePrefillRope = PerTokenScale;
    static constexpr bool kDeferPrefillSink = true;
    static_assert(Base::kPrefillSinkOffset + 64 * sizeof(float) <= offsetof(SharedMemoryPlan, smem_sP));
};
template<bool PerTokenScale>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_decode_hs64_direct_kernel(
    __grid_constant__ const FusedHs64Params<PerTokenScale> params) {
#if ACOMPUTE_VERSION >= 10500
    flashmla::dsa::hs64::hs64_attention<FusedHs64DirectTraits<PerTokenScale>, true>(params);
#endif
}

template<bool PerTokenScale>
static void run_fused_hs64_direct_output(const OutputParams<PerTokenScale>& p, int grid_rows) {
    using T = FusedHs64DirectTraits<PerTokenScale>;
    static_assert(flashmla::dsa::hs64::kPackedMain<T> == 0);
    static_assert(flashmla::dsa::hs64::kWorklist<T> == flashmla::dsa::hs64::Hs64Worklist::Single);
    static_assert(T::kQBarrierThreads == 512);
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    const FusedHs64Params<PerTokenScale> params = make_fused_hs64_decode_params<true>(p);
    auto kernel = &fused_norm_rope_attn_rope_cast_decode_hs64_direct_kernel<PerTokenScale>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(1, 1, grid_rows), T::NUM_THREADS, bytes, p.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool Norm, bool ExtraFP4, bool PerTokenScale>
static void run_fused_hs64_packed_output(const OutputParams<PerTokenScale>& p) {
    const FusedHs64Params<PerTokenScale> params = make_fused_hs64_decode_params<ExtraFP4>(p);
    using flashmla::dsa::hs64::Hs64Worklist;
    if (p.topk == 256) {
        launch_fused_hs64_packed<Norm, ExtraFP4, 256, Hs64Worklist::Batch>(params, p.stream);
    } else if (p.s_q >= (ExtraFP4 && !Norm && PerTokenScale ? 2 :
                         (ExtraFP4 && (Norm || PerTokenScale) ? 6 : 8)) * 39) {
        launch_fused_hs64_packed<Norm, ExtraFP4, 128, Hs64Worklist::Flat>(params, p.stream);
    } else {
        launch_fused_hs64_packed<Norm, ExtraFP4, 128, Hs64Worklist::Single>(params, p.stream);
    }
}

void run_fused_norm_rope_attn_rope_cast_decode_hs64_direct(const Params& p, int grid_rows) {
    if (p.per_token_scale) {
        run_fused_hs64_direct_output(OutputParams<true>{p}, grid_rows);
    } else {
        run_fused_hs64_direct_output(OutputParams<false>{p}, grid_rows);
    }
}

template<bool Norm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed(const Params& p) {
    if (p.per_token_scale) {
        run_fused_hs64_packed_output<Norm, ExtraFP4>(OutputParams<true>{p});
    } else {
        run_fused_hs64_packed_output<Norm, ExtraFP4>(OutputParams<false>{p});
    }
}

template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<false, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, false, false>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<true, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, true, false>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<false, true>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, false, true>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<true, true>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, true, true>(const Params&, const SplitKVParams&);

} // namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn
