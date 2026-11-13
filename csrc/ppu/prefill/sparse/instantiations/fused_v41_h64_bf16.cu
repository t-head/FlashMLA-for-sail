#include "sparse/fused_v41_hs64.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool Norm, bool Group, bool WideOutput, bool PerTokenScale>
using FusedHs64PrefillTraits =
    FusedHs64Traits<Norm, flashmla::dsa::hs64::Hs64PrefillTraits<512, 89, Group, false>,
                     WideOutput, !Norm && Group && WideOutput, PerTokenScale>;

template<bool Norm, bool SkipNextKPrefetch, bool PerTokenScale>
struct FusedHs64PersistentPrefillTraits
    : FusedHs64Traits<Norm, flashmla::dsa::hs64::Hs64PrefillTraits<512, 89, true, false>,
                      true, SkipNextKPrefetch, PerTokenScale> {
    static constexpr bool kPersistentPrefillQueries = true;
    // Both resident routes exclude next-query K prefetch at dispatch.
    static constexpr bool kSkipNextKPrefetch = true;
    static constexpr bool kCachePrefillRope = !SkipNextKPrefetch;
    static constexpr bool kDeferPrefillSink = !Norm && kCachePrefillRope;
    static constexpr bool kCacheRms = Norm && !SkipNextKPrefetch;
    static constexpr int kQBarrierThreads = 512;
};

template<bool PerTokenScale>
struct FusedHs64Fixed128PrefillTraits : FusedHs64PersistentPrefillTraits<false, false, PerTokenScale> {
    static constexpr int kFixedPrefillBlocks = 2;
};

template<bool PerTokenScale>
struct FusedHs64Fixed256PrefillTraits : FusedHs64PersistentPrefillTraits<false, false, PerTokenScale> {
    static constexpr int kFixedPrefillBlocks = 4;
};

template<bool Norm, bool Group, bool WideOutput, bool PerTokenScale>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_prefill_hs64_kernel(__grid_constant__ const FusedHs64Params<PerTokenScale> params) {
#if ACOMPUTE_VERSION >= 10500
    using T = FusedHs64PrefillTraits<Norm, Group, WideOutput, PerTokenScale>;
    const Flash_fwd_mla_params& base = params;
    flashmla::dsa::hs64::hs64_attention<T, false>(base, params.max_logits);
#endif
}

// Reuse the existing isolated entry for the short fixed256 body only.
// The original2048 fallback uses the existing equivalent three-bool entry.
template<bool Isolated256, bool PerTokenScale>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_prefill_hs64_fixed256_kernel(__grid_constant__ const FusedHs64Params<PerTokenScale> params) {
#if ACOMPUTE_VERSION >= 10500
    static_assert(Isolated256);
    using T = FusedHs64Fixed256PrefillTraits<PerTokenScale>;
    FusedHs64Params<PerTokenScale> fixed = params;
    fixed.q_orig = 1;
    fixed.seqlen_q = fixed.h_q = fixed.ngroups = T::kBlockM;
    fixed.h = fixed.h_h_k_ratio = 1;
    fixed.d = T::kHeadDim;
    fixed.d_v = T::kHeadDimV;
    fixed.extra_topk = -1;
    fixed.page_block_size = 1;
    fixed.q_row_stride = fixed.k_row_stride = T::kHeadDim;
    fixed.o_batch_stride = fixed.o_row_stride = 0;
    fixed.topk = 256;
    fixed.topk_len_ptr = nullptr;
    const Flash_fwd_mla_params& short_base = fixed;
    flashmla::dsa::hs64::hs64_attention<T, false>(short_base, fixed.max_logits);
#endif
}

template<bool Norm, bool SkipNextKPrefetch, bool PerTokenScale>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_prefill_hs64_persistent_kernel(__grid_constant__ const FusedHs64Params<PerTokenScale> params) {
#if ACOMPUTE_VERSION >= 10500
    using T = FusedHs64PersistentPrefillTraits<Norm, SkipNextKPrefetch, PerTokenScale>;
    // Expose the same fixed geometry as the native H64 prefill wrapper.
    FusedHs64Params<PerTokenScale> fixed = params;
    fixed.q_orig = 1;
    fixed.seqlen_q = fixed.h_q = fixed.ngroups = T::kBlockM;
    fixed.h = fixed.h_h_k_ratio = 1;
    fixed.d = T::kHeadDim;
    fixed.d_v = T::kHeadDimV;
    fixed.extra_topk = -1;
    fixed.page_block_size = 1;
    fixed.q_row_stride = fixed.k_row_stride = T::kHeadDim;
    fixed.o_batch_stride = fixed.o_row_stride = 0;
    if constexpr (!Norm && !SkipNextKPrefetch) {
        // A CTA chooses this compact body once, before its query loop.
        // Keep the existing kernel entry and all explicit-length routes.
        if (fixed.topk == 128 && fixed.topk_len_ptr == nullptr) {
            fixed.topk = 128;
            fixed.topk_len_ptr = nullptr;
            const Flash_fwd_mla_params& short_base = fixed;
            flashmla::dsa::hs64::hs64_attention<FusedHs64Fixed128PrefillTraits<PerTokenScale>, false>(
                short_base, fixed.max_logits);
            return;
        }
    }
    const Flash_fwd_mla_params& base = fixed;
    flashmla::dsa::hs64::hs64_attention<T, false>(base, fixed.max_logits);
#endif
}

template<bool Norm, bool SkipNextKPrefetch, bool PerTokenScale>
void launch_fused_prefill_hs64_persistent(const FusedHs64Params<PerTokenScale>& params, hggcStream_t stream) {
    using T = FusedHs64PersistentPrefillTraits<Norm, SkipNextKPrefetch, PerTokenScale>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &fused_norm_rope_attn_rope_cast_prefill_hs64_persistent_kernel<Norm, SkipNextKPrefetch, PerTokenScale>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const int grid_q = std::min(params.b, get_num_sm(get_current_device()));
    kernel<<<dim3(grid_q, 1, 1), T::NUM_THREADS, bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool Norm, bool Group, bool WideOutput = false, bool Isolated256 = false, bool PerTokenScale>
void launch_fused_prefill_hs64(const FusedHs64Params<PerTokenScale>& params, hggcStream_t stream) {
    static_assert(!Isolated256 || (!Norm && Group && WideOutput));
    using T = FusedHs64PrefillTraits<Norm, Group, WideOutput, PerTokenScale>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    auto kernel = [] {
        if constexpr (Isolated256) return &fused_norm_rope_attn_rope_cast_prefill_hs64_fixed256_kernel<true, PerTokenScale>;
        else return &fused_norm_rope_attn_rope_cast_prefill_hs64_kernel<Norm, Group, WideOutput, PerTokenScale>;
    }();
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const int grid_q = [&] {
        if constexpr (Isolated256) {
            return std::min(params.b, get_num_sm(get_current_device()));
        }
        return Group ? flashmla::dsa::hs64::hs64_grouped_prefill_grid_q(
            params.b, get_num_sm(get_current_device())) : params.b;
    }();
    kernel<<<dim3(grid_q, 1, 1), 512, bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool EnableNorm, bool PerTokenScale>
static void run_fused_prefill_hs64_output(const OutputParams<PerTokenScale>& p) {
    TORCH_CHECK(p.mode == FwdMode::Prefill && p.arch == 89 && p.h_q == 64);
    TORCH_CHECK(p.topk >= 128 && p.topk % 128 == 0 && p.enable_q_norm == EnableNorm);
    TORCH_CHECK(p.wv_group_size == 8 && p.n_wv_group == 8);
    if(p.s_q == 0) return;
    const int32_t* topk_length = p.topk_length;
    FusedHs64Params<PerTokenScale> params{};
    params.b = p.s_q;
    params.q_orig = 1;
    params.seqlen_q = params.h_q = params.ngroups = 64;
    params.h = params.h_h_k_ratio = 1;
    params.d = params.d_v = 512;
    params.scale_softmax = p.sm_scale;
    params.scale_softmax_log2 = p.sm_scale_log2;
    params.q_ptr = const_cast<void*>(p.q);
    params.k_ptr = const_cast<void*>(p.kv);
    params.o_ptr = p.out_fp8;
    params.o_batch_stride = params.o_row_stride = 0;
    params.softmax_lse_ptr = p.lse;
    params.indices_ptr = const_cast<int32_t*>(p.indices);
    params.topk_len_ptr = const_cast<int32_t*>(topk_length);
    params.attn_sink_ptr = const_cast<float*>(p.attn_sink);
    params.topk = p.topk;
    params.extra_topk = -1;
    params.page_block_size = 1;
    params.num_blocks = params.seqlen_k = p.s_kv;
    params.q_batch_stride = p.stride_q_s_q;
    params.q_row_stride = 512;
    params.k_batch_stride = p.stride_kv_s_kv;
    params.k_row_stride = 512;
    params.indices_batch_stride = p.stride_indices_s_q;
    params.rms_norm_eps = p.rms_norm_eps;
    params.token_positions = p.token_positions;
    params.cos_sin_cache = p.cos_sin_cache;
    params.out_fp8 = p.out_fp8;
    params.out_sf = p.out_sf;
    params.max_logits = p.max_logits;
    params.stride_q_s_q = p.stride_q_s_q;
    params.stride_out_sf_token = p.stride_out_sf_token;
    params.stride_out_sf_group = p.stride_out_sf_group;
    params.stride_out_sf_word = p.stride_out_sf_word;
    if (((p.topk <= 1024 && p.s_q >= 512) || (p.topk == 2048 && p.s_q >= 1024)) &&
        get_num_sm(get_current_device()) == 39) {
        if constexpr (!EnableNorm && PerTokenScale) {
            if (p.topk == 256 && topk_length == nullptr) {
                launch_fused_prefill_hs64<false, true, true, true>(params, p.stream);
                return;
            }
        }
        if (p.topk == 2048) launch_fused_prefill_hs64_persistent<EnableNorm, true>(params, p.stream);
        else launch_fused_prefill_hs64_persistent<EnableNorm, false>(params, p.stream);
        return;
    }
    if(p.s_q >= 1024) {
        if constexpr (!EnableNorm) {
            if (p.topk == 2048 && topk_length == nullptr) {
                launch_fused_prefill_hs64<EnableNorm, true, true>(params, p.stream);
                return;
            }
        }
        if(p.topk < 2048) launch_fused_prefill_hs64<EnableNorm, true, true>(params, p.stream);
        else launch_fused_prefill_hs64<EnableNorm, true>(params, p.stream);
    } else {
        if constexpr (!EnableNorm) {
            if (p.topk < 2048) {
                launch_fused_prefill_hs64<EnableNorm, false, true>(params, p.stream);
                return;
            }
        }
        launch_fused_prefill_hs64<EnableNorm, false>(params, p.stream);
    }
}

template<bool EnableNorm>
void run_fused_norm_rope_attn_rope_cast_prefill_hs64(const Params& p) {
    if (p.per_token_scale) {
        run_fused_prefill_hs64_output<EnableNorm>(OutputParams<true>{p});
    } else {
        run_fused_prefill_hs64_output<EnableNorm>(OutputParams<false>{p});
    }
}

template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Prefill, 64, false, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_prefill_hs64<false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Prefill, 64, true, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_prefill_hs64<true>(const Params&);

} // namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn
