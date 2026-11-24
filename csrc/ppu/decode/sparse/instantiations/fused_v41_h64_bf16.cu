#include "sparse/fused_v41_hs64.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist, int Arch, bool FloatKV, bool PerTokenScale>
struct FusedHs64PackedTraits
    : FusedHs64Traits<Norm, std::conditional_t<ExtraFP4,
        flashmla::dsa::hs64::Hs64PackedDecode<MainCapacity, Worklist, Arch, FloatKV>,
        flashmla::dsa::hs64::Hs64PackedFp8ExtraWorklist<MainCapacity, Worklist>>,
        ExtraFP4 || (!Norm && MainCapacity == 128 &&
            Worklist != flashmla::dsa::hs64::Hs64Worklist::Batch), false, PerTokenScale> {
    static_assert(Arch == 89 || (!Norm && ExtraFP4 && MainCapacity == 128 && !PerTokenScale));
    // Flat uses 39 CTAs, so its next query always selects the other RoPE slot.
    static constexpr bool kCachePrefillRope =
        Arch == 89 && !Norm && ExtraFP4 && MainCapacity == 128 && PerTokenScale &&
        Worklist == flashmla::dsa::hs64::Hs64Worklist::Flat;
    static constexpr bool kFp8LeaderPrefetch =
        !Norm && !ExtraFP4 && MainCapacity == 128 && Worklist == flashmla::dsa::hs64::Hs64Worklist::Flat;
};

}

namespace flashmla::dsa::hs64 {

template<bool Norm, bool ExtraFP4, int MainCapacity, Hs64Worklist Worklist, int Arch, bool FloatKV, bool PerTokenScale>
inline constexpr int kPackedMain<fused_norm_rope_attn_rope_cast_fwd::core_attn::FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist, Arch, FloatKV, PerTokenScale>> = MainCapacity;

template<bool Norm, bool ExtraFP4, int MainCapacity, Hs64Worklist Worklist, int Arch, bool FloatKV, bool PerTokenScale>
inline constexpr Hs64Worklist kWorklist<fused_norm_rope_attn_rope_cast_fwd::core_attn::FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist, Arch, FloatKV, PerTokenScale>> = Worklist;

}

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<typename T>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_decode_hs64_wi_kernel(
    __grid_constant__ const typename T::FusedParams params) {
#if ACOMPUTE_VERSION == 10000 || ACOMPUTE_VERSION >= 10500
    if constexpr (T::kArch == (ACOMPUTE_VERSION == 10000 ? 80 : 89)) {
        if constexpr (T::kFusedPackedFp4) {
            // The packed launcher flattens queries and fixes these capacities.
            typename T::FusedParams fixed = params;
            fixed.q_orig = 1;
            fixed.seqlen_q = fixed.ngroups = T::kBlockM;
            fixed.h = 1;
            fixed.topk = flashmla::dsa::hs64::kPackedMain<T>;
            fixed.extra_topk = 512;
            flashmla::dsa::hs64::hs64_attention<T, true>(fixed);
        } else {
            flashmla::dsa::hs64::hs64_attention<T, true>(params);
        }
    }
#endif
}

template<typename T>
static void launch_fused_hs64_wi(
    const typename T::FusedParams& params, int grid_rows, hggcStream_t stream) {
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &fused_norm_rope_attn_rope_cast_decode_hs64_wi_kernel<T>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(1, 1, grid_rows), T::NUM_THREADS, bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<int Arch>
inline constexpr int kFusedHs64PackedSmCount = Arch == 80 ? 20 : 39;

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist,
         int Arch, bool FloatKV, bool PerTokenScale>
static void launch_fused_hs64_packed(const FusedHs64Params<PerTokenScale>& params, hggcStream_t stream) {
    using T = FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist, Arch, FloatKV, PerTokenScale>;
    const int grid_rows = Worklist == flashmla::dsa::hs64::Hs64Worklist::Single
        ? params.b : kFusedHs64PackedSmCount<Arch>;
    launch_fused_hs64_wi<T>(params, grid_rows, stream);
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
    // An absent extra stream must also disable the generic loader's lookahead.
    params.extra_topk = p.extra_kv ? p.extra_topk : -1;
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
template<int Arch, bool FloatKV, bool PerTokenScale, bool RowUnits = (Arch == 89)>
struct FusedHs64DirectTraits : FusedHs64Traits<false,
    flashmla::dsa::hs64::Hs64Traits<512, true, Arch, true, true>, true, false, PerTokenScale> {
    using Base = FusedHs64Traits<false,
        flashmla::dsa::hs64::Hs64Traits<512, true, Arch, true, true>, true, false, PerTokenScale>;
    using SharedMemoryPlan = typename Base::SharedMemoryPlan;
    static_assert(!PerTokenScale || Arch == 89);
    static constexpr bool kFusedDirectQueries = true;
    static constexpr bool kUseRowUnits = RowUnits;
    static constexpr bool kV41FloatKV = FloatKV;
    // Consecutive owned queries alternate the existing two RoPE slots.
    static constexpr bool kCachePrefillRope = PerTokenScale;
    static constexpr bool kDeferPrefillSink = true;
    static_assert(Base::kPrefillSinkOffset + 64 * sizeof(float) <= offsetof(SharedMemoryPlan, smem_sP));
};
template<int Arch, bool FloatKV, bool PerTokenScale>
static void run_fused_hs64_direct_output(const OutputParams<PerTokenScale>& p, int grid_rows) {
    using T = FusedHs64DirectTraits<Arch, FloatKV, PerTokenScale>;
    static_assert(T::kQBarrierThreads == 512);
    const FusedHs64Params<PerTokenScale> params = make_fused_hs64_decode_params<true>(p);
    if constexpr (Arch == 89) {
        if (flashmla::dsa::sparse_decode_hs64_v41_cache_range_supported(
                p.num_blocks, p.page_block_size, 528, 8, p.stride_kv_s_kv) &&
            (!p.extra_kv || flashmla::dsa::sparse_decode_hs64_v41_cache_range_supported(
                p.extra_num_blocks, p.extra_page_block_size, 288, 4, p.stride_extra_kv_block))) {
            using Narrow = FusedHs64DirectTraits<Arch, FloatKV, PerTokenScale, false>;
            launch_fused_hs64_wi<Narrow>(params, grid_rows, p.stream);
            return;
        }
    }
    launch_fused_hs64_wi<T>(params, grid_rows, p.stream);
}

template<bool Norm, bool ExtraFP4, int Arch, bool FloatKV, bool PerTokenScale>
static void run_fused_hs64_packed_output(const OutputParams<PerTokenScale>& p) {
    const FusedHs64Params<PerTokenScale> params = make_fused_hs64_decode_params<ExtraFP4>(p);
    using flashmla::dsa::hs64::Hs64Worklist;
    if constexpr (Arch == 89) {
        if (p.topk == 256) {
            launch_fused_hs64_packed<Norm, ExtraFP4, 256, Hs64Worklist::Batch, Arch, FloatKV>(
                params, p.stream);
            return;
        }
    }
    constexpr int flat_waves = ExtraFP4 && !Norm && PerTokenScale ? 2 :
        (ExtraFP4 && (Norm || PerTokenScale) ? 6 : 8);
    if (p.s_q >= flat_waves * kFusedHs64PackedSmCount<Arch>) {
        launch_fused_hs64_packed<Norm, ExtraFP4, 128, Hs64Worklist::Flat, Arch, FloatKV>(
            params, p.stream);
    } else {
        launch_fused_hs64_packed<Norm, ExtraFP4, 128, Hs64Worklist::Single, Arch, FloatKV>(
            params, p.stream);
    }
}

void run_fused_norm_rope_attn_rope_cast_decode_hs64_direct(const Params& p, int grid_rows) {
    if (p.arch == 89) {
        if (p.per_token_scale)
            run_fused_hs64_direct_output<89, true>(OutputParams<true>{p}, grid_rows);
        else
            run_fused_hs64_direct_output<89, true>(OutputParams<false>{p}, grid_rows);
    } else if (p.v41_float_kv) {
        run_fused_hs64_direct_output<80, true>(OutputParams<false>{p}, grid_rows);
    } else {
        run_fused_hs64_direct_output<80, false>(OutputParams<false>{p}, grid_rows);
    }
}

template<bool Norm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed(const Params& p) {
    if constexpr (!Norm && ExtraFP4) {
        if (p.arch == 80) {
            if (p.v41_float_kv)
                run_fused_hs64_packed_output<Norm, ExtraFP4, 80, true>(OutputParams<false>{p});
            else
                run_fused_hs64_packed_output<Norm, ExtraFP4, 80, false>(OutputParams<false>{p});
            return;
        }
    }
    if (p.per_token_scale)
        run_fused_hs64_packed_output<Norm, ExtraFP4, 89, true>(OutputParams<true>{p});
    else
        run_fused_hs64_packed_output<Norm, ExtraFP4, 89, true>(OutputParams<false>{p});
}

template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<false, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<true, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<false, true>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<true, true>(const Params&);

} // namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn
