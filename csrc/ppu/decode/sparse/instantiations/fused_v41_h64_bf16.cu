#include "sparse/fused_v41_hs64.cuh"
#include "decode/sparse/fused_v41_splitkv.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist>
struct FusedHs64PackedTraits
    : FusedHs64Traits<Norm, std::conditional_t<ExtraFP4,
        flashmla::dsa::hs64::Hs64PackedDecode<MainCapacity, Worklist>,
        flashmla::dsa::hs64::Hs64PackedFp8ExtraWorklist<MainCapacity, Worklist>>,
        ExtraFP4 || (!Norm && MainCapacity == 128 &&
            Worklist != flashmla::dsa::hs64::Hs64Worklist::Batch)> {
    static constexpr bool kFp8LeaderPrefetch =
        !Norm && !ExtraFP4 && MainCapacity == 128 && Worklist == flashmla::dsa::hs64::Hs64Worklist::Flat;
};

}

namespace flashmla::dsa::hs64 {

template<bool Norm, bool ExtraFP4, int MainCapacity, Hs64Worklist Worklist>
inline constexpr int kPackedMain<fused_norm_rope_attn_rope_cast_fwd::core_attn::FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist>> = MainCapacity;

template<bool Norm, bool ExtraFP4, int MainCapacity, Hs64Worklist Worklist>
inline constexpr Hs64Worklist kWorklist<fused_norm_rope_attn_rope_cast_fwd::core_attn::FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist>> = Worklist;

}

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_decode_hs64_packed_kernel(__grid_constant__ const FusedHs64Params params) {
#if ACOMPUTE_VERSION >= 10500
    using T = FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist>;
    if constexpr (T::kFusedPackedFp4) {
        // The packed launcher flattens queries and fixes these capacities.
        FusedHs64Params fixed = params;
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

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist>
static void launch_fused_hs64_packed(const FusedHs64Params& params, hggcStream_t stream) {
    using T = FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &fused_norm_rope_attn_rope_cast_decode_hs64_packed_kernel<Norm, ExtraFP4, MainCapacity, Worklist>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const int grid_rows = Worklist == flashmla::dsa::hs64::Hs64Worklist::Single ? params.b : 39;
    kernel<<<dim3(1, 1, grid_rows), T::NUM_THREADS, bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool Norm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed(const Params& p) {
    FusedHs64Params params{};
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
    params.stride_out_sf_group = p.stride_out_sf_group;
    params.stride_out_sf_word = p.stride_out_sf_word;
    using flashmla::dsa::hs64::Hs64Worklist;
    if (p.topk == 256) {
        launch_fused_hs64_packed<Norm, ExtraFP4, 256, Hs64Worklist::Batch>(params, p.stream);
    } else if (p.s_q >= (Norm && ExtraFP4 ? 6 : 8) * 39) {
        launch_fused_hs64_packed<Norm, ExtraFP4, 128, Hs64Worklist::Flat>(params, p.stream);
    } else {
        launch_fused_hs64_packed<Norm, ExtraFP4, 128, Hs64Worklist::Single>(params, p.stream);
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
