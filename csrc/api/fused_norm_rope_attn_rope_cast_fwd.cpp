#include "common.h"
#include "fused_norm_rope_attn_rope_cast_fwd.h"
#include "ppu/decode/sparse/sparse_decode_wg.h"
#include "ppu/sparse/fused_v41.h"
#include "ppu/sparse/fused_v41_projection.h"

void get_mla_metadata_func(Mla_metadata_params &params, hggcStream_t stream);
void get_mla_uniform_sparse_metadata_func(Mla_metadata_params &params, hggcStream_t stream);

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {
template<bool Norm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed(const Params& params);
void run_fused_norm_rope_attn_rope_cast_decode_hs64_direct(const Params& params, int grid_rows);
}

namespace {

namespace fused = flashmla::fused_norm_rope_attn_rope_cast_fwd;
using Params = fused::core_attn::Params;
using FwdMode = fused::core_attn::FwdMode;

at::Tensor allocate_scale_factor(
    int64_t rows, int64_t hidden_dim, int64_t gran, const at::Tensor& like,
    std::optional<int64_t> groups = std::nullopt) {
    TORCH_CHECK(hidden_dim % (gran * 4) == 0);
    const int64_t aligned_rows = (rows + 3) / 4 * 4;
    auto sf = torch::empty({groups.value_or(1), hidden_dim / (gran * 4), aligned_rows},
                           like.options().dtype(torch::kInt32));
    sf = sf.transpose(1, 2).narrow(1, 0, rows);
    return groups.has_value() ? sf : sf.squeeze(0);
}

at::Tensor allocate_output_scale_factor(const Params& p, const at::Tensor& like) {
    if (p.per_token_scale) {
        return torch::empty({p.s_q, p.n_wv_group, 1}, like.options().dtype(torch::kFloat32));
    }
    return allocate_scale_factor(p.s_q, p.wv_group_size * 512, 32, like,
                                 p.n_wv_group).transpose(0, 1);
}

Params make_fused_params(
    const at::Tensor& q, double sm_scale, int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length, bool enable_q_norm,
    double rms_norm_eps, const at::Tensor& token_positions,
    bool is_rope_neox_style, int64_t rope_dim, const at::Tensor& cos_sin_cache,
    int64_t n_wv_group, int64_t num_per_channels, bool use_tma_aligned_col_major_sf,
    bool round_sf, bool use_packed_ue8m0) {
    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);
    KU_CHECK_NDIM(token_positions, 1);
    KU_CHECK_NDIM(cos_sin_cache, 2);
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(token_positions);
    KU_CHECK_DEVICE(cos_sin_cache);
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(token_positions, torch::kInt32);
    KU_CHECK_DTYPE(cos_sin_cache, torch::kFloat32);
    TORCH_CHECK(q.size(1) == 8 || q.size(1) == 64 || q.size(1) == 128,
                "Only 8, 64 or 128 Q heads are supported");
    TORCH_CHECK(q.size(2) == 512 && d_v == 512, "Only V4.1 head dimensions 512/512 are supported");
    TORCH_CHECK(n_wv_group > 0 && q.size(1) % n_wv_group == 0);
    TORCH_CHECK(q.size(1) / n_wv_group == 8, "WV group size must be 8");
    TORCH_CHECK(!is_rope_neox_style && rope_dim == 64);
    TORCH_CHECK(num_per_channels == 32 || num_per_channels == 4096,
                "num_per_channels must be 32 (per-32 UE8M0) or 4096 (per-token FP32)");
    const bool per_token_scale = num_per_channels == 4096;
    if (per_token_scale) {
        TORCH_CHECK(!use_tma_aligned_col_major_sf && !round_sf && !use_packed_ue8m0,
                    "Per-token output requires row-major, unrounded FP32 scales");
    } else {
        TORCH_CHECK(use_tma_aligned_col_major_sf && round_sf && use_packed_ue8m0);
    }
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    TORCH_CHECK(q.stride(1) == 512, "Q must be contiguous within each token");
    KU_CHECK_SHAPE(attn_sink, q.size(1));
    KU_CHECK_SHAPE(topk_length, q.size(0));
    KU_CHECK_SHAPE(token_positions, q.size(0));
    KU_CHECK_SHAPE(cos_sin_cache, cos_sin_cache.size(0), rope_dim);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(token_positions);
    KU_CHECK_CONTIGUOUS(cos_sin_cache);
    const auto check_device = [&](const at::Tensor& tensor) {
        TORCH_CHECK(tensor.device() == q.device(), "All fused inputs must be on the Q device");
        return true;
    };
    _check_optional_tensor(attn_sink, check_device);
    _check_optional_tensor(topk_length, check_device);
    check_device(token_positions);
    check_device(cos_sin_cache);
    const auto [major, minor] = get_compute_capability(q.get_device());
    TORCH_CHECK(major == 8 && (minor == 0 || minor == 9), "Fused attention requires PPU SM80 or SM89");
    TORCH_CHECK(q.size(1) != 8 || minor == 9, "H8 fused attention requires PPU SM89");
    TORCH_CHECK(!per_token_scale || minor == 9,
                "Per-token FP32 output scales currently require PPU SM89");

    Params p{};
    p.arch = minor == 9 ? 89 : 80;
    p.s_q = int64_stride_to_int(q.size(0));
    p.h_q = int64_stride_to_int(q.size(1));
    p.sm_scale = static_cast<float>(sm_scale);
    p.sm_scale_log2 = static_cast<float>(sm_scale * LOG_2_E);
    p.enable_q_norm = enable_q_norm;
    p.per_token_scale = per_token_scale;
    p.rms_norm_eps = static_cast<float>(rms_norm_eps);
    p.q = q.data_ptr();
    p.topk_length = get_optional_tensor_ptr<int32_t>(topk_length);
    p.attn_sink = get_optional_tensor_ptr<float>(attn_sink);
    p.token_positions = token_positions.data_ptr<int32_t>();
    p.cos_sin_cache = cos_sin_cache.data_ptr<float>();
    p.stride_q_s_q = q.stride(0);
    p.n_wv_group = int64_stride_to_int(n_wv_group);
    p.wv_group_size = 8;
    p.stream = at::cuda::getCurrentCUDAStream().stream();
    return p;
}

int logical_sm_count();

template<FwdMode Mode, int Heads, bool EnableNorm>
void launch_fused(const Params& p) {
    if constexpr (Mode == FwdMode::Prefill) {
        if (p.arch == 89 && p.s_q < logical_sm_count() && p.topk == 128) {
            fused::core_attn::run_fused_norm_rope_attn_rope_cast_prefill_small<EnableNorm>(p);
            return;
        }
    }
    if constexpr (Mode == FwdMode::Prefill && Heads == 128) {
        if (p.arch == 89) {
            fused::core_attn::run_fused_norm_rope_attn_rope_cast_prefill_h128<EnableNorm>(p);
            return;
        }
    }
    if constexpr (Mode == FwdMode::Prefill && Heads == 64) {
        if (p.arch == 89 && p.topk >= 128 && p.topk % 128 == 0) {
            fused::core_attn::run_fused_norm_rope_attn_rope_cast_prefill_hs64<EnableNorm>(p);
            return;
        }
    }
    if constexpr (Mode == FwdMode::Decode) {
        if (p.extra_fp4) {
            fused::core_attn::run_fused_norm_rope_attn_rope_cast_fwd_kernel<Mode, Heads, EnableNorm, true>(p);
            return;
        }
    }
    fused::core_attn::run_fused_norm_rope_attn_rope_cast_fwd_kernel<Mode, Heads, EnableNorm, false>(p);
}

template<FwdMode Mode>
void dispatch_fused(const Params& p) {
    if (p.h_q == 64) {
        if (p.enable_q_norm) launch_fused<Mode, 64, true>(p);
        else launch_fused<Mode, 64, false>(p);
    } else {
        if (p.enable_q_norm) launch_fused<Mode, 128, true>(p);
        else launch_fused<Mode, 128, false>(p);
    }
}

bool is_quantized_cache(const at::Tensor& cache) {
    return cache.scalar_type() == torch::kFloat8_e4m3fn ||
           cache.scalar_type() == torch::kInt8 || cache.scalar_type() == torch::kUInt8;
}

int logical_sm_count() {
    const auto* props = at::cuda::getCurrentDeviceProperties();
    return std::string(props->name).find("810E") != std::string::npos ? 20 : props->multiProcessorCount;
}

template<int Heads, bool EnableNorm, bool ExtraFP4>
void launch_fused_decode_splitkv(const Params& raw, const at::Tensor& q) {
    Params p = raw;
    const int sm_count = logical_sm_count();
    if constexpr (Heads == 8) {
        if (p.s_q >= 1 && p.topk == 128 && p.extra_topk == 512
            && !p.topk_length && !p.extra_topk_length) {
            // Complete-query CTAs avoid metadata and combine launches. Bound
            // the resident grid by the query count so tiny batches stay nonempty.
            const fused::core_attn::SplitKVParams direct{
                nullptr, nullptr, std::min(p.s_q, 2 * sm_count), nullptr, nullptr};
            fused::core_attn::run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<
                Heads, EnableNorm, ExtraFP4>(p, direct);
            return;
        }
    }
    if constexpr (Heads == 64) {
        constexpr int extra_bytes = ExtraFP4 ? 288 : 528;
        const auto aligned = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
        if (p.arch == 89 && sm_count == 39 && p.s_q >= sm_count && p.s_q <= 65535
            && (p.topk == 128 || p.topk == 256) && p.page_block_size == p.topk
            && p.stride_kv_s_kv == int64_t(p.page_block_size) * 528
            && p.extra_topk == 512 && p.extra_page_block_size == 64
            && p.stride_extra_kv_block == 64 * extra_bytes
            && flashmla::dsa::sparse_decode_hs64_v41_cache_range_supported(
                p.num_blocks, p.page_block_size, 528, 8, p.stride_kv_s_kv)
            && flashmla::dsa::sparse_decode_hs64_v41_cache_range_supported(
                p.extra_num_blocks, p.extra_page_block_size, extra_bytes,
                ExtraFP4 ? 4 : 8, p.stride_extra_kv_block)
            && aligned(p.q) && p.stride_q_s_q % 8 == 0 && aligned(p.kv) && aligned(p.extra_kv)) {
            // Packed kernels clamp both length arrays before using them.
            fused::core_attn::run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<EnableNorm, ExtraFP4>(p);
            return;
        }
    }
    if constexpr (Heads == 64 && !EnableNorm && ExtraFP4) {
        const auto aligned = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
        const auto pow2 = [](int size) { return size > 0 && (size & (size - 1)) == 0; };
        if (p.arch == 89 && sm_count == 39 && p.s_q >= sm_count && p.s_q <= 65535
            && p.topk == 128 && p.extra_topk == 512 && !p.topk_length && !p.extra_topk_length
            && pow2(p.page_block_size) && pow2(p.extra_page_block_size)
            && flashmla::dsa::sparse_decode_hs64_v41_cache_range_supported(
                p.num_blocks, p.page_block_size, 528, 8, p.stride_kv_s_kv)
            && flashmla::dsa::sparse_decode_hs64_v41_cache_range_supported(
                p.extra_num_blocks, p.extra_page_block_size, 288, 4, p.stride_extra_kv_block)
            && aligned(p.q) && p.stride_q_s_q % 8 == 0 && aligned(p.kv) && aligned(p.extra_kv)) {
            // Reuse native H64 attention with complete-query tasks and generic
            // cache addressing. The packed layout gate above keeps priority.
            fused::core_attn::run_fused_norm_rope_attn_rope_cast_decode_hs64_direct(p, sm_count);
            return;
        }
    }
    at::Tensor main_length, extra_length;
    if (p.topk_length) {
        main_length = torch::from_blob(const_cast<int*>(p.topk_length), {p.s_q},
                                      q.options().dtype(torch::kInt32)).clamp(0, p.topk);
        p.topk_length = main_length.data_ptr<int>();
    }
    if (p.extra_topk_length) {
        extra_length = torch::from_blob(const_cast<int*>(p.extra_topk_length), {p.s_q},
                                       q.options().dtype(torch::kInt32)).clamp(0, p.extra_topk);
        p.extra_topk_length = extra_length.data_ptr<int>();
    }
    const int num_sm_parts = std::min(get_num_sm_parts(Heads, 1, p.s_q, true), sm_count);
    const auto int_opts = q.options().dtype(torch::kInt32);
    auto metadata = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, int_opts);
    auto num_splits = torch::empty({p.s_q + 1}, int_opts);
    Mla_metadata_params meta{};
    meta.tile_scheduler_metadata_ptr = metadata.data_ptr<int>();
    meta.num_splits_ptr = num_splits.data_ptr<int>();
    meta.batch_size = p.s_q;
    meta.block_size_n = 64;
    meta.fixed_overhead_num_blocks = 5;
    meta.num_sm_parts = num_sm_parts;
    meta.topk = p.topk;
    meta.extra_topk = p.extra_topk;
    meta.topk_length = const_cast<int*>(p.topk_length);
    meta.extra_topk_length = const_cast<int*>(p.extra_topk_length);
    if (Heads == 64 && p.arch == 89 && p.s_q >= sm_count && p.topk == 128 && p.extra_topk == 512
        && !p.topk_length && !p.extra_topk_length) {
        get_mla_uniform_sparse_metadata_func(meta, p.stream);
    } else {
        get_mla_metadata_func(meta, p.stream);
    }
    const auto f32_opts = q.options().dtype(torch::kFloat32);
    auto lse_accum = torch::empty({p.s_q + num_sm_parts, 1, Heads}, f32_opts);
    auto o_accum = torch::empty({p.s_q + num_sm_parts, 1, Heads, 512}, f32_opts);
    const fused::core_attn::SplitKVParams splitkv{
        metadata.data_ptr<int>(), num_splits.data_ptr<int>(), num_sm_parts,
        lse_accum.data_ptr<float>(), o_accum.data_ptr<float>()};
    if constexpr (Heads == 128 && ExtraFP4) {
        if (p.arch == 89 && p.s_q >= sm_count
            && p.topk == 128 && p.extra_topk >= 128 && p.extra_topk % 64 == 0
            && p.topk_length == nullptr && p.extra_topk_length == nullptr) {
            if (fused::core_attn::run_fused_norm_rope_attn_rope_cast_decode_native_wi<EnableNorm>(p, splitkv)) return;
        }
    }
    fused::core_attn::run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<Heads, EnableNorm, ExtraFP4>(p, splitkv);
}

template<int Heads, bool EnableNorm>
void launch_fused_decode(const Params& p, const at::Tensor& q) {
    if constexpr (Heads == 128) {
        if (p.s_q >= logical_sm_count()) {
            const bool uniform_topk = !p.topk_length && !p.extra_topk_length;
            const bool long_extra = p.topk == 64 && p.extra_topk == 2048;
            const bool use_wi = p.arch == 89
                ? (p.extra_topk == 0 && p.topk <= 128) || long_extra
                : uniform_topk && !p.extra_fp4 && long_extra;
            if (use_wi) {
                launch_fused<FwdMode::Decode, Heads, EnableNorm>(p);
                return;
            }
        }
    }
    if (p.extra_fp4) launch_fused_decode_splitkv<Heads, EnableNorm, true>(p, q);
    else launch_fused_decode_splitkv<Heads, EnableNorm, false>(p, q);
}

void dispatch_fused_decode(const Params& p, const at::Tensor& q) {
    if (p.h_q == 8) {
        if (p.enable_q_norm) launch_fused_decode<8, true>(p, q);
        else launch_fused_decode<8, false>(p, q);
    } else if (p.h_q == 64) {
        if (p.enable_q_norm) launch_fused_decode<64, true>(p, q);
        else launch_fused_decode<64, false>(p, q);
    } else {
        if (p.enable_q_norm) launch_fused_decode<128, true>(p, q);
        else launch_fused_decode<128, false>(p, q);
    }
}

}

std::vector<at::Tensor> fused_norm_rope_attn_rope_cast_fwd(
    const at::Tensor& q, const at::Tensor& kv, const at::Tensor& indices,
    double sm_scale, int64_t d_v, const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length, bool enable_q_norm,
    double rms_norm_eps, const at::Tensor& token_positions, bool is_rope_neox_style,
    int64_t rope_dim, const at::Tensor& cos_sin_cache, int64_t n_wv_group,
    int64_t num_per_channels, bool use_tma_aligned_col_major_sf,
    bool round_sf, bool use_packed_ue8m0) {
    KU_CHECK_DEVICE(q);
    at::cuda::CUDAGuard guard(q.device());
    auto p = make_fused_params(q, sm_scale, d_v, attn_sink, topk_length, enable_q_norm,
        rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache,
        n_wv_group, num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    TORCH_CHECK(kv.device() == q.device() && indices.device() == q.device());
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_SHAPE(kv, kv.size(0), 1, 512);
    KU_CHECK_SHAPE(indices, p.s_q, 1, indices.size(2));
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    p.mode = FwdMode::Prefill;
    p.s_kv = int64_stride_to_int(kv.size(0));
    p.topk = int64_stride_to_int(indices.size(2));
    p.kv = kv.data_ptr();
    p.indices = indices.data_ptr<int32_t>();
    p.stride_kv_s_kv = kv.stride(0);
    p.stride_indices_s_q = indices.stride(0);

    auto out_fp8 = torch::empty({p.s_q, p.n_wv_group, p.wv_group_size * 512},
                                q.options().dtype(torch::kFloat8_e4m3fn));
    auto out_sf = allocate_output_scale_factor(p, q);
    auto max_logits = torch::empty({p.s_q, p.h_q}, q.options().dtype(torch::kFloat32));
    auto lse = torch::empty_like(max_logits);
    p.out_fp8 = static_cast<uint8_t*>(out_fp8.data_ptr());
    p.out_sf = reinterpret_cast<uint32_t*>(out_sf.data_ptr());
    p.max_logits = max_logits.data_ptr<float>();
    p.lse = lse.data_ptr<float>();
    p.stride_out_sf_token = out_sf.stride(0);
    p.stride_out_sf_group = out_sf.stride(1);
    p.stride_out_sf_word = out_sf.stride(2);
    if (p.h_q == 8) {
        if (p.enable_q_norm) fused::core_attn::run_fused_norm_rope_attn_rope_cast_prefill_small<true>(p);
        else fused::core_attn::run_fused_norm_rope_attn_rope_cast_prefill_small<false>(p);
    } else {
        dispatch_fused<FwdMode::Prefill>(p);
    }
    return {out_fp8, out_sf, max_logits, lse};
}

std::vector<at::Tensor> fused_norm_rope_attn_rope_cast_decode(
    const at::Tensor& q, const at::Tensor& kv, const at::Tensor& indices,
    double sm_scale, int64_t d_v, const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length, const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length, bool enable_q_norm,
    double rms_norm_eps, const at::Tensor& token_positions, bool is_rope_neox_style,
    int64_t rope_dim, const at::Tensor& cos_sin_cache, int64_t n_wv_group,
    int64_t num_per_channels, bool use_tma_aligned_col_major_sf,
    bool round_sf, bool use_packed_ue8m0) {
    KU_CHECK_DEVICE(q);
    at::cuda::CUDAGuard guard(q.device());
    auto p = make_fused_params(q, sm_scale, d_v, attn_sink, topk_length, enable_q_norm,
        rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache,
        n_wv_group, num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0);
    TORCH_CHECK(p.s_q > 0);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 2);
    KU_CHECK_NDIM(extra_kv, 4);
    KU_CHECK_NDIM(extra_indices, 2);
    KU_CHECK_NDIM(extra_topk_length, 1);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);
    TORCH_CHECK(kv.device() == q.device() && indices.device() == q.device());
    TORCH_CHECK(is_quantized_cache(kv), "KV must have dtype FP8 E4M3, int8 or uint8");
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);
    KU_CHECK_SHAPE(kv, kv.size(0), kv.size(1), 1, 528);
    KU_CHECK_SHAPE(indices, p.s_q, indices.size(1));
    KU_CHECK_SHAPE(extra_topk_length, p.s_q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);
    TORCH_CHECK(kv.size(1) > 0 && kv.stride(1) == 528);
    TORCH_CHECK(indices.size(1) > 0);
    TORCH_CHECK(extra_kv.has_value() == extra_indices.has_value());
    TORCH_CHECK(extra_kv.has_value() || !extra_topk_length.has_value());
    p.mode = FwdMode::Decode;
    p.num_blocks = int64_stride_to_int(kv.size(0));
    p.page_block_size = int64_stride_to_int(kv.size(1));
    p.s_kv = int64_stride_to_int(kv.size(0) * kv.size(1));
    p.topk = int64_stride_to_int(indices.size(1));
    p.kv = kv.data_ptr();
    p.indices = indices.data_ptr<int32_t>();
    p.stride_kv_s_kv = kv.stride(0);
    p.stride_indices_s_q = indices.stride(0);
    if (extra_kv.has_value()) {
        const auto& cache = *extra_kv;
        const auto& extra_idx = *extra_indices;
        TORCH_CHECK(cache.device() == q.device() && extra_idx.device() == q.device());
        TORCH_CHECK(is_quantized_cache(cache));
        TORCH_CHECK(cache.size(3) == 528 || cache.size(3) == 288);
        TORCH_CHECK(cache.size(1) > 0 && cache.stride(1) == cache.size(3));
        KU_CHECK_SHAPE(cache, cache.size(0), cache.size(1), 1, cache.size(3));
        KU_CHECK_SHAPE(extra_idx, p.s_q, extra_idx.size(1));
        if (extra_topk_length.has_value()) TORCH_CHECK(extra_topk_length->device() == q.device());
        p.extra_kv = cache.data_ptr();
        p.extra_indices = extra_idx.data_ptr<int32_t>();
        p.extra_topk_length = get_optional_tensor_ptr<int32_t>(extra_topk_length);
        p.extra_num_blocks = int64_stride_to_int(cache.size(0));
        p.extra_page_block_size = int64_stride_to_int(cache.size(1));
        p.extra_topk = int64_stride_to_int(extra_idx.size(1));
        p.stride_extra_kv_block = cache.stride(0);
        p.stride_extra_indices_s_q = extra_idx.stride(0);
        p.extra_fp4 = cache.size(3) == 288;
    }
    TORCH_CHECK(int64_t(p.topk) + p.extra_topk + 63 <= std::numeric_limits<int>::max(),
                "Combined topk exceeds the kernel index range");

    auto out_fp8 = torch::empty({p.s_q, p.n_wv_group, p.wv_group_size * 512},
                                q.options().dtype(torch::kFloat8_e4m3fn));
    auto out_sf = allocate_output_scale_factor(p, q);
    auto lse = torch::empty({p.s_q, p.h_q}, q.options().dtype(torch::kFloat32));
    p.out_fp8 = static_cast<uint8_t*>(out_fp8.data_ptr());
    p.out_sf = reinterpret_cast<uint32_t*>(out_sf.data_ptr());
    p.lse = lse.data_ptr<float>();
    p.stride_out_sf_token = out_sf.stride(0);
    p.stride_out_sf_group = out_sf.stride(1);
    p.stride_out_sf_word = out_sf.stride(2);
    dispatch_fused_decode(p, q);
    return {out_fp8, out_sf, lse};
}

std::vector<at::Tensor> permute_q_b_proj(
    const at::Tensor& q_b_proj, const at::Tensor& scale_factors, int64_t h_q, int64_t d_q) {
    KU_CHECK_NDIM(q_b_proj, 2);
    KU_CHECK_NDIM(scale_factors, 2);
    KU_CHECK_DEVICE(q_b_proj);
    KU_CHECK_DEVICE(scale_factors);
    KU_CHECK_DTYPE(q_b_proj, torch::kFloat8_e4m3fn);
    KU_CHECK_DTYPE(scale_factors, torch::kInt32);
    TORCH_CHECK(q_b_proj.device() == scale_factors.device());
    TORCH_CHECK(h_q > 0 && d_q > 0 && d_q % 16 == 0);
    TORCH_CHECK(scale_factors.size(1) > 0);
    const int64_t rank = q_b_proj.size(1);
    const int64_t gran = rank / (4 * scale_factors.size(1));
    TORCH_CHECK((gran == 32 || gran == 128) && rank % (gran * 4) == 0);
    KU_CHECK_SHAPE(q_b_proj, h_q * d_q, rank);
    KU_CHECK_SHAPE(scale_factors, h_q * d_q, rank / (gran * 4));
    KU_CHECK_LAST_DIM_CONTIGUOUS(q_b_proj);
    TORCH_CHECK(scale_factors.stride(0) == 1);
    at::cuda::CUDAGuard guard(q_b_proj.device());
    auto weight = torch::empty(q_b_proj.sizes(), q_b_proj.options());
    auto sf = allocate_scale_factor(h_q * d_q, rank, gran, q_b_proj);
    fused::permute_q_b_proj::Params p{
        static_cast<uint32_t>(h_q), static_cast<uint32_t>(d_q),
        static_cast<uint32_t>(rank), static_cast<uint32_t>(gran),
        static_cast<fused::permute_q_b_proj::fp8_e4m3*>(q_b_proj.data_ptr()),
        static_cast<uint64_t>(q_b_proj.stride(0)), scale_factors.data_ptr<int32_t>(),
        static_cast<uint64_t>(scale_factors.stride(1)),
        static_cast<fused::permute_q_b_proj::fp8_e4m3*>(weight.data_ptr()),
        static_cast<uint64_t>(weight.stride(0)), sf.data_ptr<int32_t>(),
        static_cast<uint64_t>(sf.stride(1)), at::cuda::getCurrentCUDAStream().stream()};
    fused::permute_q_b_proj::run_permute_q_b_proj_kernel(p);
    return {weight, sf};
}

std::vector<at::Tensor> permute_wv_proj(
    const at::Tensor& wv_proj, const at::Tensor& scale_factors,
    int64_t wv_group_size, int64_t d_o) {
    KU_CHECK_NDIM(wv_proj, 3);
    KU_CHECK_NDIM(scale_factors, 3);
    KU_CHECK_DEVICE(wv_proj);
    KU_CHECK_DEVICE(scale_factors);
    KU_CHECK_DTYPE(wv_proj, torch::kFloat8_e4m3fn);
    const bool channel_wise = scale_factors.scalar_type() == torch::kFloat32;
    TORCH_CHECK(channel_wise || scale_factors.scalar_type() == torch::kInt32,
                "WV scales must be per-channel FP32 or packed per-32 UE8M0");
    TORCH_CHECK(wv_proj.device() == scale_factors.device());
    TORCH_CHECK(wv_group_size > 0 && d_o == 512);
    TORCH_CHECK(scale_factors.size(2) > 0);
    const int64_t groups = wv_proj.size(0);
    const int64_t rows = wv_proj.size(1);
    const int64_t hidden = wv_group_size * d_o;
    // The K permutation always moves 32-element chunks, independently of the
    // quantization granularity. A per-channel scale is unchanged by this move.
    constexpr int64_t gran = 32;
    TORCH_CHECK(hidden % (gran * 4) == 0);
    KU_CHECK_SHAPE(wv_proj, groups, rows, hidden);
    KU_CHECK_SHAPE(scale_factors, groups, rows, channel_wise ? 1 : hidden / (gran * 4));
    KU_CHECK_LAST_DIM_CONTIGUOUS(wv_proj);
    TORCH_CHECK(channel_wise || scale_factors.stride(1) == 1);
    at::cuda::CUDAGuard guard(wv_proj.device());
    auto weight = torch::empty(wv_proj.sizes(), wv_proj.options());
    auto sf = channel_wise ? scale_factors
                          : allocate_scale_factor(rows, hidden, 32, wv_proj, groups);
    fused::permute_wv_proj::Params p{
        static_cast<uint32_t>(d_o), static_cast<uint32_t>(gran),
        static_cast<uint32_t>(wv_group_size), static_cast<uint32_t>(groups),
        static_cast<uint32_t>(rows),
        static_cast<fused::permute_wv_proj::fp8_e4m3*>(wv_proj.data_ptr()),
        static_cast<uint64_t>(wv_proj.stride(0)), static_cast<uint64_t>(wv_proj.stride(1)),
        channel_wise ? nullptr : scale_factors.data_ptr<int32_t>(),
        static_cast<uint64_t>(scale_factors.stride(0)),
        static_cast<uint64_t>(scale_factors.stride(2)),
        static_cast<fused::permute_wv_proj::fp8_e4m3*>(weight.data_ptr()),
        static_cast<uint64_t>(weight.stride(0)), static_cast<uint64_t>(weight.stride(1)),
        channel_wise ? nullptr : sf.data_ptr<int32_t>(), static_cast<uint64_t>(sf.stride(0)),
        static_cast<uint64_t>(sf.stride(2)), at::cuda::getCurrentCUDAStream().stream()};
    fused::permute_wv_proj::run_permute_wv_proj_kernel(p);
    return {weight, sf};
}
