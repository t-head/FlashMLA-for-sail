#pragma once
#include "common.h"
#include "sparse_native_fp8_config.h"
#include <cmath>
#include <c10/cuda/CUDAException.h>

namespace flashmla::sparse_native_fp8 {
inline void check_alignment(const at::Tensor &tensor, const char *name, int bytes = 16) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % bytes == 0,
        "native sparse FP8: ", name, " must be ", bytes, "-byte aligned");
}

inline void check_output_alignment(const std::optional<at::Tensor> &out) {
    // A contiguous view can still start at an unaligned storage offset. Use
    // one contract for scalar-pair stores and the split-KV merge's uint4 stores.
    if (out.has_value())
        check_alignment(out.value(), "out");
}
} // namespace flashmla::sparse_native_fp8

// ---------------------------------------------------------------------------
// Native FP8 (q8 x kv8) sparse prefill. Distinct from sparse_prefill_fwd's
// is_fp8_kv path, which dequantises the FP8 KV during the gather and still runs
// the BF16 MMA. This one keeps both operands in e4m3 and uses the PPU FP8 Tensor Cell
// path end to end.
//
// Both scales are per-tensor and are folded into the softmax exponent, so no
// per-token scale tensor is needed and the dequantisation costs nothing:
//   qk_scale_log2 = sm_scale * q_scale * kv_scale * log2(e)
//   out_scale     = kv_scale                       (V's dequant, applied in the epilogue)
//
// Head tiles, Q layout and sparse width select an internal specialization.
// Invalid indices are masked; BF16 KV is converted once into a temporary E4M3
// tensor while the caller's persistent cache retains its original dtype.
// ---------------------------------------------------------------------------
void run_sparse_prefill_fp8_fwd(const void *q, const void *kv, const int *indices, void *out,
    float *max_logits, float *lse, const float *attn_sink, const int *topk_length,
    float qk_scale_log2, float out_scale, int d_qk, int s_q, int n_tiles, int s_kv, int h_q,
    bool q_is_bf16, bool trusted_full_topk, hggcStream_t stream, const float *out_rope_freqs,
    const void *out_rope_positions, void *out_fp8, float *out_fp8_scale, int out_rope_dim,
    bool out_positions_i32, int out_total_tokens, int out_token_offset, bool static_out_scale,
    int q_stride_t, int q_stride_h);
void prepare_sparse_prefill_fp8_kv(const void *, void *, int, int, int64_t, hggcStream_t);
void prepare_sparse_prefill_index_bounds(const int *, int *, int, int, int, hggcStream_t);

void run_native_fp8_attention_decode(const void *, const void *, const int *, void *, float *,
    const float *, const int *, float, float, int, int, int, int, int, bool, int, int, int, void *,
    float *, float *, bool, int, int, hggcStream_t, int, float * = nullptr);

std::vector<at::Tensor> sparse_prefill_fp8_fwd(
    const at::Tensor &q,       // [s_q, h_q, d_qk=512|576] bf16/e4m3
    const at::Tensor &kv,      // [s_kv, h_kv=1, d_qk=512|576] bf16/e4m3
    const at::Tensor &indices, // [s_q, h_kv=1, topk] int32
    double sm_scale, double q_scale, double kv_scale, int64_t d_v,
    const std::optional<at::Tensor> &attn_sink, const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &out_opt, bool compute_aux, bool trusted_full_topk,
    const std::optional<at::Tensor> &out_rope_freqs_cis,
    const std::optional<at::Tensor> &out_rope_positions, const std::optional<at::Tensor> &out_fp8,
    const std::optional<at::Tensor> &out_fp8_scale, int64_t out_quant_heads_per_group,
    int64_t out_token_offset, double out_static_scale) {
    CHECK_DEVICE(q);
    TORCH_CHECK(std::isfinite(sm_scale) && sm_scale > 0.0 && std::isfinite(q_scale) &&
                    q_scale > 0.0 && std::isfinite(kv_scale) && kv_scale > 0.0,
        "native sparse FP8: sm_scale, q_scale and kv_scale must be finite and positive");
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(cc_major == 8 && cc_minor >= 9,
        "sparse_prefill_fp8_fwd requires sm_89 (PPU ZW-M890P); got sm_", cc_major, cc_minor);
    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3,
        "sparse_prefill_fp8_fwd: q, kv and indices must be 3D");
    TORCH_CHECK(kv.device() == q.device() && indices.device() == q.device(),
        "sparse_prefill_fp8_fwd: all tensors must be on the Q device");
    for (const auto &tensor : {attn_sink, topk_length, out_opt, out_rope_freqs_cis,
             out_rope_positions, out_fp8, out_fp8_scale}) {
        TORCH_CHECK(!tensor.has_value() || tensor->device() == q.device(),
            "sparse_prefill_fp8_fwd: all tensors must be on the Q device");
    }

    const bool q_is_bf16 = q.dtype() == torch::kBFloat16;
    TORCH_CHECK(q_is_bf16 || q.dtype() == torch::kFloat8_e4m3fn || q.dtype() == torch::kUInt8,
        "sparse_prefill_fp8_fwd: q must be bf16, float8_e4m3fn, or uint8-packed e4m3");
    const bool kv_is_bf16 = kv.dtype() == torch::kBFloat16;
    TORCH_CHECK(kv_is_bf16 || kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kUInt8,
        "sparse_prefill_fp8_fwd: kv must be bf16, float8_e4m3fn, or uint8-packed e4m3");
    TORCH_CHECK(indices.dtype() == torch::kInt32);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DEVICE(out_opt);
    KU_CHECK_DTYPE(out_opt, torch::kBFloat16);
    KU_CHECK_CONTIGUOUS(out_opt);
    flashmla::sparse_native_fp8::check_output_alignment(out_opt);
    KU_CHECK_DEVICE(out_rope_freqs_cis);
    KU_CHECK_DEVICE(out_rope_positions);
    KU_CHECK_DEVICE(out_fp8);
    KU_CHECK_DEVICE(out_fp8_scale);
    KU_CHECK_DTYPE(out_rope_freqs_cis, torch::kFloat32);
    KU_CHECK_DTYPE(out_fp8, torch::kFloat8_e4m3fn);
    KU_CHECK_DTYPE(out_fp8_scale, torch::kFloat32);
    KU_CHECK_CONTIGUOUS(out_fp8_scale);
    KU_CHECK_CONTIGUOUS(out_rope_freqs_cis);
    KU_CHECK_CONTIGUOUS(out_rope_positions);
    KU_CHECK_CONTIGUOUS(out_fp8);

    const int s_q = q.size(0);
    const int h_q = q.size(1);
    const int d_qk = q.size(2);
    const int s_kv = kv.size(0);
    const int h_kv = kv.size(1);
    const int topk = indices.size(2);

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);
    TORCH_CHECK(h_q > 0 && h_q % 8 == 0,
        "sparse_prefill_fp8_fwd: h_q must be a positive multiple of 8; got ", h_q);
    TORCH_CHECK(h_kv == 1, "sparse_prefill_fp8_fwd: h_kv must be 1; got ", h_kv);
    TORCH_CHECK(
        d_qk == 512 || d_qk == 576, "sparse_prefill_fp8_fwd: d_qk must be 512 or 576; got ", d_qk);
    TORCH_CHECK(d_v == 512, "sparse_prefill_fp8_fwd: d_v must be 512; got ", d_v);
    TORCH_CHECK(s_kv > 0, "sparse_prefill_fp8_fwd: kv must contain at least one row");
    TORCH_CHECK(topk % 64 == 0 && topk > 0 && topk <= 2176,
        "sparse_prefill_fp8_fwd: topk must be a positive multiple of 64 and <= 2176; got ", topk);
    TORCH_CHECK(q.stride(2) == 1 && q.stride(0) >= d_qk && q.stride(1) >= d_qk &&
                    q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 &&
                    q.stride(0) <= std::numeric_limits<int>::max() &&
                    q.stride(1) <= std::numeric_limits<int>::max() &&
                    reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
        "sparse_prefill_fp8_fwd: Q rows must be dense and 16-byte aligned");
    if (!q_is_bf16)
        CHECK_CONTIGUOUS(q);
    if (kv_is_bf16) {
        TORCH_CHECK(kv.stride(2) == 1 && kv.stride(0) >= d_qk && kv.stride(0) % 2 == 0 &&
                        reinterpret_cast<uintptr_t>(kv.data_ptr()) % 4 == 0,
            "sparse_prefill_fp8_fwd: BF16 KV rows must be dense and 4-byte aligned");
    } else {
        CHECK_CONTIGUOUS(kv);
        flashmla::sparse_native_fp8::check_alignment(kv, "KV");
    }
    CHECK_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(topk_length);

    const bool fused_epilogue = out_fp8.has_value();
    const bool compact_tile =
        h_q % 64 != 0 || topk > 2048 || q.stride(0) != h_q * d_qk || q.stride(1) != d_qk;
    TORCH_CHECK(!fused_epilogue || !compact_tile,
        "sparse_prefill_fp8_fwd: fused output requires contiguous 64-head tiles and topk <= 2048");
    TORCH_CHECK(fused_epilogue == out_fp8_scale.has_value(),
        "sparse_prefill_fp8_fwd: out_fp8 and out_scale must be provided together");
    TORCH_CHECK(
        !fused_epilogue || (out_rope_freqs_cis.has_value() && out_rope_positions.has_value()),
        "sparse_prefill_fp8_fwd: fused output needs rope freqs and positions");
    TORCH_CHECK(!fused_epilogue || out_quant_heads_per_group == 8,
        "sparse_prefill_fp8_fwd: fused output supports 8 heads per group");
    TORCH_CHECK(!fused_epilogue || d_qk == 512,
        "sparse_prefill_fp8_fwd: fused output is specialized for DeepSeek-V4 DQK=512");
    TORCH_CHECK(!fused_epilogue || h_q % out_quant_heads_per_group == 0,
        "sparse_prefill_fp8_fwd: h_q must be divisible by heads per group");
    TORCH_CHECK(!fused_epilogue || !compute_aux,
        "sparse_prefill_fp8_fwd: fused output requires compute_aux=False");
    TORCH_CHECK(!fused_epilogue || !out_opt.has_value(),
        "sparse_prefill_fp8_fwd: fused output requires out=None; consume out_fp8 and out_scale");
    TORCH_CHECK(out_static_scale == 0.0 || out_static_scale == 1.0 / 16.0,
        "sparse_prefill_fp8_fwd: out_static_scale must be None or 1/16");
    TORCH_CHECK(out_static_scale == 0.0 || fused_epilogue,
        "sparse_prefill_fp8_fwd: out_static_scale requires fused output");
    if (out_rope_positions.has_value()) {
        TORCH_CHECK(out_rope_positions->scalar_type() == torch::kInt32 ||
                        out_rope_positions->scalar_type() == torch::kInt64,
            "sparse_prefill_fp8_fwd: positions must be int32 or int64");
    }
    if (fused_epilogue) {
        const int total_tokens = out_fp8->size(1);
        TORCH_CHECK(out_rope_freqs_cis->dim() == 2 && out_rope_freqs_cis->size(1) == 64,
            "sparse_prefill_fp8_fwd: fused output requires [max_pos,64] RoPE freqs");
        TORCH_CHECK(
            out_rope_positions->dim() == 1 && out_rope_positions->size(0) >= out_token_offset + s_q,
            "sparse_prefill_fp8_fwd: positions do not cover the output token range");
        CHECK_SHAPE(out_fp8.value(), h_q / 8, total_tokens, 8 * (int)d_v);
        CHECK_SHAPE(out_fp8_scale.value(), h_q / 8, total_tokens);
        flashmla::sparse_native_fp8::check_alignment(out_fp8.value(), "out_fp8", 2);
        TORCH_CHECK(out_token_offset >= 0 && out_token_offset + s_q <= total_tokens,
            "sparse_prefill_fp8_fwd: fused output token range is out of bounds");
    }

    if (out_opt.has_value())
        CHECK_SHAPE(out_opt.value(), s_q, h_q, (int)d_v);
    auto opts = q.options();
    // The fused epilogue writes only out_fp8/out_fp8_scale; no BF16 output is materialized.
    at::Tensor out = fused_epilogue ? torch::empty({0}, opts.dtype(torch::kBFloat16))
                    : out_opt.has_value() ? out_opt.value()
                                         : torch::empty({s_q, h_q, (int)d_v}, opts.dtype(torch::kBFloat16));
    at::Tensor max_logits = compute_aux ? torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32))
                                        : torch::empty({0}, opts.dtype(torch::kFloat32));
    at::Tensor lse = compute_aux ? torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32))
                                 : torch::empty({0}, opts.dtype(torch::kFloat32));

    // log2(e): the kernel's softmax works in base 2 (exp2 is the only cheap form).
    const float qk_scale_log2 = (float)(sm_scale * q_scale * kv_scale * 1.4426950408889634);
    auto stream = at::cuda::getCurrentCUDAStream();
    at::Tensor kv_fp8 = kv;
    auto lengths = topk_length;
    if (kv_is_bf16 && s_q > 0) {
        kv_fp8 = torch::empty({s_kv, h_kv, d_qk}, kv.options().dtype(torch::kFloat8_e4m3fn));
        prepare_sparse_prefill_fp8_kv(
            kv.data_ptr(), kv_fp8.data_ptr(), s_kv, d_qk, kv.stride(0), stream);
        if (!lengths.has_value()) {
            lengths = torch::empty({s_q}, indices.options());
            prepare_sparse_prefill_index_bounds(
                indices.data_ptr<int>(), lengths->data_ptr<int>(), s_q, topk, s_kv, stream);
        }
    }
    // Small full-head grids leave most CUs idle while scanning the complete
    // sparse list. Reuse decode only when at least four key partitions can
    // amortize its merge; output-only and statistics callers share this route.
    if (s_q > 0 && d_qk == 512 && h_q % 64 == 0 && !q_is_bf16 && !kv_is_bf16 &&
        !fused_epilogue) {
        const auto *properties = at::cuda::getCurrentDeviceProperties();
        const int cus = properties->multiProcessorCount;
        const int splits = flashmla::sparse_native_fp8::choose_splits(s_q, h_q, topk, cus);
        if (splits >= 4) {
            auto partial_out = torch::empty({s_q, splits, h_q, (int)d_v}, out.options());
            auto partial_lse = torch::empty({s_q, splits, h_q}, lse.options());
            auto partial_max = compute_aux
                ? torch::empty({s_q, splits, h_q}, max_logits.options()) : max_logits;
            run_native_fp8_attention_decode(q.data_ptr(), kv_fp8.data_ptr(), indices.data_ptr<int>(),
                out.data_ptr(), compute_aux ? lse.data_ptr<float>() : nullptr,
                get_optional_tensor_ptr<float>(attn_sink), get_optional_tensor_ptr<int>(lengths),
                qk_scale_log2, float(kv_scale), d_qk, s_q, topk, s_kv, h_q, false,
                q.stride(0), q.stride(1), splits, partial_out.data_ptr(),
                partial_lse.data_ptr<float>(), compute_aux ? max_logits.data_ptr<float>() : nullptr,
                flashmla::sparse_native_fp8::prefer_two_stages(s_q, splits, h_q, cus), cus,
                properties->l2CacheSize, stream, 0, compute_aux ? partial_max.data_ptr<float>() : nullptr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return {out, max_logits, lse};
        }
    }
    run_sparse_prefill_fp8_fwd(q.data_ptr(), kv_fp8.data_ptr(), indices.data_ptr<int>(),
        out.data_ptr(), compute_aux ? max_logits.data_ptr<float>() : nullptr,
        compute_aux ? lse.data_ptr<float>() : nullptr, get_optional_tensor_ptr<float>(attn_sink),
        get_optional_tensor_ptr<int>(lengths), qk_scale_log2, (float)kv_scale, d_qk, s_q, topk / 64,
        s_kv, h_q, q_is_bf16, trusted_full_topk, stream,
        out_rope_freqs_cis.has_value() ? out_rope_freqs_cis->data_ptr<float>() : nullptr,
        out_rope_positions.has_value() ? out_rope_positions->data_ptr() : nullptr,
        out_fp8.has_value() ? out_fp8->data_ptr() : nullptr,
        out_fp8_scale.has_value() ? out_fp8_scale->data_ptr<float>() : nullptr,
        out_rope_freqs_cis.has_value() ? (int)out_rope_freqs_cis->size(-1) : 0,
        out_rope_positions.has_value() && out_rope_positions->scalar_type() == torch::kInt32,
        out_fp8.has_value() ? (int)out_fp8->size(1) : 0, (int)out_token_offset,
        out_static_scale != 0.0, (int)q.stride(0), (int)q.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, max_logits, lse};
}

// Native FP8 sparse decode: internal query/head tiling and aligned key splits.
void run_native_fp8_attention_merge(
    const void *, const float *, void *, float *, const float *, int, int, int, hggcStream_t, int = 0, const float * = nullptr, float * = nullptr);
void run_native_lse_natural(const float *, float *, int, int, int, hggcStream_t);

std::vector<at::Tensor> sparse_decode_fp8_fwd(const at::Tensor &q, // [n_tok, h_q, d_qk=512|576] bf16/e4m3
    const at::Tensor &kv,                                          // [s_kv, h_kv=1, d_qk] e4m3
    const at::Tensor &indices,                                     // [n_tok, h_kv=1, topk] int32
    double sm_scale, double q_scale, double kv_scale, int64_t d_v,
    const std::optional<at::Tensor> &attn_sink, const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &out_opt, int64_t lse_queries = 0) {
    CHECK_DEVICE(q);
    TORCH_CHECK(std::isfinite(sm_scale) && sm_scale > 0.0 && std::isfinite(q_scale) &&
                    q_scale > 0.0 && std::isfinite(kv_scale) && kv_scale > 0.0,
        "native sparse FP8: sm_scale, q_scale and kv_scale must be finite and positive");
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3,
        "sparse_decode_fp8_fwd: q, kv and indices must be 3D");
    TORCH_CHECK(kv.device() == q.device() && indices.device() == q.device(),
        "sparse_decode_fp8_fwd: all tensors must be on the Q device");
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(cc_major == 8 && cc_minor >= 9,
        "sparse_decode_fp8_fwd requires sm_89 (PPU ZW-M890P); got sm_", cc_major, cc_minor);
    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn || q.dtype() == torch::kUInt8 ||
                    q.dtype() == torch::kBFloat16,
        "sparse_decode_fp8_fwd: q must be bf16, float8_e4m3fn or uint8-packed e4m3");
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kUInt8,
        "sparse_decode_fp8_fwd: kv must be float8_e4m3fn or uint8-packed e4m3");
    TORCH_CHECK(indices.dtype() == torch::kInt32);

    const int n_tok = q.size(0);
    const int h_q = q.size(1);
    const int d_qk = q.size(2);
    const int s_kv = kv.size(0);
    const int h_kv = kv.size(1);
    const int topk = indices.size(2);
    TORCH_CHECK(lse_queries >= 0 && (!lse_queries ||
        n_tok % lse_queries == 0), "invalid paged LSE query count");
    CHECK_SHAPE(q, n_tok, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, n_tok, h_kv, topk);
    TORCH_CHECK(h_q > 0 && h_q % 8 == 0,
        "sparse_decode_fp8_fwd: h_q must be a positive multiple of 8; got ", h_q);
    TORCH_CHECK(h_kv == 1, "sparse_decode_fp8_fwd: h_kv must be 1; got ", h_kv);
    TORCH_CHECK(
        d_qk == 512 || d_qk == 576, "sparse_decode_fp8_fwd: d_qk must be 512 or 576; got ", d_qk);
    TORCH_CHECK(d_v == 512, "sparse_decode_fp8_fwd: d_v must be 512; got ", d_v);
    TORCH_CHECK(topk % 64 == 0 && topk > 0 && topk <= 2048,
        "sparse_decode_fp8_fwd: topk must be a positive multiple of 64 and <= 2048; got ", topk);
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(kv);
    CHECK_CONTIGUOUS(indices);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0,
        "native FP8 decode: Q and KV must be 16-byte aligned");
    TORCH_CHECK(s_kv > 0, "sparse_decode_fp8_fwd: kv must contain at least one row");
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_SHAPE(topk_length, n_tok);
    KU_CHECK_DEVICE(out_opt);
    KU_CHECK_DTYPE(out_opt, torch::kBFloat16);
    KU_CHECK_CONTIGUOUS(out_opt);
    flashmla::sparse_native_fp8::check_output_alignment(out_opt);
    KU_CHECK_SHAPE(out_opt, n_tok, h_q, 512);
    for (const auto &t : {attn_sink, topk_length, out_opt})
        TORCH_CHECK(!t.has_value() || t->device() == q.device(),
            "native FP8 decode: optional tensors must be on the Q device");
    auto output = out_opt.has_value()
                      ? out_opt.value()
                      : torch::empty({n_tok, h_q, 512}, q.options().dtype(torch::kBFloat16));
    auto lse = lse_queries
        ? torch::empty({n_tok / lse_queries, h_q, lse_queries}, q.options().dtype(torch::kFloat32))
        : torch::empty({n_tok, h_q}, q.options().dtype(torch::kFloat32));
    if (n_tok == 0)
        return {output, lse};
    const auto *properties = at::cuda::getCurrentDeviceProperties();
    const int cus = properties->multiProcessorCount;
    const int splits = flashmla::sparse_native_fp8::choose_splits(n_tok, h_q, topk, cus);
    auto part = splits > 1 ? torch::empty({n_tok, splits, h_q, 512}, output.options()) : output;
    auto plse = splits > 1 ? torch::empty({n_tok, splits, h_q}, lse.options()) : lse;

    run_native_fp8_attention_decode(q.data_ptr(), kv.data_ptr(), indices.data_ptr<int>(),
        output.data_ptr(), lse.data_ptr<float>(), get_optional_tensor_ptr<float>(attn_sink),
        get_optional_tensor_ptr<int>(topk_length),
        float(sm_scale * q_scale * kv_scale * 1.4426950408889634), float(kv_scale), d_qk, n_tok,
        topk, s_kv, h_q, q.dtype() == torch::kBFloat16, q.stride(0), q.stride(1), splits,
        part.data_ptr(), plse.data_ptr<float>(), nullptr,
        flashmla::sparse_native_fp8::prefer_two_stages(n_tok, splits, h_q, cus), cus,
        properties->l2CacheSize, at::cuda::getCurrentCUDAStream(), lse_queries);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, lse};
}

std::vector<at::Tensor> native_fp8_merge(const at::Tensor &part, const at::Tensor &plse,
    const std::optional<at::Tensor> &sink, const std::optional<at::Tensor> &out_opt) {
    CHECK_DEVICE(part);
    CHECK_DEVICE(plse);
    CHECK_CONTIGUOUS(part);
    CHECK_CONTIGUOUS(plse);
    TORCH_CHECK(part.dim() == 4 && part.size(3) == 512 && part.scalar_type() == torch::kBFloat16,
        "native merge expects BF16 [tokens,parts,heads,512]");
    flashmla::sparse_native_fp8::check_alignment(part, "part");
    at::cuda::CUDAGuard guard(part.device());
    const int n = part.size(0), s = part.size(1), h = part.size(2);
    TORCH_CHECK(
        s > 0 && h > 0 && plse.device() == part.device() && plse.scalar_type() == torch::kFloat32,
        "native merge expects same-device FP32 LSE and nonempty parts/heads");
    CHECK_SHAPE(plse, n, s, h);
    KU_CHECK_DEVICE(sink);
    KU_CHECK_DTYPE(sink, torch::kFloat32);
    KU_CHECK_CONTIGUOUS(sink);
    KU_CHECK_SHAPE(sink, h);
    KU_CHECK_DEVICE(out_opt);
    KU_CHECK_DTYPE(out_opt, torch::kBFloat16);
    KU_CHECK_CONTIGUOUS(out_opt);
    flashmla::sparse_native_fp8::check_output_alignment(out_opt);
    KU_CHECK_SHAPE(out_opt, n, h, 512);
    for (const auto &t : {sink, out_opt})
        TORCH_CHECK(!t.has_value() || t->device() == part.device(),
            "native merge tensors must share device");
    auto out = out_opt.has_value() ? out_opt.value() : torch::empty({n, h, 512}, part.options());
    auto lse = torch::empty({n, h}, plse.options());
    run_native_fp8_attention_merge(part.data_ptr(), plse.data_ptr<float>(), out.data_ptr(),
        lse.data_ptr<float>(), get_optional_tensor_ptr<float>(sink), n, s, h,
        at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}
at::Tensor native_fp8_lse_to_natural(const at::Tensor &lse, int64_t batch, int64_t queries) {
    CHECK_DEVICE(lse);
    CHECK_CONTIGUOUS(lse);
    TORCH_CHECK(lse.dim() == 2 && lse.scalar_type() == torch::kFloat32 && batch >= 0 &&
                    queries >= 0 && batch * queries == lse.size(0),
        "native LSE expects contiguous FP32 [batch*queries,heads]");
    at::cuda::CUDAGuard guard(lse.device());
    auto out = torch::empty({batch, lse.size(1), queries}, lse.options());
    run_native_lse_natural(lse.data_ptr<float>(), out.data_ptr<float>(), batch, queries,
        lse.size(1), at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void run_native_fp8_attention_extra(const void *, const void *, const int *, const void *, const int *,
    void *, float *, const float *, const int *, const int *, float, int, int, int, int, int, int,
    bool, int, int, int, void *, float *, float *, bool, hggcStream_t, int, int);

std::vector<at::Tensor> native_fp8_extra_fwd(const at::Tensor &q, const at::Tensor &kv,
    const at::Tensor &indices, const at::Tensor &extra, const at::Tensor &extra_indices,
    double sm_scale, const std::optional<at::Tensor> &sink,
    const std::optional<at::Tensor> &lengths, const std::optional<at::Tensor> &extra_lengths,
    const std::optional<at::Tensor> &out_opt, int64_t lse_queries = 0) {
    CHECK_DEVICE(q);
    at::cuda::CUDAGuard guard{q.device()};
    TORCH_CHECK(
        std::isfinite(sm_scale) && sm_scale > 0, "native extra: scale must be finite and positive");
    auto [major, minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(major == 8 && minor >= 9, "native extra requires PPU sm_89");
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && extra.dim() == 3 && indices.dim() == 3 &&
                    extra_indices.dim() == 3,
        "native extra: tensors must be 3D");
    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn || q.dtype() == torch::kUInt8 ||
                    q.dtype() == torch::kBFloat16,
        "native extra: Q must be BF16 or E4M3");
    const int n = q.size(0), h = q.size(1), d = q.size(2);
    TORCH_CHECK(lse_queries >= 0 && (!lse_queries || n % lse_queries == 0),
        "native extra: invalid paged LSE query count");
    const int w = indices.size(2), ew = extra_indices.size(2), rows = kv.size(0),
              erows = extra.size(0);
    TORCH_CHECK(
        d == 512 && h > 0 && h % 8 == 0, "native extra: requires D512 and heads divisible by 8");
    TORCH_CHECK(w > 0 && ew > 0 && w % 64 == 0 && ew % 64 == 0 && (int64_t)w + ew <= 2048,
        "native extra: widths must be positive multiples of 64, combined <=2048");
    TORCH_CHECK(rows > 0 && erows > 0 && (int64_t)rows + erows <= std::numeric_limits<int>::max(),
        "native extra: invalid KV row count");
    for (const auto &t : {q, kv, extra, indices, extra_indices}) {
        CHECK_DEVICE(t);
        CHECK_CONTIGUOUS(t);
        TORCH_CHECK(t.device() == q.device(), "native extra: device mismatch");
    }
    for (const auto &t : {kv, extra}) {
        TORCH_CHECK(t.dtype() == torch::kFloat8_e4m3fn || t.dtype() == torch::kUInt8,
            "native extra: KV must be E4M3");
        TORCH_CHECK(t.size(1) == 1 && t.size(2) == 512, "native extra: KV must be [rows,1,512]");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
            "native extra: KV must be aligned");
    }
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0, "native extra: Q must be aligned");
    for (const auto &t : {indices, extra_indices}) {
        TORCH_CHECK(t.dtype() == torch::kInt32 && t.size(0) == n && t.size(1) == 1,
            "native extra: indices must be [queries,1,width] int32");
    }
    KU_CHECK_DEVICE(sink);
    KU_CHECK_DTYPE(sink, torch::kFloat32);
    KU_CHECK_CONTIGUOUS(sink);
    KU_CHECK_SHAPE(sink, h);
    KU_CHECK_DEVICE(lengths);
    KU_CHECK_DTYPE(lengths, torch::kInt32);
    KU_CHECK_CONTIGUOUS(lengths);
    KU_CHECK_SHAPE(lengths, n);
    KU_CHECK_DEVICE(extra_lengths);
    KU_CHECK_DTYPE(extra_lengths, torch::kInt32);
    KU_CHECK_CONTIGUOUS(extra_lengths);
    KU_CHECK_SHAPE(extra_lengths, n);
    KU_CHECK_DEVICE(out_opt);
    KU_CHECK_DTYPE(out_opt, torch::kBFloat16);
    KU_CHECK_CONTIGUOUS(out_opt);
    flashmla::sparse_native_fp8::check_output_alignment(out_opt);
    KU_CHECK_SHAPE(out_opt, n, h, 512);
    for (const auto &t : {sink, lengths, extra_lengths, out_opt})
        TORCH_CHECK(
            !t.has_value() || t->device() == q.device(), "native extra: optional device mismatch");
    auto out = out_opt.has_value() ? out_opt.value()
                                   : torch::empty({n, h, 512}, q.options().dtype(torch::kBFloat16));
    auto lse = lse_queries
        ? torch::empty({n / lse_queries, h, lse_queries}, q.options().dtype(torch::kFloat32))
        : torch::empty({n, h}, q.options().dtype(torch::kFloat32));
    if (n == 0)
        return {out, lse};
    const int cus = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int splits = flashmla::sparse_native_fp8::choose_splits(n, h, w + ew, cus);
    auto part = splits > 1 ? torch::empty({n, splits, h, 512}, out.options()) : out;
    auto plse = splits > 1 ? torch::empty({n, splits, h}, lse.options()) : lse;

    run_native_fp8_attention_extra(q.data_ptr(), kv.data_ptr(), indices.data_ptr<int>(),
        extra.data_ptr(), extra_indices.data_ptr<int>(), out.data_ptr(), lse.data_ptr<float>(),
        get_optional_tensor_ptr<float>(sink), get_optional_tensor_ptr<int>(lengths),
        get_optional_tensor_ptr<int>(extra_lengths), float(sm_scale * 1.4426950408889634), n, w, ew,
        rows, erows, h, q.dtype() == torch::kBFloat16, q.stride(0), q.stride(1), splits,
        part.data_ptr(), plse.data_ptr<float>(), nullptr,
        flashmla::sparse_native_fp8::prefer_two_stages(n, splits, h, cus),
        at::cuda::getCurrentCUDAStream(), lse_queries, cus);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}

void run_native_fp8_attention_verify(const void *, const void *, const int *, void *, float *,
    const float *, const float *, const int *, float *, int *, unsigned *, int *, float, int, int,
    int, int, int, int, bool, bool, int, hggcStream_t, int);
void run_native_fp8_attention_window(const void *, const void *, const int *, const void *,
    const int *, void *, float *, const float *, const int *, const int *, int *, unsigned *,
    int *, float, int, int, int, int, int, int, int, int, hggcStream_t, int);
std::vector<at::Tensor> native_fp8_verify_fwd(const at::Tensor &q, const at::Tensor &kv,
    const at::Tensor &indices, int64_t queries_per_request, double sm_scale,
    const std::optional<at::Tensor> &sink, const std::optional<at::Tensor> &lengths,
    const std::optional<at::Tensor> &out_opt, bool shifted = false,
    const std::optional<at::Tensor> &extra = std::nullopt,
    const std::optional<at::Tensor> &extra_indices = std::nullopt,
    const std::optional<at::Tensor> &extra_lengths = std::nullopt, int64_t lse_queries = 0) {
    CHECK_DEVICE(q);
    at::cuda::CUDAGuard guard{q.device()};
    TORCH_CHECK(std::isfinite(sm_scale) && sm_scale > 0, "native verify: invalid scale");
    auto [major, minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(major == 8 && minor >= 9, "native verify requires sm_89");
    TORCH_CHECK(
        q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3, "native verify: requires 3D tensors");
    const int n = q.size(0), h = q.size(1), d = q.size(2), w = indices.size(2), rows = kv.size(0);
    TORCH_CHECK(
        (queries_per_request > 1 &&
         queries_per_request <= flashmla::sparse_native_fp8::MaxVerifyQueries) &&
            n % queries_per_request == 0,
        "native verify: invalid queries/request");
    const int qpr = queries_per_request, batch = n / qpr;
    TORCH_CHECK((d == 512 || (d == 576 && !shifted && q.dtype() != torch::kBFloat16)) &&
                    (h == 8 || h == 16), "native verify: unsupported heads/dim");
    TORCH_CHECK(
        w > 0 && w % 64 == 0 && w <= 2048 && rows > 0, "native verify: invalid width/KV rows");
    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn || q.dtype() == torch::kUInt8 ||
                    q.dtype() == torch::kBFloat16,
        "native verify: invalid Q dtype");
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kUInt8,
        "native verify: KV must be E4M3");
    for (const auto &t : {q, kv, indices}) {
        CHECK_DEVICE(t);
        CHECK_CONTIGUOUS(t);
        TORCH_CHECK(t.device() == q.device(), "native verify: device mismatch");
    }
    CHECK_SHAPE(kv, rows, 1, d);
    CHECK_SHAPE(indices, n, 1, w);
    TORCH_CHECK(indices.dtype() == torch::kInt32, "native verify: indices must be int32");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0,
        "native verify: Q/KV must be aligned");
    KU_CHECK_DEVICE(sink);
    KU_CHECK_DTYPE(sink, torch::kFloat32);
    KU_CHECK_CONTIGUOUS(sink);
    KU_CHECK_SHAPE(sink, h);
    KU_CHECK_DEVICE(lengths);
    KU_CHECK_DTYPE(lengths, torch::kInt32);
    KU_CHECK_CONTIGUOUS(lengths);
    KU_CHECK_SHAPE(lengths, n);
    KU_CHECK_DEVICE(out_opt);
    KU_CHECK_DTYPE(out_opt, torch::kBFloat16);
    KU_CHECK_CONTIGUOUS(out_opt);
    flashmla::sparse_native_fp8::check_output_alignment(out_opt);
    KU_CHECK_SHAPE(out_opt, n, h, 512);
    for (const auto &t : {sink, lengths, out_opt})
        TORCH_CHECK(
            !t.has_value() || t->device() == q.device(), "native verify: optional device mismatch");
    TORCH_CHECK(extra.has_value() == extra_indices.has_value(), "native verify: extra cache/indices required together");
    TORCH_CHECK(shifted || !extra.has_value(), "native verify: extra pool requires window union");
    TORCH_CHECK(!shifted || q.dtype() != torch::kBFloat16, "native verify: window union requires E4M3 Q");
    const int padded_width = shifted ? ((w + qpr - 1 + 63) / 64) * 64 : w;
    int ew = 0, erows = 0;
    if (extra.has_value()) {
        TORCH_CHECK(extra->dim() == 3 && extra_indices->dim() == 3, "native verify: extra tensors must be 3D");
        ew = extra_indices->size(2); erows = extra->size(0);
        TORCH_CHECK(ew > 0 && ew % 64 == 0 && erows > 0, "native verify: invalid extra shape");
        CHECK_SHAPE(extra.value(), erows, 1, 512);
        CHECK_SHAPE(extra_indices.value(), n, 1, ew);
        TORCH_CHECK(extra->dtype() == torch::kFloat8_e4m3fn && extra_indices->dtype() == torch::kInt32,
            "native verify: invalid extra dtype");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(extra->data_ptr()) % 16 == 0, "native verify: unaligned extra cache");
    }
    TORCH_CHECK(padded_width + ew <= 2048, "native verify: union exceeds supported width");
    TORCH_CHECK((int64_t)rows + erows <= INT_MAX, "native verify: too many KV rows");
    for (const auto &t : {extra, extra_indices, extra_lengths}) {
        if (t.has_value()) {
            CHECK_DEVICE(t.value()); CHECK_CONTIGUOUS(t.value());
            TORCH_CHECK(t->device() == q.device(), "native verify: extra device mismatch");
        }
    }
    KU_CHECK_DTYPE(extra_lengths, torch::kInt32);
    KU_CHECK_SHAPE(extra_lengths, n);
    auto out = out_opt.has_value() ? out_opt.value()
                                   : torch::empty({n, h, 512}, q.options().dtype(torch::kBFloat16));
    TORCH_CHECK(lse_queries == 0 || (q.dtype() != torch::kBFloat16 && lse_queries > 0 && n % lse_queries == 0),
        "native verify: invalid paged LSE query count");
    auto lse = lse_queries ? torch::empty({n / lse_queries, h, lse_queries}, q.options().dtype(torch::kFloat32))
                           : torch::empty({n, h}, q.options().dtype(torch::kFloat32));
    if (!n)
        return {out, lse};
    if (shifted) {
        auto united = torch::empty({batch, padded_width + ew}, indices.options());
        auto masks = torch::empty({batch, qpr, (padded_width + ew) / 32}, indices.options());
        auto errors = torch::empty({batch}, indices.options());
        run_native_fp8_attention_window(q.data_ptr(), kv.data_ptr(), indices.data_ptr<int>(),
            extra.has_value() ? extra->data_ptr() : nullptr, get_optional_tensor_ptr<int>(extra_indices),
            out.data_ptr(), lse.data_ptr<float>(), get_optional_tensor_ptr<float>(sink),
            get_optional_tensor_ptr<int>(lengths), get_optional_tensor_ptr<int>(extra_lengths),
            united.data_ptr<int>(), reinterpret_cast<unsigned *>(masks.data_ptr<int>()), errors.data_ptr<int>(),
            float(sm_scale * 1.4426950408889634), batch, qpr, h, w, padded_width, ew, rows, erows,
            at::cuda::getCurrentCUDAStream(), lse_queries);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {out, lse};
    }
    // One allocation retains all classifier outputs for the attention launches
    // and graph replay, without allocating three independent temporary tensors.
    const int64_t union_elements = int64_t(batch) * (padded_width + ew);
    const int64_t mask_elements = int64_t(batch) * qpr * ((padded_width + ew) / 32);
    auto workspace = torch::empty({union_elements + mask_elements + batch}, indices.options());
    auto *united = workspace.data_ptr<int>();
    auto *masks = reinterpret_cast<unsigned *>(united + union_elements);
    auto *errors = united + union_elements + mask_elements;
    auto folded_sink = sink.has_value() ? sink->repeat({qpr}) : at::Tensor();
    const auto *properties = at::cuda::getCurrentDeviceProperties();
    const int cus = properties->multiProcessorCount;
    run_native_fp8_attention_verify(q.data_ptr(), kv.data_ptr(), indices.data_ptr<int>(),
        out.data_ptr(), lse.data_ptr<float>(), get_optional_tensor_ptr<float>(sink),
        sink.has_value() ? folded_sink.data_ptr<float>() : nullptr,
        get_optional_tensor_ptr<int>(lengths), nullptr, united, masks, errors,
        float(sm_scale * 1.4426950408889634), d, batch, qpr, h, w, rows,
        q.dtype() == torch::kBFloat16, flashmla::sparse_native_fp8::prefer_two_stages(n, 1, h, cus),
        properties->l2CacheSize, at::cuda::getCurrentCUDAStream(), lse_queries);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}
