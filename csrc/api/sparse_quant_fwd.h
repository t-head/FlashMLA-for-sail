/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cstdint>

#include "common.h"
#include "prefill/sparse/sparse_quant_prefill.h"
#include "prefill/sparse/sparse_quant_prefill_narrow.h"
#include "prefill/sparse/sparse_quant_prefill_gpu_checked_fp8.h"
#if defined(FLASHMLA_ENABLE_V40_PREFILL)
#include "prefill/sparse/sparse_quant_prefill_v40.h"
#endif

// Dispatch choice only: no new public validation, allocation, or API restriction.
static bool v41_quant_narrow32_tensor_span(const at::Tensor &tensor) {
    const __int128 limit = __int128(1) << 31;
    const __int128 element_bytes = tensor.element_size();
    __int128 maximum_element_offset = 0;
    if (element_bytes <= 0) return false;
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
        if (tensor.size(dim) <= 0 || tensor.stride(dim) < 0) return false;
        const __int128 term = __int128(tensor.size(dim) - 1) * tensor.stride(dim);
        if (term >= limit / element_bytes) return false;
        maximum_element_offset += term;
        if (maximum_element_offset >= limit / element_bytes) return false;
    }
    return (maximum_element_offset + 1) * element_bytes <= limit;
}

// V4.1 packed KV prefill has its own parameters and dispatch, independent of
// the BF16 prefill API. num_tokens excludes padding in the final KV page.
static std::vector<at::Tensor>
sparse_attn_quant_prefill_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int64_t num_tokens,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &out_
) {
    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    TORCH_CHECK(kv.device() == q.device() && indices.device() == q.device(),
                "q, kv and indices must be on the same device");
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(cc_major == 8 && cc_minor == 9,
                "sparse_quant_fwd is only supported on SM89 architectures");

    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.scalar_type() == torch::kUInt8, "kv must be uint8 packed bytes");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 128 && q.size(2) == 512,
                "q must have shape [s_q, 128, 512]");
    TORCH_CHECK(q.size(0) > 0 && q.size(0) <= std::numeric_limits<int>::max() / 128,
                "s_q must be positive and s_q * 128 must fit int32");
    TORCH_CHECK(kv.dim() == 4 && kv.size(0) > 0 && kv.size(2) == 1 &&
#if defined(FLASHMLA_ENABLE_V40_PREFILL)
                (kv.size(3) == 288 || kv.size(3) == 528 || kv.size(3) == 584),
                "kv must have shape [num_blocks, page_size, 1, 288, 528 or 584]");
#else
                (kv.size(3) == 288 || kv.size(3) == 528),
                "kv must have shape [num_blocks, page_size, 1, 288 or 528]");
#endif
    const int64_t page_size = kv.size(1);
    const int64_t packed_bytes = kv.size(3);
    TORCH_CHECK(page_size > 0 && page_size <= std::numeric_limits<int>::max() &&
                (page_size & (page_size - 1)) == 0,
                "page_size must be a positive power of two that fits int32");
    TORCH_CHECK(num_tokens > 0 && num_tokens <= std::numeric_limits<int>::max() &&
                (num_tokens - 1) / page_size < kv.size(0),
                "num_tokens must be positive, fit int32 and not exceed KV capacity");
    TORCH_CHECK(indices.dim() == 3 && indices.size(0) == q.size(0) && indices.size(1) == 1,
                "indices must have shape [s_q, 1, topk]");
    TORCH_CHECK(indices.size(2) >= 128 && indices.size(2) <= std::numeric_limits<int>::max() &&
                indices.size(2) % 128 == 0,
                "topk must be a positive multiple of 128 that fits int32");
    TORCH_CHECK(q.stride(2) == 1 && indices.stride(2) == 1,
                "q and indices must have contiguous last dimensions");
    const int stride_q_s_q = int64_stride_to_int(q.stride(0));
    const int stride_q_h_q = int64_stride_to_int(q.stride(1));
    const int stride_indices_s_q = int64_stride_to_int(indices.stride(0));
    const int stride_indices_h_kv = int64_stride_to_int(indices.stride(1));
    TORCH_CHECK(stride_q_h_q >= 512 &&
                stride_q_s_q >= int64_t(127) * stride_q_h_q + 512,
                "q query rows and heads must not overlap");
    TORCH_CHECK(stride_indices_s_q >= indices.size(2) && stride_indices_h_kv >= indices.size(2),
                "indices query rows must not overlap");
    TORCH_CHECK(kv.stride(3) == 1 && kv.stride(1) == packed_bytes &&
                kv.stride(0) >= page_size * packed_bytes,
                "kv must store each packed page contiguously; padding between pages is allowed");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0,
                "q rows and heads must be aligned to 16 bytes");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0 && kv.stride(0) % 16 == 0,
                "packed KV pages must be aligned to 16 bytes");

    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_SHAPE(attn_sink, 128);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_SHAPE(topk_length, q.size(0));
    KU_CHECK_CONTIGUOUS(topk_length);
    TORCH_CHECK(!attn_sink || attn_sink->device() == q.device(),
                "attn_sink must be on the same device as q");
    TORCH_CHECK(!topk_length || topk_length->device() == q.device(),
                "topk_length must be on the same device as q");

    at::Tensor out;
    if (out_) {
        out = *out_;
        CHECK_DEVICE(out);
        TORCH_CHECK(out.device() == q.device() && out.scalar_type() == q.scalar_type(),
                    "out must have the same device and dtype as q");
        CHECK_SHAPE(out, q.size(0), 128, 512);
        CHECK_CONTIGUOUS(out);
        TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                    "out must be aligned to 16 bytes");
    } else {
        out = torch::empty({q.size(0), 128, 512}, q.options());
    }
    // The softmax core reduces the unscaled logits before applying a positive
    // scale. Move a negative scalar sign into Q so masking and maxima retain
    // that order. Keep this owning tensor alive through the current-stream launch.
    at::Tensor q_eff = q;
    float sm_scale_eff = sm_scale;
    if (sm_scale < 0.0f) {
        q_eff = q.neg();
        sm_scale_eff = -sm_scale;
        TORCH_INTERNAL_ASSERT(q_eff.sizes() == q.sizes() &&
                              q_eff.scalar_type() == q.scalar_type() &&
                              q_eff.device() == q.device());
    } else if (sm_scale == 0.0f) {
        // Preserve the negative invalid sentinel while all valid QK logits are zero.
        // Multiplication also handles negative zero without changing the input Q.
        q_eff = q.mul(0.0f);
        sm_scale_eff = 1.0f;
        TORCH_INTERNAL_ASSERT(q_eff.sizes() == q.sizes() &&
                              q_eff.scalar_type() == q.scalar_type() &&
                              q_eff.device() == q.device());
    }
    const int stride_q_eff_s_q = int64_stride_to_int(q_eff.stride(0));
    const int stride_q_eff_h_q = int64_stride_to_int(q_eff.stride(1));

    // Reserve the flags only for eligible FP8 calls. The returned tensor keeps
    // its original shape; its backing adds one immutable 128-float tail row.
    const bool gpu_checked_preeligible = packed_bytes == 528 && !topk_length &&
        indices.size(2) % 64 == 0 && q.size(0) >= 4096 && indices.size(2) >= 2048 &&
        v41_quant_narrow32_tensor_span(q_eff) && v41_quant_narrow32_tensor_span(kv) &&
        v41_quant_narrow32_tensor_span(indices) && v41_quant_narrow32_tensor_span(out);
    auto max_logits_backing = torch::empty(
        {q.size(0) + (gpu_checked_preeligible ? 1 : 0), 128}, q.options().dtype(torch::kFloat32));
    auto max_logits = gpu_checked_preeligible
        ? max_logits_backing.narrow(0, 0, q.size(0)) : max_logits_backing;
    auto lse = torch::empty_like(max_logits);

    SparseQuantPrefillParams params{};
    params.s_q = static_cast<int>(q.size(0));
    params.s_kv = static_cast<int>(num_tokens);
    params.h_q = 128; params.h_kv = 1; params.d_qk = 512; params.d_v = 512;
    params.topk = static_cast<int>(indices.size(2));
    params.sm_scale = sm_scale_eff;
    params.sm_scale_div_log2 = sm_scale_eff * LOG_2_E;
    params.q = q_eff.data_ptr(); params.kv = kv.data_ptr();
    params.indices = indices.data_ptr<int>();
    params.attn_sink = get_optional_tensor_ptr<float>(attn_sink);
    params.topk_length = get_optional_tensor_ptr<int>(topk_length);
    params.stride_q_s_q = stride_q_eff_s_q;
    params.stride_q_h_q = stride_q_eff_h_q;
    params.stride_kv_s_kv = static_cast<int>(packed_bytes);
    params.stride_kv_h_kv = static_cast<int>(packed_bytes);
    params.stride_indices_s_q = stride_indices_s_q;
    params.stride_indices_h_kv = stride_indices_h_kv;
    params.out = out.data_ptr(); params.max_logits = max_logits.data_ptr(); params.lse = lse.data_ptr();
    params.stream = at::cuda::getCurrentCUDAStream().stream();
    params.quant_page_size = static_cast<int>(page_size);
    params.quant_block_stride = kv.stride(0);

    const bool narrow32 = v41_quant_narrow32_tensor_span(q_eff) &&
        v41_quant_narrow32_tensor_span(kv) && v41_quant_narrow32_tensor_span(indices) &&
        v41_quant_narrow32_tensor_span(out) && v41_quant_narrow32_tensor_span(max_logits_backing) &&
        v41_quant_narrow32_tensor_span(lse);
#if defined(FLASHMLA_ENABLE_V40_PREFILL)
    if(packed_bytes==584) {
        if(narrow32) flashmla::v41prefill::run_sparse_quant_prefill_v40_fwd_narrow32(params);
        else flashmla::v41prefill::run_sparse_quant_prefill_v40_fwd(params);
    } else
#endif
    if (gpu_checked_preeligible && narrow32) {
        // Both launches use the current stream. Capture records the checker
        // and wrapper; all flag decisions stay on the GPU on every replay.
        flashmla::v41prefill::run_sparse_quant_prefill_gpu_checked_fp8(params);
    } else if (narrow32) {
        flashmla::v41prefill::run_sparse_quant_prefill_fwd_narrow32(params, packed_bytes == 288);
    } else {
        flashmla::v41prefill::run_sparse_quant_prefill_fwd(params, packed_bytes == 288);
    }
    return {out, max_logits, lse};
}
