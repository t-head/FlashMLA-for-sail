/******************************************************************************
 * Copyright (c) 2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 *
 * Narrow PyTorch binding for sparse-prefill kernel iteration. Keeping this
 * translation unit independent of HGGC datatype headers avoids the CUDA/HGGC
 * bfloat16 header collision in SDK 2.1.3 and lets kernel campaigns rebuild a
 * single CUDA object while reusing this API object.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <limits>
#include <optional>
#include <vector>

extern "C" void flash_mla_run_sparse_prefill_bf16(
    int s_q, int s_kv, int h_q, int h_kv, int d_qk, int d_v, int topk,
    float sm_scale, float sm_scale_div_log2,
    void* q, void* kv, int* indices, float* attn_sink, int* topk_length,
    int stride_q_s_q, int stride_q_h_q,
    int stride_kv_s_kv, int stride_kv_h_kv,
    int stride_indices_s_q, int stride_indices_h_kv,
    void* out, void* max_logits, void* lse, void* stream);

namespace {

int checked_stride(int64_t stride) {
    TORCH_CHECK(
        stride <= std::numeric_limits<int>::max(),
        "[Sparse TopK Attention] stride exceeds int32 limit: ", stride);
    return static_cast<int>(stride);
}

template <typename T>
T* optional_ptr(const std::optional<at::Tensor>& tensor) {
    return tensor.has_value() ? tensor->data_ptr<T>() : nullptr;
}

std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& out_) {
    at::cuda::CUDAGuard device_guard(q.device());

    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16);
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16);
    TORCH_CHECK(indices.scalar_type() == at::kInt);
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3);

    const int s_q = q.size(0);
    const int s_kv = kv.size(0);
    const int h_q = q.size(1);
    const int h_kv = kv.size(1);
    const int d_qk = q.size(2);
    const int topk = indices.size(2);

    TORCH_CHECK(kv.size(2) == d_qk);
    TORCH_CHECK(indices.size(0) == s_q && indices.size(1) == h_kv);
    TORCH_CHECK(q.stride(-1) == 1 && kv.stride(-1) == 1);
    TORCH_CHECK(indices.stride(-1) == 1);
    TORCH_CHECK(d_qk == 576 && d_v == 512);
    TORCH_CHECK(h_kv == 1);

    if (attn_sink.has_value()) {
        TORCH_CHECK(attn_sink->is_cuda());
        TORCH_CHECK(attn_sink->scalar_type() == at::kFloat);
        TORCH_CHECK(attn_sink->sizes() == at::IntArrayRef({h_q}));
    }
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length->is_cuda());
        TORCH_CHECK(topk_length->scalar_type() == at::kInt);
        TORCH_CHECK(topk_length->sizes() == at::IntArrayRef({s_q}));
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = *out_;
        TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16);
        TORCH_CHECK(out.sizes() == at::IntArrayRef({s_q, h_q, d_v}));
        TORCH_CHECK(out.stride(-1) == 1);
    } else {
        out = at::empty({s_q, h_q, d_v}, q.options());
    }
    auto max_logits = at::empty({s_q, h_q}, q.options().dtype(at::kFloat));
    auto lse = at::empty({s_q, h_q}, q.options().dtype(at::kFloat));

    flash_mla_run_sparse_prefill_bf16(
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * 1.44269504f,
        q.data_ptr(), kv.data_ptr(), indices.data_ptr<int>(),
        optional_ptr<float>(attn_sink), optional_ptr<int>(topk_length),
        checked_stride(q.stride(0)), checked_stride(q.stride(1)),
        checked_stride(kv.stride(0)), checked_stride(kv.stride(1)),
        checked_stride(indices.stride(0)), checked_stride(indices.stride(1)),
        out.data_ptr(), max_logits.data_ptr(), lse.data_ptr(),
        at::cuda::getCurrentCUDAStream().stream());
    return {out, max_logits, lse};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "FlashMLA sparse-prefill iteration module";
    module.def("sparse_prefill_fwd", &sparse_prefill_fwd);
}
