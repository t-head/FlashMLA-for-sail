/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include "common.h"
#include "params.h"

template<typename T>
void run_sparse_prefill_fwd_dispatch(SparsePrefillParams &params);

// New interface aligned with FlashMLA reference
static std::vector<at::Tensor>
sparse_attn_prefill_interface(
    const at::Tensor &q,           // seqlen_q x num_heads x head_size
    const at::Tensor &kv,          // seqlen_k x num_heads_k x head_size
    const at::Tensor &indices,     // seqlen_q x num_heads_k x top_k
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &out_  // seqlen_q x num_heads x d_v
) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor >= 0;
    TORCH_CHECK(is_sm8x, "Sparse Attention Forward Kernel (sparse_prefill_fwd) is only supported on SM8x architectures");
    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);

    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.dtype() == torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kv.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(indices.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    auto opts = q.options();
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q.dtype(), "out must have the same dtype as q");
        KU_CHECK_SHAPE(out, s_q, h_q, d_v);
        KU_CHECK_CONTIGUOUS(out);
        KU_CHECK_DEVICE(out);
    } else {
        out = torch::empty({s_q, h_q, d_v}, opts);
    }

    at::Tensor buf_attn_score, max_logits, lse, p_sum;
    max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    CHECK_CONTIGUOUS(max_logits);
    CHECK_CONTIGUOUS(lse);

    SparsePrefillParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * 1.44269504f,

        (void*)q.data_ptr(),
        (void*)kv.data_ptr(),
        (int*)indices.data_ptr(),

        get_optional_tensor_ptr<float>(attn_sink),
        get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (void*)out.data_ptr(),
        (void*)max_logits.data_ptr(),
        (void*)lse.data_ptr(),

        at::cuda::getCurrentCUDAStream().stream()
    };
    ppu::fmha::FmhaProfParam fmha_prof_params;
    if (ppu::fmha::ProfilingInterface::Instance().get_op_info()){
        hggcStreamCaptureStatus captureStatus;
        hggcStreamIsCapturing(params.stream, &captureStatus);
        std::string topk_len_str = "";
        if (params.topk_length) {
            topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.topk_length, params.s_q, params.stream);
        }
        fmha_prof_params.set_flash_attn_sparse_prefill_params(
            q.dtype() == torch::kBFloat16/*data_type*/,
            params.h_q/*num_heads*/, params.h_kv/*num_heads_k*/,
            params.d_qk/*head_dim*/, params.d_v/*head_dim_value*/,
            params.s_q/*seqlen_q*/, params.s_kv/*seqlen_k*/, params.topk,
            bool(params.attn_sink), topk_len_str
        );
    }
    ppu::fmha::ProfilingInterface::Instance().instrument(true, fmha_prof_params);
    run_sparse_prefill_fwd_dispatch<cutlass::bfloat16_t>(params);
    ppu::fmha::ProfilingInterface::Instance().instrument(false, fmha_prof_params);

    return {out, max_logits, lse};
}
