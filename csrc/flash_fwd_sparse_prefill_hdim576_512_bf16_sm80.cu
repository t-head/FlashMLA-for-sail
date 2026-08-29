// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template void run_sparse_prefill_fwd_dispatch<cutlass::bfloat16_t>(SparsePrefillParams &params);

extern "C" void flash_mla_run_sparse_prefill_bf16(
    int s_q, int s_kv, int h_q, int h_kv, int d_qk, int d_v, int topk,
    float sm_scale, float sm_scale_div_log2,
    void* q, void* kv, int* indices, float* attn_sink, int* topk_length,
    int stride_q_s_q, int stride_q_h_q,
    int stride_kv_s_kv, int stride_kv_h_kv,
    int stride_indices_s_q, int stride_indices_h_kv,
    void* out, void* max_logits, void* lse, void* stream) {
    SparsePrefillParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale_div_log2,
        q, kv, indices, attn_sink, topk_length,
        stride_q_s_q, stride_q_h_q,
        stride_kv_s_kv, stride_kv_h_kv,
        stride_indices_s_q, stride_indices_h_kv,
        out, max_logits, lse,
        reinterpret_cast<hggcStream_t>(stream),
    };
    run_sparse_prefill_fwd_dispatch<cutlass::bfloat16_t>(params);
}

#if !defined(FLASH_MLA_SPARSE_PREFILL_ONLY)
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, false, 576, 512>(Flash_fwd_params &params, hggcStream_t stream);
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 576, 512>(Flash_fwd_params &params, hggcStream_t stream);
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, false, 512, 512>(Flash_fwd_params &params, hggcStream_t stream);
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 512, 512>(Flash_fwd_params &params, hggcStream_t stream);
#endif
