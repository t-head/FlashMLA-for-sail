/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <type_traits>
#include "params.h"

// Keep the BF16 Prefill parameter type unchanged. This independent, flat POD
// matches the verified packed-KV kernel ABI; nested base-class parameters are
// not used for __grid_constant__ kernel serialization.
struct SparseQuantPrefillParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;

    // Input tensors
    void *__restrict__ q;    // [s_q, h_q, d_qk]
    void *__restrict__ kv;   // [s_kv, h_kv, d_qk]
    int* __restrict__ indices;   // [s_q, h_kv, topk]

    float* __restrict__ attn_sink;   // [h_q], may be nullptr
    int* __restrict__ topk_length;   // [s_q], may be nullptr

    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Output tensors
    void *__restrict__ out;   // [s_q, h_q, d_v]
    void* __restrict__ max_logits; // [s_q, h_q]
    void* __restrict__ lse; // [s_q, h_q]

    hggcStream_t stream;

    int quant_page_size;
    int64_t quant_block_stride;  // bytes between packed pages
};
static_assert(std::is_standard_layout_v<SparseQuantPrefillParams>);
static_assert(std::is_trivially_copyable_v<SparseQuantPrefillParams>);

namespace flashmla::v41prefill {

// H128, D512, one KV head; FP4 selects E2M1/E4M3, FP8 E4M3/E8M0.
// Output: BF16 attention, FP32 scaled max logits, FP32 natural-log LSE.
void run_sparse_quant_prefill_fwd(SparseQuantPrefillParams &params, bool fp4);

} // namespace flashmla::v41prefill
