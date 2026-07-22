/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <hggc_fp16.h>
#include <hggc_runtime.h>

#include <hggc_bf16.h>

struct Flash_fwd_params {
    using index_t = int64_t;

    int b, q_orig, seqlen_q, d, d_v;
    int h, h_q, h_h_k_ratio, ngroups;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int *__restrict__ cu_seqlens_k;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;
    int *__restrict__ indices_ptr;
    int *__restrict__ topk_len_ptr;
    float *__restrict__ attn_sink_ptr;

    void *__restrict__ extra_k_ptr;
    int *__restrict__ extra_indices_ptr;
    int *__restrict__ extra_topk_len_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;
    index_t indices_batch_stride;
    index_t indices_row_stride;

    index_t extra_k_batch_stride;
    index_t extra_k_row_stride;
    index_t extra_k_head_stride;
    index_t extra_indices_batch_stride;
    index_t extra_indices_row_stride;
    int extra_num_blocks;
    int extra_page_block_size;
    int extra_topk;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    int *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int num_blocks;
    int *__restrict__ num_splits_ptr;
    int num_splits;  // For split-KV version
    int seqlen_k; // real kvsize.
    int topk;

    void *__restrict__ softmax_lseaccum_ptr;
    void *__restrict__ oaccum_ptr;

    // For Holmes-LLM
    int64_t *__restrict__ hllm_block_table;
    void * workspace_ptr;
    size_t max_workspace_size;
};

using Flash_fwd_mla_params = Flash_fwd_params;

struct SparsePrefillParams {
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
};

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
    int topk;

    // int s_q;
    int extra_topk;
    int *__restrict__ topk_length;
    int *__restrict__ extra_topk_length;

};
