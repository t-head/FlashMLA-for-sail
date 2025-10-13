/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once
#include <ATen/cuda/CUDAContext.h>

struct Flash_fwd_params {
    using index_t = int64_t;

    int b, seqlen_q, d, d_v;
    int h, h_h_k_ratio, ngroups;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int *__restrict__ cu_seqlens_k;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;

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

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    int *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;
    int num_splits;  // For split-KV version
    int seqlen_k; // real kvsize.

    void *__restrict__ softmax_lseaccum_ptr;
    void *__restrict__ oaccum_ptr;

    // For Holmes-LLM
    int64_t *__restrict__ hllm_block_table;
    void * workspace_ptr;
    size_t max_workspace_size;
};

struct SparsePrefillParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;

    // Input tensors
    void *__restrict__ q;    // [s_q, h_q, d_qk]
    void *__restrict__ kv;   // [s_kv, h_kv, d_qk]
    int* __restrict__ indices;   // [s_q, h_kv, topk]

    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Output tensors
    void *__restrict__ out;   // [s_q, h_q, d_v]
    void* __restrict__ max_logits; // [s_q, h_q]
    void* __restrict__ lse; // [s_q, h_q]

    cudaStream_t stream;
};

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

static bool is_sm89_or_newer(){
    auto dprops = at::cuda::getCurrentDeviceProperties();
    return (dprops->major > 8) || (dprops->major == 8 && dprops->minor >= 9);
}

static bool use_cross_cut(int num_heads_per_head_k, int batch_size) {
// #if ACOMPUTE_VERSION == 10000
    if (!is_sm89_or_newer()) {
        if (num_heads_per_head_k <= 16) {
            return false;
        } else if (num_heads_per_head_k <= 32) {
            return batch_size >= 8;
        } else if (num_heads_per_head_k <= 64) {
            return batch_size >= 2;
        } else {
            return true;
        }
// #else
    } else {
        return true;
    }
// #endif
}

template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_params &params, cudaStream_t stream);
struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};
void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);

#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                          \
            asm("trap;");                                                                                 \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

template<typename T, int Headdim, int Headdim_V> void run_mha_fwd_splithd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<typename T> void run_sparse_prefill_fwd_dispatch(const SparsePrefillParams &params);