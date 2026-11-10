#pragma once

#include "params.h"
#include <cutlass/float8.h>

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::permute_q_b_proj {

using fp8_e4m3 = cutlass::float_e4m3_t;

struct Params {
    uint32_t h_q;
    uint32_t d_q;
    uint32_t q_lora_rank;
    uint32_t gran;

    fp8_e4m3* __restrict__ q_b_proj;
    uint64_t stride_q_b_dim0;
    int32_t* __restrict__ scale_factors;
    uint64_t stride_scale_factors_dim1;

    fp8_e4m3* __restrict__ q_b_proj_permuted;
    uint64_t stride_q_b_permuted_dim0;
    int32_t* __restrict__ scale_factors_permuted;
    uint64_t stride_scale_factors_permuted_dim1;

    hggcStream_t stream;
};

void run_permute_q_b_proj_kernel(const Params& params);

}

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::permute_wv_proj {

using fp8_e4m3 = cutlass::float_e4m3_t;

struct Params {
    uint32_t d_o;
    uint32_t input_gran;
    uint32_t wv_group_size;
    uint32_t n_wv_group;
    uint32_t d_proj_out;

    fp8_e4m3* __restrict__ wv_proj;
    uint64_t stride_wv_proj_dim0, stride_wv_proj_dim1;
    int32_t* __restrict__ scale_factors;
    uint64_t stride_scale_factors_dim0, stride_scale_factors_dim2;

    fp8_e4m3* __restrict__ wv_proj_permuted;
    uint64_t stride_wv_proj_permuted_dim0, stride_wv_proj_permuted_dim1;
    int32_t* __restrict__ scale_factors_permuted;
    uint64_t stride_scale_factors_permuted_dim0, stride_scale_factors_permuted_dim2;

    hggcStream_t stream;
};

void run_permute_wv_proj_kernel(const Params& params);

}
