#pragma once

#include "params.h"
#include "ppu/sparse/fused_v41.h"
#include <cstdint>

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::combine {

struct Params {
    int s_q, h_q, num_sm_parts;
    const int* num_splits;
    const float* lse_accum;
    const float* o_accum;
    const float* attn_sink;
    const int32_t* token_positions;
    const float* cos_sin_cache;
    uint8_t* out_fp8;
    uint32_t* out_sf;
    float* lse;
    bool per_token_scale;
    int64_t stride_out_sf_token;
    int64_t stride_out_sf_group, stride_out_sf_word;
    hggcStream_t stream;
};

void run_fused_norm_rope_attn_rope_cast_combine_kernel(const Params& params);

}
