#include "fused_v41_projection.h"
#include <cassert>

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::permute_q_b_proj {

__launch_bounds__(32)
__global__ void permute_q_b_proj_kernel(const Params params) {
    const uint32_t row_idx = blockIdx.x;
    const uint32_t head_idx = row_idx / params.d_q;
    const uint32_t head_dim_idx = row_idx % params.d_q;
    const uint32_t out_row_idx = head_dim_idx % 16 + head_idx * 16
        + (head_dim_idx / 16) * (params.h_q * 16);
    const uint32_t num_scales_per_row = params.q_lora_rank / params.gran / 4;

    const auto* input_row = reinterpret_cast<const uint8_t*>(
        params.q_b_proj + row_idx * params.stride_q_b_dim0);
    auto* output_row = reinterpret_cast<uint8_t*>(
        params.q_b_proj_permuted + out_row_idx * params.stride_q_b_permuted_dim0);
    if (((reinterpret_cast<uintptr_t>(input_row) | reinterpret_cast<uintptr_t>(output_row)) & 15) == 0) {
        const auto* input_vec = reinterpret_cast<const uint4*>(input_row);
        auto* output_vec = reinterpret_cast<uint4*>(output_row);
        for (uint32_t i = threadIdx.x; i < params.q_lora_rank / 16; i += 32) {
            output_vec[i] = input_vec[i];
        }
    } else {
        for (uint32_t i = threadIdx.x; i < params.q_lora_rank; i += 32) {
            output_row[i] = input_row[i];
        }
    }

    for (uint32_t scale_idx = threadIdx.x; scale_idx < num_scales_per_row; scale_idx += 32) {
        const int32_t cur_scale = params.scale_factors[
            row_idx + scale_idx * params.stride_scale_factors_dim1];
        params.scale_factors_permuted[
            out_row_idx + scale_idx * params.stride_scale_factors_permuted_dim1] = cur_scale;
    }
}

void run_permute_q_b_proj_kernel(const Params& params) {
    permute_q_b_proj_kernel<<<params.h_q * params.d_q, 32, 0, params.stream>>>(params);
}

}

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::permute_wv_proj {

static constexpr uint32_t INPUT_GRAN = 32;
static constexpr uint32_t CHUNK_SIZE = 32;
static constexpr uint32_t D_O = 512;

__launch_bounds__(32)
__global__ void permute_wv_proj_kernel(const Params params) {
    const uint32_t row_idx = blockIdx.x;
    const uint32_t wv_group_idx = blockIdx.y;
    const uint32_t num_sf_per_head = D_O / CHUNK_SIZE;
    const uint32_t num_sf = params.wv_group_size * num_sf_per_head;

    const auto* input_row = reinterpret_cast<const uint8_t*>(
        params.wv_proj + wv_group_idx * params.stride_wv_proj_dim0
        + row_idx * params.stride_wv_proj_dim1);
    auto* output_row = reinterpret_cast<uint8_t*>(
        params.wv_proj_permuted + wv_group_idx * params.stride_wv_proj_permuted_dim0
        + row_idx * params.stride_wv_proj_permuted_dim1);
    const bool aligned = ((reinterpret_cast<uintptr_t>(input_row)
        | reinterpret_cast<uintptr_t>(output_row)) & 15) == 0;

    for (uint32_t i = threadIdx.x; i < num_sf; i += 32) {
        const uint32_t head_idx = i / num_sf_per_head;
        const uint32_t head_dim_chunk_idx = i % num_sf_per_head;
        const uint32_t out_chunk_idx = head_dim_chunk_idx * params.wv_group_size + head_idx;
        const uint32_t input_col = i * CHUNK_SIZE;
        const uint32_t output_col = out_chunk_idx * CHUNK_SIZE;
        if (aligned) {
            const auto* input_vec = reinterpret_cast<const uint4*>(input_row + input_col);
            auto* output_vec = reinterpret_cast<uint4*>(output_row + output_col);
            output_vec[0] = input_vec[0];
            output_vec[1] = input_vec[1];
        } else {
            for (uint32_t j = 0; j < CHUNK_SIZE; ++j) {
                output_row[output_col + j] = input_row[input_col + j];
            }
        }
    }

    const uint64_t input_sf_base = wv_group_idx * params.stride_scale_factors_dim0 + row_idx;
    const uint64_t output_sf_base = wv_group_idx * params.stride_scale_factors_permuted_dim0 + row_idx;
    // One thread owns each packed output word, avoiding subword write races.
    for (uint32_t word_idx = threadIdx.x; word_idx < num_sf / 4; word_idx += 32) {
        uint32_t packed = 0;
        #pragma unroll
        for (uint32_t byte_idx = 0; byte_idx < 4; ++byte_idx) {
            const uint32_t output_sf_idx = word_idx * 4 + byte_idx;
            const uint32_t head_idx = output_sf_idx % params.wv_group_size;
            const uint32_t head_dim_chunk_idx = output_sf_idx / params.wv_group_size;
            const uint32_t input_sf_idx = head_idx * num_sf_per_head + head_dim_chunk_idx;
            const uint32_t input_word = static_cast<uint32_t>(params.scale_factors[
                input_sf_base + (input_sf_idx / 4) * params.stride_scale_factors_dim2]);
            packed |= ((input_word >> ((input_sf_idx % 4) * 8)) & 0xffu) << (byte_idx * 8);
        }
        reinterpret_cast<uint32_t*>(params.scale_factors_permuted)[
            output_sf_base + word_idx * params.stride_scale_factors_permuted_dim2] = packed;
    }
}

void run_permute_wv_proj_kernel(const Params& params) {
    assert(params.d_o == D_O);
    assert(params.input_gran == INPUT_GRAN);
    permute_wv_proj_kernel<<<dim3(params.d_proj_out, params.n_wv_group), 32, 0, params.stream>>>(params);
}

}
