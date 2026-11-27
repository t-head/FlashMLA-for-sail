#pragma once

#include "params.h"
#include <cstdint>

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

enum class FwdMode : uint8_t { Prefill, Decode };
enum class OutputFormat : uint8_t { FP8, INT8, BF16 };

struct Params {
    FwdMode mode;
    int arch;
    int s_q, s_kv, h_q, topk;
    float sm_scale, sm_scale_log2, rms_norm_eps;
    bool enable_q_norm;
    bool per_token_scale;
    OutputFormat output_format = OutputFormat::FP8;
    const void* q;
    const void* kv;
    const int32_t* indices;
    const int32_t* topk_length;
    const float* attn_sink;
    const int32_t* token_positions;
    const float* cos_sin_cache;
    int64_t stride_q_s_q;
    int64_t stride_kv_s_kv;
    int64_t stride_indices_s_q;
    int num_blocks, page_block_size;
    const void* extra_kv;
    const int32_t* extra_indices;
    const int32_t* extra_topk_length;
    int extra_num_blocks, extra_page_block_size, extra_topk;
    int64_t stride_extra_kv_block;
    int64_t stride_extra_indices_s_q;
    bool extra_fp4;
    bool v41_float_kv = kDefaultV41FloatKv;
    int n_wv_group, wv_group_size;
    uint8_t* out_fp8;
    uint32_t* out_sf;
    float* max_logits;
    float* lse;
    int64_t stride_out_sf_token, stride_out_sf_group, stride_out_sf_word;
    // run the decode five-in-one's GEMMs on the PPU1.5 E4M3xE4M3->F32
    // atom instead of the BF16 atom (Q is quantized to E4M3 after RoPE).  This
    // is the single switch between the two measurement arms; default false so
    // existing callers keep the BF16 path bit-for-bit.
    bool fp8_compute = false;
    hggcStream_t stream;
};

// Keep output-format choices constant inside kernels while the host API selects
// the format for each call through the existing quantization parameters.
template<bool PerTokenScale>
struct OutputParams : Params {
    static constexpr bool per_token_scale = PerTokenScale;
};

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const Params& params);

template<bool EnableNorm>
void run_fused_norm_rope_attn_rope_cast_prefill_hs64(const Params& params);

template<bool EnableNorm>
void run_fused_norm_rope_attn_rope_cast_prefill_h128(const Params& params);

template<bool Norm>
void run_fused_norm_rope_attn_rope_cast_prefill_small(const Params& params);

struct SplitKVParams {
    int* tile_scheduler_metadata;
    int* num_splits;
    int num_sm_parts;
    float* lse_accum;
    float* o_accum;
};

template<int Heads, bool EnableNorm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel(
    const Params& params, const SplitKVParams& splitkv);

// FP8-compute variant of the decode splitkv kernel (E4M3 QK^T and PV).
// Declared after SplitKVParams, which it takes by reference.
template<int Heads, bool EnableNorm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_decode_splitkv_fp8_kernel(
    const Params& params, const SplitKVParams& split);

template<bool Norm>
bool run_fused_norm_rope_attn_rope_cast_decode_native_wi(
    const Params& params, const SplitKVParams& splitkv);

}
