#pragma once

#include <cutlass/numeric_types.h>
#include "ppu/sparse/fused_v41.h"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::output {

__forceinline__ __device__ float quant_scale(float amax, core_attn::OutputFormat format) {
    return format == core_attn::OutputFormat::INT8 ? amax / 127.0f : amax / 448.0f;
}

__forceinline__ __device__ uint8_t pack_int8(float scaled) {
    const uint32_t bits = __float_as_uint(scaled);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0;
    const int magnitude = __float2uint_rn(fminf(fabsf(scaled), 127.0f));
    return static_cast<uint8_t>(bits >> 31 ? -magnitude : magnitude);
}

__forceinline__ __device__ void store_bf16(uint8_t* out, int64_t element_offset, float value) {
    reinterpret_cast<cutlass::bfloat16_t*>(out)[element_offset] = cutlass::bfloat16_t(value);
}

}
