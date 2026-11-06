#include "kernel.h"

#include <cmath>
#include <cutlass/cutlass.h>
#include <cutlass/float8.h>

#include "kerutils/common/common.h"
#include "kerutils/common/static_switch.h"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::combine {

constexpr int kBlockM = 8;
constexpr int kNumThreads = kBlockM * 32;
constexpr int kHeadDimV = 512;
constexpr int kElemsPerLane = kHeadDimV / 32;

__forceinline__ __device__ float2 apply_rope(float2 x, float c, float s) {
    return float2{fmaf(x.x, c, -x.y * s), fmaf(x.x, s, x.y * c)};
}

template<int kMaxSplits>
__global__ void __launch_bounds__(kNumThreads)
fused_norm_rope_attn_rope_cast_combine_kernel(const Params params) {
    __shared__ float lse_scale[kBlockM][kMaxSplits + 1];
    const int token = blockIdx.x;
    const int m_block = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int start_split = __ldg(params.num_splits + token);
    const int num_splits = __ldg(params.num_splits + token + 1) - start_split;
    FLASH_DEVICE_ASSERT(num_splits <= kMaxSplits);
    if (num_splits == 1) return;

    const int heads = params.h_q;
    const int valid_rows = min(kBlockM, heads - m_block * kBlockM);
    for (int idx = threadIdx.x; idx < num_splits * kBlockM; idx += kNumThreads) {
        const int split = idx / kBlockM;
        const int row = idx % kBlockM;
        lse_scale[row][split] = row < valid_rows
            ? params.lse_accum[int64_t(start_split + split) * heads + m_block * kBlockM + row]
            : -INFINITY;
    }
    __syncthreads();
    if (warp >= valid_rows) return;
    const int head = m_block * kBlockM + warp;

    constexpr int kLsePerLane = (kMaxSplits + 31) / 32;
    float local_lse[kLsePerLane];
    #pragma unroll
    for (int i = 0; i < kLsePerLane; ++i) {
        const int split = i * 32 + lane;
        local_lse[i] = split < num_splits ? lse_scale[warp][split] * float(M_LOG2E) : -INFINITY;
    }
    float max_lse = -INFINITY;
    #pragma unroll
    for (int i = 0; i < kLsePerLane; ++i) max_lse = max(max_lse, local_lse[i]);
    #pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
    }
    max_lse = max_lse == -INFINITY ? 0.0f : max_lse;
    float sum_lse = 0.0f;
    #pragma unroll
    for (int i = 0; i < kLsePerLane; ++i) sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
    #pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);
    }
    const float global_lse = (sum_lse == 0.f || sum_lse != sum_lse)
        ? INFINITY : log2f(sum_lse) + max_lse;
    if (lane == 0) params.lse[int64_t(token) * heads + head] = global_lse / float(M_LOG2E);
    #pragma unroll
    for (int i = 0; i < kLsePerLane; ++i) {
        const int split = i * 32 + lane;
        if (split < num_splits) lse_scale[warp][split] = exp2f(local_lse[i] - global_lse);
    }
    __syncwarp();

    float result[kElemsPerLane];
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) result[i] = 0.0f;
    const float* o_accum = params.o_accum
        + (int64_t(start_split) * heads + head) * kHeadDimV + lane;
    #pragma unroll 2
    for (int split = 0; split < num_splits; ++split) {
        const float scale = lse_scale[warp][split];
        if (scale != 0.f) {
            const float* src = o_accum + int64_t(split) * heads * kHeadDimV;
            #pragma unroll
            for (int i = 0; i < kElemsPerLane; ++i) result[i] += scale * src[i * 32];
        }
    }

    if (params.attn_sink != nullptr) {
        const float sink_log2 = __ldg(params.attn_sink + head) * float(M_LOG2E);
        const float factor = (global_lse != global_lse || global_lse == INFINITY)
            ? 1.0f : 1.0f / (1.0f + exp2f(sink_log2 - global_lse));
        #pragma unroll
        for (int i = 0; i < kElemsPerLane; ++i) result[i] *= factor;
    }

    const int position = __ldg(params.token_positions + token);
    const float* rope = params.cos_sin_cache + int64_t(position) * 64;
    const bool even = (lane & 1) == 0;
    #pragma unroll
    for (int i = 14; i < kElemsPerLane; ++i) {
        const float other = __shfl_xor_sync(uint32_t(-1), result[i], 1);
        const int pair = (i - 14) * 16 + lane / 2;
        const float2 x = even ? float2{result[i], other} : float2{other, result[i]};
        const float2 rotated = apply_rope(x, __ldg(rope + pair), -__ldg(rope + 32 + pair));
        result[i] = even ? rotated.x : rotated.y;
    }

    const int group = head / 8;
    const int in_group = head % 8;
    uint8_t* out = params.out_fp8 + int64_t(token) * heads * 512 + group * (8 * 512)
        + in_group * 32 + lane;
    uint32_t* sf_base = params.out_sf + token + int64_t(group) * params.stride_out_sf_group;
    #pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
        float amax = fabsf(result[i]);
        #pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            amax = fmaxf(amax, __shfl_xor_sync(uint32_t(-1), amax, offset));
        }
        amax = fmaxf(1e-4f, amax);
        const float scale = amax / 448.0f;
        const int exponent = int((__float_as_uint(scale) - 1u) >> 23) + 1 - 127;
        const float scale_inv = __uint_as_float(uint32_t(127 - exponent) << 23);
        out[i * (8 * 32)] = cutlass::float_e4m3_t::from_float(result[i] * scale_inv).storage;
        if (lane == 0) {
            const int sf_index = i * 8 + in_group;
            auto* sf_word = sf_base + int64_t(sf_index / 4) * params.stride_out_sf_word;
            reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
        }
    }
}

void run_fused_norm_rope_attn_rope_cast_combine_kernel(const Params& params) {
    if (params.s_q == 0) return;
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        fused_norm_rope_attn_rope_cast_combine_kernel<kMaxSplits>
            <<<dim3(params.s_q, (params.h_q + kBlockM - 1) / kBlockM), kNumThreads, 0, params.stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
