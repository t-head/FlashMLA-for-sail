#pragma once

#include <cmath>
#include <cstdint>
#include <cutlass/float8.h>

#include "sparse/fused_v41.h"
#include "ppuxx/decode/combine/fused_v41.h"
#include "decode/sparse/splitkv_mla.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

namespace splitkv {

using namespace cute;

template<int Heads, bool PagePow2>
using Traits = Flash_fwd_kernel_traits<
    512, Heads == 8 ? 16 : 64, Heads == 8 ? 32 : 64, Heads == 8 ? 8 : 16,
    /*Is_Q_in_regs=*/true, /*Share_Q_K_smem=*/false,
    cutlass::bfloat16_t, 512, /*CrossCut=*/true, /*USE_MMA_M8=*/false,
    /*AtomLayoutQ=*/Heads == 8 ? 1 : 4, /*AtomLayoutP=*/Heads == 8 ? 1 : 2,
    Heads == 8 ? 32 : 64, 2, /*kNWarps0=*/Heads == 8 ? 2 : 16, PagePow2>;

template<typename KT, bool ExtraFP4>
constexpr size_t scratch_offset() {
    return KT::kSmemSizeAccum + KT::kBlockN * 2 * sizeof(int)
        + (ExtraFP4 && KT::kBlockM > 32 ? flash::kFp4LutEntries * sizeof(uint32_t) : 0);
}

template<typename KT>
constexpr size_t scratch_bytes() {
    return (KT::kNThreads + KT::kBlockM) * sizeof(float);
}

__forceinline__ __device__ float2 apply_rope(float2 x, float c, float s) {
    return float2{fmaf(x.x, c, -x.y * s), fmaf(x.x, s, x.y * c)};
}

template<int Rows>
__forceinline__ __device__ int native_offset(int row, int dim) {
    const int tile = dim / 64;
    const int col = dim % 64;
#if ACOMPUTE_VERSION == 10000
    const int slice = col / 16;
    const int rotation = (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2;
    const int vec = (((row % 4) * 2 + (col % 16) / 8) ^ ((row / 4) & 1));
    return tile * Rows * 64 + slice * Rows * 16 + (row / 4) * 64
        + ((vec + rotation) % 8) * 8 + col % 8;
#else
    return tile * Rows * 64 + row * 64 + (col ^ ((row & 7) << 3));
#endif
}

template<typename KT, bool ExtraFP4>
__forceinline__ __device__ float* scratch(char* smem) {
    return reinterpret_cast<float*>(smem + scratch_offset<KT, ExtraFP4>());
}

template<typename KT, bool ExtraFP4, typename P>
__forceinline__ __device__ void load_q(
    const P& p, int token, int h_k_idx, char* smem, int tidx) {
    using Element = cutlass::bfloat16_t;
    constexpr int kBlockM = KT::kBlockM;
    constexpr int kParts = KT::kNThreads / kBlockM;
    constexpr int kChunks = KT::kHeadDim / 16;
    static_assert(KT::kHeadDim == 512 && kChunks % kParts == 0 && (kParts & (kParts - 1)) == 0);
    static_assert(KT::Is_Q_in_regs && !KT::Share_Q_K_smem);
    constexpr bool h8 = kBlockM == 16;
    const int row = h8 ? (tidx % 8) + (tidx / (8 * kParts)) * 8 : tidx % kBlockM;
    const int part = h8 ? (tidx / 8) % kParts : tidx / kBlockM;
    const int head = h_k_idx * kBlockM + row;
    const bool live = head < p.real_heads;
    const Element* src = reinterpret_cast<const Element*>(p.q_ptr)
        + int64_t(token) * p.stride_q_token + head * 16;
    Element* dst = reinterpret_cast<Element*>(smem);
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < kChunks / kParts; ++k) {
        const int chunk = part + k * kParts;
        uint4 raw[2]{};
        if (live) {
            const uint4* g = reinterpret_cast<const uint4*>(
                src + int64_t(chunk) * (p.real_heads * 16));
            raw[0] = __ldg(g);
            raw[1] = __ldg(g + 1);
        }
        if constexpr (h8 && !P::kEnableNorm) {
            if (chunk < 28 || !live) {
                *reinterpret_cast<uint4*>(dst + native_offset<kBlockM>(row, chunk * 16)) = raw[0];
                *reinterpret_cast<uint4*>(dst + native_offset<kBlockM>(row, chunk * 16 + 8)) = raw[1];
                continue;
            }
        }
        float x[16];
        const Element* e = reinterpret_cast<const Element*>(raw);
        #pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = float(e[i]);
        if constexpr (P::kEnableNorm) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) sum = fmaf(x[i], x[i], sum);
        }
        if (chunk >= 28 && live) {
            const int position = __ldg(p.token_positions + token);
            const float* rope = p.cos_sin_cache + int64_t(position) * 64;
            #pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const int pair = (chunk * 16 + i - 448) / 2;
                const float2 rotated = apply_rope(float2{x[i], x[i + 1]},
                    __ldg(rope + pair), __ldg(rope + 32 + pair));
                x[i] = rotated.x;
                x[i + 1] = rotated.y;
            }
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            alignas(16) Element packed[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) packed[j] = Element(x[h * 8 + j]);
            *reinterpret_cast<uint4*>(dst + native_offset<kBlockM>(row, chunk * 16 + h * 8))
                = *reinterpret_cast<const uint4*>(packed);
        }
    }
    if constexpr (P::kEnableNorm) {
        float* partials = scratch<KT, ExtraFP4>(smem);
        float* norm = partials + KT::kNThreads;
        partials[part * kBlockM + row] = sum;
        __syncthreads();
        if (tidx < kBlockM) {
            static_assert(kParts == 8 || kParts == 16);
            float sums[kParts / 8];
            #pragma unroll
            for (int group = 0; group < kParts / 8; ++group) {
                const float* group_partials = partials + group * 8 * kBlockM + tidx;
                const float s01 = __fadd_rn(group_partials[0], group_partials[kBlockM]);
                const float s23 = __fadd_rn(group_partials[2 * kBlockM], group_partials[3 * kBlockM]);
                const float s45 = __fadd_rn(group_partials[4 * kBlockM], group_partials[5 * kBlockM]);
                const float s67 = __fadd_rn(group_partials[6 * kBlockM], group_partials[7 * kBlockM]);
                sums[group] = __fadd_rn(__fadd_rn(s01, s23), __fadd_rn(s45, s67));
            }
            float sqr = sums[0];
            if constexpr (kParts == 16) {
                sqr = __fadd_rn(sums[0], sums[1]);
            }
            norm[tidx] = h_k_idx * kBlockM + tidx < p.real_heads
                ? rsqrtf(sqr / 512.0f + p.rms_norm_eps) : 1.0f;
        }
        __syncthreads();
    }
}

template<typename KT, bool ExtraFP4, typename Frag, typename ThrMma>
__forceinline__ __device__ void load_row_scale(Frag& frag, ThrMma& thr_mma, char* smem) {
    const float* norm = scratch<KT, ExtraFP4>(smem) + KT::kNThreads;
    Tensor cS = make_identity_tensor(Shape<Int<KT::kBlockM>, Int<KT::kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);
    static_assert(decltype(size(tScS))::value == decltype(size(frag))::value);
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) frag(i) = norm[get<0>(tScS(i))];
}

__forceinline__ __device__ uint16_t fp8_pair_from_float(float low, float high) {
    uint16_t result;
    asm volatile("ppu.cvt.rtte.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(result) : "f"(high), "f"(low));
    return result;
}

// Reuse the retired KV storage after the FP32 output image for this epilogue.
// Each scale belongs to one token and eight complete 512-channel heads.
template<typename KT, bool H8, bool EnableNorm = true, typename P, typename TensorO, typename Retired>
__forceinline__ __device__ void store_per_token_output(
    const P& p, int64_t token, int first_head, char* smem, TensorO& sO, Retired& retired,
    const bool* valid = nullptr, const float* lse = nullptr) {
    constexpr int kGroups = H8 ? 1 : KT::kBlockM / 8;
    constexpr int kThreadsPerGroup = KT::kNThreads / kGroups;
    constexpr int kWarpsPerGroup = kThreadsPerGroup / 32;
    constexpr int kWarps = KT::kNThreads / 32;
    constexpr size_t kStageBytes = cute::cosize(typename KT::SmemLayoutO{}) * sizeof(float);
    static_assert(KT::kBlockM % 8 == 0 && KT::kHeadDimV == 512);
    static_assert(KT::kNThreads % kGroups == 0 && kThreadsPerGroup % 32 == 0);
    static_assert(kStageBytes + (kWarps + kGroups) * sizeof(float) <= KT::kSmemSizeAccum);
    float* partials = reinterpret_cast<float*>(smem + kStageBytes);
    float* inverses = partials + kWarps;
    const int tidx = threadIdx.x;
    const int group = tidx / kThreadsPerGroup;
    const int group_lane = tidx % kThreadsPerGroup;
    const int global_group = first_head / 8 + group;
    const int position = __ldg(p.token_positions + token);
    const float* rope = p.cos_sin_cache + int64_t(position) * 64;

    auto publish_inverse = [&](float amax) {
        if constexpr (P::per_token_scale && !EnableNorm) {
            // All warps in a WV group need the same maximum. After publishing
            // partials, each warp derives its own inverse, so no CTA barrier
            // follows the independent global SF write.
            if constexpr (kWarpsPerGroup > 1) {
                if (tidx % 32 == 0) partials[tidx / 32] = amax;
                __syncthreads();
            }
            float inverse = 0.0f;
            if (tidx % 32 == 0) {
                if constexpr (kWarpsPerGroup > 1) {
                    amax = 1e-4f;
                    #pragma unroll
                    for (int warp = 0; warp < kWarpsPerGroup; ++warp)
                        amax = fmaxf(amax, partials[group * kWarpsPerGroup + warp]);
                }
                const float scale = amax / 448.0f;
                inverse = 1.0f / scale;
                if (group_lane == 0 && global_group < p.h_q / 8) {
                    reinterpret_cast<float*>(p.out_sf)[token * p.stride_out_sf_token
                        + int64_t(global_group) * p.stride_out_sf_group] = scale;
                }
            }
            return __shfl_sync(0xffffffff, inverse, 0);
        } else {
            if (tidx % 32 == 0) partials[tidx / 32] = amax;
            __syncthreads();
            if (group_lane == 0) {
                #pragma unroll
                for (int warp = 0; warp < kWarpsPerGroup; ++warp)
                    amax = fmaxf(amax, partials[group * kWarpsPerGroup + warp]);
                const float scale = amax / 448.0f;
                inverses[group] = 1.0f / scale;
                if (global_group < p.h_q / 8) {
                    reinterpret_cast<float*>(p.out_sf)[token * p.stride_out_sf_token
                        + int64_t(global_group) * p.stride_out_sf_group] = scale;
                }
            }
            __syncthreads();
            return inverses[group];
        }
    };

    if constexpr (P::per_token_scale && H8 && kThreadsPerGroup == 256) {
        // Each H8 thread owns two vectors. Keep the post-RoPE FP32 bits
        // across the scale reduction instead of storing and reloading sO.
        alignas(16) float retained[2][8]{};
        float amax = 1e-4f;
        float amax1 = 0.0f, amax2 = 0.0f, amax3 = 0.0f;
        cute::for_each(make_seq<2>{}, [&](auto slot) {
            constexpr int kSlot = decltype(slot)::value;
            const int vector = group_lane + kSlot * kThreadsPerGroup;
            const int row = group * 8 + vector / 64;
            const int col = (vector % 64) * 8;
            const int head = first_head + row;
            const bool live = head < p.h_q && (valid ? valid[row]
                : (!lse || ((__float_as_uint(lse[token * p.h_q + head]) & 0x7f800000u) != 0x7f800000u)));
            auto& values = retained[kSlot];
            if (live) {
                *reinterpret_cast<float4*>(values) = *reinterpret_cast<const float4*>(&sO(row, col));
                *reinterpret_cast<float4*>(values + 4) = *reinterpret_cast<const float4*>(&sO(row, col + 4));
            }
            if (col >= 448) {
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    const int pair = (col - 448 + i) / 2;
                    const float2 x = apply_rope(float2{values[i], values[i + 1]},
                        __ldg(rope + pair), -__ldg(rope + 32 + pair));
                    values[i] = x.x;
                    values[i + 1] = x.y;
                }
            }
            if constexpr (!EnableNorm) {
                #pragma unroll
                for (int i = 0; i < 8; i += 4) {
                    amax = fmaxf(amax, fabsf(values[i]));
                    amax1 = fmaxf(amax1, fabsf(values[i + 1]));
                    amax2 = fmaxf(amax2, fabsf(values[i + 2]));
                    amax3 = fmaxf(amax3, fabsf(values[i + 3]));
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(values[i]));
            }
        });
        if constexpr (!EnableNorm)
            amax = fmaxf(fmaxf(amax, amax1), fmaxf(amax2, amax3));
        if constexpr (P::per_token_scale && !EnableNorm) {
            // Each lane maximum is positive (floor1e-4). Its unsigned IEEE
            // bits have the same ordering; SM89 lowers this to v.red.max.u32.
            amax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(amax)));
        } else {
        #pragma unroll
        for (int delta = 16; delta > 0; delta /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        }
        const float scale_inv = publish_inverse(amax);
        cute::for_each(make_seq<2>{}, [&](auto slot) {
            constexpr int kSlot = decltype(slot)::value;
            const int vector = group_lane + kSlot * kThreadsPerGroup;
            const int row = group * 8 + vector / 64;
            const int col = (vector % 64) * 8;
            if (first_head + row >= p.h_q) return;
            const auto& values = retained[kSlot];
            uint2 packed;
            #pragma unroll
            for (int i = 0; i < 8; i += 4) {
                const uint32_t low = fp8_pair_from_float(values[i] * scale_inv,
                    values[i + 1] * scale_inv);
                const uint32_t high = fp8_pair_from_float(values[i + 2] * scale_inv,
                    values[i + 3] * scale_inv);
                (i == 0 ? packed.x : packed.y) = low | (high << 16);
            }
            const int64_t offset = token * p.h_q * 512 + int64_t(global_group) * 4096
                + (col / 32) * (8 * 32) + (row % 8) * 32 + col % 32;
            *reinterpret_cast<uint2*>(p.out_fp8 + offset) = packed;
        });
        return;
    }
    if constexpr (P::per_token_scale && !EnableNorm && !H8) {
        // The MMA accumulator is retired after staging O. Keep the transformed
        // values there until the group scale is ready instead of staging twice.
        constexpr int kVectors = 512 / kThreadsPerGroup;
        static_assert(decltype(cute::size(retired))::value >= kVectors * 8);
        float amax = 1e-4f;
        float amax1 = 0.0f, amax2 = 0.0f, amax3 = 0.0f;
        cute::for_each(make_seq<kVectors>{}, [&](auto slot) {
        constexpr int kSlot = decltype(slot)::value;
        const int vector = group_lane + kSlot * kThreadsPerGroup;
        const int row = group * 8 + vector / 64;
        const int col = (vector % 64) * 8;
        const int head = first_head + row;
        const bool live = head < p.h_q && (valid ? valid[row]
            : (!lse || ((__float_as_uint(lse[token * p.h_q + head]) & 0x7f800000u) != 0x7f800000u)));
        alignas(16) float values[8]{};
        if (live) {
            *reinterpret_cast<float4*>(values) = *reinterpret_cast<const float4*>(&sO(row, col));
            *reinterpret_cast<float4*>(values + 4) = *reinterpret_cast<const float4*>(&sO(row, col + 4));
        }
        if (col >= 448) {
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int pair = (col - 448 + i) / 2;
                const float2 x = apply_rope(float2{values[i], values[i + 1]},
                    __ldg(rope + pair), -__ldg(rope + 32 + pair));
                values[i] = x.x;
                values[i + 1] = x.y;
            }
        }
        if constexpr (!EnableNorm) {
                #pragma unroll
                for (int i = 0; i < 8; i += 4) {
                    amax = fmaxf(amax, fabsf(values[i]));
                    amax1 = fmaxf(amax1, fabsf(values[i + 1]));
                    amax2 = fmaxf(amax2, fabsf(values[i + 2]));
                    amax3 = fmaxf(amax3, fabsf(values[i + 3]));
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(values[i]));
            }
        #pragma unroll
        for (int i = 0; i < 8; ++i) retired(kSlot * 8 + i) = values[i];
        });
    if constexpr (!EnableNorm)
            amax = fmaxf(fmaxf(amax, amax1), fmaxf(amax2, amax3));
    if constexpr (P::per_token_scale && !EnableNorm) {
            // Each lane maximum is positive (floor1e-4). Its unsigned IEEE
            // bits have the same ordering; SM89 lowers this to v.red.max.u32.
            amax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(amax)));
        } else {
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        }
    const float scale_inv = publish_inverse(amax);
        cute::for_each(make_seq<kVectors>{}, [&](auto slot) {
        constexpr int kSlot = decltype(slot)::value;
        const int vector = group_lane + kSlot * kThreadsPerGroup;
        const int row = group * 8 + vector / 64;
        const int col = (vector % 64) * 8;
        if (first_head + row >= p.h_q) return;
        uint2 packed;
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
            const uint32_t low = fp8_pair_from_float(retired(kSlot * 8 + i) * scale_inv,
                retired(kSlot * 8 + i + 1) * scale_inv);
            const uint32_t high = fp8_pair_from_float(retired(kSlot * 8 + i + 2) * scale_inv,
                retired(kSlot * 8 + i + 3) * scale_inv);
            (i == 0 ? packed.x : packed.y) = low | (high << 16);
        }
        const int64_t offset = token * p.h_q * 512 + int64_t(global_group) * 4096
            + (col / 32) * (8 * 32) + (row % 8) * 32 + col % 32;
        *reinterpret_cast<uint2*>(p.out_fp8 + offset) = packed;
        });
        return;
    }
    float amax = 1e-4f;
    #pragma unroll
    for (int vector = group_lane; vector < 4096 / 8; vector += kThreadsPerGroup) {
        const int row = group * 8 + vector / 64;
        const int col = (vector % 64) * 8;
        const int head = first_head + row;
        const bool live = head < p.h_q && (valid ? valid[row]
            : (!lse || ((__float_as_uint(lse[token * p.h_q + head]) & 0x7f800000u) != 0x7f800000u)));
        alignas(16) float values[8]{};
        if (live) {
            *reinterpret_cast<float4*>(values) = *reinterpret_cast<const float4*>(&sO(row, col));
            *reinterpret_cast<float4*>(values + 4) = *reinterpret_cast<const float4*>(&sO(row, col + 4));
        }
        if (col >= 448) {
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int pair = (col - 448 + i) / 2;
                const float2 x = apply_rope(float2{values[i], values[i + 1]},
                    __ldg(rope + pair), -__ldg(rope + 32 + pair));
                values[i] = x.x;
                values[i + 1] = x.y;
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(values[i]));
        *reinterpret_cast<float4*>(&sO(row, col)) = *reinterpret_cast<const float4*>(values);
        *reinterpret_cast<float4*>(&sO(row, col + 4)) = *reinterpret_cast<const float4*>(values + 4);
    }
    if constexpr (P::per_token_scale && !EnableNorm) {
            // Each lane maximum is positive (floor1e-4). Its unsigned IEEE
            // bits have the same ordering; SM89 lowers this to v.red.max.u32.
            amax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(amax)));
        } else {
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        }
    const float scale_inv = publish_inverse(amax);
    #pragma unroll
    for (int vector = group_lane; vector < 4096 / 8; vector += kThreadsPerGroup) {
        const int row = group * 8 + vector / 64;
        const int col = (vector % 64) * 8;
        if (first_head + row >= p.h_q) continue;
        uint2 packed;
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
            const uint32_t low = fp8_pair_from_float(sO(row, col + i) * scale_inv,
                sO(row, col + i + 1) * scale_inv);
            const uint32_t high = fp8_pair_from_float(sO(row, col + i + 2) * scale_inv,
                sO(row, col + i + 3) * scale_inv);
            (i == 0 ? packed.x : packed.y) = low | (high << 16);
        }
        const int64_t offset = token * p.h_q * 512 + int64_t(global_group) * 4096
            + (col / 32) * (8 * 32) + (row % 8) * 32 + col % 32;
        *reinterpret_cast<uint2*>(p.out_fp8 + offset) = packed;
    }
}

template<typename KT>
constexpr bool h8_output_float4_layout() {
    typename KT::SmemLayoutO layout{};
    for (int head = 0; head < 8; ++head) {
        for (int col = 0; col < KT::kHeadDimV; col += 4) {
            const int offset = layout(head, col);
            if (offset % 4 != 0) return false;
            for (int i = 1; i < 4; ++i) {
                if (layout(head, col + i) != offset + i) return false;
            }
        }
    }
    return true;
}

template<typename KT, typename P, typename TensorO>
__forceinline__ __device__ void store_h8_output(const P& p, int token, int tidx, const TensorO& sO) {
    constexpr int kHeads = 8;
    constexpr int kChunks = KT::kHeadDimV / 32;
    constexpr int kLanesPerScale = 2;
    constexpr int kChannelsPerLane = 32 / kLanesPerScale;
    static_assert(KT::kNThreads == kHeads * kChunks * kLanesPerScale);
    static_assert(h8_output_float4_layout<KT>());

    // Each contiguous pair owns one 32-channel scale.
    const int scale_index = tidx / kLanesPerScale;
    const int head = scale_index % kHeads;
    const int chunk = scale_index / kHeads;
    const int lane = tidx % kLanesPerScale;
    const int col = chunk * 32 + lane * kChannelsPerLane;
    alignas(16) float values[kChannelsPerLane];
    #pragma unroll
    for (int i = 0; i < kChannelsPerLane; i += 4) {
        const float4 values4 = *reinterpret_cast<const float4*>(&sO(head, col + i));
        *reinterpret_cast<float4*>(values + i) = values4;
    }
    if (col >= 448) {
        const int position = __ldg(p.token_positions + token);
        const float* rope = p.cos_sin_cache + int64_t(position) * 64;
        #pragma unroll
        for (int i = 0; i < kChannelsPerLane; i += 2) {
            const int pair = (col - 448 + i) / 2;
            const float2 rotated = apply_rope(float2{values[i], values[i + 1]},
                __ldg(rope + pair), -__ldg(rope + 32 + pair));
            values[i] = rotated.x;
            values[i + 1] = rotated.y;
        }
    }

    float amax = 1e-4f;
    #pragma unroll
    for (int i = 0; i < kChannelsPerLane; ++i) amax = fmaxf(amax, fabsf(values[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, kLanesPerScale));
    const float scale = amax / 448.0f;
    const int exponent = int((__float_as_uint(scale) - 1u) >> 23) + 1 - 127;
    const float scale_inv = __uint_as_float(uint32_t(127 - exponent) << 23);
    uint32_t words[4];
    #pragma unroll
    for (int i = 0; i < kChannelsPerLane / 4; ++i) {
        const int d = i * 4;
        const uint32_t pair_lo = fp8_pair_from_float(values[d] * scale_inv, values[d + 1] * scale_inv);
        const uint32_t pair_hi = fp8_pair_from_float(values[d + 2] * scale_inv, values[d + 3] * scale_inv);
        words[i] = pair_lo | (pair_hi << 16);
    }
    const int64_t out_offset = int64_t(token) * kHeads * KT::kHeadDimV + scale_index * 32
        + lane * kChannelsPerLane;
    *reinterpret_cast<uint4*>(p.out_fp8 + out_offset) = make_uint4(words[0], words[1], words[2], words[3]);
    if (lane == 0) {
        auto* sf_word = p.out_sf + token + int64_t(scale_index / 4) * p.stride_out_sf_word;
        reinterpret_cast<uint8_t*>(sf_word)[scale_index % 4] = uint8_t(exponent + 127);
    }
}

template<typename KT, typename P, typename AccO, typename Softmax>
__forceinline__ __device__ void store_output(
    const P& p, int token, int h_k_idx, char* smem, AccO& acc_o, Softmax& softmax) {
    constexpr int kBlockM = KT::kBlockM;
    constexpr int kChunks = KT::kHeadDimV / 32;
    constexpr int MMA_ATOM_M = KT::USE_MMA_M8 ? 8 : 16;
    constexpr int MMA_ATOM_K_M = KT::USE_MMA_M8 ? 1 : 2;
    static_assert(KT::CrossCut && !KT::QKV_FP8 && KT::kHeadDimV == 512);
    const int tidx = threadIdx.x;
    const int warp = tidx / 32;
    const int lane = tidx % 32;
    constexpr bool kDeferredLSE = P::per_token_scale && !P::kEnableNorm;
    constexpr size_t kStageBytes = cute::cosize(typename KT::SmemLayoutO{}) * sizeof(float);
    constexpr size_t kPartialsBytes = (KT::kNThreads / 32 + kBlockM / 8) * sizeof(float);
    if constexpr (kDeferredLSE)
        static_assert(kStageBytes + kPartialsBytes + kBlockM * sizeof(bool) <= KT::kSmemSizeAccum);
    bool* row_valid = reinterpret_cast<bool*>(smem + KT::kSmemSizeAccum - kBlockM * sizeof(bool));
    auto publish_lse = [&](auto local_only) {
    if (warp < KT::AtomLayoutQ && lane % 4 == 0) {
        const int row_base = warp * MMA_ATOM_M + lane / 4;
        #pragma unroll
        for (int mi = 0; mi < size(softmax.lse); ++mi) {
            const int row = row_base + (mi / MMA_ATOM_K_M) * (MMA_ATOM_M * KT::AtomLayoutQ)
                + (mi % MMA_ATOM_K_M) * 8;
            const int head = h_k_idx * kBlockM + row;
            if constexpr (decltype(local_only)::value) {
                row_valid[row] = (__float_as_uint(softmax.lse(mi)) & 0x7f800000u) != 0x7f800000u;
            } else {
                if (head < p.real_heads) p.lse_out[int64_t(token) * p.real_heads + head] = softmax.lse(mi);
            }
        }
    }
    };
    // The staging barrier publishes local validity. Keep the public LSE store
    // after quantization so this path does not reread its own global output.
    publish_lse(cute::bool_constant<kDeferredLSE>{});
    // H8 stages O below the row scales, so its PV warps can finish rescaling independently.
    if constexpr (kBlockM != 16) __syncthreads();

    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem)), typename KT::SmemLayoutO{});
    typename KT::TiledMma tiled_mma;
    auto smem_tiled_copy_O = make_tiled_copy_C(typename KT::SmemCopyAtomOaccum{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOrO = smem_thr_copy_O.retile_S(acc_o);
    Tensor tOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, tOrO, tOsO);
    __syncthreads();

    if (p.per_token_scale) {
        store_per_token_output<KT, kBlockM == 16, P::kEnableNorm>(
            p, token, h_k_idx * kBlockM, smem, sO, acc_o,
            kDeferredLSE ? row_valid : nullptr, kDeferredLSE ? nullptr : p.lse_out);
        if constexpr (kDeferredLSE) publish_lse(cute::false_type{});
        return;
    }
    if constexpr (kBlockM == 16) {
        store_h8_output<KT>(p, token, tidx, sO);
        return;
    }

    const int position = __ldg(p.token_positions + token);
    const float* rope = p.cos_sin_cache + int64_t(position) * 64;
    #pragma unroll 1
    for (int item = tidx; item < kBlockM * kChunks; item += KT::kNThreads) {
        const int row = item % kBlockM;
        const int chunk = item / kBlockM;
        const int head = h_k_idx * kBlockM + row;
        if (head >= p.real_heads) continue;
        float values[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) values[i] = sO(row, chunk * 32 + i);
        if (chunk >= 14) {
            #pragma unroll
            for (int i = 0; i < 32; i += 2) {
                const int pair = (chunk - 14) * 16 + i / 2;
                const float2 rotated = apply_rope(float2{values[i], values[i + 1]},
                    __ldg(rope + pair), -__ldg(rope + 32 + pair));
                values[i] = rotated.x;
                values[i + 1] = rotated.y;
            }
        }
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(values[i]));
        amax = fmaxf(1e-4f, amax);
        const float scale = amax / 448.0f;
        const int exponent = int((__float_as_uint(scale) - 1u) >> 23) + 1 - 127;
        const float scale_inv = __uint_as_float(uint32_t(127 - exponent) << 23);
        alignas(16) uint8_t packed[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            packed[i] = cutlass::float_e4m3_t::from_float(values[i] * scale_inv).storage;
        }
        const int group = head / 8;
        const int in_group = head & 7;
        const int64_t out_offset = int64_t(token) * p.real_heads * 512
            + group * (8 * 512) + chunk * (8 * 32) + in_group * 32;
        uint4* out = reinterpret_cast<uint4*>(p.out_fp8 + out_offset);
        out[0] = reinterpret_cast<const uint4*>(packed)[0];
        out[1] = reinterpret_cast<const uint4*>(packed)[1];
        const int sf_index = chunk * 8 + in_group;
        auto* sf_word = p.out_sf + token + int64_t(group) * p.stride_out_sf_group
            + int64_t(sf_index / 4) * p.stride_out_sf_word;
        reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
    }
}

}

template<bool EnableNorm, bool PerTokenScale>
struct FusedSplitKVParams : Flash_fwd_params {
    float rms_norm_eps;
    const int32_t* token_positions;
    const float* cos_sin_cache;
    uint8_t* out_fp8;
    uint32_t* out_sf;
    float* lse_out;
    int64_t stride_out_sf_group;
    int64_t stride_out_sf_word;
    int64_t stride_out_sf_token;
    static constexpr bool per_token_scale = PerTokenScale;
    int64_t stride_q_token;
    int real_heads;

    static constexpr bool kFusedSplitKV = true;
    static constexpr bool kEnableNorm = EnableNorm;

    template<typename KT, bool ExtraFP4>
    static __forceinline__ __device__ void load_q(
        const FusedSplitKVParams& p, int token, int h_k_idx, char* smem, int tidx) {
        splitkv::load_q<KT, ExtraFP4>(p, token, h_k_idx, smem, tidx);
    }

    template<typename KT, bool ExtraFP4, typename Frag, typename ThrMma>
    static __forceinline__ __device__ void load_row_scale(Frag& frag, ThrMma& thr_mma, char* smem) {
        splitkv::load_row_scale<KT, ExtraFP4>(frag, thr_mma, smem);
    }

    template<typename KT, typename AccO, typename Softmax>
    static __forceinline__ __device__ void store_output(
        const FusedSplitKVParams& p, int token, int h_k_idx, char* smem, AccO& acc_o, Softmax& softmax) {
        splitkv::store_output<KT>(p, token, h_k_idx, smem, acc_o, softmax);
    }
};

template<typename KT, bool EnableNorm, bool ExtraFP4, bool PerTokenScale>
__global__ void __launch_bounds__(KT::kNThreads, 1, 1)
fused_norm_rope_attn_rope_cast_decode_splitkv_kernel(
    __grid_constant__ const FusedSplitKVParams<EnableNorm, PerTokenScale> params) {
    constexpr int kBlockN = KT::kBlockN;
    constexpr int kMetadataBlockN = 64;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    if constexpr (KT::kBlockM == 16) {
        if (!params.tile_scheduler_metadata_ptr) {
            // The H8 direct dispatch guarantees full 128 + 512 index rows.
            constexpr int kMainTopk = 128;
            constexpr int kExtraTopk = 512;
            const int rows = params.b / params.num_sm_parts;
            const int remainder = params.b % params.num_sm_parts;
            const int begin = partition_idx * rows + cute::min(partition_idx, remainder);
            const int end = begin + rows + int(partition_idx < remainder);
            #pragma unroll 1
            for (int token = begin; token < end; ++token) {
                flash::compute_attn_fp8_sparse_splitkv<KT, true, ExtraFP4>(
                    params, token, 0, 0, 0, 0, (kMainTopk + kExtraTopk) / kBlockN,
                    true, kMainTopk, kMainTopk / kBlockN, kExtraTopk);
                __syncthreads();
            }
            return;
        }
    }
    const int* metadata = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    const int4 tile = __ldg(reinterpret_cast<const int4*>(metadata));
    const int begin_idx = tile.x;
    const int begin_seqlen = tile.y;
    const int end_idx = tile.z;
    const int end_seqlen = tile.w;
    if (begin_idx >= params.b) return;
    const int begin_n_split_idx = __ldg(metadata + 4);

    #pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = params.topk_len_ptr ? params.topk_len_ptr[batch_id] : params.topk;
        int seqlen_kpad = cute::max(seqlen_k, 1);
        int extra_seqlen_k = 0;
        if (params.extra_topk > 0) {
            seqlen_kpad = cute::round_up(seqlen_kpad, kMetadataBlockN);
            extra_seqlen_k = params.extra_topk_len_ptr
                ? params.extra_topk_len_ptr[batch_id] : params.extra_topk;
        }
        const int total_k = seqlen_kpad + extra_seqlen_k;
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        const int n_block_max = batch_id == end_idx
            ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(total_k, kBlockN);
        const bool no_split = n_block_min == 0 && n_block_max == cute::ceil_div(total_k, kBlockN);
        const int ori_block_max = cute::ceil_div(seqlen_kpad, kBlockN);
        flash::compute_attn_fp8_sparse_splitkv<KT, true, ExtraFP4>(
            params, batch_id, bidh, m_block, n_split_idx,
            n_block_min, n_block_max, no_split, seqlen_k,
            ori_block_max, extra_seqlen_k);
        __syncthreads();
    }
}

template<int Heads, bool EnableNorm, bool ExtraFP4, bool PerTokenScale>
void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel_impl(
    const Params& params, const SplitKVParams& split) {
    static_assert(Heads == 8 || Heads == 64 || Heads == 128);
    TORCH_CHECK(params.mode == FwdMode::Decode && params.h_q == Heads
        && params.enable_q_norm == EnableNorm && params.extra_fp4 == ExtraFP4);
    TORCH_CHECK(params.wv_group_size == 8 && params.n_wv_group == Heads / 8);
    TORCH_CHECK(params.page_block_size > 0 && split.num_sm_parts > 0);
    TORCH_CHECK(params.stride_q_s_q % 8 == 0 && reinterpret_cast<uintptr_t>(params.q) % 16 == 0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(params.out_fp8) % 16 == 0);
    if (params.s_q == 0) return;
    const bool has_extra = params.extra_kv != nullptr && params.extra_topk > 0;
    if (has_extra) TORCH_CHECK(params.extra_page_block_size > 0);

    FusedSplitKVParams<EnableNorm, PerTokenScale> p{};
    p.b = params.s_q;
    p.q_orig = 1;
    p.seqlen_q = Heads;
    p.d = 512;
    p.d_v = 512;
    p.h = 1;
    p.h_q = Heads;
    p.h_h_k_ratio = 1;
    p.ngroups = Heads;
    p.scale_softmax = params.sm_scale;
    p.scale_softmax_log2 = params.sm_scale_log2;
    p.q_ptr = const_cast<void*>(params.q);
    p.k_ptr = const_cast<void*>(params.kv);
    p.indices_ptr = const_cast<int*>(params.indices);
    p.topk_len_ptr = const_cast<int*>(params.topk_length);
    p.attn_sink_ptr = const_cast<float*>(params.attn_sink);
    p.q_batch_stride = params.stride_q_s_q;
    p.q_row_stride = 512;
    p.k_batch_stride = params.stride_kv_s_kv;
    p.k_row_stride = 528;
    p.indices_batch_stride = params.stride_indices_s_q;
    p.page_block_size = params.page_block_size;
    p.num_blocks = params.num_blocks;
    p.topk = params.topk;
    if (has_extra) {
        p.extra_k_ptr = const_cast<void*>(params.extra_kv);
        p.extra_indices_ptr = const_cast<int*>(params.extra_indices);
        p.extra_topk_len_ptr = const_cast<int*>(params.extra_topk_length);
        p.extra_k_batch_stride = params.stride_extra_kv_block;
        p.extra_k_row_stride = ExtraFP4 ? 288 : 528;
        p.extra_indices_batch_stride = params.stride_extra_indices_s_q;
        p.extra_num_blocks = params.extra_num_blocks;
        p.extra_page_block_size = params.extra_page_block_size;
        p.extra_topk = params.extra_topk;
    } else {
        p.extra_topk = -1;
    }
    p.tile_scheduler_metadata_ptr = split.tile_scheduler_metadata;
    p.num_sm_parts = split.num_sm_parts;
    p.num_splits_ptr = split.num_splits;
    p.softmax_lseaccum_ptr = split.lse_accum;
    p.oaccum_ptr = split.o_accum;
    p.lse_to_log2 = float(M_LOG2E);
    p.rms_norm_eps = params.rms_norm_eps;
    p.token_positions = params.token_positions;
    p.cos_sin_cache = params.cos_sin_cache;
    p.out_fp8 = params.out_fp8;
    p.out_sf = params.out_sf;
    p.lse_out = params.lse;
    p.stride_out_sf_group = params.stride_out_sf_group;
    p.stride_out_sf_word = params.stride_out_sf_word;
    p.stride_out_sf_token = params.stride_out_sf_token;
    p.stride_q_token = params.stride_q_s_q;
    p.real_heads = Heads;

    IS_PAGE_POWER2(p.page_block_size, p.extra_page_block_size, [&] {
        using KT = splitkv::Traits<Heads, kPagePow2>;
        constexpr size_t smem_size = splitkv::scratch_offset<KT, ExtraFP4>() + splitkv::scratch_bytes<KT>();
        static_assert(smem_size <= 256 * 1024);
        auto kernel = &fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<KT, EnableNorm, ExtraFP4, PerTokenScale>;
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(hggcFuncSetAttribute(
                kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<dim3(cute::ceil_div(Heads, KT::kBlockM), 1, split.num_sm_parts),
            KT::kNThreads, smem_size, params.stream>>>(p);
        CHECK_CUDA_KERNEL_LAUNCH();
    });

    if constexpr (Heads == 8) {
        if (!split.tile_scheduler_metadata) return;
    }
    combine::Params c{};
    c.per_token_scale = PerTokenScale;
    c.s_q = params.s_q;
    c.h_q = Heads;
    c.num_sm_parts = split.num_sm_parts;
    c.num_splits = split.num_splits;
    c.lse_accum = split.lse_accum;
    c.o_accum = split.o_accum;
    c.attn_sink = params.attn_sink;
    c.token_positions = params.token_positions;
    c.cos_sin_cache = params.cos_sin_cache;
    c.out_fp8 = params.out_fp8;
    c.out_sf = params.out_sf;
    c.lse = params.lse;
    c.stride_out_sf_group = params.stride_out_sf_group;
    c.stride_out_sf_word = params.stride_out_sf_word;
    c.stride_out_sf_token = params.stride_out_sf_token;
    c.stream = params.stream;
    combine::run_fused_norm_rope_attn_rope_cast_combine_kernel(c);
}

template<int Heads, bool EnableNorm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel(
    const Params& params, const SplitKVParams& split) {
    if (params.per_token_scale) {
        run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel_impl<Heads, EnableNorm, ExtraFP4, true>(params, split);
    } else {
        run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel_impl<Heads, EnableNorm, ExtraFP4, false>(params, split);
    }
}

}
