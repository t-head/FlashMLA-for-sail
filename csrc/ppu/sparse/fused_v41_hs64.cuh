#pragma once

#include <cstddef>
#include <type_traits>
#include "api/common.h"
#include "fused_v41_wi.cuh"
#include "decode/sparse/fused_v41_splitkv.cuh"
#include "decode/sparse/sparse_decode_wg_hs64.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

#if ACOMPUTE_VERSION == 10000
struct Sm80SignedFp8Table {
    uint16_t words[1 << 13];

    static constexpr uint8_t encode(uint32_t bits) {
        const uint32_t magnitude = bits & 0x7fffffffu;
        const uint32_t sign = (bits >> 24) & 0x80u;
        if (magnitude > 0x7f800000u) return 0x7f;
        if (magnitude >= 0x43e00000u) return uint8_t(sign | 0x7e);
        if (magnitude >= 0x3c800000u) {
            const uint32_t rounded = magnitude - 0x3c000000u + 0x7ffffu
                + ((magnitude >> 20) & 1u);
            return uint8_t(sign | (rounded >> 20));
        }
        if (magnitude <= 0x3a800000u) return uint8_t(sign);
        // Scale by the E4M3 subnormal quantum (2^-9) and round to even.
        const uint32_t shift = 141 - (magnitude >> 23);
        const uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        const uint32_t rounded = mantissa + (1u << (shift - 1)) - 1u
            + ((mantissa >> shift) & 1u);
        return uint8_t(sign | (rounded >> shift));
    }

    constexpr Sm80SignedFp8Table(bool zero_nan = false) : words{} {
        for (uint32_t key = 0; key < (1u << 13); ++key) {
            const uint32_t boundary = key << 19;
            const uint16_t tie = encode(boundary);
            const uint16_t successor = encode(boundary | 1u);
            words[key] = uint16_t((zero_nan && tie == 0x7f ? 0 : tie)
                | ((zero_nan && successor == 0x7f ? 0 : successor) << 8));
        }
    }
};

inline __device__ const Sm80SignedFp8Table kSm80SignedFp8Table{};
inline __device__ const Sm80SignedFp8Table kSm80ZeroNanFp8Table{true};

__forceinline__ __device__ uint8_t sm80_signed_fp8_from_float(
    float value, const uint16_t* table = kSm80SignedFp8Table.words) {
    const uint32_t bits = __float_as_uint(value);
    // Issue the lookup without waiting for the sticky-bit calculation.
    const uint32_t entry = __ldg(table + (bits >> 19));
    const uint32_t shift = uint32_t((bits & 0x7ffffu) != 0) * 8;
    return uint8_t(entry >> shift);
}

__forceinline__ __device__ uint32_t sm80_pack_saturating_pair(
    uint32_t x, uint32_t y, uint32_t packed_offset) {
    const uint32_t original_pair = __byte_perm(x, y, 0x7632);
    // The midpoint-up offset has zero low bits. Saturating the two
    // magnitude halfwords yields a bounded approximation for subnormals.
    const uint32_t magnitude = __vsubus2(original_pair & 0x7fff7fffu, packed_offset);
    uint32_t signed_pair;
    asm("ppu.lop3.b32 %0, %1, %2, %3, 0xca;"
        : "=r"(signed_pair) : "r"(0x00800080u), "r"(original_pair >> 8), "r"(magnitude >> 4));
    return signed_pair;
}

__forceinline__ __device__ uint32_t sm80_pack_four_saturating(
    const float* values, uint32_t packed_offset) {
    const uint32_t low = sm80_pack_saturating_pair(
        __float_as_uint(values[0]), __float_as_uint(values[1]), packed_offset);
    const uint32_t high = sm80_pack_saturating_pair(
        __float_as_uint(values[2]), __float_as_uint(values[3]), packed_offset);
    return __byte_perm(low, high, 0x6420);
}
#endif

template<bool PerTokenScale>
struct FusedHs64Params : Flash_fwd_mla_params {
    float rms_norm_eps;
    const int32_t* token_positions;
    const float* cos_sin_cache;
    uint8_t* out_fp8;
    uint32_t* out_sf;
    static constexpr bool per_token_scale = PerTokenScale;
    float* max_logits;
    int64_t stride_q_s_q;
    int64_t stride_out_sf_token;
    int64_t stride_out_sf_group;
    int64_t stride_out_sf_word;
};

template<bool Norm, typename Base, bool WideOutput = false, bool SkipNextKPrefetch = false,
         bool PerTokenScale = false>
struct FusedHs64Traits : Base {
    using FusedParams = FusedHs64Params<PerTokenScale>;
    static constexpr bool kPerTokenScale = PerTokenScale;
    using SharedMemoryPlan = typename Base::SharedMemoryPlan;
    using SmemLayoutO_Half64x256 = typename Base::SmemLayoutO_Half;
    static constexpr bool kFusedHs64 = true;
    static constexpr bool kFusedDirectQueries = false;
    static constexpr bool kSkipNextKPrefetch = SkipNextKPrefetch;
    static constexpr bool kPersistentPrefillQueries = false;
    static constexpr bool kSinglePrefillPair = false;
    static constexpr bool kVectorPrefillOutput = false;
    static constexpr bool kAlignedQWithSink = false;
    static constexpr bool kMidQSaturatingPack = false;
    static constexpr int kFixedPrefillBlocks = 0;
    static constexpr bool kCachePrefillRope = false;
    static constexpr bool kDeferPrefillSink = false;
    static constexpr bool kFusedPackedFp4 =
        Base::kExtraFP4 && flashmla::dsa::hs64::kPackedMain<Base> != 0;
    // Native PV register coordinates differ; SM80 rotates adjacent columns
    // from the common output tile after staging its accumulators.
    static constexpr bool kRegisterOutputRope = kFusedPackedFp4 && Base::kArch == 89;
    static constexpr bool kCacheRms = Norm && kFusedPackedFp4 &&
        flashmla::dsa::hs64::kPackedMain<Base> == 128;
    static_assert(!kFusedPackedFp4 ||
                  (WideOutput && !Base::kIsPrefill && !Base::kHasExtraKTile && !Base::kParkQ5Fragment));
    static constexpr int kQBarrierThreads = Base::kIsPrefill ? Base::kQBarrierThreads : Base::NUM_THREADS;
    static constexpr size_t kNormScratchOffset = Base::kIsPrefill && Base::kParkQ5Fragment ? 200192
        : offsetof(SharedMemoryPlan, smem_cross_n_reduction)
            + sizeof(((SharedMemoryPlan*)nullptr)->smem_cross_n_reduction);
    static constexpr size_t kNormOffset = kNormScratchOffset + 512 * sizeof(float);
    static constexpr size_t kNormEnd = kNormOffset + 64 * sizeof(float);
    static constexpr size_t kPrefillRopeOffset = kNormEnd;
    static constexpr size_t kPrefillSinkOffset = kPrefillRopeOffset + 128 * sizeof(float);
    static_assert(sizeof(SharedMemoryPlan) == 262144);
    static_assert(!Base::kParkQ5Fragment || offsetof(SharedMemoryPlan, barriers_K1)
        + sizeof(((SharedMemoryPlan*)nullptr)->barriers_K1) <= kNormScratchOffset);
    static_assert(kNormEnd <= offsetof(SharedMemoryPlan, smem_sP));
    static_assert(!Base::kIsPrefill || kPrefillSinkOffset + 64 * sizeof(float)
                  <= offsetof(SharedMemoryPlan, smem_sP));
    static_assert(offsetof(SharedMemoryPlan, smem_sQ) == 0);
    static_assert(offsetof(SharedMemoryPlan, smem_sK) == 65536);
    static_assert(!Base::kIsPrefill || offsetof(SharedMemoryPlan, smem_valid_indices) ==
        (Base::kParkQ5Fragment ? 199168 : 221184));
    static_assert(offsetof(SharedMemoryPlan, smem_sP) == 204800);
    static_assert(offsetof(SharedMemoryPlan, smem_even_high) == 229376);
    static_assert(cosize_v<SmemLayoutO_Half64x256> * sizeof(float) == 65536);
    static_assert(Base::kBlockM == 64 && Base::kBlockN == 64 && Base::NUM_THREADS == 512);
    static_assert(Base::kUsePairedCubes == (Base::kIsPrefill && Base::kArch == 89) && Base::kUsePv2x4);

    static __forceinline__ __device__ int q_offset(int row, int dim) {
        if constexpr (Base::kUsePairedCubes) return paired_q_offset<64>(row, dim);
        else return native_offset<64>(row, dim);
    }

    static __forceinline__ __device__ float* prefill_rope_slot(int batch_idx) {
        extern __shared__ char wksp_buf[];
        // The persistent CTA prefetches the following row before storing the
        // current row. Keep both coefficient rows available across that overlap.
        return reinterpret_cast<float*>(wksp_buf + kPrefillRopeOffset) + (batch_idx & 1) * 64;
    }

    static __forceinline__ __device__ void cache_prefill_sink(
        const Flash_fwd_mla_params& params, int tidx) {
        if (params.attn_sink_ptr != nullptr && tidx < 64) {
            extern __shared__ char wksp_buf[];
            auto* sink_logits = reinterpret_cast<float*>(wksp_buf + kPrefillSinkOffset);
            sink_logits[tidx] = __ldg(params.attn_sink_ptr + tidx);
        }
    }

    template<typename T>
    static constexpr bool kUsePrefillRopeCache = T::kCachePrefillRope
        || (Base::kIsPrefill && Base::kArch == 80 && !Norm && !kPerTokenScale);

    template<typename T>
    static __forceinline__ __device__ void prefetch_prefill_rope(
        const FusedParams& fused, int batch_idx, int tidx) {
        if constexpr (kUsePrefillRopeCache<T>) {
            static_assert(kPrefillRopeOffset + 128 * sizeof(float)
                          <= offsetof(SharedMemoryPlan, smem_sP));
            if (tidx < 64) {
                const int position = __ldg(fused.token_positions + batch_idx);
                const float* rope = fused.cos_sin_cache + int64_t(position) * 64;
                __ppu_pipeline_memcpy_async_zfill(prefill_rope_slot(batch_idx) + tidx,
                    rope + tidx, sizeof(float), 0, 1);
            }
        }
    }

    template<typename T>
    static __forceinline__ __device__ const float* q_rope(
        const FusedParams& fused, int batch_idx) {
        if constexpr (kUsePrefillRopeCache<T>) {
            return prefill_rope_slot(batch_idx);
        } else {
            const int position = __ldg(fused.token_positions + batch_idx);
            return fused.cos_sin_cache + int64_t(position) * 64;
        }
    }

    template<typename T>
    static __forceinline__ __device__ float rope_coeff(const float* rope, int i) {
        if constexpr (kUsePrefillRopeCache<T>) return rope[i];
        else return __ldg(rope + i);
    }

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ void load_fused_q(
        const Flash_fwd_mla_params& params, int batch_idx, int, int,
        TensorQ& sQ, int tidx, int, __mbarrier_t* barrier_Q) {
        extern __shared__ char wksp_buf[];
        const auto& fused = static_cast<const FusedParams&>(params);
        auto* partials = reinterpret_cast<float*>(wksp_buf + kNormScratchOffset);
        auto* norm = reinterpret_cast<float*>(wksp_buf + kNormOffset);
        const int row = tidx % 64;
        const int part = tidx / 64;
        const auto* src = static_cast<const typename T::InputT*>(fused.q_ptr)
            + int64_t(batch_idx) * fused.stride_q_s_q;
        auto* dst = cute::raw_pointer_cast(sQ.data());
        if constexpr (!Base::kIsPrefill) {
            // The packed decode launcher checks alignment for every token.
            // Its unpaired Q tile uses the native swizzle of the FP8 decoder.
            const int head_pair = (tidx / 2) % 64;
            // Pair the source halves. Even then odd heads preserve the eight-lane bank spread.
            const int paired_head = (head_pair & ~7) | ((head_pair & 3) << 1) | ((head_pair & 4) >> 2);
            #pragma unroll
            for (int chunk = tidx; chunk < 64 * 64; chunk += Base::NUM_THREADS) {
                const int head = kFusedPackedFp4 ? paired_head : chunk % 64;
                const int dim = kFusedPackedFp4
                    ? (chunk / 128) * 16 + (chunk % 2) * 8 : (chunk / 64) * 8;
                PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                    *reinterpret_cast<const cute::uint128_t*>(src + (dim / 16) * (64 * 16) + head * 16 + dim % 16),
                    *reinterpret_cast<cute::uint128_t*>(dst + q_offset(head, dim)));
            }
            prefetch_prefill_rope<T>(fused, batch_idx, tidx);
            cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
            return;
        }
        if (T::kAlignedQWithSink || (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
            // The barrier counts the Q loaders. RMS and RoPE follow its wait
            // in finish_fused_q, so raw Q copy overlaps other work.
            if(tidx < T::kQBarrierThreads) {
                constexpr bool kCoalesceQ = WideOutput &&
                    (flashmla::dsa::hs64::kHs64PrefillGroupedQueries<T> || T::kAlignedQWithSink) &&
                    (!Norm || T::kPersistentPrefillQueries);
                static_assert(!T::kPersistentPrefillQueries || kCoalesceQ);
                constexpr int kCopyStep = T::kPersistentPrefillQueries ? 16 : 8;
                const int head_pair = (tidx / 2) % 64;
                // Pair the source halves. Even then odd heads preserve the eight-lane bank spread.
                const int copy_row = kCoalesceQ
                    ? ((head_pair & ~7) | ((head_pair & 3) << 1) | ((head_pair & 4) >> 2)) : row;
                const int dim_begin = kCoalesceQ ? (tidx / 128) * 16 + (tidx % 2) * 8 : (tidx / 64) * 128;
                #pragma unroll
                for(int i = 0; i < 128; i += kCopyStep) {
                    const int dim = dim_begin + (kCoalesceQ ? i * 4 : i);
                    PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                        *reinterpret_cast<const cute::uint128_t*>(src + (dim / 16) * (64 * 16) + copy_row * 16 + dim % 16),
                        *reinterpret_cast<cute::uint128_t*>(dst + q_offset(copy_row, dim)));
                }
                prefetch_prefill_rope<T>(fused, batch_idx, tidx);
                cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
            }
            return;
        }
        const int position = __ldg(fused.token_positions + batch_idx);
        const float* rope = fused.cos_sin_cache + int64_t(position) * 64;
        const bool aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
        float sum = 0.0f;
        #pragma unroll 1
        for(int k = 0; k < 4; ++k) {
            const int dim = (part + k * 8) * 16;
            const int offset = (dim / 16) * (64 * 16) + row * 16;
            uint4 raw[2]{};
            if(aligned) {
                raw[0] = __ldg(reinterpret_cast<const uint4*>(src + offset));
                raw[1] = __ldg(reinterpret_cast<const uint4*>(src + offset + 8));
            } else {
                auto* elements = reinterpret_cast<typename T::InputT*>(raw);
                #pragma unroll
                for(int j = 0; j < 16; ++j) elements[j] = src[offset + j];
            }
            const auto* elements = reinterpret_cast<const typename T::InputT*>(raw);
            alignas(16) typename T::InputT packed[16];
            #pragma unroll
            for(int j = 0; j < 16; j += 2) {
                float2 x{float(elements[j]), float(elements[j + 1])};
                if constexpr(Norm) {
                    sum = fmaf(x.x, x.x, sum);
                    sum = fmaf(x.y, x.y, sum);
                }
                if(dim >= 448) {
                    const int pair = (dim + j - 448) / 2;
                    x = apply_rope(x, __ldg(rope + pair), __ldg(rope + 32 + pair));
                }
                packed[j] = typename T::InputT(x.x);
                packed[j + 1] = typename T::InputT(x.y);
            }
            #pragma unroll
            for(int half = 0; half < 2; ++half) {
                const int offset = q_offset(row, dim + half * 8);
                *reinterpret_cast<uint4*>(dst + offset) = reinterpret_cast<const uint4*>(packed)[half];
            }
        }
        if constexpr (T::kCacheRms) {
            partials[row * 8 + (part ^ (row & 7))] = sum;
        } else {
            if constexpr(Norm) partials[tidx] = sum;
            __syncthreads();
            if constexpr(Norm) {
                if(tidx < 64) {
                    const float s01 = partials[tidx] + partials[64 + tidx];
                    const float s23 = partials[128 + tidx] + partials[192 + tidx];
                    const float s45 = partials[256 + tidx] + partials[320 + tidx];
                    const float s67 = partials[384 + tidx] + partials[448 + tidx];
                    const float sqr = (s01 + s23) + (s45 + s67);
                    norm[tidx] = rsqrtf(sqr / 512.0f + fused.rms_norm_eps);
                }
            }
            __syncthreads();
        }
        prefetch_prefill_rope<T>(fused, batch_idx, tidx);
        if(tidx < T::kQBarrierThreads) cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
    }

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ void finish_fused_q(
        const Flash_fwd_mla_params& params, int batch_idx, TensorQ& sQ, int tidx) {
        const auto& fused = static_cast<const FusedParams&>(params);
        const auto* src = static_cast<const typename T::InputT*>(fused.q_ptr)
            + int64_t(batch_idx) * fused.stride_q_s_q;
        if (!T::kAlignedQWithSink && (reinterpret_cast<uintptr_t>(src) & 15) != 0) return;
        extern __shared__ char wksp_buf[];
        auto* partials = reinterpret_cast<float*>(wksp_buf + kNormScratchOffset);
        auto* norm = reinterpret_cast<float*>(wksp_buf + kNormOffset);
        auto* dst = cute::raw_pointer_cast(sQ.data());
        const int row = tidx % 64;
        const int part = tidx / 64;
        const float* rope = q_rope<T>(fused, batch_idx);
        if constexpr (!Norm && (flashmla::dsa::hs64::kHs64PrefillGroupedQueries<T> ||
                                kFusedPackedFp4 || (Base::kIsPrefill && Base::kArch == 80) ||
                                (PerTokenScale && T::kFusedDirectQueries))) {
            // Split the 64 RoPE dimensions across both warpgroups.
            const int dim = 448 + part * 8;
            auto* chunk = reinterpret_cast<uint4*>(dst + q_offset(row, dim));
            const uint4 raw = *chunk;
            const auto* elements = reinterpret_cast<const typename T::InputT*>(&raw);
            alignas(16) typename T::InputT packed[8];
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                const int pair = (dim + j - 448) / 2;
                const float2 x = apply_rope(float2{float(elements[j]), float(elements[j + 1])},
                    rope_coeff<T>(rope, pair), rope_coeff<T>(rope, 32 + pair));
                packed[j] = typename T::InputT(x.x);
                packed[j + 1] = typename T::InputT(x.y);
            }
            *chunk = *reinterpret_cast<const uint4*>(packed);
            // Direct queries publish both warpgroups' rotated Q before MMA.
            if constexpr (T::kFusedDirectQueries) __syncthreads();
            return;
        }
        constexpr bool kDualRmsAccum = T::kCacheRms;
        // Split each 64-column slice across all 512 threads. Each thread
        // rotates the tail values it already loaded for the RMS sum.
        constexpr int kRmsChunk = T::kCacheRms ? 8 : 16;
        const int first = T::kCacheRms ? q_offset(row & 63, (part & 7) * 8) : 0;
        auto rms_offset = [&](int slice, int half) {
            if constexpr (T::kCacheRms) {
                static_assert(T::kBlockM == 64 && T::kHeadDim == 512);
                // All eight slices have the same column swizzle for this thread.
                if constexpr (T::kUsePairedCubes) {
                    return first + (slice & ~1) * (64 * 64) + (slice & 1) * 512;
                } else {
                    return first + slice * (64 * 64);
                }
            } else {
                return q_offset(row, (part + slice * 8) * 16 + half * 8);
            }
        };
        float sum = 0.0f;
        float odd_sum = 0.0f;
        #pragma unroll
        for(int k = 0; k < 64 / kRmsChunk; ++k) {
            const int dim = (part + k * 8) * kRmsChunk;
            alignas(16) typename T::InputT elements[kRmsChunk];
            #pragma unroll
            for(int half = 0; half < kRmsChunk / 8; ++half) {
                reinterpret_cast<uint4*>(elements)[half] = *reinterpret_cast<const uint4*>(
                    dst + rms_offset(k, half));
            }
            if constexpr(Norm) {
                #pragma unroll
                for(int j = 0; j < kRmsChunk; j += 2) {
                    const float2 x{float(elements[j]), float(elements[j + 1])};
                    sum = fmaf(x.x, x.x, sum);
                    if constexpr (kDualRmsAccum) odd_sum = fmaf(x.y, x.y, odd_sum);
                    else sum = fmaf(x.y, x.y, sum);
                }
            }
            if(T::kCacheRms ? k == 7 : dim >= 448) {
                alignas(16) typename T::InputT packed[kRmsChunk];
                #pragma unroll
                for(int j = 0; j < kRmsChunk; j += 2) {
                    const int pair = T::kCacheRms ? (part & 7) * 4 + j / 2 : (dim + j - 448) / 2;
                    const float2 x = apply_rope(float2{float(elements[j]), float(elements[j + 1])},
                        rope_coeff<T>(rope, pair), rope_coeff<T>(rope, 32 + pair));
                    packed[j] = typename T::InputT(x.x);
                    packed[j + 1] = typename T::InputT(x.y);
                }
                #pragma unroll
                for(int half = 0; half < kRmsChunk / 8; ++half) {
                    *reinterpret_cast<uint4*>(dst + rms_offset(k, half))
                        = reinterpret_cast<const uint4*>(packed)[half];
                }
            }
        }
        if constexpr (T::kCacheRms) {
            partials[row * 8 + (part ^ (row & 7))] = sum + odd_sum;
        } else if constexpr (Norm) {
            partials[tidx] = kDualRmsAccum ? sum + odd_sum : sum;
            __syncthreads();
            if(tidx < 64) {
                const float s01 = partials[tidx] + partials[64 + tidx];
                const float s23 = partials[128 + tidx] + partials[192 + tidx];
                const float s45 = partials[256 + tidx] + partials[320 + tidx];
                const float s67 = partials[384 + tidx] + partials[448 + tidx];
                const float sqr = (s01 + s23) + (s45 + s67);
                norm[tidx] = rsqrtf(sqr / 512.0f + fused.rms_norm_eps);
            }
        }
        // These paths publish Q and the RMS factors at the prolog barrier.
        if constexpr (!kFusedPackedFp4 && !T::kPersistentPrefillQueries) __syncthreads();
    }

    static __forceinline__ __device__ float2 load_rms(
        const Flash_fwd_mla_params& params, int idx_in_warpgroup) {
        extern __shared__ char wksp_buf[];
        const auto& fused = static_cast<const FusedParams&>(params);
        const auto* partials = reinterpret_cast<const float4*>(wksp_buf + kNormScratchOffset);
        const int row = flashmla::dsa::hs64::get_AorC_row_idx<Base::kAtomLayoutM>(0, idx_in_warpgroup);
        auto rms = [&](int head) {
            // XOR permutes the eight partials within the same balanced tree.
            const float4 low = partials[2 * head];
            const float4 high = partials[2 * head + 1];
            const float sqr = ((low.x + low.y) + (low.z + low.w))
                            + ((high.x + high.y) + (high.z + high.w));
            return rsqrtf(sqr / 512.0f + fused.rms_norm_eps);
        };
        return {rms(row), rms(row + 8)};
    }

    template<typename T, typename TensorP>
    static __forceinline__ __device__ void scale_fused_scores(
        TensorP& rP, int idx_in_warpgroup, float2 q_norm) {
        if constexpr (T::kCacheRms) {
            #pragma unroll
            for (int i = 0; i < size(rP); ++i) {
                flashmla::dsa::hs64::hs64_acc<Base>(rP, i) *= (i & 2) ? q_norm.y : q_norm.x;
            }
        } else if constexpr (Norm) {
            extern __shared__ char wksp_buf[];
            const auto* norm = reinterpret_cast<const float*>(wksp_buf + kNormOffset);
            typename Base::TiledMma mma;
            auto thr = mma.get_slice(idx_in_warpgroup);
            Tensor identity = make_identity_tensor(Shape<_64, _64>{});
            Tensor coords = thr.partition_C(identity);
            static_assert(decltype(size(coords))::value == decltype(size(rP))::value);
            #pragma unroll
            for(int i = 0; i < size(rP); ++i) {
                const int row = get<0>(flashmla::dsa::hs64::hs64_acc<Base>(coords, i));
                flashmla::dsa::hs64::hs64_acc<Base>(rP, i) *= norm[row];
            }
        }
    }

    template<typename T, typename TensorO>
    static __forceinline__ __device__ void store_fused_output(
        TensorO& rO, const float* rNorm, char* sO_addr,
        const Flash_fwd_mla_params& params, int batch_idx, int, int,
        int num_valid_seq_q, int warpgroup_idx, int idx_in_warpgroup) {
        static_assert(!WideOutput || (Base::kIsPrefill && (Base::kGroupPrefillQueries || !Norm))
            || kFusedPackedFp4 || (!Norm && flashmla::dsa::hs64::kPackedMain<Base> == 128
                && flashmla::dsa::hs64::kWorklist<Base> != flashmla::dsa::hs64::Hs64Worklist::Batch)
            || (T::kFusedDirectQueries && !Norm && !Base::kIsPrefill
                && Base::kHeadDim == 512 && Base::kV41FP8 && Base::kExtraFP4
                && !Base::kHasExtraKTile && !Base::kParkQ5Fragment));
        const auto& fused = static_cast<const FusedParams&>(params);
        Tensor rowcol = make_tensor(rO.data(),
            flashmla::dsa::hs64::hs64_convert_layout_acc_rowcol<T>(rO.layout()));
        static_assert(decltype(size<0>(rowcol))::value == 4);
        static_assert(decltype(size<1>(rowcol))::value == 16);
        const bool deferred_sink = T::kDeferPrefillSink &&
            ((Base::kIsPrefill && Base::kArch == 89) || T::kAlignedQWithSink ||
             params.attn_sink_ptr != nullptr);
        // The SM80 prefill caller already applies the sink scale and sets
        // all four rNorm entries to 1 before reaching this no-split hook.
        constexpr bool kSinkNormalizedInCaller =
            Base::kIsPrefill && Base::kArch == 80 && !Norm;
        if (!deferred_sink &&
            (!kSinkNormalizedInCaller || params.attn_sink_ptr == nullptr)) {
            #pragma unroll
            for(int mi = 0; mi < 4; ++mi) {
                #pragma unroll
                for(int ni = 0; ni < 16; ++ni) rowcol(mi, ni) /= rNorm[mi];
            }
        }
        typename T::TiledMmaPV mma;
        auto copy_c = make_tiled_copy_C(typename T::SmemCopyAtomOaccum{}, mma);
        auto thr = copy_c.get_slice(idx_in_warpgroup);
        Tensor src = thr.retile_S(rO);
        const int item = kFusedPackedFp4 ? idx_in_warpgroup : threadIdx.x;
        const int head = item % 64;
        const float* rope = q_rope<T>(fused, batch_idx);
        extern __shared__ char wksp_buf[];
        auto& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
        auto* sink_scales = reinterpret_cast<float*>(wksp_buf + kNormOffset);
        if constexpr (T::kDeferPrefillSink) {
            static_assert(!Norm && WideOutput);
            if (deferred_sink && item < 64) {
                // The prolog to the output epilogue published both WG sums.
                const float sum = plan.sL_reduction_wksp[head] + plan.sL_reduction_wksp[head + 128];
                float denominator = sum;
                if (params.attn_sink_ptr != nullptr) {
                    const auto* sink_logits = reinterpret_cast<const float*>(wksp_buf + kPrefillSinkOffset);
                    const float sink = sink_logits[head];
                    denominator += expf(sink - plan.smem_sM[head] * (float)M_LN2);
                }
                sink_scales[head] = sum == 0.0f || sum != sum ? __uint_as_float(0x7fc00000u)
                    : __fdividef(1.0f, denominator);
            }
        }
        if constexpr (kRegisterOutputRope) {
            static_assert(WideOutput && !T::kUsePairedCubes && T::kHeadDimV == 512 &&
                          T::kPvAtomLayoutM == 2 && T::kPvAtomLayoutN == 4);
            if (warpgroup_idx == 1) {
                // In the unpaired PV fragment, columns 448..511 occupy the
                // last four slots. Reuse each coefficient pair across four rows.
                const int pair = 8 * (idx_in_warpgroup / 64) + (idx_in_warpgroup & 3);
                const float cos0 = rope_coeff<T>(rope, pair);
                const float sin0 = -rope_coeff<T>(rope, 32 + pair);
                const float cos1 = rope_coeff<T>(rope, pair + 4);
                const float sin1 = -rope_coeff<T>(rope, 32 + pair + 4);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const float2 low = apply_rope(
                        float2{rowcol(mi, 12), rowcol(mi, 13)}, cos0, sin0);
                    const float2 high = apply_rope(
                        float2{rowcol(mi, 14), rowcol(mi, 15)}, cos1, sin1);
                    rowcol(mi, 12) = low.x;
                    rowcol(mi, 13) = low.y;
                    rowcol(mi, 14) = high.x;
                    rowcol(mi, 15) = high.y;
                }
            }
        }
        constexpr bool kNativeOutputLayout = WideOutput && T::kSinglePrefillPair
            && T::kAlignedQWithSink && Base::kIsPrefill && Base::kArch == 80
            && !Norm && !kPerTokenScale;
        // Keep the layout and packing choices independent for the multi-pair path.
        constexpr bool kVectorOutputLayout = kNativeOutputLayout || T::kVectorPrefillOutput;
        static_assert(!T::kMidQSaturatingPack || (WideOutput && !T::kSinglePrefillPair &&
            T::kAlignedQWithSink && T::kDeferPrefillSink && Base::kIsPrefill &&
            Base::kArch == 80 && !Norm && !kPerTokenScale));
        static_assert(!T::kVectorPrefillOutput || (WideOutput && T::kAlignedQWithSink &&
            !T::kSinglePrefillPair && Base::kIsPrefill && Base::kArch == 80 &&
            !Norm && !kPerTokenScale));
        auto stage_half = [&](auto& half) {
            if constexpr (kVectorOutputLayout) {
                static_assert(!T::kUsePairedCubes && T::kPvAtomLayoutM == 2 && T::kPvAtomLayoutN == 4);
                const int h0 = flashmla::dsa::hs64::get_PV_row_idx<T>(0, idx_in_warpgroup);
                const int col0 = 32 * (idx_in_warpgroup >> 7)
                    + 16 * ((idx_in_warpgroup >> 6) & 1) + (idx_in_warpgroup & 3);
                auto* output_base = reinterpret_cast<float4*>(&half(h0, col0));
                // The rotated-row swizzle only uses h[2:0] and col[5:0].
                // Subsequent fragments advance in whole 8x64 atoms.
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int row_vectors = (mi & 1) * 128 + (mi >> 1) * 512;
                    #pragma unroll
                    for (int group = 0; group < 4; ++group) {
                        const float4 output{rowcol(mi, 4 * group), rowcol(mi, 4 * group + 1),
                                            rowcol(mi, 4 * group + 2), rowcol(mi, 4 * group + 3)};
                        output_base[row_vectors + group * 1024] = output;
                    }
                }
            } else if constexpr (T::kUsePairedCubes) {
                Tensor dst = thr.partition_D(half);
                #pragma unroll
                for(int j = 0; j < size<2>(dst); ++j) {
                    #pragma unroll
                    for(int i = 0; i < size<0>(dst); ++i) {
                        const int real_i = (i & 3) | ((j & 1) << 2);
                        const int real_j = (j & ~1) | (i >> 2);
                        cute::copy(copy_c, src(real_i, _, real_j), dst(i, _, j));
                    }
                }
            } else {
                Tensor dst = thr.partition_D(half);
                cute::copy(copy_c, src, dst);
            }
        };
        // P is retired after the final PV. Unlike RMS scratch, it cannot hold
        // the following query's prefetched norm partials during this epilogue.
        static_assert(sizeof(plan.smem_sP) >= (64 + 8) * sizeof(float));
        auto* group_partials = reinterpret_cast<float*>(plan.smem_sP.data());
        auto* group_scales = group_partials + 64;
        auto flush_group_scales = [&] {
            if constexpr (kPerTokenScale && !Norm) {
                // Publish global SF with the final FP8 stores. The scale
                // publication barrier then has only shared-memory writes.
                if (threadIdx.x < 8)
                    reinterpret_cast<float*>(fused.out_sf)[int64_t(batch_idx) * fused.stride_out_sf_token
                        + int64_t(threadIdx.x) * fused.stride_out_sf_group] = group_scales[threadIdx.x];
            }
        };
        // Wide staging retires all 64 accumulator slots before either scan.
        constexpr bool kCacheProcessedOutput = PerTokenScale && WideOutput && !Norm &&
            !kFusedPackedFp4;
        constexpr bool kCacheOutputInverse = PerTokenScale && !Norm &&
            ((Base::kIsPrefill && T::kPersistentPrefillQueries) || T::kFusedDirectQueries);
        static_assert(!kCacheOutputInverse || WideOutput);
        float output_scale_inv;
        int output_cache_half = 0;
        auto quantize_half = [&](const auto& half, int tile_begin, auto scan_only) {
            if(item < 64 * 8 && head < num_valid_seq_q) {
                const int tile = item / 64 + tile_begin;
                alignas(16) float values[32];
                constexpr bool kZeroNanAtPack = (T::kSinglePrefillPair || T::kMidQSaturatingPack)
                    && T::kDeferPrefillSink && Base::kIsPrefill && Base::kArch == 80
                    && !Norm && !kPerTokenScale;
                const bool zero_nan_at_pack = kZeroNanAtPack && deferred_sink && tile < 14;
                if constexpr (kCacheProcessedOutput && !decltype(scan_only)::value) {
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) values[i] = rO(output_cache_half * 32 + i);
                } else {
                    if constexpr (WideOutput) {
                        // All load addresses can use the same 32-column tile start.
                        const int col = (tile & 7) * 32;
                        constexpr bool kReuseTileOffset = T::kCacheRms ||
                            (T::kCachePrefillRope && !kFusedPackedFp4) || T::kVectorPrefillOutput;
                        const int tile_offset = kReuseTileOffset
                            ? int(half.layout()(make_coord(head, col))) : 0;
                        if constexpr (kVectorOutputLayout) {
                            static_assert(kReuseTileOffset);
                            #pragma unroll
                            for (int i = 0; i < 32; i += 4) {
                                const float* chunk = cute::raw_pointer_cast(half.data()) + (tile_offset ^ i);
                                const float4 group = *reinterpret_cast<const float4*>(chunk);
                                const int first = (i & 16) + ((i & 15) / 4);
                                values[first] = group.x;
                                values[first + 4] = group.y;
                                values[first + 8] = group.z;
                                values[first + 12] = group.w;
                            }
                        } else {
                            #pragma unroll
                            for(int i = 0; i < 32; i += 4) {
                                const float* chunk;
                                if constexpr (kReuseTileOffset) {
                                    // The column increment changes no source bits of Swizzle<3,3,3>.
                                    chunk = cute::raw_pointer_cast(half.data()) + (tile_offset ^ i);
                                } else {
                                    chunk = &half(head, col + i);
                                }
                                *reinterpret_cast<float4*>(values + i) =
                                    *reinterpret_cast<const float4*>(chunk);
                            }
                        }
                    } else {
                        #pragma unroll
                        for(int i = 0; i < 32; ++i) values[i] = half(head, (tile & 7) * 32 + i);
                    }
                    if constexpr (T::kDeferPrefillSink) {
                        if (deferred_sink) {
                            const float factor = sink_scales[head];
                            if (zero_nan_at_pack) {
                                #pragma unroll
                                for (int i = 0; i < 32; ++i) values[i] *= factor;
                            } else {
                                #pragma unroll
                                for (int i = 0; i < 32; ++i) {
                                    const float scaled = values[i] * factor;
                                    values[i] = scaled == scaled ? scaled : 0.0f;
                                }
                            }
                        }
                    }
                    if constexpr (!kRegisterOutputRope) {
                        if(tile >= 14) {
                            #pragma unroll
                            for(int i = 0; i < 32; i += 2) {
                                const int pair = (tile - 14) * 16 + i / 2;
                                const float2 rotated = apply_rope(float2{values[i], values[i + 1]},
                                    rope_coeff<T>(rope, pair), -rope_coeff<T>(rope, 32 + pair));
                                values[i] = rotated.x;
                                values[i + 1] = rotated.y;
                            }
                        }
                    }
                    if constexpr (kCacheProcessedOutput && decltype(scan_only)::value) {
                        #pragma unroll
                        for (int i = 0; i < 32; ++i) rO(output_cache_half * 32 + i) = values[i];
                    }
                }
                constexpr bool kSaturatingPack = (kZeroNanAtPack && T::kAlignedQWithSink)
                    || T::kMidQSaturatingPack;
                float amax = 0.0f;
                bool bounded_finite = false;
                if constexpr (!kPerTokenScale || decltype(scan_only)::value) {
                    if constexpr (kSaturatingPack) {
                        // Unsigned abs-bit max also detects every Inf/NaN.
                        // fmax alone can hide NaN and cannot certify finiteness.
                        uint32_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
                        #pragma unroll
                        for (int i = 0; i < 32; i += 4) {
                            const uint32_t b0 = __float_as_uint(values[i]) & 0x7fffffffu;
                            const uint32_t b1 = __float_as_uint(values[i + 1]) & 0x7fffffffu;
                            const uint32_t b2 = __float_as_uint(values[i + 2]) & 0x7fffffffu;
                            const uint32_t b3 = __float_as_uint(values[i + 3]) & 0x7fffffffu;
                            m0 = max(m0, b0); m1 = max(m1, b1);
                            m2 = max(m2, b2); m3 = max(m3, b3);
                        }
                        const uint32_t magnitude_max = max(max(m0, m1), max(m2, m3));
                        bounded_finite = magnitude_max < 0x7f800000u;
                        if (bounded_finite) amax = __uint_as_float(magnitude_max);
                    }
                    if (!bounded_finite) {
                        if constexpr (WideOutput) {
                            float amax1 = 0.0f, amax2 = 0.0f, amax3 = 0.0f;
                            #pragma unroll
                            for(int i = 0; i < 32; i += 4) {
                                amax = fmaxf(amax, fabsf(values[i]));
                                amax1 = fmaxf(amax1, fabsf(values[i + 1]));
                                amax2 = fmaxf(amax2, fabsf(values[i + 2]));
                                amax3 = fmaxf(amax3, fabsf(values[i + 3]));
                            }
                            amax = fmaxf(fmaxf(amax, amax1), fmaxf(amax2, amax3));
                        } else {
                            #pragma unroll
                            for(int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(values[i]));
                        }
                    }
                    amax = fmaxf(1e-4f, amax);
                }
                if constexpr (decltype(scan_only)::value) return amax;
                int exponent = 0;
                float scale_inv;
                if constexpr (PerTokenScale) {
                    if constexpr (kCacheOutputInverse && !decltype(scan_only)::value)
                        scale_inv = output_scale_inv;
                    else
                        scale_inv = __fdividef(1.0f, group_scales[head / 8]);
                } else {
                    const float scale = amax / 448.0f;
                    exponent = int((__float_as_uint(scale) - 1u) >> 23) + 1 - 127;
                    scale_inv = __uint_as_float(uint32_t(127 - exponent) << 23);
                }
                alignas(16) uint8_t packed[32];
                auto* pairs = reinterpret_cast<uint16_t*>(packed);
#if ACOMPUTE_VERSION == 10000
                const uint16_t* fp8_table = kSm80SignedFp8Table.words;
                if constexpr (kZeroNanAtPack) {
                    // The positive finite reciprocal sends a pruned NaN (+0)
                    // to FP8 +0. For infinite amax, the reference multiplier
                    // is -Inf and either the original NaN or +0 produces NaN.
                    if (zero_nan_at_pack && __float_as_uint(amax) < 0x7f800000u) {
                        fp8_table = kSm80ZeroNanFp8Table.words;
                    }
                }
#endif
                bool use_saturating_pack = false;
#if ACOMPUTE_VERSION == 10000
                if constexpr (kSaturatingPack) {
                    if (bounded_finite) {
                        auto* bytes_as_words = reinterpret_cast<uint32_t*>(packed);
                        const uint32_t half_offset = uint32_t(960 + 8 * exponent) * 16u - 8u;
                        const uint32_t paired_offset = half_offset | (half_offset << 16);
                        #pragma unroll
                        for (int i = 0; i < 32; i += 4)
                            bytes_as_words[i / 4] = sm80_pack_four_saturating(values + i, paired_offset);
                        use_saturating_pack = true;
                    }
                }
#endif
                if (!use_saturating_pack) {
                    #pragma unroll
                    for(int i = 0; i < 32; i += 2) {
#if ACOMPUTE_VERSION == 10000
                        if constexpr (Base::kIsPrefill && Base::kArch == 80 && !Norm && !kPerTokenScale) {
                            const uint32_t low = sm80_signed_fp8_from_float(values[i] * scale_inv, fp8_table);
                            const uint32_t high = sm80_signed_fp8_from_float(values[i + 1] * scale_inv, fp8_table);
                            pairs[i / 2] = uint16_t(low | (high << 8));
                        } else
#endif
                        {
                            // Keep the existing converters for the other paths.
                            pairs[i / 2] = splitkv::fp8_pair_from_float(
                                values[i] * scale_inv, values[i + 1] * scale_inv);
                        }
                    }
                }
                const int group = head / 8;
                const int in_group = head & 7;
                const int64_t out_offset = int64_t(batch_idx) * (64 * 512)
                    + group * 4096 + tile * 256 + in_group * 32;
                auto* out = reinterpret_cast<uint4*>(fused.out_fp8 + out_offset);
                const auto* words = reinterpret_cast<const uint4*>(packed);
                out[0] = words[0];
                out[1] = words[1];
                if constexpr (!kPerTokenScale) {
                    const int sf_index = tile * 8 + in_group;
                    auto* sf_word = fused.out_sf + batch_idx + int64_t(group) * fused.stride_out_sf_group
                        + int64_t(sf_index / 4) * fused.stride_out_sf_word;
                    if constexpr (Base::kIsPrefill && T::kArch == 80) {
                        if (num_valid_seq_q == 64) {
                            const uint32_t encoded = uint8_t(exponent + 127);
                            const uint32_t pair = encoded
                                | (__shfl_down_sync(0xffffffff, encoded, 1, 4) << 8);
                            const uint32_t word = pair
                                | (__shfl_down_sync(0xffffffff, pair, 2, 4) << 16);
                            if ((head & 3) == 0) *sf_word = word;
                        } else {
                            reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
                        }
                    } else {
                        reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
                    }
                }
            }
            return 0.0f;
        };
        auto publish_group_scales = [&](float amax) {
            // Eight adjacent lanes cover eight heads at the same V tile. Each
            // thread first combines both of its tiles, then the CTA combines
            // the eight tile-pair partials for each 4096-element WV group.
            #pragma unroll
            for (int offset = 4; offset >= 1; offset /= 2) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, 8));
            }
            if ((threadIdx.x & 7) == 0) group_partials[threadIdx.x / 8] = amax;
            __syncthreads();
            if (threadIdx.x < 8) {
                const int group = threadIdx.x;
                float group_amax = 1e-4f;
                #pragma unroll
                for (int tile_pair = 0; tile_pair < 8; ++tile_pair) {
                    group_amax = fmaxf(group_amax, group_partials[tile_pair * 8 + group]);
                }
                const float scale = group_amax / 448.0f;
                group_scales[group] = scale;
                if constexpr (Norm) {
                    reinterpret_cast<float*>(fused.out_sf)[int64_t(batch_idx) * fused.stride_out_sf_token
                        + int64_t(group) * fused.stride_out_sf_group] = scale;
                }
            }
            __syncthreads();
        };
        using HalfLayout = typename T::SmemLayoutO_Half64x256;
        if constexpr (WideOutput) {
            // The next-Q copy uses sQ. Both retired K banks are reserved for O.
            constexpr int half_elements = cosize_v<HalfLayout>;
            static_assert(sizeof(plan.smem_sK) >= 2 * half_elements * sizeof(float));
            auto* staging = reinterpret_cast<float*>(plan.smem_sK.data());
            auto stage = make_tensor(make_smem_ptr(staging + warpgroup_idx * half_elements), HalfLayout{});
            stage_half(stage);
            if constexpr (kPerTokenScale) {
                // A WV scale spans both warpgroups, even when the old packed
                // output path needs only a warpgroup-local publication.
                if constexpr (kRegisterOutputRope && !Norm) {
                    // Each warp owns eight consecutive heads at one N tile.
                    // Reduce the normalized, inverse-rotated MMA registers
                    // directly, without scanning the staged FP32 image.
                    static_assert(T::kPvAtomLayoutM == 2 && T::kPvAtomLayoutN == 4);
                    #pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        float amax = 1e-4f, amax1 = 0.0f, amax2 = 0.0f, amax3 = 0.0f;
                        #pragma unroll
                        for (int ni = 0; ni < 16; ni += 4) {
                            amax = fmaxf(amax, fabsf(rowcol(mi, ni)));
                            amax1 = fmaxf(amax1, fabsf(rowcol(mi, ni + 1)));
                            amax2 = fmaxf(amax2, fabsf(rowcol(mi, ni + 2)));
                            amax3 = fmaxf(amax3, fabsf(rowcol(mi, ni + 3)));
                        }
                        amax = fmaxf(fmaxf(amax, amax1), fmaxf(amax2, amax3));
                        amax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(amax)));
                        const int row = flashmla::dsa::hs64::get_PV_row_idx<T>(mi, idx_in_warpgroup);
                        const int tile = warpgroup_idx * 4 + idx_in_warpgroup / 64;
                        if ((threadIdx.x & 31) == 0) group_partials[tile * 8 + row / 8] = amax;
                    }
                    __syncthreads();
                    if (threadIdx.x < 8) {
                        float amax = 1e-4f;
                        #pragma unroll
                        for (int tile = 0; tile < 8; ++tile)
                            amax = fmaxf(amax, group_partials[tile * 8 + threadIdx.x]);
                        const float scale = amax / 448.0f;
                        group_scales[threadIdx.x] = scale;
                    }
                    __syncthreads();
                    const int first_tile = warpgroup_idx * 8;
                    quantize_half(stage, first_tile, std::false_type{});
                    quantize_half(stage, first_tile + 4, std::false_type{});
                    flush_group_scales();
                    __syncthreads();
                    return;
                }
                __syncthreads();
                float amax = 0.0f;
                if constexpr (kFusedPackedFp4) {
                    const int first_tile = warpgroup_idx * 8;
                    amax = quantize_half(stage, first_tile, std::true_type{});
                    amax = fmaxf(amax, quantize_half(stage, first_tile + 4, std::true_type{}));
                } else {
                    #pragma unroll
                    for (int pass = 0; pass < 2; ++pass) {
                        auto half = make_tensor(make_smem_ptr(staging + pass * half_elements), HalfLayout{});
                        if constexpr (kCacheProcessedOutput) output_cache_half = pass;
                        amax = fmaxf(amax, quantize_half(half, pass * 8, std::true_type{}));
                    }
                }
                publish_group_scales(amax);
                if constexpr (kCacheOutputInverse) {
                    // The published FP32 scale stays unchanged across both halves.
                    output_scale_inv = __fdividef(1.0f, group_scales[head / 8]);
                }
                if constexpr (kFusedPackedFp4) {
                    const int first_tile = warpgroup_idx * 8;
                    quantize_half(stage, first_tile, std::false_type{});
                    quantize_half(stage, first_tile + 4, std::false_type{});
                } else {
                    #pragma unroll
                    for (int pass = 0; pass < 2; ++pass) {
                        auto half = make_tensor(make_smem_ptr(staging + pass * half_elements), HalfLayout{});
                        if constexpr (kCacheProcessedOutput) output_cache_half = pass;
                        quantize_half(half, pass * 8, std::false_type{});
                    }
                }
                flush_group_scales();
                // The next attention iteration may reuse P for score tiles.
                __syncthreads();
            } else if constexpr (kFusedPackedFp4) {
                // The reductions have finished using barriers 8 and 9.
                // Each warpgroup reads only its own staged output half.
                __ppu_barrier_sync(8 + warpgroup_idx, T::NUM_THREADS / 2, 15u);
                const int first_tile = warpgroup_idx * 8;
                quantize_half(stage, first_tile, std::false_type{});
                quantize_half(stage, first_tile + 4, std::false_type{});
            } else {
                __ppu_barrier_sync(0, T::NUM_THREADS, 15u);
                #pragma unroll
                for(int pass = 0; pass < 2; ++pass) {
                    auto half = make_tensor(make_smem_ptr(staging + pass * half_elements), HalfLayout{});
                    quantize_half(half, pass * 8, std::false_type{});
                }
            }
        } else {
            auto half = make_tensor(make_smem_ptr(reinterpret_cast<float*>(sO_addr)), HalfLayout{});
            if constexpr (kPerTokenScale) {
                float amax = 0.0f;
                #pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    if (warpgroup_idx == pass) stage_half(half);
                    __syncthreads();
                    amax = fmaxf(amax, quantize_half(half, pass * 8, std::true_type{}));
                    __syncthreads();
                }
                publish_group_scales(amax);
                // Half 1 is still staged. Quantize it before restoring half 0;
                // neither retired K bank nor a prefetched next Q is overwritten.
                quantize_half(half, 8, std::false_type{});
                __syncthreads();
                if (warpgroup_idx == 0) stage_half(half);
                __syncthreads();
                quantize_half(half, 0, std::false_type{});
                flush_group_scales();
                __syncthreads();
            } else {
                #pragma unroll
                for(int pass = 0; pass < 2; ++pass) {
                    if(warpgroup_idx == pass) stage_half(half);
                    __syncthreads();
                    quantize_half(half, pass * 8, std::false_type{});
                    __syncthreads();
                }
            }
        }
    }
};

}
