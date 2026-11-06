#pragma once

#include <cstddef>
#include "api/common.h"
#include "kernel.cuh"
#include "decode/sparse/sparse_decode_wg_hs64.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

struct FusedHs64Params : Flash_fwd_mla_params {
    float rms_norm_eps;
    const int32_t* token_positions;
    const float* cos_sin_cache;
    uint8_t* out_fp8;
    uint32_t* out_sf;
    float* max_logits;
    int64_t stride_q_s_q;
    int64_t stride_out_sf_group;
    int64_t stride_out_sf_word;
};

template<bool Norm, typename Base, bool WideOutput = false, bool SkipNextKPrefetch = false>
struct FusedHs64Traits : Base {
    using SharedMemoryPlan = typename Base::SharedMemoryPlan;
    using SmemLayoutO_Half64x256 = typename Base::SmemLayoutO_Half;
    static constexpr bool kFusedHs64 = true;
    static constexpr bool kSkipNextKPrefetch = SkipNextKPrefetch;
    static constexpr bool kPersistentPrefillQueries = false;
    static constexpr int kQBarrierThreads = Base::kIsPrefill ? Base::kQBarrierThreads : Base::NUM_THREADS;
    static constexpr size_t kNormScratchOffset = Base::kIsPrefill ? 200192
        : offsetof(SharedMemoryPlan, smem_cross_n_reduction)
            + sizeof(((SharedMemoryPlan*)nullptr)->smem_cross_n_reduction);
    static constexpr size_t kNormOffset = kNormScratchOffset + 512 * sizeof(float);
    static constexpr size_t kNormEnd = kNormOffset + 64 * sizeof(float);
    static_assert(sizeof(SharedMemoryPlan) == 262144);
    static_assert(!Base::kIsPrefill || offsetof(SharedMemoryPlan, barriers_K1)
        + sizeof(((SharedMemoryPlan*)nullptr)->barriers_K1) <= kNormScratchOffset);
    static_assert(kNormEnd <= offsetof(SharedMemoryPlan, smem_sP));
    static_assert(offsetof(SharedMemoryPlan, smem_sQ) == 0);
    static_assert(offsetof(SharedMemoryPlan, smem_sK) == 65536);
    static_assert(!Base::kIsPrefill || offsetof(SharedMemoryPlan, smem_valid_indices) == 199168);
    static_assert(offsetof(SharedMemoryPlan, smem_sP) == 204800);
    static_assert(offsetof(SharedMemoryPlan, smem_even_high) == 229376);
    static_assert(cosize_v<SmemLayoutO_Half64x256> * sizeof(float) == 65536);
    static_assert(Base::kBlockM == 64 && Base::kBlockN == 64 && Base::NUM_THREADS == 512);
    static_assert(Base::kUsePairedCubes == Base::kIsPrefill && Base::kUsePv2x4);
    static_assert(!WideOutput || (Base::kIsPrefill && (Base::kGroupPrefillQueries || !Norm))
        || (!Norm && flashmla::dsa::hs64::kPackedMain<Base> == 128
            && flashmla::dsa::hs64::kWorklist<Base> != flashmla::dsa::hs64::Hs64Worklist::Batch));

    static __forceinline__ __device__ int q_offset(int row, int dim) {
        if constexpr (Base::kUsePairedCubes) return paired_q_offset<64>(row, dim);
        else return native_offset<64>(row, dim);
    }

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ void load_fused_q(
        const Flash_fwd_mla_params& params, int batch_idx, int, int,
        TensorQ& sQ, int tidx, int, __mbarrier_t* barrier_Q) {
        extern __shared__ char wksp_buf[];
        const auto& fused = static_cast<const FusedHs64Params&>(params);
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
            #pragma unroll
            for (int chunk = tidx; chunk < 64 * 64; chunk += Base::NUM_THREADS) {
                const int head = chunk % 64;
                const int dim = (chunk / 64) * 8;
                PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                    *reinterpret_cast<const cute::uint128_t*>(src + (dim / 16) * (64 * 16) + head * 16 + dim % 16),
                    *reinterpret_cast<cute::uint128_t*>(dst + q_offset(head, dim)));
            }
            cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
            return;
        }
        if((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
            // The production barrier counts 256 loader threads; RMS and RoPE
            // follow its wait in finish_fused_q, so raw Q overlaps other work.
            if(tidx < 256) {
                constexpr bool kCoalesceQ = !Norm && WideOutput &&
                    flashmla::dsa::hs64::kHs64PrefillGroupedQueries<T>;
                const int head_pair = (tidx / 2) % 64;
                // Pair the source halves. Even then odd heads preserve the eight-lane bank spread.
                const int copy_row = kCoalesceQ
                    ? ((head_pair & ~7) | ((head_pair & 3) << 1) | ((head_pair & 4) >> 2)) : row;
                const int dim_begin = kCoalesceQ ? (tidx / 128) * 16 + (tidx % 2) * 8 : (tidx / 64) * 128;
                #pragma unroll
                for(int i = 0; i < 128; i += 8) {
                    const int dim = dim_begin + (kCoalesceQ ? i * 4 : i);
                    PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                        *reinterpret_cast<const cute::uint128_t*>(src + (dim / 16) * (64 * 16) + copy_row * 16 + dim % 16),
                        *reinterpret_cast<cute::uint128_t*>(dst + paired_q_offset<64>(copy_row, dim)));
                }
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
                const int q_offset = paired_q_offset<64>(row, dim + half * 8);
                *reinterpret_cast<uint4*>(dst + q_offset) = reinterpret_cast<const uint4*>(packed)[half];
            }
        }
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
        if(tidx < 256) cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
    }

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ void finish_fused_q(
        const Flash_fwd_mla_params& params, int batch_idx, TensorQ& sQ, int tidx) {
        const auto& fused = static_cast<const FusedHs64Params&>(params);
        const auto* src = static_cast<const typename T::InputT*>(fused.q_ptr)
            + int64_t(batch_idx) * fused.stride_q_s_q;
        if((reinterpret_cast<uintptr_t>(src) & 15) != 0) return;
        extern __shared__ char wksp_buf[];
        auto* partials = reinterpret_cast<float*>(wksp_buf + kNormScratchOffset);
        auto* norm = reinterpret_cast<float*>(wksp_buf + kNormOffset);
        auto* dst = cute::raw_pointer_cast(sQ.data());
        const int row = tidx % 64;
        const int part = tidx / 64;
        const int position = __ldg(fused.token_positions + batch_idx);
        const float* rope = fused.cos_sin_cache + int64_t(position) * 64;
        if constexpr (!Norm && flashmla::dsa::hs64::kHs64PrefillGroupedQueries<T>) {
            // Both warpgroups rotate the last Q tile before the prolog barrier.
            const int dim = 448 + part * 8;
            auto* chunk = reinterpret_cast<uint4*>(dst + q_offset(row, dim));
            const uint4 raw = *chunk;
            const auto* elements = reinterpret_cast<const typename T::InputT*>(&raw);
            alignas(16) typename T::InputT packed[8];
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                const int pair = (dim + j - 448) / 2;
                const float2 x = apply_rope(float2{float(elements[j]), float(elements[j + 1])},
                    __ldg(rope + pair), __ldg(rope + 32 + pair));
                packed[j] = typename T::InputT(x.x);
                packed[j + 1] = typename T::InputT(x.y);
            }
            *chunk = *reinterpret_cast<const uint4*>(packed);
            return;
        }
        float sum = 0.0f;
        #pragma unroll
        for(int k = 0; k < 4; ++k) {
            const int dim = (part + k * 8) * 16;
            alignas(16) typename T::InputT elements[16];
            #pragma unroll
            for(int half = 0; half < 2; ++half) {
                reinterpret_cast<uint4*>(elements)[half] = *reinterpret_cast<const uint4*>(
                    dst + q_offset(row, dim + half * 8));
            }
            if constexpr(Norm) {
                #pragma unroll
                for(int j = 0; j < 16; j += 2) {
                    const float2 x{float(elements[j]), float(elements[j + 1])};
                    sum = fmaf(x.x, x.x, sum);
                    sum = fmaf(x.y, x.y, sum);
                }
            }
            if(dim >= 448) {
                alignas(16) typename T::InputT packed[16];
                #pragma unroll
                for(int j = 0; j < 16; j += 2) {
                    const int pair = (dim + j - 448) / 2;
                    const float2 x = apply_rope(float2{float(elements[j]), float(elements[j + 1])},
                        __ldg(rope + pair), __ldg(rope + 32 + pair));
                    packed[j] = typename T::InputT(x.x);
                    packed[j + 1] = typename T::InputT(x.y);
                }
                #pragma unroll
                for(int half = 0; half < 2; ++half) {
                    *reinterpret_cast<uint4*>(dst + q_offset(row, dim + half * 8))
                        = reinterpret_cast<const uint4*>(packed)[half];
                }
            }
        }
        if constexpr(Norm) {
            partials[tidx] = sum;
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
        __syncthreads();
    }

    template<typename TensorP>
    static __forceinline__ __device__ void scale_fused_scores(TensorP& rP, int idx_in_warpgroup) {
        if constexpr(Norm) {
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
        const auto& fused = static_cast<const FusedHs64Params&>(params);
        Tensor rowcol = make_tensor(rO.data(),
            flashmla::dsa::hs64::hs64_convert_layout_acc_rowcol<T>(rO.layout()));
        static_assert(decltype(size<0>(rowcol))::value == 4);
        static_assert(decltype(size<1>(rowcol))::value == 16);
        #pragma unroll
        for(int mi = 0; mi < 4; ++mi) {
            #pragma unroll
            for(int ni = 0; ni < 16; ++ni) rowcol(mi, ni) /= rNorm[mi];
        }
        typename T::TiledMmaPV mma;
        auto copy_c = make_tiled_copy_C(typename T::SmemCopyAtomOaccum{}, mma);
        auto thr = copy_c.get_slice(idx_in_warpgroup);
        Tensor src = thr.retile_S(rO);
        const int item = threadIdx.x;
        const int head = item % 64;
        const int position = __ldg(fused.token_positions + batch_idx);
        const float* rope = fused.cos_sin_cache + int64_t(position) * 64;
        auto stage_half = [&](auto& half) {
            Tensor dst = thr.partition_D(half);
            if constexpr (T::kUsePairedCubes) {
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
                cute::copy(copy_c, src, dst);
            }
        };
        auto quantize_half = [&](const auto& half, int pass) {
            if(item < 64 * 8 && head < num_valid_seq_q) {
                const int tile = item / 64 + pass * 8;
                alignas(16) float values[32];
                if constexpr (WideOutput) {
                    // The swizzle preserves the three low column bits.
                    #pragma unroll
                    for(int i = 0; i < 32; i += 4)
                        *reinterpret_cast<float4*>(values + i) =
                            *reinterpret_cast<const float4*>(&half(head, (tile & 7) * 32 + i));
                } else {
                    #pragma unroll
                    for(int i = 0; i < 32; ++i) values[i] = half(head, (tile & 7) * 32 + i);
                }
                if(tile >= 14) {
                    #pragma unroll
                    for(int i = 0; i < 32; i += 2) {
                        const int pair = (tile - 14) * 16 + i / 2;
                        const float2 rotated = apply_rope(float2{values[i], values[i + 1]},
                            __ldg(rope + pair), -__ldg(rope + 32 + pair));
                        values[i] = rotated.x;
                        values[i + 1] = rotated.y;
                    }
                }
                float amax = 0.0f;
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
                amax = fmaxf(1e-4f, amax);
                const float scale = amax / 448.0f;
                const int exponent = int((__float_as_uint(scale) - 1u) >> 23) + 1 - 127;
                const float scale_inv = __uint_as_float(uint32_t(127 - exponent) << 23);
                alignas(16) uint8_t packed[32];
                auto* pairs = reinterpret_cast<uint16_t*>(packed);
                #pragma unroll
                for(int i = 0; i < 32; i += 2) {
                    pairs[i / 2] = fp8_pair_from_float(values[i] * scale_inv, values[i + 1] * scale_inv);
                }
                const int group = head / 8;
                const int in_group = head & 7;
                const int64_t out_offset = int64_t(batch_idx) * (64 * 512)
                    + group * 4096 + tile * 256 + in_group * 32;
                auto* out = reinterpret_cast<uint4*>(fused.out_fp8 + out_offset);
                const auto* words = reinterpret_cast<const uint4*>(packed);
                out[0] = words[0];
                out[1] = words[1];
                const int sf_index = tile * 8 + in_group;
                auto* sf_word = fused.out_sf + batch_idx + int64_t(group) * fused.stride_out_sf_group
                    + int64_t(sf_index / 4) * fused.stride_out_sf_word;
                reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
            }
        };
        using HalfLayout = typename T::SmemLayoutO_Half64x256;
        if constexpr (WideOutput) {
            // The next-Q copy uses sQ. Both retired K banks are reserved for O.
            extern __shared__ char wksp_buf[];
            auto& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
            constexpr int half_elements = cosize_v<HalfLayout>;
            static_assert(sizeof(plan.smem_sK) >= 2 * half_elements * sizeof(float));
            auto* staging = reinterpret_cast<float*>(plan.smem_sK.data());
            auto stage = make_tensor(make_smem_ptr(staging + warpgroup_idx * half_elements), HalfLayout{});
            stage_half(stage);
            __ppu_barrier_sync(0, T::NUM_THREADS, 15u);
            #pragma unroll
            for(int pass = 0; pass < 2; ++pass) {
                auto half = make_tensor(make_smem_ptr(staging + pass * half_elements), HalfLayout{});
                quantize_half(half, pass);
            }
        } else {
            auto half = make_tensor(make_smem_ptr(reinterpret_cast<float*>(sO_addr)), HalfLayout{});
            #pragma unroll
            for(int pass = 0; pass < 2; ++pass) {
                if(warpgroup_idx == pass) stage_half(half);
                __syncthreads();
                quantize_half(half, pass);
                __syncthreads();
            }
        }
    }
};

template<bool Norm, bool Group, bool WideOutput>
using FusedHs64PrefillTraits =
    FusedHs64Traits<Norm, flashmla::dsa::hs64::Hs64PrefillTraits<512, 89, Group, false>, WideOutput>;

template<bool SkipNextKPrefetch>
struct FusedHs64PersistentPrefillTraits
    : FusedHs64Traits<false, flashmla::dsa::hs64::Hs64PrefillTraits<512, 89, true, false>,
                      true, SkipNextKPrefetch> {
    static constexpr bool kPersistentPrefillQueries = true;
};

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist>
struct FusedHs64PackedTraits
    : FusedHs64Traits<Norm, std::conditional_t<ExtraFP4,
        flashmla::dsa::hs64::Hs64PackedDecode<MainCapacity, Worklist>,
        flashmla::dsa::hs64::Hs64PackedFp8ExtraWorklist<MainCapacity, Worklist>>,
        !Norm && MainCapacity == 128 && Worklist != flashmla::dsa::hs64::Hs64Worklist::Batch> {
    static constexpr bool kFp8LeaderPrefetch =
        !Norm && !ExtraFP4 && MainCapacity == 128 && Worklist == flashmla::dsa::hs64::Hs64Worklist::Flat;
};

}

namespace flashmla::dsa::hs64 {

template<bool Norm, bool ExtraFP4, int MainCapacity, Hs64Worklist Worklist>
inline constexpr int kPackedMain<fused_norm_rope_attn_rope_cast_fwd::core_attn::FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist>> = MainCapacity;

template<bool Norm, bool ExtraFP4, int MainCapacity, Hs64Worklist Worklist>
inline constexpr Hs64Worklist kWorklist<fused_norm_rope_attn_rope_cast_fwd::core_attn::FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist>> = Worklist;

}

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool Norm, bool Group, bool WideOutput>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_prefill_hs64_kernel(__grid_constant__ const FusedHs64Params params) {
#if ACOMPUTE_VERSION >= 10500
    using T = FusedHs64PrefillTraits<Norm, Group, WideOutput>;
    const Flash_fwd_mla_params& base = params;
    flashmla::dsa::hs64::hs64_attention<T, false>(base, params.max_logits);
#endif
}

// Isolate the fixed top-k 2048 variant so other wide kernels keep their schedule.
template<bool SkipNextKPrefetch>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_prefill_hs64_kernel(__grid_constant__ const FusedHs64Params params) {
#if ACOMPUTE_VERSION >= 10500
    using T = FusedHs64Traits<false, flashmla::dsa::hs64::Hs64PrefillTraits<512, 89, true, false>,
                             true, SkipNextKPrefetch>;
    const Flash_fwd_mla_params& base = params;
    flashmla::dsa::hs64::hs64_attention<T, false>(base, params.max_logits);
#endif
}

template<bool SkipNextKPrefetch, bool PersistentQueries>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_prefill_hs64_kernel(__grid_constant__ const FusedHs64Params params) {
    static_assert(PersistentQueries);
#if ACOMPUTE_VERSION >= 10500
    using T = FusedHs64PersistentPrefillTraits<SkipNextKPrefetch>;
    // Expose the same fixed geometry as the native H64 prefill wrapper.
    FusedHs64Params fixed = params;
    fixed.q_orig = 1;
    fixed.seqlen_q = fixed.h_q = fixed.ngroups = T::kBlockM;
    fixed.h = fixed.h_h_k_ratio = 1;
    fixed.d = T::kHeadDim;
    fixed.d_v = T::kHeadDimV;
    fixed.extra_topk = -1;
    fixed.page_block_size = 1;
    fixed.q_row_stride = fixed.k_row_stride = T::kHeadDim;
    fixed.o_batch_stride = fixed.o_row_stride = 0;
    const Flash_fwd_mla_params& base = fixed;
    flashmla::dsa::hs64::hs64_attention<T, false>(base, fixed.max_logits);
#endif
}

template<bool SkipNextKPrefetch>
void launch_fused_prefill_hs64_persistent(const FusedHs64Params& params, hggcStream_t stream) {
    using T = FusedHs64PersistentPrefillTraits<SkipNextKPrefetch>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &fused_norm_rope_attn_rope_cast_prefill_hs64_kernel<SkipNextKPrefetch, true>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const int grid_q = std::min(params.b, get_num_sm(get_current_device()));
    kernel<<<dim3(grid_q, 1, 1), T::NUM_THREADS, bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool Norm, bool Group, bool WideOutput = false, bool Wide2048 = false>
void launch_fused_prefill_hs64(const FusedHs64Params& params, hggcStream_t stream) {
    static_assert(!Wide2048 || (!Norm && Group && WideOutput));
    using T = FusedHs64PrefillTraits<Norm, Group, WideOutput>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    auto kernel = [] {
        if constexpr (Wide2048) return &fused_norm_rope_attn_rope_cast_prefill_hs64_kernel<true>;
        else return &fused_norm_rope_attn_rope_cast_prefill_hs64_kernel<Norm, Group, WideOutput>;
    }();
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const int grid_q = Group ? flashmla::dsa::hs64::hs64_grouped_prefill_grid_q(
        params.b, get_num_sm(get_current_device())) : params.b;
    kernel<<<dim3(grid_q, 1, 1), 512, bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool EnableNorm>
void run_fused_norm_rope_attn_rope_cast_prefill_hs64(const Params& p) {
    TORCH_CHECK(p.mode == FwdMode::Prefill && p.arch == 89 && p.h_q == 64);
    TORCH_CHECK(p.topk >= 512 && p.topk % 128 == 0 && p.enable_q_norm == EnableNorm);
    TORCH_CHECK(p.wv_group_size == 8 && p.n_wv_group == 8);
    if(p.s_q == 0) return;
    at::Tensor clamped_length;
    const int32_t* topk_length = p.topk_length;
    if(p.s_q >= 1024 && topk_length) {
        const auto options = torch::TensorOptions().dtype(torch::kInt32)
            .device(torch::Device(torch::kCUDA, get_current_device()));
        clamped_length = torch::from_blob(const_cast<int32_t*>(topk_length), {p.s_q}, options)
            .clamp(0, p.topk);
        topk_length = clamped_length.data_ptr<int32_t>();
    }
    FusedHs64Params params{};
    params.b = p.s_q;
    params.q_orig = 1;
    params.seqlen_q = params.h_q = params.ngroups = 64;
    params.h = params.h_h_k_ratio = 1;
    params.d = params.d_v = 512;
    params.scale_softmax = p.sm_scale;
    params.scale_softmax_log2 = p.sm_scale_log2;
    params.q_ptr = const_cast<void*>(p.q);
    params.k_ptr = const_cast<void*>(p.kv);
    params.o_ptr = p.out_fp8;
    params.o_batch_stride = params.o_row_stride = 0;
    params.softmax_lse_ptr = p.lse;
    params.indices_ptr = const_cast<int32_t*>(p.indices);
    params.topk_len_ptr = const_cast<int32_t*>(topk_length);
    params.attn_sink_ptr = const_cast<float*>(p.attn_sink);
    params.topk = p.topk;
    params.extra_topk = -1;
    params.page_block_size = 1;
    params.num_blocks = params.seqlen_k = p.s_kv;
    params.q_batch_stride = p.stride_q_s_q;
    params.q_row_stride = 512;
    params.k_batch_stride = p.stride_kv_s_kv;
    params.k_row_stride = 512;
    params.indices_batch_stride = p.stride_indices_s_q;
    params.rms_norm_eps = p.rms_norm_eps;
    params.token_positions = p.token_positions;
    params.cos_sin_cache = p.cos_sin_cache;
    params.out_fp8 = p.out_fp8;
    params.out_sf = p.out_sf;
    params.max_logits = p.max_logits;
    params.stride_q_s_q = p.stride_q_s_q;
    params.stride_out_sf_group = p.stride_out_sf_group;
    params.stride_out_sf_word = p.stride_out_sf_word;
    if(p.s_q >= 1024) {
        if constexpr (!EnableNorm) {
            if (topk_length == nullptr && (p.topk == 640 || p.topk == 2048) &&
                get_num_sm(get_current_device()) == 39) {
                if (p.topk == 2048) launch_fused_prefill_hs64_persistent<true>(params, p.stream);
                else launch_fused_prefill_hs64_persistent<false>(params, p.stream);
                return;
            }
            if (p.topk == 2048 && topk_length == nullptr) {
                launch_fused_prefill_hs64<EnableNorm, true, true, true>(params, p.stream);
                return;
            }
        }
        if(p.topk < 2048) launch_fused_prefill_hs64<EnableNorm, true, true>(params, p.stream);
        else launch_fused_prefill_hs64<EnableNorm, true>(params, p.stream);
    } else {
        if constexpr (!EnableNorm) {
            if (p.topk < 2048) {
                launch_fused_prefill_hs64<EnableNorm, false, true>(params, p.stream);
                return;
            }
        }
        launch_fused_prefill_hs64<EnableNorm, false>(params, p.stream);
    }
}

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist>
__global__ void __launch_bounds__(512, 1, 1)
fused_norm_rope_attn_rope_cast_decode_hs64_packed_kernel(__grid_constant__ const FusedHs64Params params) {
#if ACOMPUTE_VERSION >= 10500
    flashmla::dsa::hs64::hs64_attention<FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist>, true>(params);
#endif
}

template<bool Norm, bool ExtraFP4, int MainCapacity, flashmla::dsa::hs64::Hs64Worklist Worklist>
static void launch_fused_hs64_packed(const FusedHs64Params& params, hggcStream_t stream) {
    using T = FusedHs64PackedTraits<Norm, ExtraFP4, MainCapacity, Worklist>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &fused_norm_rope_attn_rope_cast_decode_hs64_packed_kernel<Norm, ExtraFP4, MainCapacity, Worklist>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const int grid_rows = Worklist == flashmla::dsa::hs64::Hs64Worklist::Single ? params.b : 39;
    kernel<<<dim3(1, 1, grid_rows), T::NUM_THREADS, bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool Norm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed(const Params& p) {
    FusedHs64Params params{};
    // Fused decode already flattens [B, Sq] and repeats lengths per query.
    params.b = p.s_q;
    params.q_orig = 1;
    params.seqlen_q = params.h_q = params.ngroups = 64;
    params.h = params.h_h_k_ratio = 1;
    params.d = params.d_v = 512;
    params.scale_softmax = p.sm_scale;
    params.scale_softmax_log2 = p.sm_scale_log2;
    params.q_ptr = const_cast<void*>(p.q);
    params.q_batch_stride = p.stride_q_s_q;
    params.q_row_stride = 512;
    params.k_ptr = const_cast<void*>(p.kv);
    params.k_batch_stride = p.stride_kv_s_kv;
    params.k_row_stride = 528;
    params.page_block_size = p.page_block_size;
    params.num_blocks = p.num_blocks;
    params.indices_ptr = const_cast<int32_t*>(p.indices);
    params.indices_batch_stride = p.stride_indices_s_q;
    params.topk_len_ptr = const_cast<int32_t*>(p.topk_length);
    params.topk = p.topk;
    params.extra_k_ptr = const_cast<void*>(p.extra_kv);
    params.extra_k_batch_stride = p.stride_extra_kv_block;
    params.extra_k_row_stride = ExtraFP4 ? 288 : 528;
    params.extra_page_block_size = p.extra_page_block_size;
    params.extra_num_blocks = p.extra_num_blocks;
    params.extra_indices_ptr = const_cast<int32_t*>(p.extra_indices);
    params.extra_indices_batch_stride = p.stride_extra_indices_s_q;
    params.extra_topk_len_ptr = const_cast<int32_t*>(p.extra_topk_length);
    params.extra_topk = p.extra_topk;
    params.attn_sink_ptr = const_cast<float*>(p.attn_sink);
    params.o_ptr = p.out_fp8;
    // The shared HS64 body forms its BF16 output view; the hook stores FP8.
    params.o_row_stride = 512 / sizeof(cutlass::bfloat16_t);
    params.o_batch_stride = 64 * params.o_row_stride;
    params.softmax_lse_ptr = p.lse;
    params.rms_norm_eps = p.rms_norm_eps;
    params.token_positions = p.token_positions;
    params.cos_sin_cache = p.cos_sin_cache;
    params.out_fp8 = p.out_fp8;
    params.out_sf = p.out_sf;
    params.stride_q_s_q = p.stride_q_s_q;
    params.stride_out_sf_group = p.stride_out_sf_group;
    params.stride_out_sf_word = p.stride_out_sf_word;
    using flashmla::dsa::hs64::Hs64Worklist;
    if (p.topk == 256) {
        launch_fused_hs64_packed<Norm, ExtraFP4, 256, Hs64Worklist::Batch>(params, p.stream);
    } else if (p.s_q >= 8 * 39) {
        launch_fused_hs64_packed<Norm, ExtraFP4, 128, Hs64Worklist::Flat>(params, p.stream);
    } else {
        launch_fused_hs64_packed<Norm, ExtraFP4, 128, Hs64Worklist::Single>(params, p.stream);
    }
}

}
