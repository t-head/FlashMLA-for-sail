#pragma once

#include <cutlass/float8.h>
#include <cmath>
#include "config.h"
#include "fused_tile_driver.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

using namespace cute;

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

template<int Rows>
__forceinline__ __device__ int paired_q_offset(int row, int dim) {
    const int tile = dim / 64;
    const int col = dim % 64;
    return (tile & ~1) * Rows * 64 + (row / 8) * 1024 + (tile & 1) * 512
        + (row & 7) * 64 + (col ^ ((row & 7) << 3));
}

template<int Rows>
__forceinline__ __device__ int paired_k_offset(int row, int dim) {
    const int tile = dim / 64;
    const int col = dim % 64;
    return (tile & ~1) * Rows * 64 + (row / 16) * 2048 + (row & 15) * 64
        + (tile & 1) * 1024 + ((col / 16) ^ ((row / 2) & 3)) * 16
        + (((col / 8) & 1) ^ (row & 1)) * 8 + col % 8;
}

__forceinline__ __device__ float fp8_value(uint8_t bits) {
    return cutlass::float_e4m3_t::to_float(cutlass::float_e4m3_t::bitcast(bits));
}

__forceinline__ __device__ float ue8m0_value(uint8_t bits) {
    const uint32_t raw = bits == 0 ? 0x00400000u
        : bits == 255 ? 0x7fffffffu : uint32_t(bits) << 23;
    return __uint_as_float(raw);
}

__forceinline__ __device__ float fp4_value(uint8_t bits) {
    constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float x = magnitudes[bits & 7];
    return bits & 8 ? -x : x;
}

__forceinline__ __device__ uint16_t fp8_pair_from_float(float low, float high) {
    uint16_t result;
    asm volatile("ppu.cvt.rtte.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(result) : "f"(high), "f"(low));
    return result;
}

template<int S, int E, typename T, bool FrontDrain, typename RawK>
__forceinline__ __device__ void CacheSupplier::load(
    const TileParams& p, RawK& raw_k, int tidx, int token_idx,
    bool valid, int block_idx, bool) {
    static_assert(E - S == 4);
    const int row = tidx / 8;
    const int lane = tidx & 7;
    const int absolute = block_idx * T::kBlockN + row;
    const bool extra = absolute >= p.main_padded_length;
    const bool fp4 = T::kExtraFP4 && extra;
    const uint8_t* bytes = nullptr;
    const typename T::InputT* bf16 = nullptr;
    if(valid) {
        if constexpr(T::kMode == FwdMode::Prefill) {
            bf16 = static_cast<const typename T::InputT*>(p.kv)
                + int64_t(token_idx) * p.stride_kv_s_kv;
        } else {
            const int page_size = extra ? p.extra_page_size : p.quant_page_size;
            const int page = token_idx / page_size;
            const int in_page = token_idx % page_size;
            const int64_t block_stride = extra ? p.extra_block_stride : p.quant_block_stride;
            const uint8_t* base = static_cast<const uint8_t*>(extra ? p.extra_kv : p.kv);
            bytes = base + int64_t(page) * block_stride + int64_t(in_page) * (fp4 ? 288 : 528);
        }
    }
    if constexpr(FrontDrain) {
        cutlass::arch::NamedBarrier::sync(256,
            static_cast<cutlass::arch::ReservedNamedBarriers>(1 + (threadIdx.x >> 8)));
    }
    auto* dst = cute::raw_pointer_cast(raw_k.data());
    #pragma unroll
    for(int tile = S; tile < E; ++tile) {
        if constexpr(T::Arch == 89 && T::kMode == FwdMode::Prefill) {
            const int dim = tile * 64 + lane * 8;
            const int offset = T::kUsePairedCubes ? paired_k_offset<T::kBlockN>(row, dim)
                : native_offset<T::kBlockN>(row, dim);
            if(!valid || (reinterpret_cast<uintptr_t>(bf16) & 15) == 0) {
                const auto* source = valid ? bf16 + dim : static_cast<const typename T::InputT*>(p.kv);
                PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                    *reinterpret_cast<const cute::uint128_t*>(source),
                    *reinterpret_cast<cute::uint128_t*>(dst + offset), valid);
            } else {
                uint4 raw{};
                auto* elements = reinterpret_cast<typename T::InputT*>(&raw);
                #pragma unroll
                for(int i = 0; i < 8; ++i) elements[i] = bf16[dim + i];
                *reinterpret_cast<uint4*>(dst + offset) = raw;
            }
        } else if constexpr(T::kVectorDecodeKV) {
            const int dim = tile * 64 + lane * 8;
            uint64_t bits = 0;
            float scale = 0.0f;
            if(valid) {
                if(fp4) {
                    const uint8_t* source = bytes + dim / 2;
                    if((reinterpret_cast<uintptr_t>(source) & 3) == 0) {
                        bits = __ldg(reinterpret_cast<const uint32_t*>(source));
                    } else {
                        #pragma unroll
                        for(int i = 0; i < 4; ++i) bits |= uint64_t(__ldg(source + i)) << (i * 8);
                    }
                    scale = fp8_value(__ldg(bytes + 256 + dim / 16));
                } else {
                    const uint8_t* source = bytes + dim;
                    if((reinterpret_cast<uintptr_t>(source) & 7) == 0) {
                        bits = __ldg(reinterpret_cast<const uint64_t*>(source));
                    } else {
                        #pragma unroll
                        for(int i = 0; i < 8; ++i) bits |= uint64_t(__ldg(source + i)) << (i * 8);
                    }
                    scale = ue8m0_value(__ldg(bytes + 512 + dim / 32));
                }
            }
            alignas(16) typename T::InputT packed[8];
            if(valid && fp4) {
                const uint4 values = flash::fp4_word_to_bf16x2_bits(uint32_t(bits));
                const __ppu_bfloat162 scale2 = __float2bfloat162_rn(scale);
                auto* output = reinterpret_cast<__ppu_bfloat162*>(packed);
                output[0] = __hmul2(*reinterpret_cast<const __ppu_bfloat162*>(&values.x), scale2);
                output[1] = __hmul2(*reinterpret_cast<const __ppu_bfloat162*>(&values.y), scale2);
                output[2] = __hmul2(*reinterpret_cast<const __ppu_bfloat162*>(&values.z), scale2);
                output[3] = __hmul2(*reinterpret_cast<const __ppu_bfloat162*>(&values.w), scale2);
            } else {
                #pragma unroll
                for(int i = 0; i < 8; ++i) {
                    packed[i] = valid ? typename T::InputT(
                        fp8_value(uint8_t(bits >> (i * 8))) * scale) : typename T::InputT(0.0f);
                }
            }
            const int offset = T::kUsePairedCubes ? paired_k_offset<T::kBlockN>(row, dim)
                : native_offset<T::kBlockN>(row, dim);
            *reinterpret_cast<uint4*>(dst + offset) = *reinterpret_cast<const uint4*>(packed);
        } else {
            #pragma unroll
            for(int i = 0; i < 8; ++i) {
                const int dim = tile * 64 + lane * 8 + i;
                typename T::InputT x = typename T::InputT(0.0f);
                if(valid) {
                    if constexpr(T::kMode == FwdMode::Prefill) {
                        x = bf16[dim];
                    } else if(fp4) {
                        const uint8_t packed = __ldg(bytes + dim / 2);
                        const uint8_t code = (packed >> ((dim & 1) * 4)) & 15;
                        x = typename T::InputT(fp4_value(code) * fp8_value(__ldg(bytes + 256 + dim / 16)));
                    } else {
                        x = typename T::InputT(fp8_value(__ldg(bytes + dim))
                            * ue8m0_value(__ldg(bytes + 512 + dim / 32)));
                    }
                }
                const int offset = T::kUsePairedCubes ? paired_k_offset<T::kBlockN>(row, dim)
                    : native_offset<T::kBlockN>(row, dim);
                dst[offset] = x;
            }
        }
    }
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1 + (threadIdx.x >> 8)));
}

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch>
template<typename T, typename TensorQ>
__forceinline__ __device__ void Config<Mode, Heads, EnableNorm, ExtraFP4, Arch>::load_q(
    const TileParams& p, int, TensorQ& q, int tidx, int warp_idx, __mbarrier_t* barrier) {
    extern __shared__ char wksp_buf[];
    auto& plan = *reinterpret_cast<typename T::SharedMemoryPlan*>(wksp_buf);
    const auto& fused = *p.fused;
    float* q_sqr_partials = nullptr;
    float* q_norm = nullptr;
    if constexpr(EnableNorm) {
        if constexpr(T::kFusedNativeDecode) {
            // All Q threads finish the reduction before the first K issue.
            q_sqr_partials = reinterpret_cast<float*>(plan.smem_sK.data());
            q_norm = reinterpret_cast<float*>(wksp_buf + 261248);
        } else {
            q_sqr_partials = plan.q_sqr_partials.data();
            q_norm = plan.q_norm.data();
        }
    }
    const int head = tidx / 4;
    const int dim_begin = (tidx & 3) * 128;
    const auto* src = static_cast<const typename T::InputT*>(p.q);
    auto* dst = cute::raw_pointer_cast(q.data());
    const int position = __ldg(fused.token_positions + p.token_idx);
    const float* rope = fused.cos_sin_cache + int64_t(position) * 64;
    float sum = 0.0f;
    bool async_q = false;
    if constexpr(Arch == 89) {
        const bool aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
        if(aligned) {
            // All raw chunks are in flight before any use; RoPE is applied after
            // the RMS reads, so every value and FMA order matches the serial loop.
            async_q = true;
            const bool live = head < p.real_heads;
            #pragma unroll
            for(int i = 0; i < 128; i += 8) {
                const int dim = dim_begin + i;
                const int offset = T::kUsePairedCubes ? paired_q_offset<T::kBlockM>(head, dim)
                    : native_offset<T::kBlockM>(head, dim);
                const auto* source = live
                    ? src + (dim / 16) * (p.real_heads * 16) + head * 16 + dim % 16 : src;
                PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                    *reinterpret_cast<const cute::uint128_t*>(source),
                    *reinterpret_cast<cute::uint128_t*>(dst + offset), live);
            }
            cute::cp_async_fence();
            cute::cp_async_wait<0>();
            if constexpr(EnableNorm) {
                #pragma unroll
                for(int i = 0; i < 128; i += 8) {
                    const int dim = dim_begin + i;
                    const int offset = T::kUsePairedCubes ? paired_q_offset<T::kBlockM>(head, dim)
                        : native_offset<T::kBlockM>(head, dim);
                    const uint4 raw = *reinterpret_cast<const uint4*>(dst + offset);
                    const auto* elements = reinterpret_cast<const typename T::InputT*>(&raw);
                    #pragma unroll
                    for(int j = 0; j < 8; j += 2) {
                        const float2 x{float(elements[j]), float(elements[j + 1])};
                        sum = fmaf(x.x, x.x, sum);
                        sum = fmaf(x.y, x.y, sum);
                    }
                }
            }
        } else {
        #pragma unroll 1
        for(int i = 0; i < 128; i += 8) {
            const int dim = dim_begin + i;
            uint4 raw{};
            if(head < p.real_heads) {
                const int offset = (dim / 16) * (p.real_heads * 16) + head * 16 + dim % 16;
                if(aligned) {
                    raw = __ldg(reinterpret_cast<const uint4*>(src + offset));
                } else {
                    auto* elements = reinterpret_cast<typename T::InputT*>(&raw);
                    #pragma unroll
                    for(int j = 0; j < 8; ++j) elements[j] = src[offset + j];
                }
            }
            const auto* elements = reinterpret_cast<const typename T::InputT*>(&raw);
            alignas(16) typename T::InputT packed[8];
            #pragma unroll
            for(int j = 0; j < 8; j += 2) {
                float2 x{float(elements[j]), float(elements[j + 1])};
                if constexpr(EnableNorm) {
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
            const int offset = T::kUsePairedCubes ? paired_q_offset<T::kBlockM>(head, dim)
                : native_offset<T::kBlockM>(head, dim);
            *reinterpret_cast<uint4*>(dst + offset) = *reinterpret_cast<const uint4*>(packed);
        }
        }
    } else {
        #pragma unroll 1
        for(int i = 0; i < 128; i += 2) {
            const int dim = dim_begin + i;
            float2 x{0.0f, 0.0f};
            if(head < p.real_heads) {
                const int offset = (dim / 16) * (p.real_heads * 16) + head * 16 + dim % 16;
                x = float2{float(src[offset]), float(src[offset + 1])};
            }
            if constexpr(EnableNorm) {
                sum = fmaf(x.x, x.x, sum);
                sum = fmaf(x.y, x.y, sum);
            }
            if(dim >= 448) {
                const int pair = (dim - 448) / 2;
                x = apply_rope(x, __ldg(rope + pair), __ldg(rope + 32 + pair));
            }
            const int offset0 = T::kUsePairedCubes ? paired_q_offset<T::kBlockM>(head, dim)
                : native_offset<T::kBlockM>(head, dim);
            const int offset1 = T::kUsePairedCubes ? paired_q_offset<T::kBlockM>(head, dim + 1)
                : native_offset<T::kBlockM>(head, dim + 1);
            dst[offset0] = typename T::InputT(x.x);
            dst[offset1] = typename T::InputT(x.y);
        }
    }
    if constexpr(EnableNorm) q_sqr_partials[tidx] = sum;
    __syncthreads();
    if constexpr(Arch == 89) {
        if(async_q) {
            // RMS partials are published; spread the 1024 RoPE chunks over the CTA.
            #pragma unroll
            for(int chunk = tidx; chunk < T::kBlockM * 8; chunk += T::NUM_THREADS) {
                const int rope_head = chunk / 8;
                const int dim = 448 + (chunk % 8) * 8;
                const int offset = T::kUsePairedCubes ? paired_q_offset<T::kBlockM>(rope_head, dim)
                    : native_offset<T::kBlockM>(rope_head, dim);
                const uint4 raw = *reinterpret_cast<const uint4*>(dst + offset);
                const auto* elements = reinterpret_cast<const typename T::InputT*>(&raw);
                alignas(16) typename T::InputT packed[8];
                #pragma unroll
                for(int j = 0; j < 8; j += 2) {
                    const int pair = (dim + j - 448) / 2;
                    const float2 x = apply_rope(float2{float(elements[j]), float(elements[j + 1])},
                        __ldg(rope + pair), __ldg(rope + 32 + pair));
                    packed[j] = typename T::InputT(x.x);
                    packed[j + 1] = typename T::InputT(x.y);
                }
                *reinterpret_cast<uint4*>(dst + offset) = *reinterpret_cast<const uint4*>(packed);
            }
        }
    }
    if constexpr(EnableNorm) {
        if(tidx < T::kBlockM) {
            const float* partials = q_sqr_partials + tidx * 4;
            const float sqr = (partials[0] + partials[1]) + (partials[2] + partials[3]);
            q_norm[tidx] = tidx < p.real_heads ? rsqrtf(sqr / 512.0f + fused.rms_norm_eps) : 1.0f;
        }
    }
    __syncthreads();
    if constexpr(T::kUsePairedCubes) {
        cutlass::arch::cpasync_barrier_arrive_noinc(barrier);
    } else if(warp_idx == 0) {
        cutlass::arch::cpasync_barrier_arrive_noinc(barrier);
    }
}

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch>
template<typename T, typename TensorQ>
__forceinline__ __device__ int Config<Mode, Heads, EnableNorm, ExtraFP4, Arch>::issue_q(
    const TileParams& p, int m_block_idx, TensorQ& q, int tidx, int warp_idx, __mbarrier_t* barrier) {
    if constexpr(Arch == 89 && T::kUsePairedCubes) {
        const auto* src = static_cast<const typename T::InputT*>(p.q);
        if((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
            const int position = __ldg(p.fused->token_positions + p.token_idx);
            const int head = tidx / 4;
            const int dim_begin = (tidx & 3) * 128;
            const bool live = head < p.real_heads;
            auto* dst = cute::raw_pointer_cast(q.data());
            #pragma unroll
            for(int i = 0; i < 128; i += 8) {
                const int dim = dim_begin + i;
                const int offset = paired_q_offset<T::kBlockM>(head, dim);
                const auto* source = live
                    ? src + (dim / 16) * (p.real_heads * 16) + head * 16 + dim % 16 : src;
                PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                    *reinterpret_cast<const cute::uint128_t*>(source),
                    *reinterpret_cast<cute::uint128_t*>(dst + offset), live);
            }
            // Like the production paired loader, raw Q lands while K is issued;
            // finish_q applies RMS and RoPE after barrier_Q completes.
            cutlass::arch::cpasync_barrier_arrive_noinc(barrier);
            return position;
        }
    }
    load_q<T>(p, m_block_idx, q, tidx, warp_idx, barrier);
    return 0;
}

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch>
template<typename T, typename TensorQ>
__forceinline__ __device__ void Config<Mode, Heads, EnableNorm, ExtraFP4, Arch>::finish_q(
    const TileParams& p, TensorQ& q, int tidx, int position) {
    if constexpr(Arch == 89 && T::kUsePairedCubes) {
        const auto* src = static_cast<const typename T::InputT*>(p.q);
        if((reinterpret_cast<uintptr_t>(src) & 15) != 0) return;
        const auto& fused = *p.fused;
        auto* dst = cute::raw_pointer_cast(q.data());
        if constexpr(EnableNorm) {
            extern __shared__ char wksp_buf[];
            const int head = tidx / 4;
            const int dim_begin = (tidx & 3) * 128;
            float sum = 0.0f;
            #pragma unroll
            for(int i = 0; i < 128; i += 8) {
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    dst + paired_q_offset<T::kBlockM>(head, dim_begin + i));
                const auto* elements = reinterpret_cast<const typename T::InputT*>(&raw);
                #pragma unroll
                for(int j = 0; j < 8; j += 2) {
                    const float2 x{float(elements[j]), float(elements[j + 1])};
                    sum = fmaf(x.x, x.x, sum);
                    sum = fmaf(x.y, x.y, sum);
                }
            }
            // The quad shuffle forms the same (p0 + p1) + (p2 + p3) tree.
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            if((tidx & 3) == 0) {
                float* q_norm = nullptr;
                if constexpr(T::kFusedNativeDecode) {
                    q_norm = reinterpret_cast<float*>(wksp_buf + 261248);
                } else {
                    q_norm = reinterpret_cast<typename T::SharedMemoryPlan*>(wksp_buf)->q_norm.data();
                }
                q_norm[head] = head < p.real_heads ? rsqrtf(sum / 512.0f + fused.rms_norm_eps) : 1.0f;
            }
            __syncthreads();
        }
        const float* rope = fused.cos_sin_cache + int64_t(position) * 64;
        #pragma unroll
        for(int chunk = tidx; chunk < T::kBlockM * 8; chunk += T::NUM_THREADS) {
            const int dim = 448 + (chunk % 8) * 8;
            const int offset = paired_q_offset<T::kBlockM>(chunk / 8, dim);
            const uint4 raw = *reinterpret_cast<const uint4*>(dst + offset);
            const auto* elements = reinterpret_cast<const typename T::InputT*>(&raw);
            alignas(16) typename T::InputT packed[8];
            #pragma unroll
            for(int j = 0; j < 8; j += 2) {
                const int pair = (dim + j - 448) / 2;
                const float2 x = apply_rope(float2{float(elements[j]), float(elements[j + 1])},
                    __ldg(rope + pair), __ldg(rope + 32 + pair));
                packed[j] = typename T::InputT(x.x);
                packed[j + 1] = typename T::InputT(x.y);
            }
            *reinterpret_cast<uint4*>(dst + offset) = *reinterpret_cast<const uint4*>(packed);
        }
        __syncthreads();
    }
}

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch>
template<typename T, typename TensorO>
__forceinline__ __device__ void Config<Mode, Heads, EnableNorm, ExtraFP4, Arch>::store_output(
    const TileParams& p, TensorO& output, typename T::SharedMemoryPlan& plan,
    int warpgroup_idx, int idx_in_warpgroup) {
    const auto& fused = *p.fused;
    const int head = threadIdx.x / 4;
    const int lane = threadIdx.x & 3;
    __syncthreads();
    const float li = plan.sL_reduction_wksp[head];
    const float mi = plan.smem_sM[head];
    const bool valid = li > 0.0f && mi > ::flashmla::sparse_prefill::MAX_INIT_VAL_SM;
    float multiplier = 0.0f;
    if(valid && head < p.real_heads) {
        const float sink = fused.attn_sink ? __ldg(fused.attn_sink + head) * float(M_LOG2E) : -INFINITY;
        multiplier = __fdividef(1.0f, li + exp2f(sink - mi));
    }
    if(lane == 0 && head < p.real_heads) {
        const int64_t index = int64_t(p.token_idx) * p.real_heads + head;
        fused.lse[index] = valid ? fmaf(mi, float(M_LN2), logf(li)) : INFINITY;
        if constexpr(Mode == FwdMode::Prefill) {
            fused.max_logits[index] = valid ? mi * float(M_LN2) : -INFINITY;
        }
    }
    const int position = __ldg(fused.token_positions + p.token_idx);
    const float* rope = fused.cos_sin_cache + int64_t(position) * 64;
    __syncthreads();

    Tensor staged = make_tensor(make_smem_ptr(reinterpret_cast<float*>(plan.smem_sQ.data())),
        typename T::SmemLayoutO{});
    Tensor half = local_tile(staged, Shape<_128, _256>{}, make_coord(_0{}, warpgroup_idx));
    typename T::TiledMma mma;
    auto copy_c = make_tiled_copy_C(typename T::SmemCopyAtomOaccum{}, mma);
    auto copy_thr = copy_c.get_slice(idx_in_warpgroup);
    Tensor src = copy_thr.retile_S(output);
    Tensor dst = copy_thr.partition_D(half);
    if constexpr(T::kUsePairedCubes) {
        #pragma unroll
        for(int j = 0; j < size<2>(dst); ++j) {
            #pragma unroll
            for(int i = 0; i < size<0>(dst); ++i) {
                const int real_i = (i & 3) | (j & 4);
                const int real_j = i + j - real_i;
                cute::copy(copy_c, src(real_i, _, real_j), dst(i, _, j));
            }
        }
    } else {
        cute::copy(copy_c, src, dst);
    }
    __syncthreads();

    if(head < p.real_heads) {
        // Tiles are independent; unrolling lets their reads and casts overlap.
        #pragma unroll
        for(int step = 0; step < 4; ++step) {
            const int tile = lane + step * 4;
            alignas(16) float values[32];
            if constexpr(T::kVectorOutputLoad) {
                #pragma unroll
                for(int i = 0; i < 32; i += 4) {
                    const float4 chunk = multiplier == 0.0f ? float4{0.0f, 0.0f, 0.0f, 0.0f}
                        : *reinterpret_cast<const float4*>(&staged(head, tile * 32 + i));
                    *reinterpret_cast<float4*>(values + i) = chunk;
                }
            } else {
                #pragma unroll
                for(int i = 0; i < 32; ++i) values[i] = multiplier == 0.0f ? 0.0f : staged(head, tile * 32 + i);
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
            // fmaxf over non-negative values is order independent.
            float amax0 = 0.0f, amax1 = 0.0f, amax2 = 0.0f, amax3 = 0.0f;
            #pragma unroll
            for(int i = 0; i < 32; i += 4) {
                amax0 = fmaxf(amax0, fabsf(values[i]));
                amax1 = fmaxf(amax1, fabsf(values[i + 1]));
                amax2 = fmaxf(amax2, fabsf(values[i + 2]));
                amax3 = fmaxf(amax3, fabsf(values[i + 3]));
            }
            float amax = fmaxf(fmaxf(amax0, amax1), fmaxf(amax2, amax3));
            amax = fmaxf(1e-4f, amax * multiplier);
            const float scale = amax / 448.0f;
            const int exponent = int((__float_as_uint(scale) - 1u) >> 23) + 1 - 127;
            const float scale_inv = __uint_as_float(uint32_t(127 - exponent) << 23);
            const float cast_multiplier = multiplier * scale_inv;
            alignas(Arch == 89 ? 16 : 1) uint8_t packed[32];
            if constexpr(Arch == 89) {
                auto* pairs = reinterpret_cast<uint16_t*>(packed);
                #pragma unroll
                for(int i = 0; i < 32; i += 2) {
                    pairs[i / 2] = fp8_pair_from_float(
                        values[i] * cast_multiplier, values[i + 1] * cast_multiplier);
                }
            } else {
                #pragma unroll
                for(int i = 0; i < 32; ++i) {
                    packed[i] = cutlass::float_e4m3_t::from_float(values[i] * cast_multiplier).storage;
                }
            }
            const int group = head / 8;
            const int in_group = head & 7;
            const int64_t out_offset = int64_t(p.token_idx) * p.real_heads * 512
                + group * (8 * 512) + tile * (8 * 32) + in_group * 32;
            if constexpr(Arch == 89) {
                auto* out = reinterpret_cast<uint4*>(fused.out_fp8 + out_offset);
                const auto* words = reinterpret_cast<const uint4*>(packed);
                out[0] = words[0];
                out[1] = words[1];
            } else {
                #pragma unroll
                for(int i = 0; i < 32; ++i) fused.out_fp8[out_offset + i] = packed[i];
            }
            const int sf_index = tile * 8 + in_group;
            auto* sf_word = fused.out_sf + p.token_idx + int64_t(group) * fused.stride_out_sf_group
                + int64_t(sf_index / 4) * fused.stride_out_sf_word;
            reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
        }
    }
}

template<typename T>
__forceinline__ __device__ TileParams make_tile(const Params& fused, int token) {
    TileParams p{};
    p.fused = &fused;
    p.token_idx = token;
    p.real_heads = T::kRealHeads;
    p.s_q = 1;
    p.h_q = T::kBlockM;
    p.h_kv = 1;
    p.d_qk = p.d_v = 512;
    p.s_kv = T::kMode == FwdMode::Prefill ? fused.s_kv : fused.num_blocks * fused.page_block_size;
    p.q = const_cast<cutlass::bfloat16_t*>(static_cast<const cutlass::bfloat16_t*>(fused.q))
        + int64_t(token) * fused.stride_q_s_q;
    p.kv = const_cast<void*>(fused.kv);
    p.indices = const_cast<int32_t*>(fused.indices) + int64_t(token) * fused.stride_indices_s_q;
    p.stride_q_s_q = fused.stride_q_s_q;
    p.stride_q_h_q = 512;
    p.stride_kv_s_kv = fused.stride_kv_s_kv;
    p.sm_scale = fused.sm_scale;
    p.sm_scale_div_log2 = fused.sm_scale_log2;
    p.main_storage_topk = fused.topk;
    p.main_length = fused.topk_length ? __ldg(fused.topk_length + token) : fused.topk;
    p.main_length = min(max(p.main_length, 0), p.main_storage_topk);
    p.main_padded_length = fused.topk;
    if constexpr(T::kMode == FwdMode::Decode) {
        p.quant_page_size = fused.page_block_size;
        p.quant_block_stride = fused.stride_kv_s_kv;
        p.extra_kv = const_cast<void*>(fused.extra_kv);
        p.extra_indices = fused.extra_topk > 0
            ? const_cast<int32_t*>(fused.extra_indices) + int64_t(token) * fused.stride_extra_indices_s_q : nullptr;
        p.extra_storage_topk = fused.extra_topk;
        p.extra_length = fused.extra_topk_length ? __ldg(fused.extra_topk_length + token) : fused.extra_topk;
        p.extra_length = min(max(p.extra_length, 0), p.extra_storage_topk);
        p.extra_s_kv = fused.extra_num_blocks * fused.extra_page_block_size;
        p.extra_page_size = fused.extra_page_block_size;
        p.extra_block_stride = fused.stride_extra_kv_block;
    }
    p.topk = p.extra_storage_topk > 0 ? fused.topk + p.extra_length : p.main_length;
    p.start_block = 0;
    p.end_block = cute::ceil_div(max(p.topk, 1), 64) * 2;
    p.topk = max(p.topk, 1);
    p.partial_output = false;
    return p;
}

template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
fused_norm_rope_attn_rope_cast_fwd_kernel(__grid_constant__ const Params params) {
    const int token = blockIdx.x;
    const TileParams tile = make_tile<T>(params, token);
    fused_wi::tile_body<T>(tile, 0);
}

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch>
void launch(const Params& params) {
    using T = Config<Mode, Heads, EnableNorm, ExtraFP4, Arch>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryOutPut);
    if constexpr(Mode == FwdMode::Prefill && Arch == 89) {
        if(params.topk_length == nullptr) {
            using V = NativePrefillConfig<Heads, EnableNorm, Arch>;
            auto native_kernel = &fused_norm_rope_attn_rope_cast_fwd_kernel<V>;
            C10_CUDA_CHECK(hggcFuncSetAttribute(native_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
            native_kernel<<<dim3(params.s_q, 1, 1), V::NUM_THREADS, bytes, params.stream>>>(params);
            CHECK_CUDA_KERNEL_LAUNCH();
            return;
        }
    }
    if constexpr(Mode == FwdMode::Decode && Arch == 89 && Heads == 128) {
        if(params.extra_topk > 0) {
            using V = VectorDecodeConfig<Heads, EnableNorm, ExtraFP4, Arch>;
            auto vector_kernel = &fused_norm_rope_attn_rope_cast_fwd_kernel<V>;
            C10_CUDA_CHECK(hggcFuncSetAttribute(vector_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
            vector_kernel<<<dim3(params.s_q, 1, 1), V::NUM_THREADS, bytes, params.stream>>>(params);
            CHECK_CUDA_KERNEL_LAUNCH();
            return;
        }
    }
    auto kernel = &fused_norm_rope_attn_rope_cast_fwd_kernel<T>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(params.s_q, 1, 1), T::NUM_THREADS, bytes, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const Params& params) {
    TORCH_CHECK(params.mode == Mode && params.h_q == Heads && params.enable_q_norm == EnableNorm);
    TORCH_CHECK(params.wv_group_size == 8 && params.n_wv_group == Heads / 8);
    if constexpr(Mode == FwdMode::Decode) TORCH_CHECK(params.extra_fp4 == ExtraFP4);
    if(params.s_q == 0) return;
    if(params.arch == 80) {
        launch<Mode, Heads, EnableNorm, ExtraFP4, 80>(params);
    } else {
        TORCH_CHECK(params.arch == 89, "Fused WI requires SM80 or SM89");
        launch<Mode, Heads, EnableNorm, ExtraFP4, 89>(params);
    }
}

}
