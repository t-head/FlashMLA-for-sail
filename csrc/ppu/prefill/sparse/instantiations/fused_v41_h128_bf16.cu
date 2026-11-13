#include <cmath>
#include <math_constants.h>
#include <type_traits>

#include "api/common.h"
#include "sparse/fused_v41_wi.cuh"
#include "decode/sparse/fused_v41_splitkv.cuh"
#include "prefill/sparse/sparse_prefill_std.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool PerTokenScale>
struct FusedH128Params : SparsePrefillParams {
    OutputParams<PerTokenScale> fused;
};

template<bool Norm, bool PerTokenScale>
struct FusedH128PrefillTraits : DSA_Traits<cutlass::bfloat16_t, 512, 89> {
    using Base = DSA_Traits<cutlass::bfloat16_t, 512, 89>;
    using FusedConfig = Config<FwdMode::Prefill, 128, Norm, false, 89, PerTokenScale>;
    using KernelParams = FusedH128Params<PerTokenScale>;
    static constexpr bool kFusedH128 = true;
    static constexpr bool kFusedNativeDecode = false;
    static constexpr bool kVectorOutputLoad = true;

    template<int Group>
    static __forceinline__ __device__ void sync_warpgroup() {
        // Use physical IDs: the user-ID overload adds five, so mGroup1 (11)
        // would become 16 and alias the CTA barrier used by fused Q/epilogue.
        cutlass::arch::NamedBarrier::arrive_and_wait(Base::NUM_THREADS / 2,
            static_cast<cutlass::arch::ReservedNamedBarriers>(Group + 1));
    }

    struct SharedMemoryPlan : Base::SharedMemoryPlan {
        cute::array_aligned<float, 512> q_sqr_partials;
        cute::array_aligned<float, 128> q_norm;
    };

    static_assert(Base::kUsePairedCubes);
    static_assert(Base::NUM_THREADS == 512 && Base::kBlockM == 128
                  && Base::kQRowsPerLoad == 8);
    static_assert(std::is_same_v<typename Base::SmemLayoutQ,
                                 typename FusedConfig::SmemLayoutQ>);
    static_assert(std::is_same_v<typename Base::SmemLayoutO,
                                 typename FusedConfig::SmemLayoutO>);
    static_assert(std::is_same_v<typename Base::SmemCopyAtomOaccum,
                                 typename FusedConfig::SmemCopyAtomOaccum>);
    static_assert(std::is_same_v<typename Base::TiledMma,
                                 typename FusedConfig::TiledMma>);
    static_assert(sizeof(SharedMemoryPlan) < 256 * 1024);

    static __forceinline__ __device__ float score_scale(float scale, int row) {
        if constexpr (Norm) {
            extern __shared__ char wksp_buf[];
            const auto& plan = *reinterpret_cast<const SharedMemoryPlan*>(wksp_buf);
            return scale * plan.q_norm[row];
        } else {
            return scale;
        }
    }

    static __forceinline__ __device__ TileParams make_tile(
        const FusedH128Params<PerTokenScale>& params, int token) {
        TileParams tile{};
        tile.fused = &params.fused;
        tile.token_idx = token;
        tile.real_heads = 128;
        tile.q = const_cast<cutlass::bfloat16_t*>(
            static_cast<const cutlass::bfloat16_t*>(params.fused.q))
            + int64_t(token) * params.fused.stride_q_s_q;
        return tile;
    }

    template<typename TensorQ>
    static __forceinline__ __device__ void load_q(
        const FusedH128Params<PerTokenScale>& params, int m_block_idx, TensorQ& q,
        int tidx, int warp_idx, __mbarrier_t* barrier) {
        const TileParams tile = make_tile(params, m_block_idx);
        FusedConfig::template load_q<FusedH128PrefillTraits<Norm, PerTokenScale>>(
            tile, m_block_idx, q, tidx, warp_idx, barrier);
    }

    template<typename TensorQ>
    static __forceinline__ __device__ int issue_q(
        const FusedH128Params<PerTokenScale>& params, int m_block_idx, TensorQ& q,
        int tidx, int warp_idx, __mbarrier_t* barrier) {
        const TileParams tile = make_tile(params, m_block_idx);
        return FusedConfig::template issue_q<FusedH128PrefillTraits<Norm, PerTokenScale>>(
            tile, m_block_idx, q, tidx, warp_idx, barrier);
    }

    template<typename TensorQ>
    static __forceinline__ __device__ void finish_q(
        const FusedH128Params<PerTokenScale>& params, int m_block_idx, TensorQ& q, int tidx, int position) {
        const TileParams tile = make_tile(params, m_block_idx);
        FusedConfig::template finish_q<FusedH128PrefillTraits<Norm, PerTokenScale>>(tile, q, tidx, position);
    }

    template<typename TensorO>
    static __forceinline__ __device__ void store_output(
        const FusedH128Params<PerTokenScale>& params, TensorO& output, SharedMemoryPlan& plan,
        int warpgroup_idx, int idx_in_warpgroup) {
        const int m_block_idx = blockIdx.x;
        const TileParams tile = make_tile(params, m_block_idx);
        FusedConfig::template store_output<FusedH128PrefillTraits<Norm, PerTokenScale>>(
            tile, output, plan, warpgroup_idx, idx_in_warpgroup);
    }
};

template<bool EnableNorm, bool PerTokenScale>
void run_fused_norm_rope_attn_rope_cast_prefill_h128_impl(const Params& p) {
    TORCH_CHECK(p.mode == FwdMode::Prefill && p.arch == 89 && p.h_q == 128);
    TORCH_CHECK(p.enable_q_norm == EnableNorm);
    TORCH_CHECK(p.wv_group_size == 8 && p.n_wv_group == 16);
    if (p.s_q == 0) return;

    at::Tensor clamped_length;
    const int32_t* topk_length = p.topk_length;
    if (topk_length) {
        const auto options = torch::TensorOptions().dtype(torch::kInt32)
            .device(torch::Device(torch::kCUDA, get_current_device()));
        clamped_length = torch::from_blob(const_cast<int32_t*>(topk_length), {p.s_q}, options)
            .clamp(0, p.topk);
        topk_length = clamped_length.data_ptr<int32_t>();
    }

    FusedH128Params<PerTokenScale> params{};
    params.s_q = p.s_q;
    params.s_kv = p.s_kv;
    params.h_q = 128;
    params.h_kv = 1;
    params.d_qk = params.d_v = 512;
    params.topk = p.topk;
    params.sm_scale = p.sm_scale;
    params.sm_scale_div_log2 = p.sm_scale_log2;
    params.q = const_cast<void*>(p.q);
    params.kv = const_cast<void*>(p.kv);
    params.indices = const_cast<int32_t*>(p.indices);
    params.attn_sink = const_cast<float*>(p.attn_sink);
    params.topk_length = const_cast<int32_t*>(topk_length);
    params.stride_q_s_q = int64_stride_to_int(p.stride_q_s_q);
    params.stride_q_h_q = 512;
    params.stride_kv_s_kv = int64_stride_to_int(p.stride_kv_s_kv);
    params.stride_kv_h_kv = 512;
    params.stride_indices_s_q = int64_stride_to_int(p.stride_indices_s_q);
    params.stride_indices_h_kv = p.topk;
    params.out = p.out_fp8;
    params.max_logits = p.max_logits;
    params.lse = p.lse;
    params.stream = p.stream;
    params.fused = OutputParams<PerTokenScale>{p};
    params.fused.topk_length = topk_length;

    using T = FusedH128PrefillTraits<EnableNorm, PerTokenScale>;
    constexpr size_t bytes = std::max(sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));
    auto kernel = &::flash_sparse_prefill_fwd_wg_kernel<T, true, false, FusedH128Params<PerTokenScale>>;
    // Match the ordinary D512 launcher. Each CTA still validates its own row
    // before selecting the unmasked body.
    if (p.topk >= 4 * (2 * T::kBlockN))
        kernel = &::flash_sparse_prefill_fwd_wg_kernel<T, true, true, FusedH128Params<PerTokenScale>>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(
        kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(p.s_q, 1, 1), T::NUM_THREADS, bytes, p.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<bool EnableNorm>
void run_fused_norm_rope_attn_rope_cast_prefill_h128(const Params& p) {
    if (p.per_token_scale) run_fused_norm_rope_attn_rope_cast_prefill_h128_impl<EnableNorm, true>(p);
    else run_fused_norm_rope_attn_rope_cast_prefill_h128_impl<EnableNorm, false>(p);
}

namespace small_prefill {

using BaseTraits = Flash_fwd_kernel_traits<
    512, 16, 64, 16, false, false, cutlass::bfloat16_t, 512,
    true, false, 1, 1, 64, 2, 4, false, false,
    Flash_kernel_traits<false, cutlass::bfloat16_t>>;

using H8BaseTraits = Flash_fwd_kernel_traits<
    512, 16, 32, 8, false, false, cutlass::bfloat16_t, 512,
    true, false, 1, 1, 32, 2, 2, false, false,
    Flash_kernel_traits<false, cutlass::bfloat16_t>>;

template<typename KT>
constexpr bool vector_q_layout() {
    typename KT::SmemLayoutAtomQ atom{};
    for (int row = 0; row < 8; ++row) {
        for (int dim = 0; dim < 64; dim += 8) {
            const int offset = atom(row, dim);
            if (offset % 8 != 0) return false;
            for (int i = 1; i < 8; ++i) {
                if (int(atom(row, dim + i)) != offset + i) return false;
            }
        }
    }
    return true;
}

template<typename KT>
constexpr bool unique_stat_rows() {
    int count[KT::kBlockM]{};
    constexpr int atom_m = KT::USE_MMA_M8 ? 8 : 16;
    constexpr int atom_k_m = KT::USE_MMA_M8 ? 1 : 2;
    constexpr int rows_per_lane = atom_k_m * KT::kBlockM / (atom_m * KT::AtomLayoutQ);
    for (int warp = 0; warp < KT::AtomLayoutQ; ++warp) {
        for (int lane = 0; lane < 32; lane += 4) {
            for (int mi = 0; mi < rows_per_lane; ++mi) {
                const int row = warp * atom_m + lane / 4
                    + (mi / atom_k_m) * (atom_m * KT::AtomLayoutQ)
                    + (mi % atom_k_m) * 8;
                if (row >= KT::kBlockM) return false;
                ++count[row];
            }
        }
    }
    for (int row = 0; row < KT::kBlockM; ++row) {
        if (count[row] != 1) return false;
    }
    return true;
}

__forceinline__ __device__ float2 apply_rope(float2 x, float c, float s) {
    return float2{fmaf(x.x, c, -x.y * s), fmaf(x.x, s, x.y * c)};
}

__forceinline__ __device__ uint16_t fp8_pair_from_float(float low, float high) {
    uint16_t result;
    asm volatile("ppu.cvt.rtte.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(result) : "f"(high), "f"(low));
    return result;
}

}

template<bool PerTokenScale>
struct FusedSmallPrefillParams : SparsePrefillParams {
    OutputParams<PerTokenScale> fused;
};

template<bool Norm, bool H8, bool PerTokenScale>
struct FusedSmallPrefillTraits
    : std::conditional_t<H8, small_prefill::H8BaseTraits, small_prefill::BaseTraits> {
    using Base = std::conditional_t<H8, small_prefill::H8BaseTraits, small_prefill::BaseTraits>;
    using PrefillParams = FusedSmallPrefillParams<PerTokenScale>;
    static constexpr bool kFusedSmallPrefill = true;
    static constexpr size_t kStdSmemEnd = Base::kSmemSize + 2 * Base::kBlockN * sizeof(bool);
    static constexpr size_t kScratchOffset = (kStdSmemEnd + 15) / 16 * 16;
    static constexpr int kQParts = 32;
    static constexpr size_t kScratchBytes = (H8 ? 8 * 8 + Base::kBlockM : Base::kBlockM * (kQParts + 1)) * sizeof(float);
    static constexpr size_t kStageBytes = Base::kBlockM * Base::kHeadDimV * sizeof(float);
    static constexpr size_t kSmallSmemBytes = std::max(
        kScratchOffset + (H8 && !Norm ? Base::kBlockM * sizeof(bool) : kScratchBytes), kStageBytes);

    static_assert(Base::kBlockM == 16 && Base::kBlockN == (H8 ? 32 : 64)
                  && Base::kNThreads == (H8 ? 256 : 512));
    static_assert(!Base::Is_Q_in_regs && !Base::Share_Q_K_smem && !Base::QKV_FP8);
    static_assert(Base::AtomLayoutQ == 1 && Base::AtomLayoutP == 1
                  && Base::kNWarps0 == (H8 ? 2 : 4));
    static_assert(Base::kHeadDim == 512 && Base::kHeadDimV == 512);
    static_assert(std::is_same_v<typename Base::SmemLayoutQ,
        decltype(cute::tile_to_shape(typename Base::SmemLayoutAtomQ{}, cute::Shape<cute::_16, cute::_512>{}))>);
    static_assert(small_prefill::vector_q_layout<Base>());
    static_assert(small_prefill::unique_stat_rows<Base>());
    static_assert(cute::cosize(typename Base::SmemLayoutO{}) * sizeof(float) == kStageBytes);
    static_assert(kStdSmemEnd == (H8 ? 83584 : 150720) && kScratchOffset == kStdSmemEnd);
    static_assert(kStageBytes == 32 * 1024 && kStageBytes <= Base::kSmemQSize + Base::kSmemKVSize / 2);
    static_assert(kScratchOffset >= kStageBytes && kSmallSmemBytes == (H8 ? (Norm ? 83904 : 83600) : 152832));
    static_assert(Base::kBlockM * kQParts % Base::kNThreads == 0);
    static_assert(kSmallSmemBytes <= 256 * 1024);

    template<typename KT, typename TensorQ>
    static __forceinline__ __device__ void load_fused_q(
        const PrefillParams& params, TensorQ& q, char* smem, int tidx,
        int m_block, int64_t s_q_idx) {
        using Element = typename KT::InputT;
        float* partials = reinterpret_cast<float*>(smem + kScratchOffset);
        float* norm = partials + (H8 ? 8 * 8 : KT::kBlockM * kQParts);
        #pragma unroll
        for (int item = tidx; item < (H8 ? 8 : KT::kBlockM) * kQParts; item += KT::kNThreads) {
            const int row = H8 ? item % 8 : item % KT::kBlockM;
            const int part = H8 ? item / 8 : item / KT::kBlockM;
            const int head = m_block * KT::kBlockM + row;
            const int dim_begin = part * (512 / kQParts);
            const auto* src = static_cast<const Element*>(params.q)
                + s_q_idx * params.stride_q_s_q;
            auto* dst = reinterpret_cast<Element*>(cute::raw_pointer_cast(q.data()));
            const bool aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
            float sum = 0.0f;
            #pragma unroll 1
            for (int i = 0; i < 512 / kQParts; i += 8) {
                const int dim = dim_begin + i;
                uint4 raw{};
                if (H8 || head < params.h_q) {
                    const int offset = (dim / 16) * ((H8 ? 8 : params.h_q) * 16) + head * 16 + dim % 16;
                    if (aligned) {
                        raw = __ldg(reinterpret_cast<const uint4*>(src + offset));
                    } else {
                        auto* elements = reinterpret_cast<Element*>(&raw);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) elements[j] = src[offset + j];
                    }
                }
                const auto* elements = reinterpret_cast<const Element*>(&raw);
                if constexpr (Norm) {
                    #pragma unroll
                    for (int j = 0; j < 8; j += 2) {
                        sum = fmaf(float(elements[j]), float(elements[j]), sum);
                        sum = fmaf(float(elements[j + 1]), float(elements[j + 1]), sum);
                    }
                }
                const int offset = cute::Swizzle<KT::kSwizzle, 3, 3>{}(q.layout()(row, dim));
                if (dim < 448 || (!H8 && head >= params.h_q)) {
                    *reinterpret_cast<uint4*>(dst + offset) = raw;
                } else {
                    const int position = __ldg(params.fused.token_positions + s_q_idx);
                    const float* rope = params.fused.cos_sin_cache + int64_t(position) * 64;
                    alignas(16) Element packed[8];
                    #pragma unroll
                    for (int j = 0; j < 8; j += 2) {
                        const int pair = (dim + j - 448) / 2;
                        const float2 x = small_prefill::apply_rope(float2{float(elements[j]), float(elements[j + 1])},
                            __ldg(rope + pair), __ldg(rope + 32 + pair));
                        packed[j] = Element(x.x);
                        packed[j + 1] = Element(x.y);
                    }
                    *reinterpret_cast<uint4*>(dst + offset) = *reinterpret_cast<const uint4*>(packed);
                }
                if constexpr (H8) {
                    const int dummy_offset = cute::Swizzle<KT::kSwizzle, 3, 3>{}(q.layout()(row + 8, dim));
                    *reinterpret_cast<uint4*>(dst + dummy_offset) = uint4{};
                }
            }
            if constexpr (Norm) {
                if constexpr (H8) {
                    // The warp holds four consecutive parts for each of eight heads.
                    const float pair = sum + __shfl_down_sync(0xffffffff, sum, 8);
                    const float group = pair + __shfl_down_sync(0xffffffff, pair, 16);
                    if (tidx % 32 < 8) partials[(tidx / 32) * 8 + row] = group;
                } else {
                    partials[row * kQParts + part] = sum;
                }
            }
        }
        if constexpr (Norm) {
            __syncthreads();
            if constexpr (H8) {
                if (tidx < 8) {
                    const float* groups = partials + tidx;
                    const float sqr = ((groups[0] + groups[8]) + (groups[16] + groups[24]))
                        + ((groups[32] + groups[40]) + (groups[48] + groups[56]));
                    norm[tidx] = rsqrtf(sqr / 512.0f + params.fused.rms_norm_eps);
                    norm[tidx + 8] = 1.0f;
                }
                // The first QK barrier publishes these factors before scale_scores.
            } else {
                if (tidx < KT::kBlockM) {
                    const float* row_partials = partials + tidx * kQParts;
                    const float s0 = (row_partials[0] + row_partials[1]) + (row_partials[2] + row_partials[3]);
                    const float s1 = (row_partials[4] + row_partials[5]) + (row_partials[6] + row_partials[7]);
                    const float s2 = (row_partials[8] + row_partials[9]) + (row_partials[10] + row_partials[11]);
                    const float s3 = (row_partials[12] + row_partials[13]) + (row_partials[14] + row_partials[15]);
                    const float s4 = (row_partials[16] + row_partials[17]) + (row_partials[18] + row_partials[19]);
                    const float s5 = (row_partials[20] + row_partials[21]) + (row_partials[22] + row_partials[23]);
                    const float s6 = (row_partials[24] + row_partials[25]) + (row_partials[26] + row_partials[27]);
                    const float s7 = (row_partials[28] + row_partials[29]) + (row_partials[30] + row_partials[31]);
                    const float sqr = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
                    norm[tidx] = m_block * KT::kBlockM + tidx < params.h_q
                        ? rsqrtf(sqr / 512.0f + params.fused.rms_norm_eps) : 1.0f;
                }
                __syncthreads();
            }
        }
        // The first QK tile synchronizes Q before reading it.
    }

    template<typename KT, typename AccS, typename ThrMma>
    static __forceinline__ __device__ void scale_scores(AccS& acc_s, ThrMma& thr_mma_s, char* smem) {
        if constexpr (Norm) {
            const float* norm = reinterpret_cast<const float*>(smem + kScratchOffset) + (H8 ? 8 * 8 : KT::kBlockM * kQParts);
            auto identity = cute::make_identity_tensor(cute::Shape<cute::Int<KT::kBlockM>, cute::Int<KT::kBlockN>>{});
            auto coords = thr_mma_s.partition_C(identity);
            static_assert(decltype(cute::size(coords))::value == decltype(cute::size(acc_s))::value);
            #pragma unroll
            for (int i = 0; i < cute::size(acc_s); ++i) acc_s(i) *= norm[cute::get<0>(coords(i))];
        }
    }

    template<typename KT, typename AccO, typename LSE, typename MLogits>
    static __forceinline__ __device__ void store_fused_output(
        const PrefillParams& params, AccO& acc_o, const LSE& lse, const MLogits& mlogits,
        char* smem, int tidx, int m_block, int64_t s_q_idx, int topk_length) {
        using namespace cute;
        const auto& p = params.fused;
        constexpr bool kRegisterOutput = H8 && !Norm && PerTokenScale
            && ACOMPUTE_VERSION >= 10500;
        bool* valid = reinterpret_cast<bool*>(smem + kScratchOffset);
        constexpr int atom_m = KT::USE_MMA_M8 ? 8 : 16;
        constexpr int atom_k_m = KT::USE_MMA_M8 ? 1 : 2;
        static_assert(decltype(size(lse))::value == atom_k_m * KT::kBlockM / (atom_m * KT::AtomLayoutQ));
        const int warp = tidx / 32;
        const int row_lse_base = warp % KT::AtomLayoutQ * atom_m + (tidx % 32) / 4;
        if (warp < KT::kNWarps0 && warp / KT::AtomLayoutQ == 0 && tidx % 4 == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = row_lse_base + (mi / atom_k_m) * (atom_m * KT::AtomLayoutQ)
                    + (mi % atom_k_m) * 8;
                const int head = m_block * KT::kBlockM + row;
                const bool live = topk_length > 0 && (__float_as_uint(lse(mi)) & 0x7f800000u) != 0x7f800000u;
                valid[row] = live;
                if constexpr (!kRegisterOutput) {
                    if (head < (H8 ? 8 : params.h_q)) {
                        const int64_t index = s_q_idx * params.h_q + head;
                        p.lse[index] = live ? lse(mi) * CUDART_LN2_F : INFINITY;
                        p.max_logits[index] = live ? mlogits(mi) * CUDART_LN2_F : -INFINITY;
                    }
                }
            }
        }
        // A nonempty H8 row drained its final K group in the attention loop.
        // The staging barrier below also publishes the valid-row flags.
        if (!H8 || topk_length == 0) {
            flash::cp_async_wait<0>();
            __syncthreads();
        }

        if constexpr (kRegisterOutput) {
            // Adjacent columns stay paired in the native M16 accumulator.
            // Keep the eight live rows in registers through scale reduction,
            // then store FP8 pairs directly without an output staging barrier.
            static_assert(KT::kBlockM == 16 && KT::kHeadDimV == 512
                          && KT::kNThreads == 256 && KT::AtomLayoutP == 1);
            typename KT::TiledMma mma;
            auto coords = mma.get_thread_slice(tidx).partition_C(
                make_identity_tensor(Shape<Int<KT::kBlockM>, Int<KT::kHeadDimV>>{}));
            static_assert(decltype(size(coords))::value == decltype(size(acc_o))::value);
            const int position = __ldg(p.token_positions + s_q_idx);
            const float* rope = p.cos_sin_cache + int64_t(position) * 64;
            // Publish the valid-row flags before consuming any accumulator.
            __syncthreads();
            float amax = 1e-4f;
            #pragma unroll
            for (int i = 0; i < size(acc_o); i += 2) {
                const int row = get<0>(coords(i));
                const int col = get<1>(coords(i));
                if (row < 8) {
                    float2 values = valid[row] ? float2{acc_o(i), acc_o(i + 1)}
                                              : float2{0.0f, 0.0f};
                    if (col >= 448) {
                        const int pair = (col - 448) / 2;
                        values = small_prefill::apply_rope(values,
                            __ldg(rope + pair), -__ldg(rope + 32 + pair));
                    }
                    acc_o(i) = values.x;
                    acc_o(i + 1) = values.y;
                    amax = fmaxf(amax, fmaxf(fabsf(values.x), fabsf(values.y)));
                }
            }
            amax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(amax)));
            auto* partials = reinterpret_cast<float*>(smem + 4096);
            if (tidx % 32 == 0) partials[tidx / 32] = amax;
            __syncthreads();
            float inverse = 0.0f;
            if (tidx % 32 == 0) {
                #pragma unroll
                for (int warp = 0; warp < KT::kNThreads / 32; ++warp)
                    amax = fmaxf(amax, partials[warp]);
                const float scale = amax / 448.0f;
                inverse = 1.0f / scale;
                if (tidx == 0)
                    reinterpret_cast<float*>(p.out_sf)[s_q_idx * p.stride_out_sf_token] = scale;
            }
            inverse = __shfl_sync(0xffffffff, inverse, 0);
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
            using Mma = typename KT::TiledMma;
            using NativeCLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
            static_assert(std::is_same_v<typename Mma::LayoutC_TV, NativeCLayout>);
            using NativeMma = TiledMMA<typename Mma::Atom,
                Layout<Shape<_1, _8, _1>>, Tile<_16, _128, _16>>;
            static_assert(std::is_same_v<Mma, NativeMma>);
#endif
            static_assert(decltype(size(acc_o))::value == 32);
            #pragma unroll
            for (int i = 0; i < size(acc_o); i += 8) {
                const uint32_t low = small_prefill::fp8_pair_from_float(
                    acc_o(i) * inverse, acc_o(i + 1) * inverse);
                const uint32_t high = small_prefill::fp8_pair_from_float(
                    acc_o(i + 4) * inverse, acc_o(i + 5) * inverse);
                const uint32_t low_word = low | (__shfl_xor_sync(0xffffffff, low, 1) << 16);
                const uint32_t high_word = high | (__shfl_xor_sync(0xffffffff, high, 1) << 16);
                const uint4 packed{low_word, __shfl_xor_sync(0xffffffff, low_word, 2),
                    high_word, __shfl_xor_sync(0xffffffff, high_word, 2)};
                if (tidx % 4 == 0) {
                    const int row = get<0>(coords(i));
                    const int col = get<1>(coords(i));
                    const int offset = (col / 32) * (8 * 32) + row * 32 + col % 32;
                    *reinterpret_cast<uint4*>(p.out_fp8 + s_q_idx * 4096 + offset) = packed;
                }
            }
            // Publish independent global statistics after every CTA barrier.
            if (warp < KT::kNWarps0 && warp / KT::AtomLayoutQ == 0 && tidx % 4 == 0) {
                #pragma unroll
                for (int mi = 0; mi < size(lse); ++mi) {
                    const int row = row_lse_base + (mi / atom_k_m) * (atom_m * KT::AtomLayoutQ)
                        + (mi % atom_k_m) * 8;
                    if (row < 8) {
                        const int64_t index = s_q_idx * params.h_q + row;
                        p.lse[index] = valid[row] ? lse(mi) * CUDART_LN2_F : INFINITY;
                        p.max_logits[index] = valid[row] ? mlogits(mi) * CUDART_LN2_F : -INFINITY;
                    }
                }
            }
            return;
        }

        Tensor staged = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem)), typename KT::SmemLayoutO{});
        typename KT::TiledMma mma;
        auto copy_c = make_tiled_copy_C(typename KT::SmemCopyAtomOaccum{}, mma);
        auto copy_thr = copy_c.get_thread_slice(tidx);
        Tensor src = copy_thr.retile_S(acc_o);
        Tensor dst = copy_thr.partition_D(staged);
        cute::copy(copy_c, src, dst);
        __syncthreads();

        if constexpr (PerTokenScale) {
            splitkv::store_per_token_output<KT, H8, Norm>(
                p, s_q_idx, m_block * KT::kBlockM, smem, staged, acc_o, valid);
            return;
        }
        const int position = __ldg(p.token_positions + s_q_idx);
        const float* rope = p.cos_sin_cache + int64_t(position) * 64;
        if constexpr (H8) {
            // Each four-lane group owns one head's 32-channel quantization block.
            const int lane = tidx % 4;
            const int row = (tidx / 4) % 8;
            #pragma unroll
            for (int tile = tidx / 32; tile < 16; tile += KT::kNThreads / 32) {
                const int col = tile * 32 + lane * 8;
                alignas(16) float values[8]{};
                if (valid[row]) {
                    *reinterpret_cast<float4*>(values) = *reinterpret_cast<const float4*>(&staged(row, col));
                    *reinterpret_cast<float4*>(values + 4) = *reinterpret_cast<const float4*>(&staged(row, col + 4));
                }
                if (tile >= 14) {
                    #pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        const int pair = (col - 448 + i) / 2;
                        const float2 rotated = small_prefill::apply_rope(float2{values[i], values[i + 1]},
                            __ldg(rope + pair), -__ldg(rope + 32 + pair));
                        values[i] = rotated.x;
                        values[i + 1] = rotated.y;
                    }
                }
                float amax = 1e-4f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(values[i]));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
                const float scale = amax / 448.0f;
                const int exponent = int((__float_as_uint(scale) - 1u) >> 23) + 1 - 127;
                const float scale_inv = __uint_as_float(uint32_t(127 - exponent) << 23);
                uint2 packed;
                #pragma unroll
                for (int i = 0; i < 8; i += 4) {
                    const uint32_t low = small_prefill::fp8_pair_from_float(values[i] * scale_inv, values[i + 1] * scale_inv);
                    const uint32_t high = small_prefill::fp8_pair_from_float(values[i + 2] * scale_inv, values[i + 3] * scale_inv);
                    (i == 0 ? packed.x : packed.y) = low | (high << 16);
                }
                const int64_t out_offset = s_q_idx * (8 * 512) + tile * (8 * 32) + row * 32 + lane * 8;
                *reinterpret_cast<uint2*>(p.out_fp8 + out_offset) = packed;
                if (lane == 0) {
                    const int sf_index = tile * 8 + row;
                    auto* sf_word = p.out_sf + s_q_idx + int64_t(sf_index / 4) * p.stride_out_sf_word;
                    reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
                }
            }
            return;
        }

        #pragma unroll 1
        for (int item = tidx; item < KT::kBlockM * 16; item += KT::kNThreads) {
            const int row = item % KT::kBlockM;
            const int tile = item / KT::kBlockM;
            const int head = m_block * KT::kBlockM + row;
            if (head >= params.h_q) continue;
            alignas(16) float values[32];
            #pragma unroll
            for (int i = 0; i < 32; i += 4) {
                const float4 chunk = valid[row] ? *reinterpret_cast<const float4*>(&staged(row, tile * 32 + i))
                    : float4{0.0f, 0.0f, 0.0f, 0.0f};
                *reinterpret_cast<float4*>(values + i) = chunk;
            }
            if (tile >= 14) {
                #pragma unroll
                for (int i = 0; i < 32; i += 2) {
                    const int pair = (tile - 14) * 16 + i / 2;
                    const float2 rotated = small_prefill::apply_rope(float2{values[i], values[i + 1]},
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
            auto* pairs = reinterpret_cast<uint16_t*>(packed);
            #pragma unroll
            for (int i = 0; i < 32; i += 2) {
                pairs[i / 2] = small_prefill::fp8_pair_from_float(values[i] * scale_inv, values[i + 1] * scale_inv);
            }
            const int group = head / 8;
            const int in_group = head & 7;
            const int64_t out_offset = s_q_idx * params.h_q * 512
                + group * (8 * 512) + tile * (8 * 32) + in_group * 32;
            auto* out = reinterpret_cast<uint4*>(p.out_fp8 + out_offset);
            const auto* words = reinterpret_cast<const uint4*>(packed);
            out[0] = words[0];
            out[1] = words[1];
            const int sf_index = tile * 8 + in_group;
            auto* sf_word = p.out_sf + s_q_idx + int64_t(group) * p.stride_out_sf_group
                + int64_t(sf_index / 4) * p.stride_out_sf_word;
            reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
        }
    }
};

template<bool Norm, bool PerTokenScale>
void run_fused_norm_rope_attn_rope_cast_prefill_small_impl(const Params& p) {
    TORCH_CHECK(p.mode == FwdMode::Prefill && p.arch == 89);
    if (p.h_q == 8) {
        TORCH_CHECK(p.topk > 0 && p.topk % small_prefill::BaseTraits::kBlockN == 0,
                    "H8 fused prefill requires a positive top-k divisible by 64");
        TORCH_CHECK((reinterpret_cast<uintptr_t>(p.kv) & 15) == 0 && p.stride_kv_s_kv % 8 == 0,
                    "H8 fused prefill requires 16-byte-aligned KV rows");
    } else {
        TORCH_CHECK((p.h_q == 64 || p.h_q == 128) && p.topk == 128);
    }
    TORCH_CHECK(p.enable_q_norm == Norm);
    TORCH_CHECK(p.wv_group_size == 8 && p.n_wv_group == p.h_q / 8);
    if (p.s_q == 0) return;

    at::Tensor clamped_length;
    const int32_t* topk_length = p.topk_length;
    if (topk_length) {
        const auto options = torch::TensorOptions().dtype(torch::kInt32)
            .device(torch::Device(torch::kCUDA, get_current_device()));
        clamped_length = torch::from_blob(const_cast<int32_t*>(topk_length), {p.s_q}, options)
            .clamp(0, p.topk);
        topk_length = clamped_length.data_ptr<int32_t>();
    }

    FusedSmallPrefillParams<PerTokenScale> params{};
    params.s_q = p.s_q;
    params.s_kv = p.s_kv;
    params.h_q = p.h_q;
    params.h_kv = 1;
    params.d_qk = params.d_v = 512;
    params.topk = p.topk;
    params.sm_scale = p.sm_scale;
    params.sm_scale_div_log2 = p.sm_scale_log2;
    params.q = const_cast<void*>(p.q);
    params.kv = const_cast<void*>(p.kv);
    params.indices = const_cast<int32_t*>(p.indices);
    params.attn_sink = const_cast<float*>(p.attn_sink);
    params.topk_length = const_cast<int32_t*>(topk_length);
    params.stride_q_s_q = int64_stride_to_int(p.stride_q_s_q);
    params.stride_q_h_q = 512;
    params.stride_kv_s_kv = int64_stride_to_int(p.stride_kv_s_kv);
    params.stride_kv_h_kv = 512;
    params.stride_indices_s_q = int64_stride_to_int(p.stride_indices_s_q);
    params.stride_indices_h_kv = p.topk;
    params.out = p.out_fp8;
    params.max_logits = p.max_logits;
    params.lse = p.lse;
    params.stream = p.stream;
    params.fused = OutputParams<PerTokenScale>{p};
    params.fused.topk_length = topk_length;

    const auto launch = [&](auto h8) {
        using T = FusedSmallPrefillTraits<Norm, decltype(h8)::value, PerTokenScale>;
        constexpr size_t bytes = T::kSmallSmemBytes;
        auto kernel = &flash::flash_sparse_prefill_fwd_kernel<T, true, false, true, false, FusedSmallPrefillParams<PerTokenScale>>;
        C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
        kernel<<<dim3(p.s_q * cute::ceil_div(p.h_q, T::kBlockM), 1, 1), T::kNThreads, bytes, p.stream>>>(params);
        CHECK_CUDA_KERNEL_LAUNCH();
    };
    if (p.h_q == 8) launch(std::true_type{});
    else launch(std::false_type{});
}

template<bool Norm>
void run_fused_norm_rope_attn_rope_cast_prefill_small(const Params& p) {
    if (p.per_token_scale) run_fused_norm_rope_attn_rope_cast_prefill_small_impl<Norm, true>(p);
    else run_fused_norm_rope_attn_rope_cast_prefill_small_impl<Norm, false>(p);
}

template void run_fused_norm_rope_attn_rope_cast_prefill_small<false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Prefill, 128, false, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_prefill_h128<false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_prefill_small<true>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Prefill, 128, true, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_prefill_h128<true>(const Params&);

} // namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn
