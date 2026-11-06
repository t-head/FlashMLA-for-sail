#pragma once

#include <type_traits>
#include "api/common.h"
#include "kernel.cuh"
#include "prefill/sparse/sparse_prefill_wg.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

struct FusedH128Params : SparsePrefillParams {
    Params fused;
};

template<bool Norm>
struct FusedH128PrefillTraits : DSA_Traits<cutlass::bfloat16_t, 512, 89> {
    using Base = DSA_Traits<cutlass::bfloat16_t, 512, 89>;
    using FusedConfig = Config<FwdMode::Prefill, 128, Norm, false, 89>;
    using KernelParams = FusedH128Params;
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
        const FusedH128Params& params, int token) {
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
        const FusedH128Params& params, int m_block_idx, TensorQ& q,
        int tidx, int warp_idx, __mbarrier_t* barrier) {
        const TileParams tile = make_tile(params, m_block_idx);
        FusedConfig::template load_q<FusedH128PrefillTraits<Norm>>(
            tile, m_block_idx, q, tidx, warp_idx, barrier);
    }

    template<typename TensorQ>
    static __forceinline__ __device__ int issue_q(
        const FusedH128Params& params, int m_block_idx, TensorQ& q,
        int tidx, int warp_idx, __mbarrier_t* barrier) {
        const TileParams tile = make_tile(params, m_block_idx);
        return FusedConfig::template issue_q<FusedH128PrefillTraits<Norm>>(
            tile, m_block_idx, q, tidx, warp_idx, barrier);
    }

    template<typename TensorQ>
    static __forceinline__ __device__ void finish_q(
        const FusedH128Params& params, int m_block_idx, TensorQ& q, int tidx, int position) {
        const TileParams tile = make_tile(params, m_block_idx);
        FusedConfig::template finish_q<FusedH128PrefillTraits<Norm>>(tile, q, tidx, position);
    }

    template<typename TensorO>
    static __forceinline__ __device__ void store_output(
        const FusedH128Params& params, TensorO& output, SharedMemoryPlan& plan,
        int warpgroup_idx, int idx_in_warpgroup) {
        const int m_block_idx = blockIdx.x;
        const TileParams tile = make_tile(params, m_block_idx);
        FusedConfig::template store_output<FusedH128PrefillTraits<Norm>>(
            tile, output, plan, warpgroup_idx, idx_in_warpgroup);
    }
};

template<bool EnableNorm>
void run_fused_norm_rope_attn_rope_cast_prefill_h128(const Params& p) {
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

    FusedH128Params params{};
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
    params.fused = p;
    params.fused.topk_length = topk_length;

    using T = FusedH128PrefillTraits<EnableNorm>;
    constexpr size_t bytes = std::max(sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));
    auto kernel = &::flash_sparse_prefill_fwd_wg_kernel<T, true, false, FusedH128Params>;
    // Match the ordinary D512 launcher. Each CTA still validates its own row
    // before selecting the unmasked body.
    if (p.topk >= 4 * (2 * T::kBlockN))
        kernel = &::flash_sparse_prefill_fwd_wg_kernel<T, true, true, FusedH128Params>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(
        kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(p.s_q, 1, 1), T::NUM_THREADS, bytes, p.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
