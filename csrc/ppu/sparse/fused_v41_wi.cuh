/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cmath>
#include <cutlass/float8.h>

#include "fused_v41.h"
#include "prefill/sparse/traits.h"
#include "prefill/sparse/sparse_prefill_wg.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

struct TileParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;
    void *q, *kv;
    int *indices;
    float *attn_sink;
    int *topk_length;
    int64_t stride_q_s_q, stride_q_h_q;
    int64_t stride_kv_s_kv, stride_kv_h_kv;
    int64_t stride_indices_s_q, stride_indices_h_kv;
    void *out, *max_logits, *lse;
    hggcStream_t stream;
    int quant_page_size;
    int64_t quant_block_stride;

    void *extra_kv;
    int *extra_indices;
    int extra_s_kv, extra_page_size;
    int64_t extra_block_stride;
    int main_length, extra_length, main_storage_topk, extra_storage_topk;
    int main_padded_length, start_block, end_block;
    bool partial_output;
    const Params* fused;
    int token_idx;
    int real_heads;
};

struct CacheSupplier {
    template<int S, int E, typename T, bool FrontDrain, typename RawK>
    static __forceinline__ __device__ void load(
        const TileParams& p, RawK& raw_k, int tidx, int token_idx,
        bool valid, int block_idx, bool async_preissued);
};

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch_, bool PerTokenScale = false>
struct Config : ::DSA_Traits<cutlass::bfloat16_t, 512, Arch_> {
    using Base = ::DSA_Traits<cutlass::bfloat16_t, 512, Arch_>;
    static constexpr int Arch = Arch_;
    static constexpr bool kQuantCache = true;
    using Params = TileParams;
    using KernelParams = OutputParams<PerTokenScale>;
    static constexpr bool per_token_scale = PerTokenScale;
    using CacheAdapter = CacheSupplier;
    static constexpr int Arch_value = ACOMPUTE_VERSION == 10000 ? 80 : Arch;
    static constexpr FwdMode kMode = Mode;
    static constexpr int kRealHeads = Heads;
    static constexpr bool kEnableNorm = EnableNorm;
    static constexpr bool kExtraFP4 = ExtraFP4;
    static constexpr bool kDecode = true;
    static constexpr bool kFusedNativeDecode = false;
    static constexpr bool kTokenInterleavedCache = false;
    static constexpr bool kVectorDecodeKV = false;
    static constexpr bool kVectorOutputLoad = Arch == 89;
    static constexpr bool kFusedQLoad = true;
    static constexpr bool kFusedEpilogue = true;
    static constexpr bool kRegisterPreissue = false;
    static constexpr bool kUsePairedCubes = Arch == 89 && ACOMPUTE_VERSION >= 10500;
    static constexpr bool kCacheLastQ = !kUsePairedCubes;
    static constexpr int kQRowsPerLoad = kUsePairedCubes ? 8 : Base::kBlockM;
    static constexpr bool kTaskAllValidGuard = false;
    static constexpr bool kTaskAllValidVotes = false;
    static constexpr bool kLengthGuardAllowed = false;

    struct SharedMemoryPlan : Base::SharedMemoryPlan {
        cute::array_aligned<float, Base::NUM_THREADS> q_sqr_partials;
        cute::array_aligned<float, Base::kBlockM> q_norm;
    };
    static_assert(Heads == 64 || Heads == 128);
    static_assert(Arch == 80 || Arch == 89);
    static_assert(Mode == FwdMode::Decode || !ExtraFP4);
    static_assert(sizeof(SharedMemoryPlan) < 256 * 1024);
    static_assert(sizeof(typename Base::SharedMemoryOutPut) == 256 * 1024);

    static __forceinline__ __device__ float score_scale(float scale, int row) {
        if constexpr(EnableNorm) {
            extern __shared__ char wksp_buf[];
            const auto& plan = *reinterpret_cast<const SharedMemoryPlan*>(wksp_buf);
            return scale * plan.q_norm[row];
        } else {
            return scale;
        }
    }

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ void load_q(
        const TileParams& p, int m_block_idx, TensorQ& q,
        int tidx, int warp_idx, __mbarrier_t* barrier);

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ int issue_q(
        const TileParams& p, int m_block_idx, TensorQ& q,
        int tidx, int warp_idx, __mbarrier_t* barrier);

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ void finish_q(
        const TileParams& p, TensorQ& q, int tidx, int position);

    template<typename T, typename TensorO>
    static __forceinline__ __device__ void store_output(
        const TileParams& p, TensorO& output, typename T::SharedMemoryPlan& plan,
        int warpgroup_idx, int idx_in_warpgroup);
};

template<int Heads, bool EnableNorm, bool ExtraFP4, int Arch, bool PerTokenScale = false>
struct VectorDecodeConfig : Config<FwdMode::Decode, Heads, EnableNorm, ExtraFP4, Arch, PerTokenScale> {
    static constexpr bool kVectorDecodeKV = true;
};

template<int Heads, bool EnableNorm, int Arch, bool PerTokenScale = false>
struct NativePrefillConfig : Config<FwdMode::Prefill, Heads, EnableNorm, false, Arch, PerTokenScale> {
    static constexpr bool kDecode = Arch != 89;
    static constexpr bool kQuantCache = kDecode;
};

}

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn::fused_wi {

using namespace cute;
using cutlass::arch::NamedBarrier;
namespace prefill = ::flashmla::sparse_prefill;

template<typename T>
__forceinline__ __device__ int token_index(
    const TileParams& params, const int* prefill_indices,
    int block_idx, int real_end_block_idx, int tidx) {
    if constexpr (T::kDecode) {
        const int column = tidx / 8;
        const int absolute = block_idx * T::kBlockN + column;
        const bool extra = absolute >= params.main_padded_length;
        const int position = extra ? absolute - params.main_padded_length : absolute;
        const int storage = extra ? params.extra_storage_topk : params.main_storage_topk;
        const int length = extra ? params.extra_length : params.main_length;
        if (position < 0 || position >= storage || position >= length) return -1;
        const int* indices = extra ? params.extra_indices : params.indices;
        return __ldg(indices + position);
    } else {
        const int position = block_idx * T::kBlockN + tidx / 8;
        return block_idx < real_end_block_idx && position < params.main_storage_topk
            ? __ldg(prefill_indices + block_idx * T::kBlockN) : -1;
    }
}

template<typename T, bool WriteVI, typename VI>
__forceinline__ __device__ void prepare_k(
    const TileParams& params, int token, int block_idx, int seqlen_k,
    int tidx, VI& valid_indices, int vi_buf, bool& valid) {
    if constexpr (T::kDecode) {
        const int absolute = block_idx * T::kBlockN + tidx / 8;
        const bool extra = absolute >= params.main_padded_length;
        const int position = extra ? absolute - params.main_padded_length : absolute;
        const int length = extra ? params.extra_length : params.main_length;
        const int storage = extra ? params.extra_storage_topk : params.main_storage_topk;
        const int capacity = extra ? params.extra_s_kv : params.s_kv;
        valid = token >= 0 && token < capacity && position >= 0 && position < length && position < storage;
        if constexpr (WriteVI) {
            if (tidx % 8 == 0) valid_indices(vi_buf, tidx / 8) = valid;
        }
    } else {
        valid = token >= 0 && token < params.s_kv;
        if constexpr (WriteVI) {
            if (tidx % 8 == 0)
                valid_indices(vi_buf, tidx / 8) = valid && (block_idx * T::kBlockN + tidx / 8 < seqlen_k);
        }
    }
}

template<int S, int E, typename T, bool Prefetch = true, bool FrontDrain = true, typename K>
__forceinline__ __device__ void issue_k(
    K& raw_k, __mbarrier_t* barrier, bool valid, const int* indices,
    int prefetch_block, int real_end_block_idx, int& next_token, int tidx,
    const TileParams& params, int token, int load_block) {
    T::CacheAdapter::template load<S, E, T, FrontDrain>(
        params, raw_k, tidx, token, valid, load_block, false);
    cutlass::arch::cpasync_barrier_arrive_noinc(barrier);
    if constexpr (Prefetch) {
        if (prefetch_block < real_end_block_idx) {
            next_token = token_index<T>(params, indices, prefetch_block, real_end_block_idx, tidx);
        } else {
            next_token = -1;
        }
    }
}

template<typename T, int WG>
__forceinline__ __device__ void prolog_sync() {
    // Match CacheSupplier's physical IDs without CUTLASS's user-ID offset.
    NamedBarrier::arrive_and_wait(T::NUM_THREADS / 2,
        static_cast<cutlass::arch::ReservedNamedBarriers>(WG + 1));
}

template<typename T, bool Last, typename Q, typename K, typename KV, typename P,
         typename Stats, typename RQ, typename RP, typename RO, typename VI>
__forceinline__ __device__ void wg0_step(
    Q& sQ, K sK, KV& cur_sK0, KV& cur_sK1, KV& nxt_sK0,
    P& sP0, P& sP1, Stats& sM, Stats& sScale0, Stats& sScale1,
    RQ& rQ8, RP& rP0, RO& rO0, float rL[2],
    __mbarrier_t* barriers_K0, __mbarrier_t* barriers_K1, bool& phase,
    const TileParams& params, const int* indices, int seqlen_k, int block_idx,
    int end_block_idx, int real_end_block_idx, int tidx, int warp,
    int& kv_idx, int& next0, int& next1, bool& valid0, bool& valid1, VI& vi) {
    const int start_token_idx = block_idx * T::kBlockN;
    auto sV0L = prefill::get_half_V<T, 0>(cur_sK0);
    auto sV1L = prefill::get_half_V<T, 0>(cur_sK1);
    auto nxt_sK1 = cur_sK0;

    auto rPb = prefill::wg0_bunch_0<T, Last>(rP0, rO0, sScale0, sM, rL,
        params.sm_scale_div_log2, start_token_idx, tidx, vi, (block_idx / 2) % 2);
    NamedBarrier::arrive(T::NUM_THREADS, prefill::NamedBarriers::sScale0Ready);
    if constexpr (!Last) {
        issue_k<0, 4, T>(nxt_sK0, &barriers_K0[0], valid0, indices,
            block_idx + 4, real_end_block_idx, next0, tidx, params, next0, block_idx + 2);
    }
    prefill::wg0_scale0_rO0<T>(rO0, sScale0, tidx);
    prefill::dsa_warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, tidx, warp);

    NamedBarrier::arrive_and_wait(T::NUM_THREADS, prefill::NamedBarriers::sScale1Ready);
    if (!Last && __builtin_expect(block_idx + 3 < end_block_idx, true)) {
        issue_k<0, 4, T, true, false>(nxt_sK1, &barriers_K1[0], valid1, indices,
            block_idx + 5, real_end_block_idx, next1, tidx, params, next1, block_idx + 3);
    }
    prefill::wg0_scale_rP0<T>(sScale1, rP0, rPb, tidx);
    prefill::save_rP0_to_sP<T>(rPb, sP0, tidx);
    NamedBarrier::arrive(T::NUM_THREADS, prefill::NamedBarriers::sP0Ready);
    prefill::wg0_rescale_rO0<T>(rO0, sScale1, rL, tidx);
    prefill::dsa_warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, tidx, warp);

    if constexpr (!Last) {
        cute::clear(rP0);
        prefill::dsa_warpgroup_cooperative_qkt_gemm<T, 0>(
            sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, phase, tidx, warp);
        if (block_idx + 4 < end_block_idx) {
            prepare_k<T, true>(params, next0, block_idx + 4, seqlen_k, tidx,
                vi, (block_idx / 2) % 2, valid0);
        }
        if (block_idx + 5 < end_block_idx) {
            prepare_k<T, false>(params, next1, block_idx + 5, seqlen_k, tidx, vi, 0, valid1);
        }
        prefill::dsa_warpgroup_cooperative_qkt_gemm<T, 2>(
            sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, phase, tidx, warp);
    }
    kv_idx = (kv_idx + 2) % 3;
    cur_sK0 = sK(_, _, kv_idx);
    cur_sK1 = sK(_, _, (kv_idx + 1) % 3);
    nxt_sK0 = sK(_, _, (kv_idx + 2) % 3);
}

template<typename T, bool Last, typename Q, typename K, typename KV, typename P,
         typename Stats, typename RQ, typename RP, typename RO, typename VI>
__forceinline__ __device__ void wg1_step(
    Q& sQ, K sK, KV& cur_sK1, KV& cur_sK0, KV& nxt_sK1,
    P& sP0, P& sP1, Stats& sM, Stats& sScale0, Stats& sScale1,
    RQ& rQ8, RP& rP1, RO& rO1, float rL[2],
    __mbarrier_t* barriers_K0, __mbarrier_t* barriers_K1, bool& phase,
    const TileParams& params, const int* indices, int seqlen_k, int block_idx,
    int end_block_idx, int real_end_block_idx, int tidx, int warp,
    int& kv_idx, int& next0, int& next1, bool& valid0, bool& valid1, VI& vi) {
    const int start_token_idx = block_idx * T::kBlockN;
    auto nxt_sK0 = cur_sK1;
    auto sV0R = prefill::get_half_V<T, 1>(cur_sK0);
    auto sV1R = prefill::get_half_V<T, 1>(cur_sK1);

    // Native-prefill supply precedes curmax; other scopes finish after Scale0.
    if constexpr (!T::kQuantCache && !Last) {
        issue_k<4, T::NUM_TILES, T>(nxt_sK1, &barriers_K1[1], valid0, indices,
            block_idx + 5, real_end_block_idx, next1, tidx, params, next1, block_idx + 3);
    }
    float r_cur_max[2];
    prefill::dsa_wg1_bunch_0_pre<T, false, Last>(r_cur_max, rP1, params.sm_scale_div_log2,
        start_token_idx + T::kBlockN, tidx, vi, 2 + (block_idx / 2) % 2);
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, prefill::NamedBarriers::sScale0Ready);
    if constexpr (T::kQuantCache && !Last) {
        issue_k<4, T::NUM_TILES, T, true, false>(nxt_sK1, &barriers_K1[1], valid0, indices,
            block_idx + 5, real_end_block_idx, next1, tidx, params, next1, block_idx + 3);
    }
    auto rP1b = prefill::wg1_bunch_0<T, false, Last>(sScale1, rO1, sM, rL, sScale0, rP1,
        params.sm_scale_div_log2, start_token_idx + T::kBlockN, tidx,
        vi, 2 + (block_idx / 2) % 2, r_cur_max);
    prefill::save_rP1_to_sP<T>(rP1b, sP1, tidx);
    NamedBarrier::arrive(T::NUM_THREADS, prefill::NamedBarriers::sScale1Ready);
    prefill::wg1_scale0_rO1<T>(rO1, sScale0, sScale1, tidx);
    prefill::dsa_warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV1R, rO1, tidx, warp);

    NamedBarrier::arrive_and_wait(T::NUM_THREADS, prefill::NamedBarriers::sP0Ready);
    if constexpr (!Last) {
        issue_k<4, T::NUM_TILES, T, true, false>(nxt_sK0, &barriers_K0[1], valid1, indices,
            block_idx + 4, real_end_block_idx, next0, tidx, params, next0, block_idx + 2);
        if (block_idx + 5 < end_block_idx) {
            prepare_k<T, true>(params, next1, block_idx + 5, seqlen_k, tidx,
                vi, 2 + (block_idx / 2) % 2, valid0);
        }
        if (block_idx + 4 < end_block_idx) {
            prepare_k<T, false>(params, next0, block_idx + 4, seqlen_k, tidx, vi, 0, valid1);
        }
    }
    prefill::dsa_warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, tidx, warp);
    if constexpr (!Last) {
        cute::clear(rP1);
        prefill::dsa_warpgroup_cooperative_qkt_gemm<T, 1>(
            sQ, nxt_sK0, nxt_sK1, rP1, rQ8, barriers_K1, phase, tidx, warp);
    }
    kv_idx = (kv_idx + 2) % 3;
    cur_sK1 = sK(_, _, kv_idx);
    cur_sK0 = sK(_, _, (kv_idx + 1) % 3);
    nxt_sK1 = sK(_, _, (kv_idx + 2) % 3);
}

template<typename T>
__forceinline__ __device__ void tile_body(const TileParams& params, int m_block_idx) {
    static_assert(!T::kRegisterPreissue && T::kFusedQLoad && T::kFusedEpilogue);
    static_assert(T::kBlockM == 128 && T::kBlockN == 32 && T::kHeadDim == 512);
    static_assert(T::NUM_THREADS == 512 && DSA_SIM_AIU);
    const int warpgroup_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 256);
    const int idx_in_warpgroup = threadIdx.x % 256;
    const int warp_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 32);
    const int tidx = threadIdx.x;
    using InputT = typename T::InputT;
    typename T::TiledMma tiled_mma;
    extern __shared__ char wksp_buf[];
    auto& plan = *reinterpret_cast<typename T::SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_sQ.data()), typename T::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_sK.data()), typename T::SmemLayoutK{});
    auto p_base = [&]() {
        if constexpr (T::kUsePairedCubes) {
            return make_smem_ptr(plan.smem_sP.data());
        } else {
            return flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::PAGE_BLOCK_SIZE>>{})
                (_, _, _0{}, Int<T::NUM_TILES - 1>{}).data();
        }
    };
    Tensor sP0 = make_tensor(p_base(), typename T::SmemLayoutP0{});
    Tensor sP1 = make_tensor(sP0.data() + sP0.size(), typename T::SmemLayoutP0{});
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<2 * T::BLOCK_SIZE_M>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor vi = make_tensor(make_smem_ptr(plan.smem_valid_indices.data()),
        Shape<_4, Int<T::kBlockN>>{}, Stride<Int<T::kBlockN>, _1>{});
    const int q_idx = m_block_idx / (params.h_q / T::kBlockM);
    const int* indices = params.indices + int64_t(q_idx) * params.stride_indices_s_q + idx_in_warpgroup / 8;
    int next0 = -1, next1 = -1;
    bool valid0 = false, valid1 = false;
    constexpr int kBlockN = T::kBlockN;
    auto* barrier_Q = &plan.barrier_Q;
    auto* barriers_K0 = plan.barriers_K0;
    auto* barriers_K1 = plan.barriers_K1;
    if (threadIdx.x == 0) {
        __mbarrier_init(barrier_Q, 32 * (T::kBlockM / T::kQRowsPerLoad));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            __mbarrier_init(&barriers_K0[i], 256);
            __mbarrier_init(&barriers_K1[i], 256);
        }
    }
    __syncthreads();

    int seqlen_k = params.topk_length ? __ldg(params.topk_length + q_idx) : params.topk;
    int start_block_idx = 0;
    int real_end_block_idx = cute::ceil_div(seqlen_k, kBlockN);
    int end_block_idx = cute::ceil_div(cute::round_up(seqlen_k, 2 * kBlockN), kBlockN);
    if constexpr (T::kDecode) {
        seqlen_k = params.topk;
        start_block_idx = params.start_block;
        real_end_block_idx = cute::ceil_div(seqlen_k, kBlockN);
        end_block_idx = params.end_block;
    }
    if (warpgroup_idx == 0) {
        next0 = token_index<T>(params, indices, start_block_idx, real_end_block_idx, idx_in_warpgroup);
    } else {
        next1 = token_index<T>(params, indices, start_block_idx + 1, real_end_block_idx, idx_in_warpgroup);
    }
    bool cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;
    const int fused_q_position = T::template issue_q<T>(params, m_block_idx, sQ, tidx, warp_idx, barrier_Q);
    Tensor cur_sK0 = sK(_, _, 0);
    Tensor cur_sK1 = sK(_, _, 1);
    Tensor nxt_sK0 = sK(_, _, 2);

    if (warpgroup_idx == 0 && seqlen_k != 0) {
        bool valid;
        prepare_k<T, true>(params, next0, start_block_idx, seqlen_k, idx_in_warpgroup,
            vi, (start_block_idx / 2) % 2, valid);
        issue_k<4, T::NUM_TILES, T, false, false>(cur_sK1, &barriers_K0[1], valid, indices,
            0, real_end_block_idx, next0, idx_in_warpgroup, params, next0, start_block_idx);
        issue_k<0, 4, T, false, false>(cur_sK0, &barriers_K0[0], valid, indices,
            0, real_end_block_idx, next0, idx_in_warpgroup, params, next0, start_block_idx);
        next0 = token_index<T>(params, indices, start_block_idx + 2, real_end_block_idx, idx_in_warpgroup);
        next1 = token_index<T>(params, indices, start_block_idx + 3, real_end_block_idx, idx_in_warpgroup);
        prolog_sync<T, 0>();
        if (start_block_idx + 2 < end_block_idx) {
            prepare_k<T, true>(params, next0, start_block_idx + 2, seqlen_k, idx_in_warpgroup,
                vi, (start_block_idx / 2 + 1) % 2, valid0);
        }
        if (start_block_idx + 3 < end_block_idx) {
            prepare_k<T, false>(params, next1, start_block_idx + 3, seqlen_k, idx_in_warpgroup, vi, 0, valid1);
        }
    }
    if (start_block_idx + 1 < end_block_idx) {
        if (warpgroup_idx == 1) {
            bool valid;
            prepare_k<T, true>(params, next1, start_block_idx + 1, seqlen_k, idx_in_warpgroup,
                vi, 2 + (start_block_idx / 2) % 2, valid);
            issue_k<4, T::NUM_TILES, T, false, false>(cur_sK0, &barriers_K1[1], valid, indices,
                0, real_end_block_idx, next0, idx_in_warpgroup, params, next1, start_block_idx + 1);
            issue_k<0, 4, T, false, false>(cur_sK1, &barriers_K1[0], valid, indices,
                0, real_end_block_idx, next0, idx_in_warpgroup, params, next1, start_block_idx + 1);
            next0 = token_index<T>(params, indices, start_block_idx + 2, real_end_block_idx, idx_in_warpgroup);
            next1 = token_index<T>(params, indices, start_block_idx + 3, real_end_block_idx, idx_in_warpgroup);
            prolog_sync<T, 1>();
            if (start_block_idx + 3 < end_block_idx) {
                prepare_k<T, true>(params, next1, start_block_idx + 3, seqlen_k, idx_in_warpgroup,
                    vi, 2 + (start_block_idx / 2 + 1) % 2, valid0);
            }
            if (start_block_idx + 2 < end_block_idx) {
                prepare_k<T, false>(params, next0, start_block_idx + 2, seqlen_k, idx_in_warpgroup, vi, 0, valid1);
            }
        }
    }

    Tensor rO = partition_fragment_C(typename T::TiledMma{}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV / 2>>{});
    float rL[2];
    rL[0] = rL[1] = 0.0f;
    cute::fill(rO, 0.);
    if (threadIdx.x < size(sM)) sM[threadIdx.x] = prefill::MAX_INIT_VAL_SM;
    while (!cutlass::arch::test_wait(barrier_Q, cur_phase_Q, 1)) {
        prefill::kernel_sleep_ns();
    }
    cur_phase_Q = (cur_phase_Q + 1) & 1;
    T::template finish_q<T>(params, sQ, tidx, fused_q_position);
    Tensor rQ8 = make_tensor<InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
    if constexpr (T::kCacheLastQ) {
        prefill::retrieve_rP_from_sP<T>(rQ8,
            local_tile(sQ, Shape<_128, _64>{}, Coord<_0, Int<T::NUM_TILES - 1>>{}), idx_in_warpgroup);
    }

    if (warpgroup_idx == 0) {
        Tensor rP0 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});
        const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);
        int idx = 0;
        int block_idx = start_block_idx;
        if (seqlen_k != 0) {
            cute::clear(rP0);
            prefill::dsa_warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK0, cur_sK1,
                rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
        }
#define FUSED_WI_WG0_STEP(Last) \
        wg0_step<T, Last>(sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, \
            rQ8, rP0, rO, rL, barriers_K0, barriers_K1, cur_phase_K0, params, indices, seqlen_k, \
            block_idx, end_block_idx, real_end_block_idx, idx_in_warpgroup, wg_idx, idx, \
            next0, next1, valid0, valid1, vi)
        #pragma unroll 1
        for (; block_idx < end_block_idx - 2; block_idx += 2) {
            FUSED_WI_WG0_STEP(false);
        }
        if (block_idx < end_block_idx) {
            FUSED_WI_WG0_STEP(true);
        }
#undef FUSED_WI_WG0_STEP
    } else {
        Tensor rP1 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});
        const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);
        if (start_block_idx + 1 < end_block_idx) {
            prefill::dsa_warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK1, cur_sK0,
                rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
        }
        int idx = 0;
        int block_idx = start_block_idx;
#define FUSED_WI_WG1_STEP(Last) \
        wg1_step<T, Last>(sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, \
            rQ8, rP1, rO, rL, barriers_K0, barriers_K1, cur_phase_K1, params, indices, seqlen_k, \
            block_idx, end_block_idx, real_end_block_idx, idx_in_warpgroup, wg_idx, idx, \
            next0, next1, valid0, valid1, vi)
        #pragma unroll 1
        for (; block_idx < end_block_idx - 2; block_idx += 2) {
            FUSED_WI_WG1_STEP(false);
        }
        if (block_idx < end_block_idx) {
            FUSED_WI_WG1_STEP(true);
        }
#undef FUSED_WI_WG1_STEP
    }

    rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
    rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
    rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
    rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
    const int my_row = prefill::get_AorC_row_idx(0, idx_in_warpgroup);
    if (idx_in_warpgroup % 4 == 0) {
        sL[my_row + warpgroup_idx * 128] = rL[0];
        sL[my_row + 8 + warpgroup_idx * 128] = rL[1];
    }
    __syncthreads();
    if (warpgroup_idx == 0) {
        rL[0] += sL[my_row + 128];
        rL[1] += sL[my_row + 8 + 128];
    } else {
        if (idx_in_warpgroup % 4 == 0) {
            sL[my_row] += rL[0];
            sL[my_row + 8] += rL[1];
        }
        __syncwarp();
        rL[0] = sL[my_row];
        rL[1] = sL[my_row + 8];
    }
    T::template store_output<T>(params, rO, plan, warpgroup_idx, idx_in_warpgroup);
}

} // namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn::fused_wi

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

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch, bool PerTokenScale>
template<typename T, typename TensorQ>
__forceinline__ __device__ void Config<Mode, Heads, EnableNorm, ExtraFP4, Arch, PerTokenScale>::load_q(
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

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch, bool PerTokenScale>
template<typename T, typename TensorQ>
__forceinline__ __device__ int Config<Mode, Heads, EnableNorm, ExtraFP4, Arch, PerTokenScale>::issue_q(
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

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch, bool PerTokenScale>
template<typename T, typename TensorQ>
__forceinline__ __device__ void Config<Mode, Heads, EnableNorm, ExtraFP4, Arch, PerTokenScale>::finish_q(
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

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch, bool PerTokenScale>
template<typename T, typename TensorO>
__forceinline__ __device__ void Config<Mode, Heads, EnableNorm, ExtraFP4, Arch, PerTokenScale>::store_output(
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

    float group_scale_inv = 0.0f;
    if constexpr(PerTokenScale) {
        // Four lanes cover one head; a warp covers one complete WV group.
        float group_amax = 0.0f;
        if(head < p.real_heads && multiplier != 0.0f) {
            #pragma unroll
            for(int step = 0; step < 4; ++step) {
                const int tile = lane + step * 4;
                #pragma unroll
                for(int i = 0; i < 32; i += 2) {
                    float2 value{staged(head, tile * 32 + i), staged(head, tile * 32 + i + 1)};
                    if(tile >= 14) {
                        const int pair = (tile - 14) * 16 + i / 2;
                        value = apply_rope(value, __ldg(rope + pair), -__ldg(rope + 32 + pair));
                    }
                    group_amax = fmaxf(group_amax, fmaxf(fabsf(value.x), fabsf(value.y)));
                }
            }
            group_amax *= multiplier;
        }
        #pragma unroll
        for(int offset = 16; offset >= 1; offset /= 2) {
            group_amax = fmaxf(group_amax, __shfl_xor_sync(0xffffffffu, group_amax, offset));
        }
        const float scale = fmaxf(1e-4f, group_amax) / 448.0f;
        group_scale_inv = __fdividef(1.0f, scale);
        if((threadIdx.x & 31) == 0 && head < p.real_heads) {
            reinterpret_cast<float*>(fused.out_sf)[int64_t(p.token_idx) * fused.stride_out_sf_token
                + int64_t(head / 8) * fused.stride_out_sf_group] = scale;
        }
    }

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
            int exponent = 0;
            float cast_multiplier = multiplier * group_scale_inv;
            if constexpr(!PerTokenScale) {
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
                exponent = int((__float_as_uint(scale) - 1u) >> 23) + 1 - 127;
                const float scale_inv = __uint_as_float(uint32_t(127 - exponent) << 23);
                cast_multiplier = multiplier * scale_inv;
            }
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
            if constexpr(!PerTokenScale) {
                const int sf_index = tile * 8 + in_group;
                auto* sf_word = fused.out_sf + p.token_idx + int64_t(group) * fused.stride_out_sf_group
                    + int64_t(sf_index / 4) * fused.stride_out_sf_word;
                reinterpret_cast<uint8_t*>(sf_word)[sf_index % 4] = uint8_t(exponent + 127);
            }
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
fused_norm_rope_attn_rope_cast_fwd_kernel(__grid_constant__ const typename T::KernelParams params) {
    const int token = blockIdx.x;
    const TileParams tile = make_tile<T>(params, token);
    fused_wi::tile_body<T>(tile, 0);
}

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch, bool PerTokenScale>
void launch(const Params& params) {
    const OutputParams<PerTokenScale> kernel_params{params};
    using T = Config<Mode, Heads, EnableNorm, ExtraFP4, Arch, PerTokenScale>;
    constexpr size_t bytes = sizeof(typename T::SharedMemoryOutPut);
    if constexpr(Mode == FwdMode::Prefill && Arch == 89) {
        if(params.topk_length == nullptr) {
            using V = NativePrefillConfig<Heads, EnableNorm, Arch, PerTokenScale>;
            auto native_kernel = &fused_norm_rope_attn_rope_cast_fwd_kernel<V>;
            C10_CUDA_CHECK(hggcFuncSetAttribute(native_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
            native_kernel<<<dim3(params.s_q, 1, 1), V::NUM_THREADS, bytes, params.stream>>>(kernel_params);
            CHECK_CUDA_KERNEL_LAUNCH();
            return;
        }
    }
    if constexpr(Mode == FwdMode::Decode && Arch == 89 && Heads == 128) {
        if(params.extra_topk > 0) {
            using V = VectorDecodeConfig<Heads, EnableNorm, ExtraFP4, Arch, PerTokenScale>;
            auto vector_kernel = &fused_norm_rope_attn_rope_cast_fwd_kernel<V>;
            C10_CUDA_CHECK(hggcFuncSetAttribute(vector_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
            vector_kernel<<<dim3(params.s_q, 1, 1), V::NUM_THREADS, bytes, params.stream>>>(kernel_params);
            CHECK_CUDA_KERNEL_LAUNCH();
            return;
        }
    }
    auto kernel = &fused_norm_rope_attn_rope_cast_fwd_kernel<T>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(params.s_q, 1, 1), T::NUM_THREADS, bytes, params.stream>>>(kernel_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const Params& params) {
    TORCH_CHECK(params.mode == Mode && params.h_q == Heads && params.enable_q_norm == EnableNorm);
    TORCH_CHECK(params.wv_group_size == 8 && params.n_wv_group == Heads / 8);
    if constexpr(Mode == FwdMode::Decode) TORCH_CHECK(params.extra_fp4 == ExtraFP4);
    if(params.s_q == 0) return;
    if(params.arch == 80) {
        TORCH_CHECK(!params.per_token_scale, "Per-token FP32 output scales currently require PPU SM89");
        launch<Mode, Heads, EnableNorm, ExtraFP4, 80, false>(params);
    } else {
        TORCH_CHECK(params.arch == 89, "Fused WI requires SM80 or SM89");
        if (params.per_token_scale) launch<Mode, Heads, EnableNorm, ExtraFP4, 89, true>(params);
        else launch<Mode, Heads, EnableNorm, ExtraFP4, 89, false>(params);
    }
}

}
