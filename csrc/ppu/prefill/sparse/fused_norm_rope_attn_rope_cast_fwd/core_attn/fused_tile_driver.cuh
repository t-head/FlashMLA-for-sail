/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include "config.h"
#include "prefill/sparse/sparse_prefill_wg.cuh"

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
