#pragma once

#include <type_traits>
#include "decode/sparse/sparse_decode_wg.cuh"
#include "kernel.cuh"
#include "../combine/kernel.h"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

struct FusedNativeDecodeParams : Flash_fwd_params {
    core_attn::Params fused;
};

template<bool Norm>
struct FusedNativeDecodeTraits
    : flashmla::dsa::v41::SparseDecodeV41DirectScopeNarrowNativeFP4WG1PublishFirstRemotePVTraits {
    using Base = flashmla::dsa::v41::SparseDecodeV41DirectScopeNarrowNativeFP4WG1PublishFirstRemotePVTraits;
    using FusedConfig = Config<FwdMode::Decode, 128, Norm, true, 89>;
    using KernelParams = FusedNativeDecodeParams;
    using Params = TileParams;
    using SharedMemoryPlan = typename Base::SharedMemoryPlan;
    static constexpr bool kFusedNativeDecode = true;
    static constexpr bool kTokenInterleavedCache = true;
    static constexpr bool kFusedQLoad = true;
    static constexpr bool kFusedEpilogue = true;
    static constexpr bool kVectorOutputLoad = true;
    static constexpr int kQNormOffset = flashmla::dsa::v41::V41_FP4_ASYNC_STAGE_BASE
        + Base::NUM_THREADS * flashmla::dsa::v41::V41_FP4_ASYNC_STAGE_STRIDE;

    static_assert(Base::kUsePairedCubes && Base::kSingleTaskCTA);
    static_assert(Base::NUM_THREADS == 512 && Base::kBlockM == 128 && Base::kQRowsPerLoad == 8);
    static_assert(!Base::kDecodeFP8ScopeRow32 && !Base::kDecodePageBase64);
    static_assert(std::is_same_v<typename Base::CacheAdapter, flashmla::dsa::v41::SparseDecodeV41CacheAdapter>);
    static_assert(std::is_same_v<typename Base::Scheduler, flashmla::dsa::v41::SparseDecodeV41DirectScheduler>);
    static_assert(std::is_same_v<typename Base::SmemLayoutQ, typename FusedConfig::SmemLayoutQ>);
    static_assert(std::is_same_v<typename Base::SmemLayoutO, typename FusedConfig::SmemLayoutO>);
    static_assert(std::is_same_v<typename Base::SmemCopyAtomOaccum, typename FusedConfig::SmemCopyAtomOaccum>);
    static_assert(std::is_same_v<typename Base::TiledMma, typename FusedConfig::TiledMma>);
    static_assert(sizeof(SharedMemoryPlan) == 248880);
    static_assert(sizeof(SharedMemoryPlan) < flashmla::dsa::v41::V41_FP4_ASYNC_STAGE_BASE);
    static_assert(flashmla::dsa::v41::V41_FP4_ASYNC_STAGE_BASE == 248960);
    static_assert(flashmla::dsa::v41::V41_FP4_ASYNC_STAGE_STRIDE == 24);
    static_assert(Base::NUM_THREADS * sizeof(float) == 2048);
    static_assert(cute::cosize_v<typename Base::SmemLayoutK> * sizeof(typename Base::InputT) >= 2048);
    static_assert(kQNormOffset == 261248);
    static_assert(kQNormOffset + Base::kBlockM * sizeof(float) == 261760);
    static_assert(kQNormOffset + Base::kBlockM * sizeof(float) < 262144);
    static_assert(sizeof(typename Base::SharedMemoryOutPut) == 262144);

    static __forceinline__ __device__ float score_scale(float scale, int row) {
        if constexpr(Norm) {
            extern __shared__ char wksp_buf[];
            const float* q_norm = reinterpret_cast<const float*>(wksp_buf + kQNormOffset);
            return scale * q_norm[row];
        } else {
            return scale;
        }
    }

    static __forceinline__ __device__ uint4 fp4_word_to_bf16x2_bits(uint32_t word) {
        const uint32_t hsel=((word>>1)&0x77777777u)|(word&0x11111111u);
        const uint32_t selectors0=word&0x7777u;
        const uint32_t selectors1=(word>>16)&0x7777u;
        const uint32_t low0=__byte_perm(0xc0800000u,0xc0804000u,selectors0);
        uint32_t high0;
        asm volatile("ppu.prmt.b32 %0, %1, %2, %3;\n"
            : "=r"(high0) : "n"(0x40403f00u), "n"(0xc0c0bf80u), "r"(hsel));
        const uint32_t low1=__byte_perm(0xc0800000u,0xc0804000u,selectors1);
        const uint32_t high1=__byte_perm(0x40403f00u,0xc0c0bf80u,hsel>>16);
        return uint4{__byte_perm(low0,high0,0x5140),__byte_perm(low0,high0,0x7362),
            __byte_perm(low1,high1,0x5140),__byte_perm(low1,high1,0x7362)};
    }

    static __forceinline__ __device__ uint4 fp8_words_to_scaled_bf16x8_bits(
        uint32_t first_word, uint32_t second_word, __nv_bfloat162 scale2) {
        const uint32_t words[2]={first_word,second_word};
        uint32_t out_bits[4]{};
        #pragma unroll
        for(int word=0;word<2;++word) {
            const __nv_fp8x4_e4m3 vals4=*reinterpret_cast<const __nv_fp8x4_e4m3*>(&words[word]);
            const float4 vals=(float4)vals4;
            const __nv_bfloat162 lo=__hmul2(__float22bfloat162_rn({vals.x,vals.y}),scale2);
            const __nv_bfloat162 hi=__hmul2(__float22bfloat162_rn({vals.z,vals.w}),scale2);
            out_bits[word*2]=*reinterpret_cast<const uint32_t*>(&lo);
            out_bits[word*2+1]=*reinterpret_cast<const uint32_t*>(&hi);
        }
        return uint4{out_bits[0],out_bits[1],out_bits[2],out_bits[3]};
    }

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ void load_q(
        const TileParams& p, int m_block_idx, TensorQ& q,
        int tidx, int warp_idx, __mbarrier_t* barrier) {
        FusedConfig::template load_q<T>(p, m_block_idx, q, tidx, warp_idx, barrier);
    }

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ int issue_q(
        const TileParams& p, int m_block_idx, TensorQ& q,
        int tidx, int warp_idx, __mbarrier_t* barrier) {
        return FusedConfig::template issue_q<T>(p, m_block_idx, q, tidx, warp_idx, barrier);
    }

    template<typename T, typename TensorQ>
    static __forceinline__ __device__ void finish_q(
        const TileParams& p, TensorQ& q, int tidx, int position) {
        FusedConfig::template finish_q<T>(p, q, tidx, position);
    }
};

template<bool Norm>
bool run_fused_norm_rope_attn_rope_cast_decode_native_wi(
    const Params& params, const SplitKVParams& split) {
    using T = FusedNativeDecodeTraits<Norm>;
    FusedNativeDecodeParams p{};
    // The API has flattened tokens: retain b=s_q, never reconstruct an old batch.
    p.b = params.s_q;
    p.q_orig = 1;
    p.seqlen_q = 128;
    p.d = 512;
    p.d_v = 512;
    p.h = 1;
    p.h_q = 128;
    p.h_h_k_ratio = 1;
    p.ngroups = 128;
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
    p.extra_k_ptr = const_cast<void*>(params.extra_kv);
    p.extra_indices_ptr = const_cast<int*>(params.extra_indices);
    p.extra_topk_len_ptr = const_cast<int*>(params.extra_topk_length);
    p.extra_k_batch_stride = params.stride_extra_kv_block;
    p.extra_k_row_stride = 288;
    p.extra_indices_batch_stride = params.stride_extra_indices_s_q;
    p.extra_num_blocks = params.extra_num_blocks;
    p.extra_page_block_size = params.extra_page_block_size;
    p.extra_topk = params.extra_topk;
    p.tile_scheduler_metadata_ptr = split.tile_scheduler_metadata;
    p.num_sm_parts = split.num_sm_parts;
    p.num_splits_ptr = split.num_splits;
    p.softmax_lseaccum_ptr = split.lse_accum;
    p.oaccum_ptr = split.o_accum;
    p.lse_to_log2 = float(M_LOG2E);
    p.o_ptr = params.out_fp8;
    // make_task's NoSplit view is typed BF16; the fused hook alone writes FP8.
    p.o_batch_stride = 128 * 512 / sizeof(typename T::InputT);
    p.o_row_stride = 512 / sizeof(typename T::InputT);
    p.softmax_lse_ptr = params.lse;
    p.fused = params;

    if(!flashmla::dsa::v41::sparse_decode_v41_fp4_cache_offset32(p)
        || !flashmla::dsa::v41::sparse_decode_v41_direct_grid_gate(p)) return false;
    TORCH_CHECK(params.mode == FwdMode::Decode && params.arch == 89 && params.h_q == 128
        && params.enable_q_norm == Norm && params.extra_fp4);
    TORCH_CHECK(params.wv_group_size == 8 && params.n_wv_group == 16);
    TORCH_CHECK(params.stride_q_s_q % 8 == 0 && reinterpret_cast<uintptr_t>(params.q) % 16 == 0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(params.out_fp8) % 16 == 0);

    flashmla::dsa::v41::launch_sparse_decode_wg_v41_main<T>(p, params.stream);

    combine::Params c{};
    c.s_q = params.s_q;
    c.h_q = 128;
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
    c.stream = params.stream;
    combine::run_fused_norm_rope_attn_rope_cast_combine_kernel(c);
    return true;
}

}
