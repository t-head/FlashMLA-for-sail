#pragma once

#include "kernel.h"
#include "prefill/sparse/traits.h"

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

template<FwdMode Mode, int Heads, bool EnableNorm, bool ExtraFP4, int Arch_>
struct Config : ::DSA_Traits<cutlass::bfloat16_t, 512, Arch_> {
    using Base = ::DSA_Traits<cutlass::bfloat16_t, 512, Arch_>;
    static constexpr int Arch = Arch_;
    static constexpr bool kQuantCache = true;
    using Params = TileParams;
    using KernelParams = core_attn::Params;
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

template<int Heads, bool EnableNorm, bool ExtraFP4, int Arch>
struct VectorDecodeConfig : Config<FwdMode::Decode, Heads, EnableNorm, ExtraFP4, Arch> {
    static constexpr bool kVectorDecodeKV = true;
};

template<int Heads, bool EnableNorm, int Arch>
struct NativePrefillConfig : Config<FwdMode::Prefill, Heads, EnableNorm, false, Arch> {
    static constexpr bool kDecode = Arch != 89;
    static constexpr bool kQuantCache = kDecode;
};

}
