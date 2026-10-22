#pragma once

#include "params.h"
#include "prefill/sparse/sparse_quant_prefill.h"

namespace flashmla::v41prefill {

// One Decode metadata task projected onto the common quantized attention tile.
// Query and cache pointers are views of the original tensors; packed data and
// their scale planes are never copied or concatenated for this projection.
struct QuantDecodeTileParams {
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
};

struct QuantDecodeScheduler;
struct V41QuantRowAddress64;
struct V41QuantRowAddress32;

struct QuantDecodeCacheAdapter {
    template<typename T>
    static __forceinline__ __device__ V41QuantRowAddress32 prepare_scope_row32(
        const typename T::Params &, int token_idx, bool valid, int loaded_block);

    template<int S, int E, typename T>
    static __forceinline__ __device__ auto register_load_scope_row32(
        const typename T::Params &, int tidx, int token_idx, bool valid,
        int loaded_block, const V41QuantRowAddress32 &row);

    template<int S, int E, typename T, bool FRONT_DRAIN, typename RawK>
    static __forceinline__ __device__ void load_scope_row32(
        const typename T::Params &, RawK &, int tidx, int token_idx,
        bool valid, int loaded_block, const V41QuantRowAddress32 &row);


    template<typename T>
    static __forceinline__ __device__ typename T::InputT *prepare_page_base64(
        const typename T::Params &, int token_idx, bool valid, int loaded_block);

    template<int S, int E, typename T>
    static __forceinline__ __device__ auto register_load_page64(
        const typename T::Params &, int tidx, int token_idx, bool valid,
        int loaded_block, const typename T::InputT *page_base);

    template<int S, int E, typename T, bool FRONT_DRAIN, typename RawK>
    static __forceinline__ __device__ void load_page64(
        const typename T::Params &, RawK &, int tidx, int token_idx,
        bool valid, int loaded_block, const typename T::InputT *page_base);

    template<int S, int E, typename T>
    static __forceinline__ __device__ auto register_load(
        const typename T::Params &, int tidx, int token_idx, bool valid, int block_idx);

    template<int S, int E, typename T, bool FRONT_DRAIN, typename RawK, typename Packet>
    static __forceinline__ __device__ void register_finish(
        const typename T::Params &, RawK &, int tidx, int token_idx,
        bool valid, const Packet &packet);

    template<int S, int E, typename T, bool FRONT_DRAIN, typename RawK>
    static __forceinline__ __device__ void load(
        const typename T::Params &, RawK &, int tidx, int token_idx,
        bool valid, int block_idx, bool async_preissued=false);

    template<int S, int E, typename T>
    static __forceinline__ __device__ void async_issue(
        const typename T::Params &, int tidx, int token_idx, bool valid, int block_idx);
};

} // namespace flashmla::v41prefill
