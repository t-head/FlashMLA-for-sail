/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include "common.h"
#include "params.h"
#include "dense_decode.h"
#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
#include "sparse_decode.h"
#endif

void get_mla_metadata_func(Mla_metadata_params &params, hggcStream_t stream);

static std::vector<at::Tensor>
get_mla_metadata(
    const std::optional<at::Tensor> &seqlens_k,
    const int num_heads_per_head_k,
    const int num_heads_k,
    const std::optional<int> num_heads_q_,
    const bool is_fp8_kvcache,
    const std::optional<int> topk,
    const std::optional<int> extra_topk,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &extra_topk_length
) {
    bool is_sparse_attn = topk.has_value();

    TORCH_CHECK(seqlens_k.has_value(), "seqlens_k must be provided");
    int batch_size = seqlens_k.value().size(0);
    if (!is_sparse_attn) {
        CHECK_DEVICE(seqlens_k.value());
        TORCH_CHECK(seqlens_k.value().is_contiguous());
        TORCH_CHECK(seqlens_k.value().dtype() == torch::kInt32);
    }

    int num_tokens_per_head_k = num_heads_per_head_k;
    if (is_sparse_attn) {
        TORCH_CHECK(num_heads_q_.has_value(), "num_heads_q must be provided when topk is provided");
        int num_heads_q = num_heads_q_.value();
        TORCH_CHECK(num_heads_q % num_heads_k == 0);
        num_tokens_per_head_k = num_heads_q / num_heads_k;
    }

    int num_sm_parts = get_num_sm_parts(num_tokens_per_head_k, num_heads_k, batch_size, is_sparse_attn);

    int block_size_n;
    if (is_sparse_attn) {
        block_size_n = 64;
    } else {
        if (!is_sm89_or_newer()) {
            block_size_n = use_cross_cut(num_tokens_per_head_k, batch_size)
                         ? (num_tokens_per_head_k > 32 && num_tokens_per_head_k <= 64 ? 64 : 32) : 16;
        } else {
            block_size_n = 64;
        }
    }

    static constexpr int fixed_overhead_num_blocks = 5;

    TORCH_CHECK(seqlens_k.has_value() || is_sparse_attn);
    auto options = seqlens_k.has_value() ? seqlens_k.value().options() : torch::TensorOptions().dtype(torch::kInt32).device(topk_length.value().device());
    auto tile_scheduler_metadata = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::empty({batch_size + 1}, options);

    int *seqlens_k_ptr = seqlens_k.has_value() ? seqlens_k.value().data_ptr<int>() : nullptr;

    at::cuda::CUDAGuard device_guard{seqlens_k.has_value() ? seqlens_k.value().device() : topk_length.value().device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_splits_ptr = num_splits.data_ptr<int>();
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    params.topk = is_sparse_attn ? topk.value() : -1;
    params.extra_topk = extra_topk.value_or(0);
    params.topk_length = topk_length.has_value() ? topk_length.value().data_ptr<int>() : nullptr;
    params.extra_topk_length = extra_topk_length.has_value() ? extra_topk_length.value().data_ptr<int>() : nullptr;

    get_mla_metadata_func(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

static std::vector<at::Tensor>
mha_fwd_kvcache_mla(
    at::Tensor &q,
    const at::Tensor &kcache,
    c10::optional<const at::Tensor> &vcache_,
    const int head_size_v,
    const std::optional<at::Tensor> &seqlens_k,
    const std::optional<at::Tensor> &block_table,
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,
    const at::Tensor &num_splits,
    const bool &is_fp8,
    const std::optional<at::Tensor> &indices,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &extra_k_cache,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    const std::optional<at::Tensor> &out_
) {
    std::optional<at::Tensor> tile_meta = tile_scheduler_metadata;
    std::optional<at::Tensor> num_splits_opt = num_splits;

    if (indices.has_value()) {
#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
        auto result = sparse_attn_decode_interface(
            q,
            kcache,
            head_size_v,
            indices.value(),
            attn_sink,
            topk_length,
            tile_meta,
            num_splits_opt,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
            softmax_scale,
            out_
        );
        return {std::get<0>(result), std::get<1>(result)};
#else
        TORCH_CHECK(false, "Sparse decode not enabled (FLASHMLA_C_ENABLE_DECODE_SPARSE not defined)");
#endif
    } else {
        auto result = dense_attn_decode_interface(
            q,
            kcache,
            head_size_v,
            seqlens_k.value(),
            block_table.value(),
            softmax_scale,
            is_causal,
            tile_meta,
            num_splits_opt,
            out_
        );
        return {std::get<0>(result), std::get<1>(result)};
    }
}
