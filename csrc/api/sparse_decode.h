/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE

#include "common.h"
#include "params.h"
#include "kerutils/common/static_switch.h"

template<typename T, bool IsFP8, int Headdim, int Headdim_V>
void run_sparse_decode_fwd_dispatch(Flash_fwd_params &params, hggcStream_t stream);
void get_mla_metadata_func(Mla_metadata_params &params, hggcStream_t stream);

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    at::Tensor &q,                               // batch_size x seqlen_q_ori x num_heads_ori x head_size
    const at::Tensor &kv,                        // num_blocks x page_block_size x num_heads_k x head_size
    const int head_size_v,
    const at::Tensor &indices,                   // batch_size x seqlen_q_ori x topk
    const std::optional<at::Tensor> &attn_sink,  // num_heads_q
    const std::optional<at::Tensor> &topk_length, // batch_size
    std::optional<at::Tensor> &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    std::optional<at::Tensor> &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &extra_kv,            // extra_num_blocks x extra_page_block_size x num_heads_k x head_size
    const std::optional<at::Tensor> &extra_indices,       // batch_size x seqlen_q_ori x extra_topk
    const std::optional<at::Tensor> &extra_topk_length,   // batch_size
    const float softmax_scale,
    const std::optional<at::Tensor> &out_                 // batch_size x seqlen_q_ori x num_heads_ori x head_size_v
) {
    bool is_fp8 = (kv.dtype() == torch::kFloat8_e4m3fn ||
                   kv.dtype() == torch::kInt8 ||
                   kv.dtype() == torch::kUInt8);

    // ========== Phase 1: Lazy metadata ==========
    if (!tile_scheduler_metadata.has_value()) {
        const auto sizes = q.sizes();
        const int batch_size = sizes[0];
        const int num_heads_ori = sizes[2];
        const int num_heads_k = kv.size(2);
        TORCH_CHECK(num_heads_ori % num_heads_k == 0);
        const int ngroups = num_heads_ori / num_heads_k;

        int num_sm_parts = get_num_sm_parts(ngroups, num_heads_k, batch_size, /*is_sparse_attn=*/true);

        // btv105 only use cross_cut method.
        int block_size_n = 64;

        static constexpr int fixed_overhead_num_blocks = 5;
        auto options = q.options().dtype(torch::kInt32);
        auto tile_scheduler_metadata_t = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
        auto num_splits_t = torch::empty({batch_size + 1}, options);

        at::cuda::CUDAGuard device_guard{(char)q.get_device()};
        auto meta_stream = at::cuda::getCurrentCUDAStream().stream();
        Mla_metadata_params meta_params = {};
        meta_params.seqlens_k_ptr = nullptr;
        meta_params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_t.data_ptr<int>();
        meta_params.num_splits_ptr = num_splits_t.data_ptr<int>();
        meta_params.batch_size = batch_size;
        meta_params.block_size_n = block_size_n;
        meta_params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
        meta_params.num_sm_parts = num_sm_parts;
        meta_params.topk = indices.size(-1);
        meta_params.extra_topk = extra_indices.has_value() ? (int)extra_indices->size(-1) : 0;
        meta_params.topk_length = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
        meta_params.extra_topk_length = extra_topk_length.has_value() ? extra_topk_length->data_ptr<int>() : nullptr;

        get_mla_metadata_func(meta_params, meta_stream);

        tile_scheduler_metadata = tile_scheduler_metadata_t;
        num_splits = num_splits_t;
    }

    // ========== Phase 2: Kernel dispatch ==========
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor >= 0;
    TORCH_CHECK(is_sm8x);

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kBFloat16, "Sparse only supports BFloat16");
    if (!is_fp8) {
        TORCH_CHECK(kv.dtype() == q_dtype, "query and key must have the same dtype");
    } else {
        TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8, "key must have dtype fp8_e4m3fn or int8 or uint8");
    }
    CHECK_DEVICE(q); CHECK_DEVICE(kv);
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kv.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_ori = sizes[2];
    const int head_size = sizes[3];
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");
    TORCH_CHECK(head_size == 576 || head_size == 512);

    const int num_blocks = kv.size(0);
    const int page_block_size = kv.size(1);
    const int num_heads_k = kv.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (is_fp8) {
        int bytes_per_token;
        if (head_size == 576 && head_size_v == 512) {
            // V3.2 style
            bytes_per_token = 512 + 64*2 + (512/128)*4;
        } else if (head_size == 512 && head_size_v == 512) {
            // MODEL1 style
            bytes_per_token = 448 + 64*2 + (448/64)*1 + 1;
        } else {
            TORCH_CHECK(false, "Unsupported head sizes for is_fp8_kvcache == True");
        }
        CHECK_SHAPE(kv, num_blocks, page_block_size, num_heads_k, bytes_per_token);
        TORCH_CHECK(num_heads_k == 1, "Currently the number of k heads must be 1 when is_fp8_kvcache is True");
        TORCH_CHECK(kv.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True");
    }

    const int topk = indices.size(-1);
    CHECK_DEVICE(indices);
    CHECK_SHAPE(indices, batch_size, seqlen_q_ori, topk);
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must have dtype int32");
    TORCH_CHECK(indices.stride(-1) == 1, "indices must have contiguous last dimension");

    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;

    q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
            .reshape({batch_size, seqlen_q, num_heads, head_size});

    auto opts = q.options();
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "out must have the same dtype as q");
        KU_CHECK_SHAPE(out, batch_size, seqlen_q_ori, num_heads_ori, head_size_v);
        out = out.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size_v}).transpose(2, 3)
                .reshape({batch_size, seqlen_q, num_heads, head_size_v});
        KU_CHECK_CONTIGUOUS(out);
        KU_CHECK_DEVICE(out);
    } else {
        out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts);
    }
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    const at::Tensor &vcache = kv;

    Flash_fwd_params params {};

    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.q_orig = seqlen_q_ori;
    params.cu_seqlens_k = nullptr;
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = false;
    params.topk = topk;

    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);

    params.q_ptr = q.data_ptr();
    params.k_ptr = kv.data_ptr();
    params.v_ptr = vcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.indices_ptr = indices.data_ptr<int>();
    params.attn_sink_ptr = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    params.topk_len_ptr = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
    params.extra_k_ptr = extra_kv.has_value() ? extra_kv->data_ptr() : nullptr;
    params.extra_indices_ptr = extra_indices.has_value() ? extra_indices->data_ptr<int>() : nullptr;
    params.extra_topk_len_ptr = extra_topk_length.has_value() ? extra_topk_length->data_ptr<int>() : nullptr;

    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kv.stride(0);
    params.v_batch_stride = vcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kv.stride(-3);
    params.v_row_stride = vcache.stride(-3);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kv.stride(-2);
    params.v_head_stride = vcache.stride(-2);
    params.o_head_stride = out.stride(-2);
    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = page_block_size;
    params.num_blocks = num_blocks;
    params.indices_batch_stride = indices.stride(0);
    params.indices_row_stride = indices.stride(1);
    params.extra_k_batch_stride = extra_kv.has_value() ? extra_kv->stride(0) : 0;
    params.extra_k_row_stride = extra_kv.has_value() ? extra_kv->stride(-3) : 0;
    params.extra_k_head_stride = extra_kv.has_value() ? extra_kv->stride(-2) : 0;
    params.extra_indices_batch_stride = extra_indices.has_value() ? extra_indices->stride(0) : 0;
    params.extra_indices_row_stride = extra_indices.has_value() ? extra_indices->stride(1) : 0;
    params.extra_num_blocks = extra_kv.has_value() ? extra_kv->size(0) : 0;
    params.extra_page_block_size = extra_kv.has_value() ? extra_kv->size(1) : 0;
    params.extra_topk = extra_indices.has_value() ? (int)extra_indices->size(-1) : -1;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // export PPU_LIB_SHOW_PARAMS=1
    ppu::fmha::FmhaProfParam fmha_prof_params;
    if (ppu::fmha::ProfilingInterface::Instance().get_op_info()) {
        hggcStreamCaptureStatus captureStatus;
        hggcStreamIsCapturing(stream, &captureStatus);

        std::string topk_len_str = "";
        std::string extra_topk_len_str = "";
        if (captureStatus != hggcStreamCaptureStatusNone) {
            GraphCaptureModeSuspender protector(hggcStreamCaptureModeRelaxed);
            // Create a temporary side stream not associated with the capture
            hggcStream_t side_stream;
            if (hggcStreamCreateWithFlags(&side_stream, hggcStreamNonBlocking) == hggcSuccess) {
                // Perform Device-to-Host copy on the side stream
                // This won't be recorded in the graph
                if (params.topk_len_ptr) {
                    topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.topk_len_ptr, params.b, side_stream);
                }
                if (params.extra_topk_len_ptr) {
                    extra_topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.extra_topk_len_ptr, params.b, side_stream);
                }
                // Synchronize the side stream (Safe because of Relaxed mode)
                hggcStreamDestroy(side_stream);
            }
        } else {
            // Construct the string for profiling/UT analysis
            if (params.topk_len_ptr) {
                topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.topk_len_ptr, params.b, stream);
            }
            if (params.extra_topk_len_ptr) {
                extra_topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.extra_topk_len_ptr, params.b, stream);
            }
        }

        fmha_prof_params.set_flash_attn_params(
                q_dtype == torch::kBFloat16/*data_type*/,
                params.is_causal/*custom_mask*/, params.b/*batch_size*/,
                num_heads_ori/*num_heads*/, num_heads_k/*num_heads_k*/,
                params.d/*head_dim*/, params.d_v/*head_dim_value*/,
                seqlen_q_ori/*seqlen_q*/, /*seqlen_kv_str=*/"",
                topk, is_fp8, bool(params.attn_sink_ptr), topk_len_str,
                params.extra_topk, extra_topk_len_str
            );
    }

    // tile_scheduler
    TORCH_CHECK(tile_scheduler_metadata->dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(tile_scheduler_metadata->size(1) == TileSchedulerMetaDataSize);
    CHECK_DEVICE(tile_scheduler_metadata.value());
    CHECK_CONTIGUOUS(tile_scheduler_metadata.value());
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata->data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata->size(0);
    TORCH_CHECK(num_splits->dtype() == torch::kInt32, "num_splits must have dtype int32");
    CHECK_DEVICE(num_splits.value());
    CHECK_CONTIGUOUS(num_splits.value());
    params.num_splits_ptr = num_splits->data_ptr<int>();

    // splitkv
    //at::Tensor softmax_lse_accum, out_accum;
    //std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
    //    params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
    //    head_size, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);
    at::Tensor softmax_lse_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor out_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q, head_size_v}, opts.dtype(at::kFloat));
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();

    ppu::fmha::ProfilingInterface::Instance().instrument(true, fmha_prof_params);
    // TORCH_CHECK(head_size == 576);

    if (is_fp8) {
        // TORCH_CHECK(false, "Only FP8 kvcahe is supported for sparse MLA on SM90"); // TODO
        if (head_size == 576) {
            // V3.2 style
            run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 576, 512>(params, stream);
        } else {
            // MODEL1 style
            run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 512, 512>(params, stream);
        }
    } else {
        if (head_size == 576) {
            // V3.2 style
            run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, false, 576, 512>(params, stream);
        } else {
            // MODEL1 style
            run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, false, 512, 512>(params, stream);
        }
    }

    ppu::fmha::ProfilingInterface::Instance().instrument(false, fmha_prof_params);

    out = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(2, 3)
            .reshape({batch_size, num_heads_ori, seqlen_q_ori});

    return std::make_tuple(out, softmax_lse, tile_scheduler_metadata, num_splits);
}

#endif // FLASHMLA_C_ENABLE_DECODE_SPARSE
