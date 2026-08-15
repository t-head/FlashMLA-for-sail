/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include "common.h"
#include "params.h"
#include "kerutils/host/host.h"
#include "kerutils/common/static_switch.h"

template<typename T, int Headdim, int Headdim_V>
void run_mha_fwd_splithd_splitkv_dispatch(Flash_fwd_params &params, hggcStream_t stream);
void get_mla_metadata_func(Mla_metadata_params &params, hggcStream_t stream);

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
dense_attn_decode_interface(
    at::Tensor &q,                               // batch_size x seqlen_q_ori x num_heads_ori x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    std::optional<at::Tensor> &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    std::optional<at::Tensor> &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &out_                 // batch_size x seqlen_q_ori x num_heads_ori x head_size_v
) {
    // ========== Phase 1: Lazy metadata ==========
    if (!tile_scheduler_metadata.has_value()) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        const auto sizes = q.sizes();
        const int batch_size = sizes[0];
        const int seqlen_q_ori = sizes[1];
        const int num_heads_ori = sizes[2];
        const int num_heads_k = kcache.size(2);
        const int ngroups = num_heads_ori / num_heads_k;
        const int seqlen_q = seqlen_q_ori * ngroups;

        CHECK_DEVICE(seqlens_k);
        // TORCH_CHECK(seqlens_k.is_contiguous());
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);

        int num_sm_parts = get_num_sm_parts(seqlen_q, num_heads_k, batch_size, /*is_sparse_attn=*/false);

        int block_size_n;
        if (!is_sm89_or_newer()) {
            block_size_n = use_cross_cut(seqlen_q, batch_size)
                         ? (seqlen_q > 32 && seqlen_q <= 64 ? 64 : 32) : 16;
        } else {
            // btv105 only use cross_cut method.
            block_size_n = 64;
        }

        static constexpr int fixed_overhead_num_blocks = 5;
        auto options = seqlens_k.options();
        auto tile_scheduler_metadata_t = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
        auto num_splits_t = torch::empty({batch_size + 1}, options);

        at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
        Mla_metadata_params params = {};
        params.seqlens_k_ptr = seqlens_k.data_ptr<int>();
        params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_t.data_ptr<int>();
        params.num_splits_ptr = num_splits_t.data_ptr<int>();
        params.batch_size = batch_size;
        params.block_size_n = block_size_n;
        params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
        params.num_sm_parts = num_sm_parts;
        params.topk = -1;
        params.extra_topk = 0;
        params.topk_length = nullptr;
        params.extra_topk_length = nullptr;

        get_mla_metadata_func(params, stream);

        tile_scheduler_metadata = tile_scheduler_metadata_t;
        num_splits = num_splits_t;
    }

    // ========== Phase 2: Kernel dispatch ==========
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor >= 0;
    TORCH_CHECK(is_sm8x);

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    CHECK_DEVICE(q); CHECK_DEVICE(kcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_ori = sizes[2];
    const int head_size = sizes[3];
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;

    q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
            .reshape({batch_size, seqlen_q, num_heads, head_size});

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size);

    int max_num_blocks_per_seq = block_table.size(1);
    CHECK_DEVICE(block_table);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    CHECK_DEVICE(seqlens_k);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(seqlens_k, batch_size);

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

    const at::Tensor &vcache = kcache;

    Flash_fwd_params params {};

    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.q_orig = seqlen_q_ori;
    params.cu_seqlens_k = seqlens_k.data_ptr<int>();
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.topk = -1;

    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.v_ptr = vcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.indices_ptr = nullptr;
    params.attn_sink_ptr = nullptr;
    params.topk_len_ptr = nullptr;
    params.extra_k_ptr = nullptr;
    params.extra_indices_ptr = nullptr;
    params.extra_topk_len_ptr = nullptr;
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.v_batch_stride = vcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(-3);
    params.v_row_stride = vcache.stride(-3);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(-2);
    params.v_head_stride = vcache.stride(-2);
    params.o_head_stride = out.stride(-2);
    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;
    params.num_blocks = num_blocks;
    params.indices_batch_stride = 0;
    params.indices_row_stride = 0;
    params.extra_k_batch_stride = 0;
    params.extra_k_row_stride = 0;
    params.extra_k_head_stride = 0;
    params.extra_indices_batch_stride = 0;
    params.extra_indices_row_stride = 0;
    params.extra_num_blocks = 0;
    params.extra_page_block_size = 0;
    params.extra_topk = -1;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // export PPU_LIB_SHOW_PARAMS=1
    ppu::fmha::FmhaProfParam fmha_prof_params;
    if (ppu::fmha::ProfilingInterface::Instance().get_op_info()) {
        hggcStreamCaptureStatus captureStatus;
        hggcStreamIsCapturing(stream, &captureStatus);

        std::string seqlen_kv_str = "";
        if (captureStatus != hggcStreamCaptureStatusNone) {
            GraphCaptureModeSuspender protector(hggcStreamCaptureModeRelaxed);
            // Create a temporary side stream not associated with the capture
            hggcStream_t side_stream;
            if (hggcStreamCreateWithFlags(&side_stream, hggcStreamNonBlocking) == hggcSuccess) {
                // Perform Device-to-Host copy on the side stream
                // This won't be recorded in the graph
                seqlen_kv_str = fmha_prof_params.nvtx_param2str<int>(params.cu_seqlens_k, params.b, side_stream);
                // Synchronize the side stream (Safe because of Relaxed mode)
                hggcStreamDestroy(side_stream);
            }
        } else {
            // Construct the string for profiling/UT analysis
            seqlen_kv_str = fmha_prof_params.nvtx_param2str<int>(params.cu_seqlens_k, params.b, stream);
        }

        fmha_prof_params.set_flash_attn_params(
                q_dtype == torch::kBFloat16/*data_type*/,
                params.is_causal/*custom_mask*/, params.b/*batch_size*/,
                num_heads_ori/*num_heads*/, num_heads_k/*num_heads_k*/,
                params.d/*head_dim*/, params.d_v/*head_dim_value*/,
                seqlen_q_ori/*seqlen_q*/, seqlen_kv_str/*seqlen_kv*/,
                /*topk=*/-1, /*is_fp8=*/false, /*has_attn_sink=*/false,
                /*topk_len_str=*/"", /*extra_topk=*/-1, /*extra_topk_len_str=*/""
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

    TORCH_CHECK(head_size == 576);
    if (q_dtype == torch::kBFloat16) {
        run_mha_fwd_splithd_splitkv_dispatch<cutlass::bfloat16_t, 576, 512>(params, stream);
    }
    #ifndef FLASH_MLA_DISABLE_FP16
    else if (q_dtype == torch::kHalf) {
        run_mha_fwd_splithd_splitkv_dispatch<cutlass::half_t, 576, 512>(params, stream);
    }
    #endif
    else {
        TORCH_CHECK(false, "Unsupported tensor dtype for query");
    }

    ppu::fmha::ProfilingInterface::Instance().instrument(false, fmha_prof_params);

    out = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(2, 3)
            .reshape({batch_size, num_heads_ori, seqlen_q_ori});

    return std::make_tuple(out, softmax_lse, tile_scheduler_metadata, num_splits);
}
