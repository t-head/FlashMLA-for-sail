/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>
#include <limits>
#include <ATen/cuda/CUDAContext.h>

#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"
#include "fmha_profiling_interface.hpp"

// Check whether the given tensor or optional tensor satisfies the given condition
// If tensor_or_opt is a tensor, check_fn is applied directly
// If tensor_or_opt is an optional tensor, check_fn is applied only when the optional has value
template<typename T>
static inline bool _check_optional_tensor(const T& tensor_or_opt, const std::function<bool(const at::Tensor&)>& check_fn) {
    if constexpr (std::is_same<T, at::Tensor>::value) {
        return check_fn(tensor_or_opt);
    } else {
        if (tensor_or_opt.has_value()) {
            return check_fn(tensor_or_opt.value());
        } else {
            return true;
        }
    }
}

// Get the pointer of the given tensor
// Return (PtrT*)tensor.data_ptr() if the tensor has a backend storage, nullptr otherwise
template<typename PtrT>
static inline PtrT* get_tensor_ptr(const at::Tensor& tensor) {
    if (tensor.has_storage()) {
        return (PtrT*)tensor.data_ptr();
    } else {
        return nullptr;
    }
}

// Get the pointer of the given tensor or optional tensor
// Return (PtrT*)tensor.data_ptr() if tensor_or_opt has value and points to a valid tensor, return nullptr otherwise
template<typename PtrT, typename T>
static inline PtrT* get_optional_tensor_ptr(const T& tensor_or_opt) {
    if constexpr (std::is_same<T, at::Tensor>::value) {
        return get_tensor_ptr<PtrT>(tensor_or_opt);
    } else {
        if (tensor_or_opt.has_value()) {
            return get_tensor_ptr<PtrT>(*tensor_or_opt);
        } else {
            return nullptr;
        }
    }
}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Check whether the given tensor (or optional<tensor>) is on cuda
#define KU_CHECK_DEVICE(tensor) TORCH_CHECK(_check_optional_tensor(tensor, [](const at::Tensor& t) { return t.is_cuda(); }), #tensor " must be on CUDA")
// Check whether the given tensor (or optional<tensor>) has the given number of dimensions
#define KU_CHECK_NDIM(tensor, ndim) TORCH_CHECK(_check_optional_tensor(tensor, [&](const at::Tensor& t) { return t.dim() == (ndim); }), #tensor " must have " #ndim " dimensions")
// Check whether the given tensor (or optional<tensor>) has the given shape
#define KU_CHECK_SHAPE(tensor, ...) TORCH_CHECK(_check_optional_tensor(tensor, [&](const at::Tensor& t) { return t.sizes() == torch::IntArrayRef({__VA_ARGS__}); }), #tensor " must have shape (" #__VA_ARGS__ ")")
// Check whether the given tensor (or optional<tensor>) is contiguous
#define KU_CHECK_CONTIGUOUS(tensor) TORCH_CHECK(_check_optional_tensor(tensor, [](const at::Tensor& t) { return t.is_contiguous(); }), #tensor " must be contiguous")
// Check whether the last dimention of the given tensor (or optional<tensor>)
#define KU_CHECK_LAST_DIM_CONTIGUOUS(tensor) TORCH_CHECK(_check_optional_tensor(tensor, [](const at::Tensor& t) { return t.size(-1) == 1 || t.stride(-1) == 1; }), #tensor " must have contiguous last dimension")
// Check whether the given tensor (or optional<tensor>) has the specified dtype
#define KU_CHECK_DTYPE(tensor, target_dtype) TORCH_CHECK(_check_optional_tensor(tensor, [](const at::Tensor& t) { return t.dtype() == (target_dtype); }), #tensor " must have dtype " #target_dtype)


struct GraphCaptureModeSuspender {
    hggcStreamCaptureMode original_mode;

    explicit GraphCaptureModeSuspender(hggcStreamCaptureMode relaxed_mode = hggcStreamCaptureModeRelaxed) {
        original_mode = relaxed_mode;
        // Exchange current thread's mode with relaxed_mode, and store previous mode in original_mode
        hggcThreadExchangeStreamCaptureMode(&original_mode);
    }

    ~GraphCaptureModeSuspender() {
        // Restore the saved original mode back to the current thread
        hggcThreadExchangeStreamCaptureMode(&original_mode);
    }

    // Disable copying to prevent accidental double-restoration
    GraphCaptureModeSuspender(const GraphCaptureModeSuspender&) = delete;
    GraphCaptureModeSuspender& operator=(const GraphCaptureModeSuspender&) = delete;
};

std::vector<at::Tensor>
mha_fwd_kvcache_mla(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    c10::optional<const at::Tensor> &vcache_,    // num_blocks x page_block_size x num_heads_k x head_size_v
    const int head_size_v,
    const std::optional<at::Tensor> &seqlens_k,  // batch_size
    const std::optional<at::Tensor> &block_table,// batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits                 // batch_size + 1
#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
    ,
    const bool &is_fp8,
    const std::optional<at::Tensor> &indices,    // None, or batch_size x seqlen_q x topk
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &extra_k_cache,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    const std::optional<at::Tensor> &out_
#endif
) {
#ifndef FLASHMLA_C_ENABLE_DECODE_SPARSE
    bool is_fp8 = false;
    const std::optional<at::Tensor> indices = std::nullopt;
#endif
    bool is_sparse_attn = indices.has_value();
    int topk = is_sparse_attn ? indices->size(-1) : -1;
    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    // bool is_sm75 = cc_major == 7 && cc_minor == 5;
    bool is_sm8x = cc_major == 8 && cc_minor >= 0;
    TORCH_CHECK(is_sm8x);

    at::Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    if (!is_fp8) {
        TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
        TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");
    } else {
        TORCH_CHECK(kcache.dtype() == torch::kFloat8_e4m3fn || kcache.dtype() == torch::kInt8 || kcache.dtype() == torch::kUInt8, "key must have dtype fp8_e4m3fn or int8 or uint8");
    }
    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_ori = sizes[2];
    const int head_size = sizes[3];
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

    // const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);

    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    //const int seqlen_k = max_num_blocks_per_seq * page_block_size;
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
            .reshape({batch_size, seqlen_q, num_heads, head_size});

    int head_size_k = head_size;
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    if (!is_fp8) {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    } else {
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
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, bytes_per_token);
        TORCH_CHECK(num_heads_k == 1, "Currently the number of k heads must be 1 when is_fp8_kvcache is True");
        TORCH_CHECK(kcache.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True");
    }

    if (vcache_.has_value()) { CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_v); }

    if (is_sparse_attn) {
        CHECK_DEVICE(indices.value());
        CHECK_SHAPE(indices.value(), batch_size, seqlen_q_ori, topk);
    } else {
        int max_num_blocks_per_seq = block_table->size(1);
        CHECK_DEVICE(block_table.value());
        CHECK_SHAPE(block_table.value(), batch_size, max_num_blocks_per_seq);
        TORCH_CHECK(block_table->dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table->stride(-1) == 1, "block_table must have contiguous last dimension");

        TORCH_CHECK(seqlens_k->dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k.value());
        CHECK_CONTIGUOUS(seqlens_k.value());
        CHECK_SHAPE(seqlens_k.value(), batch_size);
    }

    TORCH_CHECK(!is_sparse_attn || indices->dtype() == torch::kInt32, "indices must have dtype int32");
    TORCH_CHECK(!is_sparse_attn || indices->stride(-1) == 1, "indices must have contiguous last dimension");

    auto opts = q.options();
    // at::Tensor out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts);
    at::Tensor out;

    if (out_.has_value()) {
        out = out_.value(); // batch_size, seqlen_q_ori, num_heads_ori, head_size
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

    Flash_fwd_params params {};

    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.q_orig = seqlen_q_ori;
    params.cu_seqlens_k = is_sparse_attn ? nullptr : seqlens_k->data_ptr<int>();
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.topk = topk;

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
    params.indices_ptr = is_sparse_attn ? indices->data_ptr<int>() : nullptr;
    params.attn_sink_ptr = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    params.topk_len_ptr = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
    params.extra_k_ptr = extra_k_cache.has_value() ? extra_k_cache.value().data_ptr() : nullptr;
    params.extra_indices_ptr = extra_indices.has_value() ? extra_indices->data_ptr<int>() : nullptr;
    params.extra_topk_len_ptr = extra_topk_length.has_value() ? extra_topk_length->data_ptr<int>() : nullptr;
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
    params.block_table = is_sparse_attn ? nullptr : block_table->data_ptr<int>();
    params.block_table_batch_stride = is_sparse_attn ? 0 : block_table->stride(0);
    params.page_block_size = page_block_size;
    params.num_blocks = num_blocks;
    params.indices_batch_stride = is_sparse_attn ? indices->stride(0) : 0;
    params.indices_row_stride = is_sparse_attn ? indices->stride(1) : 0;
    //params.seqlen_k = seqlen_k;
    params.extra_k_batch_stride = extra_k_cache.has_value() ? extra_k_cache->stride(0) : 0;
    params.extra_k_row_stride = extra_k_cache.has_value() ? extra_k_cache->stride(-3) : 0;
    params.extra_k_head_stride = extra_k_cache.has_value() ? extra_k_cache->stride(-2) : 0;
    params.extra_indices_batch_stride = extra_indices.has_value() ? extra_indices->stride(0) : 0;
    params.extra_indices_row_stride = extra_indices.has_value() ? extra_indices->stride(1) : 0;
    params.extra_num_blocks = extra_k_cache.has_value() ? extra_k_cache->size(0) : 0;
    params.extra_page_block_size = extra_k_cache.has_value() ? extra_k_cache->size(1) : 0;
    params.extra_topk = extra_indices.has_value() ? extra_indices->size(-1) : -1;

    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    // export PPU_LIB_SHOW_PARAMS=1
    ppu::fmha::FmhaProfParam fmha_prof_params;
    if (ppu::fmha::ProfilingInterface::Instance().get_op_info()){
        // check if cuda graph captured
        hggcStreamCaptureStatus captureStatus;
        hggcStreamIsCapturing((hggcStream_t)stream, &captureStatus);

        std::string seqlen_kv_str = "";
        std::string topk_len_str = "";
        std::string extra_topk_len_str = "";
        if (captureStatus != hggcStreamCaptureStatusNone) {
            GraphCaptureModeSuspender protector(hggcStreamCaptureModeRelaxed); // !!!!!! fmha when cudagraph, DON'T DELET

            hggcStream_t side_stream;
            if (hggcStreamCreateWithFlags(&side_stream, hggcStreamNonBlocking) == hggcSuccess) {
                if (!is_sparse_attn) {
                    seqlen_kv_str = fmha_prof_params.nvtx_param2str<int>(params.cu_seqlens_k, params.b, side_stream);
                }
                if (params.topk_len_ptr) {
                    topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.topk_len_ptr, params.b, side_stream);
                }
                if (params.extra_topk_len_ptr) {
                    extra_topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.extra_topk_len_ptr, params.b, side_stream);
                }
                hggcStreamDestroy(side_stream);
            }
        } else {
            if (!is_sparse_attn) {
                seqlen_kv_str = fmha_prof_params.nvtx_param2str<int>(params.cu_seqlens_k, params.b, (hggcStream_t)stream);
            }
            if (params.topk_len_ptr) {
                topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.topk_len_ptr, params.b, (hggcStream_t)stream);
            }
            if (params.extra_topk_len_ptr) {
                extra_topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.extra_topk_len_ptr, params.b, (hggcStream_t)stream);
            }
        }

        fmha_prof_params.set_flash_attn_params(
                q_dtype == torch::kBFloat16/*data_type*/,
                params.is_causal/*custom_mask*/, params.b/*batch_size*/,
                num_heads_ori/*num_heads*/, num_heads_k/*num_heads_k*/,
                params.d/*head_dim*/, params.d_v/*head_dim_value*/,
                seqlen_q_ori/*seqlen_q*/, seqlen_kv_str/*seqlen_kv*/,
                topk, is_fp8, bool(params.attn_sink_ptr), topk_len_str,
                params.extra_topk, extra_topk_len_str
            );
    }

    // tile_scheduler
    TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");
    CHECK_DEVICE(num_splits);
    CHECK_CONTIGUOUS(num_splits);
    params.num_splits_ptr = num_splits.data_ptr<int>();
    at::Tensor softmax_lse_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor out_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q, head_size_v}, opts.dtype(at::kFloat));
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();
     // splitkv
    //at::Tensor softmax_lse_accum, out_accum;
    //std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
    //    params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
    //    head_size, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);

    ppu::fmha::ProfilingInterface::Instance().instrument(true, fmha_prof_params);
    // TORCH_CHECK(head_size == 576);

    if (is_sparse_attn) {
        TORCH_CHECK(head_size == 576 || head_size == 512);
            TORCH_CHECK(q_dtype == torch::kBFloat16, "Sparse FP8 MLA only supports BFloat16 on SM8X");
        if (is_fp8) {
            // TORCH_CHECK(false, "Only FP8 kvcahe is supported for sparse MLA on SM90"); // TODO
            if (head_size == 576) {
                // V3.2 style
                run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 576, 512>(params, stream);
            } else {
                // TODO.
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
    }
    else if (is_fp8) {
        TORCH_CHECK(false, "Dense FP8 MLA is not supported on SM8X");
    }
    else if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(head_size == 576);
        run_mha_fwd_splithd_splitkv_dispatch<cutlass::bfloat16_t, 576, 512>(params, stream);
    }
    #ifndef FLASH_MLA_DISABLE_FP16
    else if (q_dtype == torch::kHalf) {
        TORCH_CHECK(head_size == 576);
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
    return {out, softmax_lse};
}

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int
get_num_sm_parts(
    const int num_heads_per_head_k,
    const int num_heads_k,
    const int batch,
    bool is_sparse_attn = false
) {
    // This should match the logic in the MLA kernel.
    //static constexpr int block_size_m = 64;
    int block_size_m;

    auto dprops = at::cuda::getCurrentDeviceProperties();
    int sm_count = dprops->multiProcessorCount;
    // static set occpuancy priori knowledge.
    int occupancy;
    // #if ACOMPUTE_VERSION == 10000
    if (std::string(dprops->name).find("810E") != std::string::npos) {
        sm_count = 20;
    }
    if (is_sparse_attn) {
        block_size_m = 64;
        occupancy = 1;
    } else if (!is_sm89_or_newer()) {
        block_size_m = num_heads_per_head_k > 64 ? 128 : (num_heads_per_head_k <= 32 ? (num_heads_per_head_k + 16 - 1) / 16 * 16: 64);
        occupancy = block_size_m == 8 ? 7 : block_size_m == 16 ? 7 : block_size_m == 32 ? 4 : 1;
    } else {
        // btv105 small head size use corss split
        block_size_m = num_heads_per_head_k <= 16 ? 16 : num_heads_per_head_k <= 32 ? 32 : ((num_heads_per_head_k % 128 == 0) || (num_heads_per_head_k > 256)) ? 128 : 64;
        occupancy = 1;
    }
// #endif

    // to avoid too big empty sm split parts when batch is small
    int num_sm_parts = num_heads_per_head_k > 128 && batch < 4 ?
        (occupancy * sm_count) / num_heads_k / cutlass::ceil_div(num_heads_per_head_k, block_size_m) :
        (occupancy * sm_count) / gcd(cutlass::ceil_div(num_heads_per_head_k, block_size_m) * num_heads_k, occupancy * sm_count);

    // make sure num_sm_parts <= 320 && can be divided by sm_count
    num_sm_parts = num_sm_parts <= 320 ? num_sm_parts : ((320 / sm_count) * sm_count);
    return num_sm_parts;
}

std::vector<at::Tensor>
get_mla_metadata(
    const std::optional<at::Tensor> &seqlens_k,
    const int num_heads_per_head_k,
    const int num_heads_k
#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
    ,
    const std::optional<int> num_heads_q_,
    const bool is_fp8_kvcache,
    const std::optional<int> topk,
    const std::optional<int> extra_topk,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &extra_topk_length
#endif
) {
#ifndef FLASHMLA_C_ENABLE_DECODE_SPARSE
    const std::optional<int> num_heads_q_ = std::nullopt;
    bool is_fp8_kvcache = false;
    const std::optional<int> topk = std::nullopt;
    const std::optional<int> extra_topk = std::nullopt;
    const std::optional<at::Tensor> topk_length = std::nullopt;
    const std::optional<at::Tensor> extra_topk_length = std::nullopt;
#endif
    bool is_sparse_attn = topk.has_value();

    int batch_size = seqlens_k->size(0);
    int *seqlens_k_ptr = is_sparse_attn ? nullptr : seqlens_k->data_ptr<int>();
    auto options = seqlens_k->options();

    int num_tokens_per_head_k = num_heads_per_head_k;
    if (is_sparse_attn) {
        TORCH_CHECK(num_heads_q_.has_value(), "num_heads_q must be provided when topk is provided");
        // TORCH_CHECK(is_fp8_kvcache, "Sparse BF16 MLA is not supported on SM90");
        int num_heads_q = num_heads_q_.value();
        TORCH_CHECK(num_heads_q % num_heads_k == 0);
        num_tokens_per_head_k = num_heads_q / num_heads_k;
        // int seqlen_q_ori = num_heads_per_head_k * num_heads_k / num_heads_q;
        // batch_size_per_head_k = seqlen_q_ori * batch_size;
    } else {
        CHECK_DEVICE(seqlens_k.value());
        // TORCH_CHECK(seqlens_k->is_contiguous());
        TORCH_CHECK(seqlens_k->dtype() == torch::kInt32);
    }

    int num_sm_parts = get_num_sm_parts(num_tokens_per_head_k, num_heads_k, batch_size, is_sparse_attn);

    //static constexpr int block_size_n = 64;
    int block_size_n;
    if (is_sparse_attn) {
        block_size_n = 64;
    } else if (!is_sm89_or_newer()) {
        block_size_n = use_cross_cut(num_tokens_per_head_k, batch_size)
                     ? (num_tokens_per_head_k > 32 && num_tokens_per_head_k <= 64 ? 64 : 32) : 16;
    } else {
        // btv105 only use cross_cut method.
        block_size_n = 64;
    }

    static constexpr int fixed_overhead_num_blocks = 5;

    auto tile_scheduler_metadata = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::empty({batch_size + 1}, options);
    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();

    at::cuda::CUDAGuard device_guard{(char)seqlens_k->get_device()};
    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    Mla_metadata_params params = {};

    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n; // DeodeImplMeta->block_size_topk
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    params.topk = is_sparse_attn ? topk.value() : -1;
    params.extra_topk = extra_topk.has_value() ? extra_topk.value() : 0;
    params.topk_length = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
    params.extra_topk_length = extra_topk_length.has_value() ? extra_topk_length->data_ptr<int>() : nullptr;
    get_mla_metadata_func(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

int
flash_mla_get_workspace_size(
    const int batch_size,
    const int seqlen_q_ori,
    const int num_heads_per_head_k,
    const int num_heads_k,
    const int head_size_v
) {
    int num_sm_parts = get_num_sm_parts(num_heads_per_head_k, batch_size, num_heads_k);

    constexpr size_t size_per_elemnet_int = sizeof(int32_t);
    size_t metadata_workspace_size = size_per_elemnet_int * num_sm_parts * TileSchedulerMetaDataSize;
    size_t num_splits_workspace_size = size_per_elemnet_int * (batch_size + 1) * TileSchedulerMetaDataSize;

    constexpr size_t size_per_elemnet_float = sizeof(float);
    const int ngroups = num_heads_per_head_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    size_t softmax_lse_workspace_size = size_per_elemnet_float * batch_size * num_heads * seqlen_q;
    size_t softmax_lse_accum_workspace_size = size_per_elemnet_float * (batch_size + num_sm_parts) * num_heads * seqlen_q;
    size_t out_accum_workspace_size = size_per_elemnet_float * (batch_size + num_sm_parts) * num_heads * seqlen_q * head_size_v;

    size_t total_workspace_size = metadata_workspace_size + num_splits_workspace_size
        + softmax_lse_workspace_size + softmax_lse_accum_workspace_size + out_accum_workspace_size;

    return total_workspace_size;
}

std::vector<at::Tensor>
get_mla_metadata_with_workspace(
    const at::Tensor &seqlens_k,
    const int num_heads_per_head_k,
    const int num_heads_k,
    void *workspace_ptr,
    size_t &used_workspace_size,
    size_t max_workspace_size = std::numeric_limits<size_t>::max()
) {
    CHECK_DEVICE(seqlens_k);
    TORCH_CHECK(seqlens_k.is_contiguous());
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);

    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data_ptr<int>();
    auto options = seqlens_k.options();

    int num_sm_parts = get_num_sm_parts(num_heads_per_head_k, batch_size, num_heads_k);

    // static constexpr int block_size_n = 64;
    int block_size_n;
    if (!is_sm89_or_newer()) {
        block_size_n = use_cross_cut(num_heads_per_head_k, batch_size)
            ? num_heads_per_head_k > 32 && num_heads_per_head_k <= 64 ? 64 : 32 : 16;
    } else {
        // btv105 only use cross_cut method.
        block_size_n = 64;
    }

    static constexpr int fixed_overhead_num_blocks = 5;

    constexpr size_t size_per_elemnet = sizeof(int32_t);
    size_t metadata_workspace_size = size_per_elemnet * num_sm_parts * TileSchedulerMetaDataSize;
    used_workspace_size = metadata_workspace_size;
    void *metadata_workspace_ptr = workspace_ptr;
    size_t num_splits_workspace_size = size_per_elemnet * (batch_size + 1) * TileSchedulerMetaDataSize;
    void *num_splits_workspace_ptr = metadata_workspace_ptr + used_workspace_size;
    used_workspace_size += num_splits_workspace_size;
    TORCH_CHECK(used_workspace_size < max_workspace_size);

    // auto tile_scheduler_metadata = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
    // auto num_splits = torch::empty({batch_size + 1}, options);
    at::Tensor tile_scheduler_metadata = at::from_blob(metadata_workspace_ptr, {num_sm_parts, TileSchedulerMetaDataSize}, options);
    at::Tensor num_splits = at::from_blob(num_splits_workspace_ptr, {batch_size + 1}, options);

    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();

    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    params.topk = -1;
    params.extra_topk = 0;
    params.topk_length = nullptr;
    params.extra_topk_length =  nullptr;

    get_mla_metadata_func(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

int
mha_fwd_kvcache_mla_with_workspace(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    c10::optional<const at::Tensor> &vcache_,    // num_blocks x page_block_size x num_heads_k x head_size_v
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    c10::optional<at::Tensor> &out_,
    void *workspace_ptr,
    size_t max_workspace_size = std::numeric_limits<size_t>::max()
) {
    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    // bool is_sm75 = cc_major == 7 && cc_minor == 5;
    bool is_sm8x = cc_major == 8 && cc_minor >= 0;
    TORCH_CHECK(is_sm8x);

    at::Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == torch::kInt64, "block_table must have dtype torch.int64");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_ori = sizes[2];
    const int head_size = sizes[3];
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    //const int seqlen_k = max_num_blocks_per_seq * page_block_size;
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
            .reshape({batch_size, seqlen_q, num_heads, head_size});

    int head_size_k = head_size;
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    if (vcache_.has_value()) { CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_v); }
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);

    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    CHECK_DEVICE(seqlens_k);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(seqlens_k, batch_size);

    auto opts = q.options();
    // at::Tensor out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts);
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have head_size_v last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
    } else {
        out = torch::empty({ batch_size, seqlen_q, num_heads, head_size_v }, opts);
    }

    // at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    constexpr size_t size_per_elemnet = sizeof(float);
    size_t softmax_lse_workspace_size = size_per_elemnet * batch_size * num_heads * seqlen_q;
    size_t used_workspace_size = softmax_lse_workspace_size;
    void *softmax_lse_workspace_ptr = workspace_ptr;
    at::Tensor softmax_lse = at::from_blob(softmax_lse_workspace_ptr, {batch_size, num_heads, seqlen_q}, at::dtype(at::kFloat).device(at::kCUDA));

    Flash_fwd_params params {};

    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.cu_seqlens_k = seqlens_k.data_ptr<int>();
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;

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
    params.block_table = nullptr;
    params.hllm_block_table = block_table.data_ptr<int64_t>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;
    params.num_blocks = num_blocks;
    //params.seqlen_k = seqlen_k;
    params.workspace_ptr = workspace_ptr;
    params.max_workspace_size = max_workspace_size;

    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    // export PPU_LIB_SHOW_PARAMS=1
    ppu::fmha::FmhaProfParam fmha_prof_params;
    if (ppu::fmha::ProfilingInterface::Instance().get_op_info()){
        // check if cuda graph captured
        hggcStreamCaptureStatus captureStatus;
        hggcStreamIsCapturing((hggcStream_t)stream, &captureStatus);

        std::string seqlen_kv_str;
        if (captureStatus != hggcStreamCaptureStatusNone) {
            // printf("dump info not supported in cuda graph mode\n");
            GraphCaptureModeSuspender protector(hggcStreamCaptureModeRelaxed);

            hggcStream_t side_stream;
            // Create a temporary side stream not associated with the capture
            if (hggcStreamCreateWithFlags(&side_stream, hggcStreamNonBlocking) == hggcSuccess) {
                std::vector<int> tmp(params.b);

                // Perform Device-to-Host copy on the side stream
                // This won't be recorded in the graph
                hggcMemcpyAsync(tmp.data(), params.cu_seqlens_k, sizeof(int) * params.b, hggcMemcpyDeviceToHost, side_stream);

                // Synchronize the side stream (Safe because of Relaxed mode)
                hggcStreamSynchronize(side_stream);

                // Construct the string for profiling/UT analysis
                std::ostringstream oss;
                oss << "[";
                for (int i = 0; i < params.b; ++i) {
                    oss << tmp[i];
                    if (i < params.b - 1) oss << ",";
                }
                oss << "]";
                seqlen_kv_str = oss.str();

                hggcStreamDestroy(side_stream);
            }
        } else {
            std::vector<int> tmp(params.b);
            hggcMemcpyAsync(tmp.data(), params.cu_seqlens_k, sizeof(int) * params.b, hggcMemcpyDeviceToHost, (hggcStream_t)stream);

            hggcStreamSynchronize((hggcStream_t)stream);

            std::ostringstream oss;
            oss << "[";
            for (int i = 0; i < params.b; ++i) {
                oss << tmp[i];
                if (i < params.b - 1) oss << ",";
            }
            oss << "]";
            seqlen_kv_str = oss.str();
        }

        fmha_prof_params.set_flash_attn_params(
                q_dtype == torch::kFloat16/*data_type*/,
                params.is_causal/*custom_mask*/, params.b/*batch_size*/,
                num_heads_ori/*num_heads*/, num_heads_k/*num_heads_k*/,
                params.d/*head_dim*/, params.d_v/*head_dim_value*/,
                seqlen_q_ori/*seqlen_q*/, seqlen_kv_str/*seqlen_kv*/
            );
    }

    // tile_scheduler
    TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");
    CHECK_DEVICE(num_splits);
    CHECK_CONTIGUOUS(num_splits);
    params.num_splits_ptr = num_splits.data_ptr<int>();
    // at::Tensor softmax_lse_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    // at::Tensor out_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q, head_size_v}, opts.dtype(at::kFloat));
    size_t softmax_lse_accum_workspace_size = size_per_elemnet * (batch_size + params.num_sm_parts) * num_heads * seqlen_q;
    void *softmax_lse_accum_workspace_ptr = workspace_ptr + used_workspace_size;
    used_workspace_size += softmax_lse_accum_workspace_size;
    at::Tensor softmax_lse_accum = at::from_blob(softmax_lse_accum_workspace_ptr, {batch_size + params.num_sm_parts, num_heads, seqlen_q}, at::dtype(at::kFloat).device(at::kCUDA));
    size_t out_accum_workspace_size = size_per_elemnet * (batch_size + params.num_sm_parts) * num_heads * seqlen_q * head_size_v;
    void *out_accum_workspace_ptr = workspace_ptr + used_workspace_size;
    at::Tensor out_accum = at::from_blob(out_accum_workspace_ptr, {batch_size + params.num_sm_parts, num_heads, seqlen_q, head_size_v}, at::dtype(at::kFloat).device(at::kCUDA));
    used_workspace_size += out_accum_workspace_size;
    TORCH_CHECK(used_workspace_size < max_workspace_size);

    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();
    // splitkv
    //at::Tensor softmax_lse_accum, out_accum;
    //std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
    //    params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
    //    head_size, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);

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
    out = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});

    return 0;
}

inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        TORCH_CHECK(false, "[Sparse TopK Attention] Stride exceeds int32 limit: ", orig_stride);
    }
    return static_cast<int>(orig_stride);
}

std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,           // seqlen_q x num_heads x head_size
    const at::Tensor &kv,          // seqlen_k x num_heads_k x head_size
    const at::Tensor &indices,     // seqlen_q x num_heads_k x top_k
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &out_
) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor >= 0;
    TORCH_CHECK(is_sm8x, "Sparse Attention Forward Kernel (sparse_prefill_fwd) is only supported on SM8x architectures");
    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);

    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.dtype() == torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kv.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(indices.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    // at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    // CHECK_CONTIGUOUS(out);
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        KU_CHECK_DTYPE(out, torch::kBFloat16);
        KU_CHECK_SHAPE(out, s_q, h_q, d_v);
        KU_CHECK_LAST_DIM_CONTIGUOUS(out);
        KU_CHECK_DEVICE(out);
    } else {
        out = torch::empty({s_q, h_q, d_v}, opts);
    }

    at::Tensor buf_attn_score, max_logits, lse, p_sum;
    max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    CHECK_CONTIGUOUS(max_logits);
    CHECK_CONTIGUOUS(lse);

    SparsePrefillParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * 1.44269504f,

        (void*)q.data_ptr(),
        (void*)kv.data_ptr(),
        (int*)indices.data_ptr(),

        // attn_sink.has_value() ? (float*)attn_sink.value().data_ptr<float>() : nullptr,
        // topk_length.has_value() ? (int*)topk_length.value().data_ptr<int>() : nullptr,
        get_optional_tensor_ptr<float>(attn_sink),
        get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (void*)out.data_ptr(),
        (void*)max_logits.data_ptr(),
        (void*)lse.data_ptr(),

        (hggcStream_t)at::cuda::getCurrentCUDAStream().stream()
    };
    //TODO:   // export PPU_LIB_SHOW_PARAMS=1
    ppu::fmha::FmhaProfParam fmha_prof_params;
    if (ppu::fmha::ProfilingInterface::Instance().get_op_info()){
        // check if cuda graph captured
        hggcStreamCaptureStatus captureStatus;
        hggcStreamIsCapturing((hggcStream_t)params.stream, &captureStatus);
        std::string topk_len_str = "";
        if (params.topk_length) {
            topk_len_str = fmha_prof_params.nvtx_param2str<int>(params.topk_length, params.s_q, (hggcStream_t)params.stream);
        }
        fmha_prof_params.set_flash_attn_sparse_prefill_params(
            q.dtype() == torch::kBFloat16/*data_type*/,
            params.h_q/*num_heads*/, params.h_kv/*num_heads_k*/,
            params.d_qk/*head_dim*/, params.d_v/*head_dim_value*/,
            params.s_q/*seqlen_q*/, params.s_kv/*seqlen_k*/, params.topk,
            bool(params.attn_sink), topk_len_str
        );
    }
    ppu::fmha::ProfilingInterface::Instance().instrument(true, fmha_prof_params);
    run_sparse_prefill_fwd_dispatch<cutlass::bfloat16_t>(params);
    ppu::fmha::ProfilingInterface::Instance().instrument(false, fmha_prof_params);

    return {out, max_logits, lse};
}

#ifndef FLASH_MLA_CPP_INFER_BUILD

#ifdef FLASH_MLA_STANDALONE_BUILD

#include <torch/python.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("get_mla_metadata", &get_mla_metadata);
    m.def("fwd_kvcache_mla", &mha_fwd_kvcache_mla);
    m.def("sparse_prefill_fwd", &sparse_prefill_fwd);
}

#else

#include <Python.h>
#include "pytorch_shim.h"

TORCH_LIBRARY(_flashmla_C, m) {
    m.def("get_mla_metadata", make_pytorch_shim(&get_mla_metadata));
    m.impl("get_mla_metadata", torch::kCUDA, make_pytorch_shim(&get_mla_metadata));

    m.def("fwd_kvcache_mla", make_pytorch_shim(&mha_fwd_kvcache_mla));
    m.impl("fwd_kvcache_mla", torch::kCUDA, make_pytorch_shim(&mha_fwd_kvcache_mla));

    m.def("sparse_prefill_fwd", make_pytorch_shim(&sparse_prefill_fwd));
    m.impl("sparse_prefill_fwd", torch::kCUDA, make_pytorch_shim(&sparse_prefill_fwd));
}

PyMODINIT_FUNC PyInit__flashmla_C() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_flashmla_C", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
#endif // FLASH_MLA_STANDALONE_BUILD

#endif // FLASH_MLA_CPP_INFER_BUILD
