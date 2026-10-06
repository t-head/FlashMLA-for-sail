/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
// Forward-declare hggcStream_t so function signatures match between .cu and .cpp
typedef struct HGstream_st* hggcStream_t;
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>
#include <algorithm>
#include <limits>

#include "kerutils/host/host.h"
#include "kerutils/host/hardware_info.h"
#include "fmha_profiling_interface.hpp"
#include "kerutils/supplemental/torch_tensors.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static bool use_cross_cut(int num_heads_per_head_k, int batch_size) {
    if (!is_sm89_or_newer()) {
        if (num_heads_per_head_k <= 16) {
            return false;
        } else if (num_heads_per_head_k <= 32) {
            return batch_size >= 8;
        } else if (num_heads_per_head_k <= 64) {
            return batch_size >= 2;
        } else {
            return true;
        }
    } else {
        return true;
    }
}

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

static constexpr float LOG_2_E = 1.44269504f;

inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        TORCH_CHECK(false, "[Sparse TopK Attention] Stride exceeds int32 limit: ", orig_stride);
    }
    return static_cast<int>(orig_stride);
}

static inline int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

static inline int
get_num_sm_parts(
    const int num_heads_per_head_k,
    const int num_heads_k,
    const int batch,
    bool is_sparse_attn
) {
    // This should match the logic in the MLA kernel.
    //static constexpr int block_size_m = 64;
    int block_size_m;

    auto dprops = at::cuda::getCurrentDeviceProperties();
    int sm_count = dprops->multiProcessorCount;
    // static set occupancy priori knowledge.
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
        // btv105 small head size use cross split
        block_size_m = num_heads_per_head_k <= 16 ? 16 : num_heads_per_head_k <= 32 ? 32 : ((num_heads_per_head_k % 128 == 0) || (num_heads_per_head_k > 256)) ? 128 : 64;
        occupancy = 1;
    }
// #endif

    // to avoid too big empty sm split parts when batch is small
    int num_sm_parts = num_heads_per_head_k > 128 && batch < 4 ?
        std::max(1, (occupancy * sm_count) / num_heads_k / cutlass::ceil_div(num_heads_per_head_k, block_size_m)) :
        (occupancy * sm_count) / gcd(cutlass::ceil_div(num_heads_per_head_k, block_size_m) * num_heads_k, occupancy * sm_count);

    // make sure num_sm_parts <= 320 && can be divided by sm_count
    num_sm_parts = num_sm_parts <= 320 ? num_sm_parts : ((320 / sm_count) * sm_count);
    return num_sm_parts;
}
