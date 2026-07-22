#pragma once

// /usr/local/cuda/include/cuda_math_constants.h
#define CUDART_LN2_F 0.693147180559945309f

namespace Config {

static constexpr int BLOCK_SIZE_M = 128;
static constexpr int PAGE_BLOCK_SIZE = 64;
static constexpr int BLOCK_SIZE_N = 32;

static constexpr int HEAD_DIM_K = 576;
static constexpr int HEAD_DIM_V = 512;

}

enum NamedBarriers : int {
    sScale0Ready = 5,
    sScale1Ready = 6,
    sP0Ready = 7,
    rO1sP0sV0RIssued = 8,
    sMInitialized = 9,
    mGroup0 = 10,
    mGroup1 = 11,
    tma_copy_issued = 12
};
