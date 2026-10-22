// Private asynchronous global validator. No host copy, allocation or sync.
#include <c10/cuda/CUDAException.h>
#include "../sparse_quant_prefill_gpu_checked_fp8.h"
namespace flashmla::v41prefill {
namespace {
constexpr unsigned kCheckBlocks = 128;
constexpr unsigned kCheckThreads = 256;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarps = kCheckThreads / kWarpSize;
__global__ void quant_allvalid_indices_check_kernel(SparseQuantPrefillParams params) {
    static_assert(sizeof(SparseQuantPrefillParams) == 152);
    __shared__ int warp_valid[kWarps];
    int lane_valid = 1;
    for (unsigned row = blockIdx.x; row < unsigned(params.s_q); row += kCheckBlocks) {
        const int *row_indices = params.indices + int64_t(row) * params.stride_indices_s_q;
        for (unsigned col = threadIdx.x; col < unsigned(params.topk); col += kCheckThreads) {
            const unsigned index = unsigned(row_indices[col]);
            lane_valid &= int(index < unsigned(params.s_kv));
        }
    }
    const int warp_allvalid = __all_sync(0xffffffffu, lane_valid);
    const unsigned lane = threadIdx.x & (kWarpSize - 1);
    const unsigned warp = threadIdx.x / kWarpSize;
    if (lane == 0) warp_valid[warp] = warp_allvalid;
    __syncthreads();
    if (warp == 0) {
        const int block_valid = lane < kWarps ? warp_valid[lane] : 1;
        const int block_allvalid = __all_sync(0xffffffffu, block_valid);
        if (lane == 0) {
            float *flags = static_cast<float *>(params.max_logits) + int64_t(params.s_q) * params.h_q;
            flags[blockIdx.x] = block_allvalid ? 1.0f : 0.0f;
        }
    }
}
}
void launch_sparse_quant_allvalid_tail_check(const SparseQuantPrefillParams &params) {
    quant_allvalid_indices_check_kernel<<<kCheckBlocks, kCheckThreads, 0, params.stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
