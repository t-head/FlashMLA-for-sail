// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "decode/sparse/splitkv_mla.cuh"

template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, false, 512, 512, false, false>(Flash_fwd_params &params, hggcStream_t stream);
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 512, 512, false, false>(Flash_fwd_params &params, hggcStream_t stream);
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 512, 512, true, false>(Flash_fwd_params &params, hggcStream_t stream);
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 512, 512, true, true>(Flash_fwd_params &params, hggcStream_t stream);

namespace flashmla::dsa::v41 {
namespace {
using H8DirectTraits = Flash_fwd_kernel_traits<
    512, 16, 32, 8, true, false, cutlass::bfloat16_t, 512,
    true, false, 1, 1, 32, 2, 2, true>;

__global__ void __launch_bounds__(H8DirectTraits::kNThreads, 1, 1)
flash_sparse_decode_fwd_h8_fp8_fp4_direct_kernel(__grid_constant__ const Flash_fwd_params params) {
    constexpr int kMainTopK = 128;
    constexpr int kExtraTopK = 512;
    constexpr int kBlockN = H8DirectTraits::kBlockN;
    const int part = blockIdx.z;
    const int queries = params.b * params.q_orig;
    const int rows = queries / params.num_sm_parts;
    const int remainder = queries % params.num_sm_parts;
    const int begin = part * rows + cute::min(part, remainder);
    const int end = begin + rows + int(part < remainder);
    #pragma unroll 1
    for (int token = begin; token < end; ++token) {
        // Each CTA owns complete queries and writes BF16 output directly.
        const int batch = token / params.q_orig;
        const int query = token % params.q_orig;
        flash::compute_attn_fp8_sparse_splitkv<H8DirectTraits, true, true, true>(
            params, batch, 0, query, 0, 0, (kMainTopK + kExtraTopK) / kBlockN,
            true, kMainTopK, kMainTopK / kBlockN, kExtraTopK);
        __syncthreads();
    }
}
} // namespace

void run_h8_fp8_fp4_direct(Flash_fwd_params &params, hggcStream_t stream) {
    constexpr auto shared_bytes = H8DirectTraits::kSmemSizeAccum
        + H8DirectTraits::kBlockN * 2 * sizeof(int);
    auto kernel = &flash_sparse_decode_fwd_h8_fp8_fp4_direct_kernel;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    kernel<<<dim3(1, 1, params.num_sm_parts), H8DirectTraits::kNThreads, shared_bytes, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
} // namespace flashmla::dsa::v41
