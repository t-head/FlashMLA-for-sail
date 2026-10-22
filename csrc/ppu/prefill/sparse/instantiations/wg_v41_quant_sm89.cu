/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/
#include "../quant_wi.cuh"

namespace flashmla::v41prefill {

template<bool FP4>
static void launch_quant_prefill(SparseQuantPrefillParams &params) {
    using T = QuantTraits<FP4>;
    auto kernel = &sparse_quant_wi_kernel<T, true>;
    constexpr size_t bytes = std::max(sizeof(typename T::SharedMemoryPlan),
                                      sizeof(typename T::SharedMemoryOutPut));
    static_assert(bytes == 262144);
    static_assert(sizeof(typename T::SharedMemoryPlan) == 248880);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(params.s_q, 1, 1), T::NUM_THREADS, bytes, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

void run_sparse_quant_prefill_fwd(SparseQuantPrefillParams &params, bool fp4) {
    if (fp4) {
        launch_quant_prefill<true>(params);
    } else {
        launch_quant_prefill<false>(params);
    }
}

} // namespace flashmla::v41prefill
