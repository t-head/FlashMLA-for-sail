#define FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32 1
#define FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE 1
// The private macro applies only to this TU; never a global compiler flag.
#define FLASHMLA_V41_QUANT_ADDRESS32 1
#include "../quant_wi.cuh"
#include "../sparse_quant_prefill_narrow.h"

namespace flashmla::v41prefill {
template<bool FP4> struct QuantNarrowAddressTraits : QuantTraits<FP4> {
    static constexpr bool kNarrowAddress = true;
};
template<bool FP4> static void launch_quant_prefill_narrow32(SparseQuantPrefillParams &params) {
    using T = QuantNarrowAddressTraits<FP4>;
    auto kernel = &sparse_quant_wi_kernel<T, true>;
    constexpr size_t bytes = std::max(sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));
    static_assert(bytes == 262144);
    static_assert(sizeof(typename T::SharedMemoryPlan) == 248880);
    static_assert(sizeof(SparseQuantPrefillParams) == 152);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(params.s_q, 1, 1), T::NUM_THREADS, bytes, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
void run_sparse_quant_prefill_fwd_narrow32(SparseQuantPrefillParams &params, bool fp4) {
    if (fp4) launch_quant_prefill_narrow32<true>(params);
    else launch_quant_prefill_narrow32<false>(params);
}
}
