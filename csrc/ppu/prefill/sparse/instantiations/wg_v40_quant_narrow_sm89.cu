#define FLASHMLA_V40_PREFILL_MODEL1 1
#define FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE 1
#define FLASHMLA_V41_QUANT_ADDRESS32 1
#include "../quant_traits.h"
namespace flashmla::v41prefill {
struct V40QuantNarrowTraits : QuantTraits<false> {
    static constexpr bool kV40Cache = true;
};
}
#include "../quant_wi.cuh"
#include "../sparse_quant_prefill_v40.h"
namespace flashmla::v41prefill {
void run_sparse_quant_prefill_v40_fwd_narrow32(SparseQuantPrefillParams &params) {
    using T=V40QuantNarrowTraits;
    auto kernel=&sparse_quant_wi_kernel<T,true>;
    constexpr size_t bytes=std::max(sizeof(typename T::SharedMemoryPlan),sizeof(typename T::SharedMemoryOutPut));
    static_assert(bytes==262144 && sizeof(typename T::SharedMemoryPlan)==248880);
    static_assert(sizeof(SparseQuantPrefillParams)==152);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    kernel<<<dim3(params.s_q,1,1),T::NUM_THREADS,bytes,params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
}
