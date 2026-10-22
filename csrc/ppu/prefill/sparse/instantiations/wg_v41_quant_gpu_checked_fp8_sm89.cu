#define FLASHMLA_V41_FP8_WG1_REMOTE_PV_FIRST_PREFETCH 1
#define FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32 1
#define FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE 1
// Private TU-local entry transformation. Existing global kernels are unchanged.
#define FLASHMLA_V41_QUANT_ADDRESS32 1
#define FLASHMLA_V41_QUANT_ALLVALID_NO_LENGTHS 1
#define FLASHMLA_V41_QUANT_DEVICE_BODY 1
#define FLASHMLA_V41_QUANT_FP8_PROVEN_PACKET_VALID 1
#include "../quant_wi.cuh"
#include "../sparse_quant_prefill_gpu_checked_fp8.h"
namespace flashmla::v41prefill {
template<bool AllValid> struct GpuCheckedFP8NarrowTraits : QuantTraits<false> {
    static constexpr bool kNarrowAddress = true;
    static constexpr bool kAllValidNoLengths = AllValid;
};
__global__ void __launch_bounds__(512, 1, 1)
quant_prefill_wi_gpu_checked_fp8_kernel(__grid_constant__ const SparseQuantPrefillParams params) {
    static_assert(sizeof(SparseQuantPrefillParams) == 152);
    const float *flags = static_cast<const float *>(params.max_logits) + int64_t(params.s_q) * params.h_q;
    const unsigned lane = threadIdx.x & 31u;
    int lane_allvalid = 1;
    // Every full warp reads the same 128 immutable flags. The warp AND is
    // therefore identical across all 16 warps and both warpgroups of this CTA.
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        lane_allvalid &= int(flags[lane + 32u * i] == 1.0f);
    }
    const int warp_allvalid = __all_sync(0xffffffffu, lane_allvalid);
    const int allvalid = __builtin_ppu_to_uniform_b32(warp_allvalid);
    if (allvalid) {
        quant_prefill_wi_device_body<GpuCheckedFP8NarrowTraits<true>, true>(params);
        return;
    }
    quant_prefill_wi_device_body<GpuCheckedFP8NarrowTraits<false>, true>(params);
}
void run_sparse_quant_prefill_gpu_checked_fp8(SparseQuantPrefillParams &params) {
    using TrueT = GpuCheckedFP8NarrowTraits<true>;
    using FalseT = GpuCheckedFP8NarrowTraits<false>;
    constexpr size_t bytes = std::max(sizeof(typename TrueT::SharedMemoryPlan),sizeof(typename TrueT::SharedMemoryOutPut));
    static_assert(bytes == 262144 && sizeof(typename TrueT::SharedMemoryPlan) == 248880);
    static_assert(sizeof(typename TrueT::SharedMemoryPlan) == sizeof(typename FalseT::SharedMemoryPlan));
    static_assert(sizeof(typename TrueT::SharedMemoryOutPut) == sizeof(typename FalseT::SharedMemoryOutPut));
    static_assert(TrueT::NUM_THREADS == 512 && FalseT::NUM_THREADS == 512);
    static_assert(!TrueT::kFP4Cache && !FalseT::kFP4Cache);
    C10_CUDA_CHECK(cudaFuncSetAttribute(quant_prefill_wi_gpu_checked_fp8_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    launch_sparse_quant_allvalid_tail_check(params);
    quant_prefill_wi_gpu_checked_fp8_kernel<<<dim3(params.s_q,1,1),512,bytes,params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
}
