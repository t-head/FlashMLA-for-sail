#include "decode/sparse/sparse_decode_wg.cuh"

template<typename T>
static void launch_v32_packed_wg(Flash_fwd_params &params, hggcStream_t stream) {
    static_assert(T::kV32FP8 && !T::IsFP8 && T::kBytesPerToken == 656 &&
                  T::kMetadataStripeFactor == 1 &&
                  std::is_same_v<WgKAddr<T>, cutlass::bfloat16_t*>);
    constexpr size_t smem_size = std::max(
        sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));
    static_assert(smem_size <= 256 * 1024);
    auto kernel = &flash_sparse_decode_wg_kernel<T>;
    CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    const int num_m_blocks = params.seqlen_q / T::kBlockM;
    const int launch_parts = std::is_same_v<T, V32SingleTaskDecodeTraits>
        ? int(int64_t(params.b) + params.num_sm_parts - 1) : params.num_sm_parts;
    const char *show_log = std::getenv("show_log");
    if (show_log && isdigit(*show_log)) {
        hggcFuncAttributes attr;
        CHECK_CUDA(hggcFuncGetAttributes(&attr, kernel));
        printf("[sparse_decode_wg:v32_packed656_fp32_bf16_wg0_packet] "
               "QK9/PV8 PV4x2 grid=[%d,1,%d] smem=%zu vreg=%d stack=%zu metadata_quantum=64\n",
               num_m_blocks, launch_parts, smem_size, attr.numRegs, attr.localSizeBytes);
    }
    hggcLaunchAttribute attributes[1];
    attributes[0].id = hggcLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    hggcLaunchConfig_t config = {
        dim3(num_m_blocks, 1, launch_parts),
        dim3(T::NUM_THREADS, 1, 1), smem_size, stream, attributes, 1
    };
    hggcLaunchKernelEx(&config, kernel, params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

void run_flash_sparse_decode_wg_kernel_v32_fp8(Flash_fwd_params &params, hggcStream_t stream) {
    const int64_t tasks_bound = int64_t(params.b) + params.num_sm_parts - 1;
    const bool single_task = (params.q_orig == 4 || params.q_orig == 5) &&
        (params.b >= 128 || int64_t(params.b) >= 8 * int64_t(params.num_sm_parts)) && params.topk >= 2048 &&
        params.num_sm_parts > 1 && !params.topk_len_ptr &&
        tasks_bound > 0 && tasks_bound <= 65535;
    if (single_task) {
        launch_v32_packed_wg<V32SingleTaskDecodeTraits>(params, stream);
    } else {
        launch_v32_packed_wg<V32OptimizedDecodeTraits>(params, stream);
    }
    run_flash_mla_combine_kernel<cutlass::bfloat16_t>(params, stream);
}

// FP8 path: BF16 + sm89 instantiation (IsFP8=true, default).
// Uses BF16 in SMEM; FP8 is dequantised on the fly per warpgroup.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 128>(Flash_fwd_params &params, hggcStream_t stream);

// BF16 path: BF16 + sm89 instantiation (IsFP8=false).
// KV cache is contiguous BF16; no dequant needed.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, false, 128>(Flash_fwd_params &params, hggcStream_t stream);

// BF16 M64 is instantiated by sparse_decode_wg_hs64_bf16.cu. Keep the
// original generic BlockM=64 instantiation only for the unchanged FP8 path.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 64>(Flash_fwd_params &params, hggcStream_t stream);

// V4.1 quantizes all 512 dimensions; extra KV may use FP4 data with E4M3 scales.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 128, true, false>(Flash_fwd_params &params, hggcStream_t stream);
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 64, true, false>(Flash_fwd_params &params, hggcStream_t stream);
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 128, true, true>(Flash_fwd_params &params, hggcStream_t stream);
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 64, true, true>(Flash_fwd_params &params, hggcStream_t stream);
