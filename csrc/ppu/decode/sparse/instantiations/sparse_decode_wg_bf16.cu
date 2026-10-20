#include "decode/sparse/sparse_decode_wg.cuh"

// FP8 path: BF16 + sm89 instantiation (IsFP8=true, default).
// Uses BF16 in SMEM; FP8 is dequantised on the fly per warpgroup.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 128>(Flash_fwd_params &params, hggcStream_t stream);

// BF16 path: BF16 + sm89 instantiation (IsFP8=false).
// KV cache is contiguous BF16; no dequant needed.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, false, 128>(Flash_fwd_params &params, hggcStream_t stream);

// BF16 M64 is instantiated by sparse_decode_wg_hs64_bf16.cu. Keep the
// original generic BlockM=64 instantiation only for the unchanged FP8 path.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 64>(Flash_fwd_params &params, hggcStream_t stream);

// V4.1 uses all 512 dimensions as quantized data; extra KV may be block-segmented FP4.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 128, true, false>(Flash_fwd_params &params, hggcStream_t stream);
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 64, true, false>(Flash_fwd_params &params, hggcStream_t stream);
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 128, true, true>(Flash_fwd_params &params, hggcStream_t stream);
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 64, true, true>(Flash_fwd_params &params, hggcStream_t stream);

// Common Prefill/Decode paired-cube quantized WI engine.
#include "decode/sparse/quant_decode_wi.cuh"
