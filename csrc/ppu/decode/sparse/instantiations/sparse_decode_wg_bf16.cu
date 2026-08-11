#include "decode/sparse/sparse_decode_wg.cuh"
// Declaration with the default template arguments (IsFP8=true, BlockM=128) used
// by the explicit instantiations below; the .cuh does not include it.
#include "decode/sparse/sparse_decode_wg.h"

// FP8 path: BF16 + sm89 instantiation (IsFP8=true, default).
// Uses BF16 in SMEM; FP8 is dequantised on the fly per warpgroup.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true>(Flash_fwd_params &params, hggcStream_t stream);

// BF16 path: BF16 + sm89 instantiation (IsFP8=false).
// KV cache is contiguous BF16; no dequant needed.
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, false>(Flash_fwd_params &params, hggcStream_t stream);

// BlockM=64 Cross layout instantiations
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, true, 64>(Flash_fwd_params &params, hggcStream_t stream);
template void run_flash_sparse_decode_wg_kernel<cutlass::bfloat16_t, 89, false, 64>(Flash_fwd_params &params, hggcStream_t stream);
