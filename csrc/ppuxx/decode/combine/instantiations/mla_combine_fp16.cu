#ifndef FLASH_MLA_DISABLE_FP16
#include "../combine.cuh"

template void run_flash_mla_combine_kernel<cutlass::half_t>(Flash_fwd_mla_params &params, hggcStream_t stream);
#endif
