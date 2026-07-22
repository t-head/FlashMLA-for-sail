#include "../combine.cuh"

template void run_flash_mla_combine_kernel<cutlass::bfloat16_t>(Flash_fwd_mla_params &params, hggcStream_t stream);
