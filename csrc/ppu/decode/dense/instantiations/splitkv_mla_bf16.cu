#include "decode/dense/splitkv_mla.h"

template void run_flash_splitkv_mla_kernel<cutlass::bfloat16_t, 80>(Flash_fwd_mla_params &params, hggcStream_t stream);
template void run_flash_splitkv_mla_kernel<cutlass::bfloat16_t, 89>(Flash_fwd_mla_params &params, hggcStream_t stream);
