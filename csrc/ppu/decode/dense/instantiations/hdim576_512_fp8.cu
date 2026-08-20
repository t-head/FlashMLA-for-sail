#include "decode/dense/splitkv_mla.h"

template void run_mha_fwd_splithd_splitkv_dispatch<cutlass::float_e4m3_t, 576, 512>(Flash_fwd_params &params, hggcStream_t stream);
