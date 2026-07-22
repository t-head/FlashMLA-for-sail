#include "decode/sparse/splitkv_mla.cuh"

template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, false, 576, 512>(Flash_fwd_params &params, hggcStream_t stream);
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true, 576, 512>(Flash_fwd_params &params, hggcStream_t stream);
