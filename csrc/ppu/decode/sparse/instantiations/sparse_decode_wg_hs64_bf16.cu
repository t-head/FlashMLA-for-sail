#include "decode/sparse/sparse_decode_wg_hs64.cuh"

template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<512, 80>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<576, 80>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<512, 89>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<576, 89>(SparsePrefillParams &params);
