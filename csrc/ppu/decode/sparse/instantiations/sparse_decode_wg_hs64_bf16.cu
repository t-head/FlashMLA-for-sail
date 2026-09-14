#include "decode/sparse/sparse_decode_wg_hs64.cuh"

template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<512>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<576>(SparsePrefillParams &params);
