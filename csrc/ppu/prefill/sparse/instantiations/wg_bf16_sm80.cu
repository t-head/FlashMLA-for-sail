#include "../sparse_prefill.h"

template void run_flash_sparse_prefill_fwd_wg<cutlass::bfloat16_t, 80, 576>(SparsePrefillParams &params);
template void run_flash_sparse_prefill_fwd_wg<cutlass::bfloat16_t, 80, 512>(SparsePrefillParams &params);
