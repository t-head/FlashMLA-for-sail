#ifndef FLASH_MLA_DISABLE_FP16
#include "../sparse_prefill.h"

template void run_flash_sparse_prefill_fwd_wg<cutlass::half_t, 89, 576>(SparsePrefillParams &params);
template void run_flash_sparse_prefill_fwd_wg<cutlass::half_t, 89, 512>(SparsePrefillParams &params);
#endif
