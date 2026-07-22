#include "prefill/sparse/sparse_prefill.h"

template void run_sparse_prefill_fwd_dispatch<cutlass::bfloat16_t>(SparsePrefillParams &params);
