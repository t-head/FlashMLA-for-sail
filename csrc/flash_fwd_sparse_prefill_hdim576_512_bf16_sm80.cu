// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template void run_sparse_prefill_fwd_dispatch<cutlass::bfloat16_t>(SparsePrefillParams &params);

template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, false>(Flash_fwd_params &params, cudaStream_t stream);
template void run_sparse_decode_fwd_dispatch<cutlass::bfloat16_t, true>(Flash_fwd_params &params, cudaStream_t stream);