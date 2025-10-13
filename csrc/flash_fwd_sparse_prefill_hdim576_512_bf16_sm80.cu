// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template void run_sparse_prefill_fwd_dispatch<cutlass::bfloat16_t>(const SparsePrefillParams &params);
