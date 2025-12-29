#pragma once

// #include "params.h"
#include "flash.h"

template<typename ElementT>
void run_flash_mla_combine_kernel(Flash_fwd_params &params, cudaStream_t stream);
