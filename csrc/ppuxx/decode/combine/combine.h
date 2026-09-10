#pragma once

#include "params.h"

template<typename ElementT>
void run_flash_mla_combine_kernel(Flash_fwd_mla_params &params, hggcStream_t stream, bool lse_in_log2 = true);
