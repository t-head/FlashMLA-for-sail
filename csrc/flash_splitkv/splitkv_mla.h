#pragma once

// #include "params.h"
#include "flash.h"

template<typename InputT, int Arch>
void run_flash_splitkv_mla_kernel(Flash_fwd_mla_params &params, hggcStream_t stream);
