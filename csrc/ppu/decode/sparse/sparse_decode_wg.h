#pragma once

#include "params.h"

template<typename InputT, int Arch, bool IsFP8 = true, int BlockM = 128>
void run_flash_sparse_decode_wg_kernel(Flash_fwd_params &params, hggcStream_t stream);
