#pragma once

#include "flash.h"

template<typename InputT, int Arch, int HEAD_DIM>
void run_flash_sparse_prefill_fwd_wg(SparsePrefillParams &params);