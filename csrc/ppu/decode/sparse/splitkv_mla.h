#pragma once

#include "params.h"

template<typename T, bool IsFP8, int Headdim, int Headdim_V>
void run_sparse_decode_fwd_dispatch(Flash_fwd_params &params, hggcStream_t stream);
