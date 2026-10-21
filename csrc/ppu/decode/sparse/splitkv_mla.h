#pragma once

#include "params.h"

template<typename T, bool IsFP8, int Headdim, int Headdim_V,
         bool IsV41FP8 = false, bool IsExtraFP4 = false>
void run_sparse_decode_fwd_dispatch(Flash_fwd_params &params, hggcStream_t stream);
