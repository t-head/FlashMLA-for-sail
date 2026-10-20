#pragma once

#include "params.h"

namespace flashmla::v41prefill {

// Decode and Prefill share the paired-cube quantized WI tile engine. Decode
// retains its metadata partitions, FP32 partial outputs, and combine kernel.
void run_sparse_quant_decode_wi_fp8(Flash_fwd_params &, hggcStream_t);
void run_sparse_quant_decode_wi_fp4_extra(Flash_fwd_params &, hggcStream_t);

} // namespace flashmla::v41prefill
