#pragma once
#include "sparse_quant_prefill.h"
namespace flashmla::v41prefill {
void run_sparse_quant_prefill_fwd_narrow32(SparseQuantPrefillParams &, bool fp4);
}
