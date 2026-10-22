#pragma once
#include "sparse_quant_prefill.h"
namespace flashmla::v41prefill {
void run_sparse_quant_prefill_v40_fwd(SparseQuantPrefillParams &);
void run_sparse_quant_prefill_v40_fwd_narrow32(SparseQuantPrefillParams &);
}
