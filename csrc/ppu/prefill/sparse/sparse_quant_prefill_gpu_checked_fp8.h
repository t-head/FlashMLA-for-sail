#pragma once
#include "sparse_quant_prefill.h"
namespace flashmla::v41prefill {
// Private FP8 route: flags occupy one extra row beyond max_logits logical Sq.
// Checker and checked attention enqueue in the same stream, with no host read.
void launch_sparse_quant_allvalid_tail_check(const SparseQuantPrefillParams &params);
void run_sparse_quant_prefill_gpu_checked_fp8(SparseQuantPrefillParams &params);
}
