#include "../kernel_h128.cuh"
#include "../kernel_small_prefill.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {
template void run_fused_norm_rope_attn_rope_cast_prefill_small<true>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Prefill, 128, true, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_prefill_h128<true>(const Params&);
}
