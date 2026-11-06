#include "../kernel_hs64.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {
template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Prefill, 64, true, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_prefill_hs64<true>(const Params&);
}
