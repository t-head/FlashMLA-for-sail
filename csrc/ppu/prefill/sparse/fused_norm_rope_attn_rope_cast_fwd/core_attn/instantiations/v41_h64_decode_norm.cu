#include "../kernel_hs64.cuh"
#include "../kernel_splitkv.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {
template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Decode, 64, true, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_hs64_packed<true, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, true, false>(const Params&, const SplitKVParams&);
}
