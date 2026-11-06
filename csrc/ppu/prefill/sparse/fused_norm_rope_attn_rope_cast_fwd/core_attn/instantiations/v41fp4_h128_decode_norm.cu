#include "../kernel_native_decode.cuh"
#include "../kernel_splitkv.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {
template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Decode, 128, true, true>(const Params&);
template bool run_fused_norm_rope_attn_rope_cast_decode_native_wi<true>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<128, true, true>(const Params&, const SplitKVParams&);
}
