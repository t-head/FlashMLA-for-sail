#include "../kernel.cuh"
#include "../kernel_splitkv.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {
template void run_fused_norm_rope_attn_rope_cast_fwd_kernel<FwdMode::Decode, 128, false, false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<128, false, false>(const Params&, const SplitKVParams&);
}
