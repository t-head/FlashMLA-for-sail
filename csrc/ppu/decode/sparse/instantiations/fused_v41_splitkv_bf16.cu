#include "decode/sparse/fused_v41_splitkv.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<8, false, false>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<8, false, true>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<8, true, false>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<8, true, true>(const Params&, const SplitKVParams&);

template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, false, false>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, false, true>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, true, false>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<64, true, true>(const Params&, const SplitKVParams&);

template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<128, false, false>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<128, false, true>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<128, true, false>(const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_kernel<128, true, true>(const Params&, const SplitKVParams&);
}
