// FP8 five-in-one instantiation (h64)
//
// Mirrors the splitkv instantiations in fused_v41_h64_bf16.cu but selects the
// FP8-compute entry point: Q is quantized to E4M3 after RoPE and both GEMMs run
// on the PPU1.5 E4M3xE4M3->F32 atom (PPU0015_16x16x32_F32E4M3E4M3F32_TN, K=32).
//
// The ACOMPUTE_VERSION raise is required and must come FIRST: the wheel is built
// with -DACOMPUTE_VERSION=10000, and kernel_traits.h only selects the PPU1.5 FP8
// atom / 128-wide swizzle layouts under ACOMPUTE_VERSION >= 10500.  This is the
// same whole-TU override that csrc/ppu/sparse_native_fp8_arch.h applies for the
// native FP8 kernels and that sparse_decode_wg_traits.h applies around the HS64
// atom includes.  Without it the FP8 path silently degrades to the BF16/PPU1.0
// configuration (see evidence/feasibility/README.md, the "elided body" trap).
#if defined(ACOMPUTE_VERSION)
#undef ACOMPUTE_VERSION
#endif
#define ACOMPUTE_VERSION 10500

#include "sparse/fused_v41_hs64.cuh"
#include "decode/sparse/fused_v41_splitkv.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

// Norm x ExtraFP4 coverage: an earlier probe measured EnableNorm=false in this build, and
// found BOTH ExtraFP4 instances live (they come from whether a layer is
// compressed), so all four combinations are instantiated.
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_fp8_kernel<64, false, false>(
    const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_fp8_kernel<64, true, false>(
    const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_fp8_kernel<64, false, true>(
    const Params&, const SplitKVParams&);
template void run_fused_norm_rope_attn_rope_cast_decode_splitkv_fp8_kernel<64, true, true>(
    const Params&, const SplitKVParams&);

} // namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn