#include "decode/sparse/sparse_decode_wg_hs64.cuh"

template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<512, 80>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<576, 80>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<512, 89>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<576, 89>(SparsePrefillParams &params);

namespace flashmla::dsa::hs64 {
void run_flash_sparse_decode_wg_kernel_hs64(
    Flash_fwd_params &params, hggcStream_t stream,
    bool is_v41_fp8, bool is_extra_fp4) {
    if (is_v41_fp8) {
        FLASH_ASSERT(params.d == 512 && params.ngroups == 64 && params.q_orig > 1);
        if (is_extra_fp4) {
            run_flash_sparse_decode_wg_kernel_hs64_hdim<512, true, 89, true, true>(
                params, stream);
        } else {
            run_flash_sparse_decode_wg_kernel_hs64_hdim<512, true, 89, true, false>(
                params, stream);
        }
        return;
    }
    const auto [major, minor] = get_compute_capability(get_current_device());
    if (major > 8 || (major == 8 && minor >= 9)) {
        run_flash_sparse_decode_wg_kernel_hs64_arch<89>(params, stream);
    } else {
        run_flash_sparse_decode_wg_kernel_hs64_arch<80>(params, stream);
    }
}
} // namespace flashmla::dsa::hs64
