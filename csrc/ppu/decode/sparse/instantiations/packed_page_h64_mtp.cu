#include "decode/sparse/sparse_decode_wg_hs64.cuh"

namespace flashmla::dsa::hs64 {
static bool supports_v41_packed_flat_mtp(const Flash_fwd_params &params) {
    const int sq = params.q_orig;
    if (!(sq == 3 || sq == 4 || sq == 5) || params.b <= 0 || params.b > 65535 / sq ||
        params.d != 512 || params.d_v != 512 || params.h != 1 || params.ngroups != 64 ||
        params.seqlen_q != 64 * sq || params.topk != 128 || params.extra_topk != 512 ||
        params.page_block_size != 128 || params.k_batch_stride != 128 * 528 ||
        params.extra_page_block_size != 64 || params.extra_k_batch_stride != 64 * 288)
        return false;
    const auto affine = [](int64_t batch_pitch, int64_t row_pitch, int rows) {
        return batch_pitch > 0 && row_pitch > 0 &&
               batch_pitch % rows == 0 && batch_pitch / rows == row_pitch;
    };
    return affine(params.q_batch_stride, params.q_row_stride, sq * 64) &&
           affine(params.o_batch_stride, params.o_row_stride, sq * 64) &&
           affine(params.indices_batch_stride, params.indices_row_stride, sq) &&
           affine(params.extra_indices_batch_stride, params.extra_indices_row_stride, sq);
}

void run_v41_packed_page_direct_mtp(Flash_fwd_params &params, hggcStream_t stream) {
    // Sq1 enters this wrapper only on the validated 39-CU target. Reserve
    // the resident worklist for at least sixteen full waves of independent rows.
    if (params.q_orig == 1 &&
        params.b >= 16 * 39) {
        launch_flash_sparse_decode_wg_hs64<Hs64PackedDecode<128, Hs64Worklist::Batch>, true>(params, stream);
        return;
    }
    const int sq = params.q_orig;
    if (supports_v41_packed_flat_mtp(params)) {
        // Public Q, O, index and LSE buffers are affine in t = b * Sq + q.
        // Both optional length arrays still have B elements. q_orig retains Sq.
        Flash_fwd_params view = params;
        view.b *= sq;
        view.seqlen_q = 64;
        view.q_batch_stride /= sq;
        view.o_batch_stride /= sq;
        view.indices_batch_stride = view.indices_row_stride;
        view.extra_indices_batch_stride = view.extra_indices_row_stride;
        launch_flash_sparse_decode_wg_hs64<Hs64PackedDecode<128, Hs64Worklist::Flat>, true>(view, stream);
        return;
    }
    launch_flash_sparse_decode_wg_hs64<Hs64PackedDecode<128, Hs64Worklist::Single>, true>(params, stream);
}
} // namespace flashmla::dsa::hs64
