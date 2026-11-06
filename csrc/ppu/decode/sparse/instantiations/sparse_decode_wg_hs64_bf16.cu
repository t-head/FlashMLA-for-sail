#include "decode/sparse/sparse_decode_wg_hs64.cuh"

template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<512, 80>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<576, 80>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<512, 89>(SparsePrefillParams &params);
template void flashmla::dsa::hs64::run_flash_sparse_prefill_fwd_hs64<576, 89>(SparsePrefillParams &params);

namespace flashmla::dsa::hs64 {
static bool supports_v41_packed_flat_mtp(const Flash_fwd_params &params, int extra_bytes) {
    const int sq = params.q_orig;
    if (!(sq == 3 || sq == 4 || sq == 5) || params.b <= 0 || params.b > 65535 / sq ||
        params.d != 512 || params.d_v != 512 || params.h != 1 || params.ngroups != 64 ||
        params.seqlen_q != 64 * sq || params.topk != 128 || params.extra_topk != 512 ||
        params.page_block_size != 128 || params.k_batch_stride != 128 * 528 ||
        params.extra_page_block_size != 64 || params.extra_k_batch_stride != 64 * extra_bytes)
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

template<int MainCapacity, Hs64Worklist W, bool ExtraFP4>
using PackedCodec = std::conditional_t<ExtraFP4, Hs64PackedDecode<MainCapacity, W>,
    std::conditional_t<MainCapacity == 128 && W == Hs64Worklist::Flat,
        Hs64PackedFp8ExtraDecode, Hs64PackedFp8ExtraWorklist<MainCapacity, W>>>;

template<bool ExtraFP4>
static void run_v41_packed_page_decode_typed(Flash_fwd_params &params, hggcStream_t stream) {
    if (params.topk == 256) {
        launch_flash_sparse_decode_wg_hs64<PackedCodec<256, Hs64Worklist::Batch, ExtraFP4>, true>(params, stream);
        return;
    }
    // Sq1 enters this wrapper only on the validated 39-CU target. Reserve
    // the resident worklist for at least sixteen full waves of independent rows.
    if (params.q_orig == 1 &&
        params.b >= 16 * 39) {
        launch_flash_sparse_decode_wg_hs64<PackedCodec<128, Hs64Worklist::Batch, ExtraFP4>, true>(params, stream);
        return;
    }
    const int sq = params.q_orig;
    if (supports_v41_packed_flat_mtp(params, ExtraFP4 ? 288 : 528)) {
        // Public Q, O, index and LSE buffers are affine in t = b * Sq + q.
        // Both optional length arrays still have B elements. q_orig retains Sq.
        Flash_fwd_params view = params;
        view.b *= sq;
        view.seqlen_q = 64;
        view.q_batch_stride /= sq;
        view.o_batch_stride /= sq;
        view.indices_batch_stride = view.indices_row_stride;
        view.extra_indices_batch_stride = view.extra_indices_row_stride;
        launch_flash_sparse_decode_wg_hs64<PackedCodec<128, Hs64Worklist::Flat, ExtraFP4>, true>(view, stream);
        return;
    }
    launch_flash_sparse_decode_wg_hs64<PackedCodec<128, Hs64Worklist::Single, ExtraFP4>, true>(params, stream);
}

void run_v41_packed_page_decode(Flash_fwd_params &params, hggcStream_t stream) {
    if (params.extra_k_row_stride == 528)
        run_v41_packed_page_decode_typed<false>(params, stream);
    else
        run_v41_packed_page_decode_typed<true>(params, stream);
}

} // namespace flashmla::dsa::hs64
