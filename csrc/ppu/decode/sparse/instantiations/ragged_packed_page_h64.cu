#include "decode/sparse/sparse_decode_wg_hs64.cuh"

namespace flashmla::dsa::hs64 {
void run_v41_packed_page_direct_ragged(Flash_fwd_params &params, hggcStream_t stream) {
    launch_flash_sparse_decode_wg_hs64<Hs64PackedDecode<256, Hs64Worklist::Batch>, true>(params, stream);
}
} // namespace flashmla::dsa::hs64
