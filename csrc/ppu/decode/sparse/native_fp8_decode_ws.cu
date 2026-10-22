// Native FP8 sparse decode/verify warp-specialized kernels and launchers.

#include "ppu/sparse_native_fp8_attention.cuh"

void run_native_d512_h8_fallback(const void *, const void *, const int *, void *, float *, float *,
    const float *, const int *, const int *, float, int, int, int, hggcStream_t, int, int);

void run_native_d512_h16_fallback(const void *, const void *, const int *, void *, float *, float *,
    const float *, const int *, const int *, float, int, int, int, hggcStream_t, int, int);

void run_native_d512_dual_compact(const void *, const void *, const int *, const void *,
    const int *, void *, float *, const float *, const int *, const int *, const int *,
    float, int, int, int, int, int, int, hggcStream_t, int, int);

namespace flashmla_fp8_attention {
using namespace flashmla::sparse_native_fp8::ws;
using Pipeline = flashmla::sparse_native_fp8::ws::Pipeline<DecodePolicy>;
// WI = Warp-Interleave: overlap producer/consumer work using the shared-slot
// handoff schedule implemented by produce() and consume() in the header.
template <int D, int Stages, int BM, int Rows, bool Late = false, int Prefix = 0,
    bool Dual = false, bool Gated = false, int GatherLanes = 0, bool PairedV = false>
__global__ __launch_bounds__(2 * (BM / 16) * 4 * 32, BM <= 32 ? 2 : 1) void native_fp8_attention_kernel(
    const void *qraw, const F8 *kv, const int *ids, cutlass::bfloat16_t *out, float *maxima,
    float *lse, const float *sink, const int *lengths, float scale, float outscale, int width,
    int kvrows, int heads, bool q_bf16, int qstride_t, int qstride_h, int splits, const F8 *extra,
    const int *extra_ids, const int *extra_lengths, int primary_width, int extra_width,
    int primary_rows, int lse_queries, const int *errors, int qpr) {
    constexpr int G = (BM / 16) * 4 * 32, Threads = 2 * G;
    const int head_groups = (heads + BM - 1) / BM;
    const int query_part = blockIdx.x / head_groups;
    const int tid = threadIdx.x, query = query_part / splits, part = query_part % splits,
              head = (blockIdx.x % head_groups) * BM, active = min(BM, heads - head);
    if constexpr (Gated) {
        if (!errors[query / qpr]) return;
    }
    const int tile_begin = (width / 64) * part / splits,
              tile_end = (width / 64) * (part + 1) / splits;
    const int original_width = width;
    if constexpr (Dual) {
        ids += (long)query * primary_width;
        extra_ids += (long)query * extra_width;
    } else
        ids += (long)query * original_width + tile_begin * 64;
    out += (((long)query * splits + part) * heads + head) * 512;
    if (sink)
        sink += head;
    if (maxima)
        maxima += ((long)query * splits + part) * heads + head;

    int main_length = 0, extra_length = 0, global_length;
    if constexpr (Dual) {
        main_length = lengths ? max(0, min(primary_width, lengths[query])) : primary_width;
        extra_length = extra_lengths ? max(0, min(extra_width, extra_lengths[query])) : extra_width;
        global_length = extra_length > 0 ? primary_width + extra_length : main_length;
    } else
        global_length = lengths ? max(0, min(original_width, lengths[query])) : original_width;
    width = (tile_end - tile_begin) * 64;
    const int length = COOP_UNIF(max(0, min(width, global_length - tile_begin * 64)));
    extern __shared__ char bytes[];
    using AttentionPipeline = flashmla::sparse_native_fp8::ws::Pipeline<DecodePolicy, PairedV>;
    State s = AttentionPipeline::template shared_state<D, Stages, BM, true>(bytes);
    auto sq = make_tensor(make_smem_ptr(s.q), LQ<D, BM>{});
    for (int i = tid; i < BM * (D / 16); i += Threads) {
        const int r = i / (D / 16), c = i % (D / 16) * 16;
        uint128_t packed{};
        if (r < active) {
            if (q_bf16) {
                const auto *q = static_cast<const __ppu_bfloat16_raw *>(qraw) +
                                (long)query * qstride_t + (long)(head + r) * qstride_h + c;
                auto *pairs = reinterpret_cast<__hg_fp8x2_storage_t *>(&packed);
                CUTE_UNROLL
                for (int j = 0; j < 8; j++)
                    pairs[j] = flashmla::sparse_native_fp8::pack_e4m3_pair(
                        *reinterpret_cast<const __ppu_bfloat162_raw *>(q + j * 2));
            } else
                packed = *reinterpret_cast<const uint128_t *>(
                    static_cast<const F8 *>(qraw) + ((long)query * heads + head + r) * D + c);
        }
        *reinterpret_cast<uint128_t *>(&sq(r, c)) = packed;
    }
    if (tid < BM) {
        s.max[tid] = Neg;
        s.sum[tid] = 0.f;
    }
    for (int i = tid; i < width; i += Threads) {
        int id = -1;
        if constexpr (Dual) {
            const int col = tile_begin * 64 + i;
            if (col < primary_width) {
                if (col < main_length) {
                    int x = ids[col];
                    if ((unsigned)x < (unsigned)primary_rows)
                        id = x;
                }
            } else if (col - primary_width < extra_length) {
                int x = extra_ids[col - primary_width];
                if ((unsigned)x < (unsigned)(kvrows - primary_rows))
                    id = x + primary_rows;
            }
        } else
            id = i < length ? ids[i] : -1;
        s.idx[i] = id;
        const unsigned valid = __ballot_sync(0xffffffff, (unsigned)id < (unsigned)kvrows);
        if ((tid & 31) == 0)
            s.idx[2048 + (i >> 5)] = int(valid);
    }
    __ppu_barrier_sync(0, Threads, 15u);
    if (COOP_UNIF((role<BM>())) == 0) {
        AttentionPipeline::template produce<D, Stages, BM, true, Prefix, Dual, false, GatherLanes>(
            s, kv, length, kvrows, scale, extra, primary_rows);
        __ppu_barrier_sync(0, Threads, 15u);
        const int gt = role_lane<BM>();
        if (gt < active) {
            if (maxima)
                maxima[gt] = s.max[gt];
            if (lse)
                flashmla::sparse_native_fp8::store_lse(lse, query * splits + part, head + gt,
                    heads, lse_queries, s.sum[gt] > 0 ? s.max[gt] + log2f(s.sum[gt] / 448.f) : -INFINITY);
        }
    } else
        AttentionPipeline::template consume<D, Stages, BM, Rows, Late>(
            s, out, sink, length, outscale, active);
}

template <int D, int ST, int BM, int Rows = 1, bool Late = false, int Prefix = 0,
    bool Dual = false, bool Gated = false, int GatherLanes = 0, bool PairedV = false>
void launch_attention(const void *q, const void *kv, const int *idx, void *out, float *mx, float *lse,
    const float *sink, const int *lengths, float scale, float os, int queries, int width,
    int kvrows, int heads, bool qbf, int qst, int qsh, hggcStream_t stream, int splits = 1,
    const void *extra = nullptr, const int *extra_ids = nullptr, const int *extra_lengths = nullptr,
    int primary_width = 0, int extra_width = 0, int primary_rows = 0, int lse_queries = 0,
    const int *errors = nullptr, int qpr = 1) {
    constexpr int Smem = Pipeline::smem_bytes<D, ST, BM, true>() + (Prefix ? BM * (D - Prefix * 32) - 24 : 0);
    auto kernel = native_fp8_attention_kernel<D, ST, BM, Rows, Late, Prefix,
        Dual, Gated, GatherLanes, PairedV>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Smem);
    kernel<<<dim3(int64_t(queries) * splits * ((heads + BM - 1) / BM)), 2 * (BM / 16) * 4 * 32, Smem, stream>>>(
        q, static_cast<const F8 *>(kv), idx, static_cast<cutlass::bfloat16_t *>(out), mx, lse, sink,
        lengths, scale, os, width, kvrows, heads, qbf, qst, qsh, splits,
        static_cast<const F8 *>(extra), extra_ids, extra_lengths, primary_width, extra_width,
        primary_rows, lse_queries, errors, qpr);
}

// Only the validated Warp-Interleave (WI) double/triple-buffer paths
// are production choices.
template <int D, int BM, bool Dual = false, bool Gated = false>
void dispatch_attention(bool two_stage, const void *q, const void *kv, const int *idx, void *out,
    float *maxima, float *lse, const float *sink, const int *lengths, float scale, float out_scale,
    int queries, int width, int kv_rows, int heads, bool q_bf16, int q_stride_t, int q_stride_h,
    hggcStream_t stream, int splits, const void *extra = nullptr,
    const int *extra_indices = nullptr, const int *extra_lengths = nullptr, int main_width = 0,
    int extra_width = 0, int main_rows = 0, int lse_queries = 0,
    const int *errors = nullptr, int qpr = 1, bool cache_resident = false) {
    constexpr int Prefix = BM == 32 ? 12 : 0;
    auto launch = two_stage ? launch_attention<D, 2, BM, 1, false, Prefix, Dual, Gated>
                            : launch_attention<D, 3, BM, 1, false, Prefix, Dual, Gated>;
    // Keep narrow gathers for short D512 partitions; longer partitions amortize
    // the repeated rows needed for an eight-lane cooperative gather. D576 also
    // benefits from the wider gather when processing an unsplit list.
    constexpr int MinWideGatherTiles = 8;
    // A resident KV pool favors fewer gather address/row passes on the
    // three-stage small-head path. Streaming pools retain wider transactions.
    const bool narrow_cached = D == 576 && BM == 16 && !two_stage &&
                               splits == 1 && cache_resident;
    if (!q_bf16 && !narrow_cached && ((D == 576 && splits == 1) ||
            width / splits >= MinWideGatherTiles * flashmla::sparse_native_fp8::TileKeys)) {
        launch = two_stage ? launch_attention<D, 2, BM, 1, false, Prefix, Dual, Gated, 8>
                           : launch_attention<D, 3, BM, 1, false, Prefix, Dual, Gated, 8>;
        if constexpr (!Dual && !Gated) {
            // M890P measurements favor the paired transpose at 32 key tiles.
            // Short partitions keep the original V fragments.
            constexpr int MinPairedVTiles = 32;
            if (splits == 1 && width >= MinPairedVTiles * flashmla::sparse_native_fp8::TileKeys)
                launch = two_stage ? launch_attention<D, 2, BM, 1, false, Prefix, false, false, 8, true>
                                   : launch_attention<D, 3, BM, 1, false, Prefix, false, false, 8, true>;
        }
    }
    launch(q, kv, idx, out, maxima, lse, sink, lengths, scale, out_scale, queries, width, kv_rows,
        heads, q_bf16, q_stride_t, q_stride_h, stream, splits, extra, extra_indices, extra_lengths,
        main_width, extra_width, main_rows, lse_queries, errors, qpr);
}
} // namespace flashmla_fp8_attention

namespace flashmla_fp8_attention {
template <int D, int ST, int H, bool QB, bool Dual = false, bool Window = false, int BM = 64>
__global__ __launch_bounds__(2 * (BM / 16) * 4 * 32, BM <= 32 ? 2 : 1) void native_verify_stream_kernel(const void *qraw,
    const F8 *kv, const int *ids, const unsigned *masks, cutlass::bfloat16_t *out, float *maxima,
    float *lse, const float *sink, const int *errors, float scale, int width, int kvrows, int qpr,
    const F8 *extra, int primary_rows, int lse_queries) {
    // Qualify the native single-KV folded path independently of BF16-query
    // compatibility, short windows and dual-KV verification.
    using VerifyPipeline = flashmla::sparse_native_fp8::ws::Pipeline<DecodePolicy,
        !QB && !Dual && !Window && (BM >= 32)>;
    constexpr int G = (BM / 16) * 4 * 32, Threads = 2 * G;
    const int batch = blockIdx.x, head = blockIdx.y * BM, heads = qpr * H,
              active = min(BM, heads - head), tid = threadIdx.x;
    if (errors[batch])
        return;
    extern __shared__ char bytes[];
    State state = VerifyPipeline::template shared_state<D, ST, BM, true>(bytes);
    auto sq = make_tensor(make_smem_ptr(state.q), LQ<D, BM>{});
    for (int i = tid; i < BM * (D / 16); i += Threads) {
        const int r = i / (D / 16), c = i % (D / 16) * 16;
        uint128_t packed{};
        if (r < active) {
            const long offset = ((long)batch * heads + head + r) * D + c;
            if constexpr (QB) {
                const auto *src = static_cast<const __ppu_bfloat16_raw *>(qraw) + offset;
                auto *pairs = reinterpret_cast<__hg_fp8x2_storage_t *>(&packed);
                CUTE_UNROLL
                for (int j = 0; j < 8; j++)
                    pairs[j] = flashmla::sparse_native_fp8::pack_e4m3_pair(
                        *reinterpret_cast<const __ppu_bfloat162_raw *>(src + 2 * j));
            } else
                packed =
                    *reinterpret_cast<const uint128_t *>(static_cast<const F8 *>(qraw) + offset);
        }
        *reinterpret_cast<uint128_t *>(&sq(r, c)) = packed;
    }
    for (int i = tid; i < width; i += Threads)
        state.idx[i] = ids[(long)batch * width + i];
    if (tid < BM) {
        state.max[tid] = Neg;
        state.sum[tid] = 0.f;
    }
    __ppu_barrier_sync(0, Threads, 15u);
    if (COOP_UNIF((role<BM>())) == 0) {
        VerifyPipeline::template produce<D, ST, BM, true, 0, Dual, true>(state, kv, width, kvrows,
            scale, extra, primary_rows, masks + (long)batch * qpr * (width / 32), width / 32, H, head,
            heads);
        __ppu_barrier_sync(0, Threads, 15u);
        const int gt = role_lane<BM>();
        if (gt < active) {
            long row = (long)batch * heads + head + gt;
            if (maxima)
                maxima[row] = state.max[gt];
            const float value = state.sum[gt] > 0 ? state.max[gt] + log2f(state.sum[gt] / 448.f) : -INFINITY;
            flashmla::sparse_native_fp8::store_lse(lse, batch * qpr + (head + gt) / H,
                (head + gt) % H, H, lse_queries, value);
        }
    } else
        VerifyPipeline::template consume<D, ST, BM, 1, true, Window ? H : 0>(
            state, out + ((long)batch * heads + head) * 512,
            sink ? sink + (Window ? 0 : head) : nullptr, width, 1.f, active);
}
template <int D, int H, bool QB, bool Dual = false, bool Window = false, int BM = 64>
void launch_verify_stream(const void *q, const void *kv, const int *ids, const unsigned *masks,
    void *out, float *mx, float *lse, const float *sink, const int *errors, float scale, int batch,
    int width, int rows, int qpr, hggcStream_t stream,
    const void *extra = nullptr, int primary_rows = 0, int lse_queries = 0) {
    if constexpr (!Window && BM == 64) {
        const int folded_heads = qpr * H;
        // Avoid a mostly padded 64-row tile for short verify groups.
        if (folded_heads <= 32) {
            auto launch = folded_heads <= 16 ? launch_verify_stream<D, H, QB, Dual, Window, 16>
                                            : launch_verify_stream<D, H, QB, Dual, Window, 32>;
            launch(q, kv, ids, masks, out, mx, lse, sink, errors, scale, batch,
                width, rows, qpr, stream, extra, primary_rows, lse_queries);
            return;
        }
    }
    constexpr int ST = 3, Smem = Pipeline::smem_bytes<D, ST, BM, true>();
    auto kernel = native_verify_stream_kernel<D, ST, H, QB, Dual, Window, BM>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Smem);
    kernel<<<dim3(batch, (qpr * H + BM - 1) / BM), 2 * (BM / 16) * 4 * 32, Smem, stream>>>(q, static_cast<const F8 *>(kv),
        ids, masks, static_cast<cutlass::bfloat16_t *>(out), mx, lse, sink, errors, scale, width,
        rows, qpr, static_cast<const F8 *>(extra), primary_rows, lse_queries);
}

} // namespace flashmla_fp8_attention

namespace flashmla_fp8_attention {
template <int ST, int H = 8>
__global__ __launch_bounds__(256, 2) void native_verify_unfolded_kernel(const void *qraw,
    const F8 *kv, const int *ids, cutlass::bfloat16_t *out, float *maxima, float *lse,
    const float *sink, const int *lengths, float scale, int width, int kvrows, bool q_bf16,
    const int *errors, int request_queries) {
    constexpr int D = 512, BM = 16, Stages = ST, Rows = 1, Prefix = 0, heads = H, splits = 1,
                  qstride_t = H * 512, qstride_h = 512;
    // Enable Warp-Interleave (WI) for the independent-query verify fallback.
    constexpr bool Late = false;
    const float outscale = 1.f;

    constexpr int G = (BM / 16) * 4 * 32, Threads = 2 * G;
    const int tid = threadIdx.x, query = blockIdx.x / splits, part = blockIdx.x % splits,
              head = blockIdx.y * BM, active = min(BM, heads - head);
    if (!errors[query / request_queries])
        return;
    const int tile_begin = (width / 64) * part / splits,
              tile_end = (width / 64) * (part + 1) / splits;
    const int original_width = width;
    ids += (long)query * original_width + tile_begin * 64;
    out += (((long)query * splits + part) * heads + head) * 512;
    if (sink)
        sink += head;
    if (maxima)
        maxima += ((long)query * splits + part) * heads + head;
    if (lse)
        lse += ((long)query * splits + part) * heads + head;

    const int global_length =
        lengths ? max(0, min(original_width, lengths[query])) : original_width;
    width = (tile_end - tile_begin) * 64;
    const int length = COOP_UNIF(max(0, min(width, global_length - tile_begin * 64)));
    extern __shared__ char bytes[];
    State s = Pipeline::shared_state<D, Stages, BM, true>(bytes);
    auto sq = make_tensor(make_smem_ptr(s.q), LQ<D, BM>{});
    for (int i = tid; i < BM * (D / 16); i += Threads) {
        const int r = i / (D / 16), c = i % (D / 16) * 16;
        uint128_t packed{};
        if (r < active) {
            if (q_bf16) {
                const auto *q = static_cast<const __ppu_bfloat16_raw *>(qraw) +
                                (long)query * qstride_t + (long)(head + r) * qstride_h + c;
                auto *pairs = reinterpret_cast<__hg_fp8x2_storage_t *>(&packed);
                CUTE_UNROLL
                for (int j = 0; j < 8; j++)
                    pairs[j] = flashmla::sparse_native_fp8::pack_e4m3_pair(
                        *reinterpret_cast<const __ppu_bfloat162_raw *>(q + j * 2));
            } else
                packed = *reinterpret_cast<const uint128_t *>(
                    static_cast<const F8 *>(qraw) + ((long)query * heads + head + r) * D + c);
        }
        *reinterpret_cast<uint128_t *>(&sq(r, c)) = packed;
    }
    if (tid < BM) {
        s.max[tid] = Neg;
        s.sum[tid] = 0.f;
    }
    for (int i = tid; i < width; i += Threads) {
        const int id = i < length ? ids[i] : -1;
        s.idx[i] = id;
        const unsigned valid = __ballot_sync(0xffffffff, (unsigned)id < (unsigned)kvrows);
        if ((tid & 31) == 0)
            s.idx[2048 + (i >> 5)] = int(valid);
    }
    __ppu_barrier_sync(0, Threads, 15u);
    if (COOP_UNIF((role<BM>())) == 0) {
        Pipeline::produce<D, Stages, BM, true, Prefix>(s, kv, length, kvrows, scale);
        __ppu_barrier_sync(0, Threads, 15u);
        const int gt = role_lane<BM>();
        if (gt < active) {
            if (maxima)
                maxima[gt] = s.max[gt];
            if (lse)
                lse[gt] = s.sum[gt] > 0 ? s.max[gt] + log2f(s.sum[gt] / 448.f) : -INFINITY;
        }
    } else
        Pipeline::consume<D, Stages, BM, Rows, Late>(s, out, sink, length, outscale, active);
}
template <int ST, int H = 8>
void launch_verify_unfolded(const void *q, const void *kv, const int *idx, void *out, float *mx,
    float *lse, const float *sink, const int *lengths, const int *errors, float scale, int queries,
    int width, int rows, bool qbf, hggcStream_t stream, int request_queries) {
    constexpr int Smem = Pipeline::smem_bytes<512, ST, 16, true>();
    auto kernel = native_verify_unfolded_kernel<ST, H>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Smem);
    kernel<<<queries, 256, Smem, stream>>>(q, static_cast<const F8 *>(kv), idx,
        static_cast<cutlass::bfloat16_t *>(out), mx, lse, sink, lengths, scale, width, rows, qbf,
        errors, request_queries);
}
} // namespace flashmla_fp8_attention

template <int H, bool Dual>
void launch_native_window(const void *q, const void *kv, const int *ids,
    const void *extra, const int *extra_ids, void *out, float *lse, const float *sink,
    const int *lengths, const int *extra_lengths, int *united, unsigned *masks,
    int *errors, float scale, int batch, int qpr, int width, int padded_width,
    int extra_width, int primary_rows, int extra_rows, hggcStream_t stream, int lse_queries) {
    using namespace flashmla_fp8_attention;
    auto folded = qpr * H <= 16 ? launch_verify_stream<512, H, false, Dual, true, 16>
                : qpr * H <= 32 ? launch_verify_stream<512, H, false, Dual, true, 32>
                               : launch_verify_stream<512, H, false, Dual, true, 64>;
    folded(q, kv, united, masks, out, nullptr,
        lse, sink, errors, scale, batch, padded_width + extra_width,
        primary_rows + extra_rows, qpr, stream, extra, primary_rows, lse_queries);
    if constexpr (Dual) {
        run_native_d512_dual_compact(q, kv, ids, extra, extra_ids, out, lse, sink,
            lengths, extra_lengths, errors, scale, batch * qpr, H, width, extra_width,
            primary_rows, extra_rows, stream, lse_queries, qpr);
    } else {
        launch_attention<512, 2, 16, 1, false, 0, Dual, true>(q, kv, ids, out,
            nullptr, lse, sink, lengths, scale, 1.f, batch * qpr, width + extra_width,
            primary_rows + extra_rows, H, false, H * 512, 512, stream, 1,
            extra, extra_ids, extra_lengths, width, extra_width, primary_rows, lse_queries, errors, qpr);
    }
}

void run_native_fp8_attention_ws(int dim, bool dual, bool two_stage,
    const void *q, const void *kv, const int *idx, void *out, float *maxima, float *lse,
    const float *sink, const int *lengths, float scale, float out_scale,
    int queries, int width, int kv_rows, int heads, bool q_bf16, int q_stride_t, int q_stride_h,
    hggcStream_t stream, int splits, const void *extra, const int *extra_indices,
    const int *extra_lengths, int main_width, int extra_width, int main_rows,
    int lse_queries, const int *errors, int qpr, bool cache_resident) {
    using namespace flashmla_fp8_attention;
    auto launch = dim == 512 ? (heads <= 16 ? dispatch_attention<512, 16> : dispatch_attention<512, 32>)
                             : (heads <= 16 ? dispatch_attention<576, 16> : dispatch_attention<576, 32>);
    if (dual)
        launch = heads <= 16 ? dispatch_attention<512, 16, true> : dispatch_attention<512, 32, true>;
    launch(two_stage, q, kv, idx, out, maxima, lse, sink, lengths, scale, out_scale,
        queries, width, kv_rows, heads, q_bf16, q_stride_t, q_stride_h, stream, splits,
        extra, extra_indices, extra_lengths, main_width, extra_width, main_rows,
        lse_queries, errors, qpr, cache_resident);
}

void run_native_fp8_verify_ws(const void *q, const void *kv, const int *idx, void *out,
    float *lse, const float *sink, const float *fold_sink, const int *lengths, float *maxima,
    int *united, unsigned *masks, int *errors, float scale, int dim, int batch, int qpr, int heads,
    int width, int kvrows, bool qbf, bool two_stage, int cache_bytes,
    hggcStream_t stream, int lse_queries) {
    if (dim == 576) {
        using namespace flashmla_fp8_attention;
        auto folded = heads == 16 ? launch_verify_stream<576, 16, false>
                                  : launch_verify_stream<576, 8, false>;
        folded(q, kv, united, masks, out, maxima, lse, fold_sink, errors, scale,
            batch, width, kvrows, qpr, stream, nullptr, 0, lse_queries);
        // Pools with effective KV reuse share ordinary decode's stage/gather
        // selection. Streaming pools retain the wider three-stage pipeline.
        constexpr int SparsePoolToHistoryRatio = 4;
        const int64_t pool_rows = std::max<int64_t>(0, int64_t(kvrows) -
            int64_t(batch) * flashmla::sparse_native_fp8::TileKeys);
        const bool sparse_pool = pool_rows >= int64_t(batch) * width * SparsePoolToHistoryRatio;
        if (flashmla::sparse_native_fp8::prefer_kv_reuse(int64_t(kvrows) * dim, cache_bytes) &&
            !sparse_pool) {
            dispatch_attention<576, 16, false, true>(two_stage, q, kv, idx, out, maxima, lse,
                sink, lengths, scale, 1.f, batch * qpr, width, kvrows, heads, false,
                heads * 576, 576, stream, 1, nullptr, nullptr, nullptr, 0, 0, 0,
                lse_queries, errors, qpr, int64_t(kvrows) * dim <= cache_bytes);
        } else {
            launch_attention<576, 3, 16, 1, false, 0, false, true, 16>(q, kv, idx, out,
                maxima, lse, sink, lengths, scale, 1.f, batch * qpr, width, kvrows,
                heads, false, heads * 576, 576, stream, 1, nullptr, nullptr, nullptr,
                0, 0, 0, lse_queries, errors, qpr);
        }
        return;
    }
    if (heads == 16) {
        // Narrow a half-empty tail while the pool benefits from KV reuse.
        // Streaming pools use 64-row tiles to avoid rereading KV across more CTAs.
        const int tail = (qpr * heads) % 64;
        const bool compact_tail = flashmla::sparse_native_fp8::prefer_kv_reuse(
            int64_t(kvrows) * dim, cache_bytes) && tail > 0 && tail <= 32;
        using namespace flashmla_fp8_attention;
        auto folded = qbf ? launch_verify_stream<512, 16, true>
                          : launch_verify_stream<512, 16, false>;
        if (compact_tail)
            folded = qbf ? launch_verify_stream<512, 16, true, false, false, 32>
                         : launch_verify_stream<512, 16, false, false, false, 32>;
        folded(q, kv, united, masks, out, maxima, lse, fold_sink, errors, scale,
            batch, width, kvrows, qpr, stream, nullptr, 0, lse_queries);
        if (!qbf)
            run_native_d512_h16_fallback(q, kv, idx, out, maxima, lse, sink, lengths, errors, scale,
                batch * qpr, width / 64, kvrows, stream, lse_queries, qpr);
        else if (two_stage)
            flashmla_fp8_attention::launch_verify_unfolded<2, 16>(q, kv, idx, out, maxima, lse, sink,
                lengths, errors, scale, batch * qpr, width, kvrows, qbf, stream, qpr);
        else
            flashmla_fp8_attention::launch_verify_unfolded<3, 16>(q, kv, idx, out, maxima, lse, sink,
                lengths, errors, scale, batch * qpr, width, kvrows, qbf, stream, qpr);
        return;
    }
    if (qbf)
        flashmla_fp8_attention::launch_verify_stream<512, 8, true>(q, kv, united, masks, out, maxima,
            lse, fold_sink, errors, scale, batch, width, kvrows, qpr, stream, nullptr, 0, lse_queries);
    else
        flashmla_fp8_attention::launch_verify_stream<512, 8, false>(q, kv, united, masks, out,
            maxima, lse, fold_sink, errors, scale, batch, width, kvrows, qpr, stream, nullptr, 0, lse_queries);
    if (!qbf) {
        run_native_d512_h8_fallback(q, kv, idx, out, maxima, lse, sink, lengths, errors, scale,
            batch * qpr, width / 64, kvrows, stream, lse_queries, qpr);
        return;
    }
#define FALLBACK(ST)                                                                               \
    flashmla_fp8_attention::launch_verify_unfolded<ST>(q, kv, idx, out, maxima, lse, sink, lengths,   \
        errors, scale, batch * qpr, width, kvrows, qbf, stream, qpr)
    if (two_stage) {
        FALLBACK(2);
    } else {
        FALLBACK(3);
    }
#undef FALLBACK
}

void run_native_fp8_window_ws(const void *q, const void *kv, const int *ids,
    const void *extra, const int *extra_ids, void *out, float *lse, const float *sink,
    const int *lengths, const int *extra_lengths, int *united, unsigned *masks,
    int *errors, float scale, int batch, int qpr, int heads, int width,
    int padded_width, int extra_width, int primary_rows, int extra_rows, hggcStream_t stream, int lse_queries) {
    auto launch = heads == 8 ? (extra ? launch_native_window<8, true> : launch_native_window<8, false>)
                            : (extra ? launch_native_window<16, true> : launch_native_window<16, false>);
    launch(q, kv, ids, extra, extra_ids, out, lse, sink, lengths, extra_lengths,
        united, masks, errors, scale, batch, qpr, width, padded_width, extra_width,
        primary_rows, extra_rows, stream, lse_queries);
}
