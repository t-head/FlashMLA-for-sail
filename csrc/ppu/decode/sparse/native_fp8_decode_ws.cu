// Native FP8 sparse decode/verify warp-specialized kernels and launchers.

#include "ppu/sparse_native_fp8_attention.cuh"

void run_native_d512_h8_fallback(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_d512_h16_fallback(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_d512_dual_compact(const flashmla::sparse_native_fp8::AttentionParams &params);

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

template <int D, int ST, int BM, int Rows = 1, bool Late = false, int Prefix = 0, bool Dual = false,
          bool Gated = false, int GatherLanes = 0, bool PairedV = false>
void launch_attention(const flashmla::sparse_native_fp8::AttentionParams &params) {
    constexpr int Smem =
        Pipeline::smem_bytes<D, ST, BM, true>() + (Prefix ? BM * (D - Prefix * 32) - 24 : 0);
    auto kernel = native_fp8_attention_kernel<D, ST, BM, Rows, Late, Prefix, Dual, Gated,
                                              GatherLanes, PairedV>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Smem);
    kernel<<<dim3(int64_t(params.queries) * params.splits * ((params.heads + BM - 1) / BM)),
             2 * (BM / 16) * 4 * 32, Smem, params.stream>>>(
        params.q, static_cast<const F8 *>(params.kv), params.idx,
        static_cast<cutlass::bfloat16_t *>(params.out), params.maxima, params.lse, params.sink,
        params.lengths, params.scale, params.os, params.width, params.kvrows, params.heads,
        params.qbf, params.qst, params.qsh, params.splits, static_cast<const F8 *>(params.extra),
        params.extra_idx, params.extra_lengths, params.primary_width, params.extra_width,
        params.primary_rows, params.lse_queries, params.errors, params.qpr);
}

// Only the validated Warp-Interleave (WI) double/triple-buffer paths
// are production choices.
template <int D, int BM, bool Dual = false, bool Gated = false>
void dispatch_attention(const flashmla::sparse_native_fp8::AttentionParams &params) {
    constexpr int Prefix = BM == 32 ? 12 : 0;
    auto launch = params.two_stage ? launch_attention<D, 2, BM, 1, false, Prefix, Dual, Gated>
                                   : launch_attention<D, 3, BM, 1, false, Prefix, Dual, Gated>;
    // Keep narrow gathers for short D512 partitions; longer partitions amortize
    // the repeated rows needed for an eight-lane cooperative gather. D576 also
    // benefits from the wider gather when processing an unsplit list.
    constexpr int MinWideGatherTiles = 8;
    // A resident KV pool favors fewer gather address/row passes on the
    // three-stage small-head path. Streaming pools retain wider transactions.
    const bool narrow_cached =
        D == 576 && BM == 16 && !params.two_stage && params.splits == 1 && params.cache_resident;
    if (!params.qbf && !narrow_cached &&
        ((D == 576 && params.splits == 1) ||
         params.width / params.splits >=
             MinWideGatherTiles * flashmla::sparse_native_fp8::TileKeys)) {
        launch = params.two_stage ? launch_attention<D, 2, BM, 1, false, Prefix, Dual, Gated, 8>
                                  : launch_attention<D, 3, BM, 1, false, Prefix, Dual, Gated, 8>;
        if constexpr (!Dual && !Gated) {
            // M890P measurements favor the paired transpose at 32 key tiles.
            // Short partitions keep the original V fragments.
            constexpr int MinPairedVTiles = 32;
            if (params.splits == 1 &&
                params.width >= MinPairedVTiles * flashmla::sparse_native_fp8::TileKeys)
                launch = params.two_stage
                             ? launch_attention<D, 2, BM, 1, false, Prefix, false, false, 8, true>
                             : launch_attention<D, 3, BM, 1, false, Prefix, false, false, 8, true>;
        }
    }
    launch(params);
}
} // namespace flashmla_fp8_attention

namespace flashmla_fp8_attention {
template <int D, int ST, int H, bool Dual = false, bool Window = false, int BM = 64>
__global__ __launch_bounds__(2 * (BM / 16) * 4 * 32, BM <= 32 ? 2 : 1) void native_verify_stream_kernel(const void *qraw,
    const F8 *kv, const int *ids, const unsigned *masks, cutlass::bfloat16_t *out, float *maxima,
    float *lse, const float *sink, const int *errors, float scale, int width, int kvrows, int qpr,
    const F8 *extra, int primary_rows, int lse_queries) {
    // Paired V loads apply to the native single-KV folded path; short windows
    // and dual-KV verification retain their original fragment layout.
    using VerifyPipeline = flashmla::sparse_native_fp8::ws::Pipeline<DecodePolicy,
        !Dual && !Window && (BM >= 32)>;
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
template <int D, int H, bool Dual = false, bool Window = false, int BM = 64>
void launch_verify_stream(const flashmla::sparse_native_fp8::AttentionParams &params) {
    if constexpr (!Window && BM == 64) {
        const int folded_heads = params.qpr * H;
        // Avoid a mostly padded 64-row tile for short verify groups.
        if (folded_heads <= 32) {
            auto launch = folded_heads <= 16 ? launch_verify_stream<D, H, Dual, Window, 16>
                                             : launch_verify_stream<D, H, Dual, Window, 32>;
            launch(params);
            return;
        }
    }
    constexpr int ST = 3, Smem = Pipeline::smem_bytes<D, ST, BM, true>();
    auto kernel = native_verify_stream_kernel<D, ST, H, Dual, Window, BM>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Smem);
    kernel<<<dim3(params.batch, (params.qpr * H + BM - 1) / BM), 2 * (BM / 16) * 4 * 32, Smem,
             params.stream>>>(
        params.q, static_cast<const F8 *>(params.kv), params.idx, params.masks,
        static_cast<cutlass::bfloat16_t *>(params.out), params.maxima, params.lse, params.sink,
        params.errors, params.scale, params.width, params.kvrows, params.qpr,
        static_cast<const F8 *>(params.extra), params.primary_rows, params.lse_queries);
}

} // namespace flashmla_fp8_attention

template <int H, bool Dual>
void launch_native_window(const flashmla::sparse_native_fp8::AttentionParams &params) {
    using namespace flashmla_fp8_attention;
    auto folded = params.qpr * H <= 16   ? launch_verify_stream<512, H, Dual, true, 16>
                  : params.qpr * H <= 32 ? launch_verify_stream<512, H, Dual, true, 32>
                                         : launch_verify_stream<512, H, Dual, true, 64>;
    {
        auto view = params;
        view.idx = params.united;
        view.maxima = nullptr;
        view.width = params.padded_width + params.extra_width;
        view.kvrows = params.kvrows + params.extra_rows;
        view.primary_rows = params.kvrows;
        folded(view);
    }
    if constexpr (Dual)
        run_native_d512_dual_compact(params);
    else
        launch_attention<512, 2, 16, 1, false, 0, false, true>(params);
}

void run_native_fp8_attention_ws(const flashmla::sparse_native_fp8::AttentionParams &params) {
    using namespace flashmla_fp8_attention;
    auto launch =
        params.dim == 512
            ? (params.heads <= 16 ? dispatch_attention<512, 16> : dispatch_attention<512, 32>)
            : (params.heads <= 16 ? dispatch_attention<576, 16> : dispatch_attention<576, 32>);
    if (params.extra)
        launch = params.heads <= 16 ? dispatch_attention<512, 16, true>
                                    : dispatch_attention<512, 32, true>;
    launch(params);
}

void run_native_fp8_verify_ws(const flashmla::sparse_native_fp8::AttentionParams &params) {
    using namespace flashmla_fp8_attention;
    auto folded_params = params;
    folded_params.idx = params.united;
    folded_params.sink = params.fold_sink;
    if (params.dim == 576) {
        auto folded = params.heads == 16 ? launch_verify_stream<576, 16>
                                         : launch_verify_stream<576, 8>;
        folded(folded_params);
        // Streaming pools retain the wider three-stage pipeline. Pools with
        // effective KV reuse share ordinary decode's stage/gather selection.
        constexpr int SparsePoolToHistoryRatio = 4;
        const int64_t pool_rows =
            std::max<int64_t>(0, int64_t(params.kvrows) -
                                     int64_t(params.batch) * flashmla::sparse_native_fp8::TileKeys);
        const bool sparse_pool =
            pool_rows >= int64_t(params.batch) * params.width * SparsePoolToHistoryRatio;
        if (flashmla::sparse_native_fp8::prefer_kv_reuse(int64_t(params.kvrows) * params.dim,
                                                         params.cache_bytes) &&
            !sparse_pool) {
            auto independent = params;
            independent.cache_resident = int64_t(params.kvrows) * params.dim <= params.cache_bytes;
            dispatch_attention<576, 16, false, true>(independent);
        } else {
            launch_attention<576, 3, 16, 1, false, 0, false, true, 16>(params);
        }
        return;
    }
    if (params.heads == 16) {
        // Narrow a half-empty tail while the pool benefits from KV reuse.
        // Streaming pools use 64-row tiles to avoid rereading KV across CTAs.
        const int tail = (params.qpr * params.heads) % 64;
        const bool compact_tail = flashmla::sparse_native_fp8::prefer_kv_reuse(
                                      int64_t(params.kvrows) * params.dim, params.cache_bytes) &&
                                  tail > 0 && tail <= 32;
        auto folded = launch_verify_stream<512, 16>;
        if (compact_tail)
            folded = launch_verify_stream<512, 16, false, false, 32>;
        folded(folded_params);
        run_native_d512_h16_fallback(params);
        return;
    }
    launch_verify_stream<512, 8>(folded_params);
    run_native_d512_h8_fallback(params);
}

void run_native_fp8_window_ws(const flashmla::sparse_native_fp8::AttentionParams &params) {
    auto launch =
        params.heads == 8
            ? (params.extra ? launch_native_window<8, true> : launch_native_window<8, false>)
            : (params.extra ? launch_native_window<16, true> : launch_native_window<16, false>);
    launch(params);
}
