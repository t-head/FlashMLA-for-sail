// Native FP8 sparse decode/verify routing and auxiliary kernels.
// Standard CTA and warp-specialized kernels share this translation unit.

#include "ppu/sparse_native_fp8_attention.cuh"

// Launch declarations for the kernels below.
void run_native_fp8_attention_ws(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_fp8_verify_ws(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_fp8_window_ws(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_d512_compact(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_d512_h16_fallback(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_d512_h16_shared(const flashmla::sparse_native_fp8::AttentionParams &params);

bool try_sparse_fp8_std(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_d512_dual_compact(const flashmla::sparse_native_fp8::AttentionParams &params);

template <bool WriteMax, bool WriteLse>
__global__ void native_attention_merge_kernel(const cutlass::bfloat16_t *part, const float *plse,
    cutlass::bfloat16_t *out, float *lse, const float *sink, int tokens, int splits, int heads,
    int lse_queries, const float *partial_maxima, float *maxima) {
    constexpr int Values = 8;
    const int row = blockIdx.x, column = threadIdx.x * Values;
    if (row >= static_cast<long>(tokens) * heads)
        return;
    const int token = row / heads, head = row % heads;
    float maximum = -INFINITY;
    for (int j = 0; j < splits; j++)
        maximum = fmaxf(maximum, plse[((long)token * splits + j) * heads + head]);
    const bool active = maximum != -INFINITY;
    const float safe_max = active ? maximum : 0.f;
    float sum = 0.f;
    float accum[Values] = {};
    for (int j = 0; j < splits; j++) {
        const long base = ((long)token * splits + j) * heads + head;
        const float weight = active ? exp2f(plse[base] - safe_max) : 0.f;
        sum += weight;
        if (weight > 0.f) {
            const uint4 packed = *reinterpret_cast<const uint4 *>(part + base * 512 + column);
            const auto *values = reinterpret_cast<const cutlass::bfloat16_t *>(&packed);
#pragma unroll
            for (int i = 0; i < Values; ++i)
                accum[i] += weight * float(values[i]);
        }
    }
    float denominator = sum;
    if (sink)
        denominator += exp2f(sink[head] * 1.4426950408889634f - safe_max);
    const float inv = denominator > 0.f ? 1.f / denominator : 0.f;
    uint4 packed;
    auto *values = reinterpret_cast<cutlass::bfloat16_t *>(&packed);
#pragma unroll
    for (int i = 0; i < Values; ++i)
        values[i] = cutlass::bfloat16_t(accum[i] * inv);
    *reinterpret_cast<uint4 *>(out + static_cast<long>(row) * 512 + column) = packed;
    if (column == 0) {
        if constexpr (WriteLse)
            flashmla::sparse_native_fp8::store_lse(lse, token, head, heads, lse_queries,
                sum > 0.f ? safe_max + log2f(sum) : -INFINITY);
        if constexpr (WriteMax) {
            // Prefill exposes the ceil-quantized running maximum, not max(LSE).
            // Reduce the per-partition maxima independently of normalization.
            float max_logit = -INFINITY;
            for (int j = 0; j < splits; ++j)
                max_logit = fmaxf(max_logit, partial_maxima[((long)token * splits + j) * heads + head]);
            maxima[row] = max_logit;
        }
    }
}

void run_native_fp8_attention_merge(const flashmla::sparse_native_fp8::MergeParams &params) {
    const int blocks = params.tokens * params.heads;
    if (blocks) {
        auto kernel = params.maxima ? native_attention_merge_kernel<true, true>
                      : params.lse  ? native_attention_merge_kernel<false, true>
                                    : native_attention_merge_kernel<false, false>;
        kernel<<<blocks, 64, 0, params.stream>>>(
            static_cast<const cutlass::bfloat16_t *>(params.part), params.plse,
            static_cast<cutlass::bfloat16_t *>(params.out), params.lse, params.sink, params.tokens,
            params.splits, params.heads, params.lse_queries, params.partial_maxima, params.maxima);
    }
}

// Split and dual-pool routes share the same WS output view and merge contract.
static void launch_ws_and_merge(const flashmla::sparse_native_fp8::AttentionParams &params) {
    auto view = params;
    if (params.splits > 1) {
        view.out = params.part;
        view.lse = params.plse;
        view.maxima = params.partial_maxima;
        view.sink = nullptr;
        view.lse_queries = 0;
    }
    if (params.extra) {
        view.primary_width = params.width;
        view.primary_rows = params.kvrows;
        view.width += params.extra_width;
        view.kvrows += params.extra_rows;
    }
    view.cache_resident =
        !params.extra && int64_t(params.kvrows) * params.dim <= params.cache_bytes;
    run_native_fp8_attention_ws(view);
    if (params.splits > 1)
        run_native_fp8_attention_merge(flashmla::sparse_native_fp8::MergeParams(params));
}

void run_native_fp8_attention_decode(const flashmla::sparse_native_fp8::AttentionParams &params) {
    auto standard = params;
    standard.trusted_full_topk = false;

    if (params.splits == 1 && !params.qbf && params.heads % 64 == 0 && try_sparse_fp8_std(standard))
        return;

    // Tiled attention requires at least eight N64 tiles, so it cannot consume
    // the short full-head cases. Keep long-list compact fallbacks after it.
    const bool short_full_heads =
        params.width == 128 && params.heads % 64 == 0 &&
        int64_t(params.queries) * (params.heads / 64) > params.compute_units;
    const bool compact_fallback =
        (params.width <= 128 &&
         (params.heads == 8 || params.heads == 16 ||
          (params.heads == 64 && params.queries >= 64))) ||
        (params.width >= 512 &&
         ((params.queries >= 96 &&
           (params.heads == 16 || (params.heads == 8 && params.width == 2048))) ||
          (params.queries >= 192 && params.heads == 64)));
    if (params.dim == 512 && params.splits == 1 && !params.qbf &&
        (short_full_heads || compact_fallback)) {
        run_native_d512_compact(standard);
        return;
    }

    launch_ws_and_merge(params);
}

// Two independent persistent KV pools, one online softmax and one PV output.
void run_native_fp8_attention_extra(const flashmla::sparse_native_fp8::AttentionParams &params) {
    const bool small_heads =
        (params.heads == 8 || params.heads == 16) && params.queries > 2 * params.compute_units;
    const bool full_heads = params.heads % 64 == 0 &&
                            int64_t(params.queries) * (params.heads / 64) > params.compute_units;
    if (params.splits == 1 && !params.qbf && (small_heads || full_heads)) {
        run_native_d512_dual_compact(params);
        return;
    }
    launch_ws_and_merge(params);
}
__global__ void native_verify_union_kernel(const int *ids, const int *lengths, int *united,
    unsigned *masks, int *errors, int qpr, int width, int kvrows) {
    const int batch = blockIdx.x, tid = threadIdx.x;
    // Every thread checks the same first slot, so rejection is uniform without
    // a block reduction. A match is only a hint: the full classifier below must
    // still validate every query and column before choosing the shared path.
    if (qpr >= 2 && width > 0) {
        const bool first_live = !lengths || lengths[batch * qpr] > 0;
        const bool next_live = !lengths || lengths[batch * qpr + 1] > 0;
        const int first = first_live ? ids[(long)batch * qpr * width] : -1;
        const int next = next_live ? ids[((long)batch * qpr + 1) * width] : -1;
        if ((unsigned)first < (unsigned)kvrows && (unsigned)next < (unsigned)kvrows &&
            first != next) {
            if (tid == 0)
                errors[batch] = 1;
            return;
        }
    }
    bool incompatible = false;
    for (int start = 0; start < width; start += blockDim.x) {
        const int col = start + tid;
        int chosen = -1;
        for (int q = 0; q < qpr; q++) {
            const int length = lengths ? max(0, min(width, lengths[batch * qpr + q])) : width;
            const int token =
                col < width && col < length ? ids[((long)batch * qpr + q) * width + col] : -1;
            const bool valid = (unsigned)token < (unsigned)kvrows;
            if (valid) {
                if (chosen >= 0 && chosen != token)
                    incompatible = true;
                else
                    chosen = token;
            }
            const unsigned bits = __ballot_sync(0xffffffff, valid);
            if ((tid & 31) == 0 && col < width)
                masks[((long)batch * qpr + q) * (width / 32) + (col / 32)] = bits;
        }
        if (col < width)
            united[(long)batch * width + col] = chosen;
        // Independent per-query TopK generally disagrees in the first slice. All
        // threads leave together; skipped folded CTAs never read partial scratch.
        if (start == 0 && __syncthreads_or(incompatible)) {
            if (tid == 0)
                errors[batch] = 1;
            return;
        }
    }
    const int mismatch = __syncthreads_or(incompatible);
    if (tid == 0)
        errors[batch] = mismatch;
}

// Both consumers process the original per-query indices. This sampled agreement
// only selects CTA residency; it never authorizes a shared-index computation.
__global__ void native_kv_reuse_hint_kernel(const int *ids, const int *lengths, int *errors,
    int batch, int qpr, int width, int kvrows) {
    const int request = blockIdx.x * blockDim.x + threadIdx.x;
    if (request >= batch)
        return;
    const int query = request * qpr;
    const int first = !lengths || lengths[query] > 0 ? ids[int64_t(query) * width] : -1;
    const int next = !lengths || lengths[query + 1] > 0 ? ids[int64_t(query + 1) * width] : -1;
    errors[request] = (unsigned)first < (unsigned)kvrows &&
                      (unsigned)next < (unsigned)kvrows && first != next;
}

void run_native_fp8_attention_verify(const flashmla::sparse_native_fp8::AttentionParams &params) {
    const int folded_heads = params.qpr * params.heads, tail = folded_heads % 64;
    // A nearly empty final fold adds a full KV pass to a streaming pool.
    // Compact queries avoid that padding. A cheap reuse hint selects residency;
    // both branches preserve each query's original indices and lengths.
    if (params.dim == 512 && params.heads == 16 && !params.qbf && folded_heads > 64 && tail > 0 &&
        tail <= params.heads &&
        !flashmla::sparse_native_fp8::prefer_kv_reuse(int64_t(params.kvrows) * params.dim,
                                                      params.cache_bytes)) {
        native_kv_reuse_hint_kernel<<<(params.batch + 255) / 256, 256, 0, params.stream>>>(
            params.idx, params.lengths, params.errors, params.batch, params.qpr, params.width,
            params.kvrows);
        run_native_d512_h16_shared(params);
        run_native_d512_h16_fallback(params);
        return;
    }

    // Wider classifiers help small partial-head request grids. Large grids
    // already expose enough independent requests and retain the smaller CTA.
    const bool wide_classifier = params.dim == 576 && params.heads != 8 &&
        params.heads != 16 && params.batch < 4 * params.compute_units;
    native_verify_union_kernel<<<params.batch, wide_classifier ? 512 : 256, 0, params.stream>>>(
        params.idx, params.lengths, params.united, params.masks, params.errors, params.qpr,
        params.width, params.kvrows);
    run_native_fp8_verify_ws(params);
}

// Align adjacent sliding windows and the common extra-pool list. Every physical
// index is checked on device on each replay; incompatible requests use the gated
// independent-query kernel below. Invalid entries retain each query's own mask.
__global__ void native_window_union_kernel(const int *ids, const int *lengths,
    const int *extra_ids, const int *extra_lengths, int *united, unsigned *masks,
    int *errors, int qpr, int width, int padded_width, int extra_width,
    int primary_rows, int extra_rows) {
    const int batch = blockIdx.x, tid = threadIdx.x, total = padded_width + extra_width;
    bool incompatible = false;
    for (int start = 0; start < total; start += blockDim.x) {
        const int col = start + tid;
        int chosen = -1;
        for (int q = 0; q < qpr; ++q) {
            const int query = batch * qpr + q;
            const bool secondary = col >= padded_width;
            const int source_col = secondary ? col - padded_width : col - q;
            const int source_width = secondary ? extra_width : width;
            const int *source_lengths = secondary ? extra_lengths : lengths;
            const int length = source_lengths ? max(0, min(source_width, source_lengths[query])) : source_width;
            int token = -1;
            if (col < total && source_col >= 0 && source_col < length) {
                const int *source = secondary ? extra_ids : ids;
                const int value = source[(long)query * source_width + source_col];
                if ((unsigned)value < (unsigned)(secondary ? extra_rows : primary_rows))
                    token = value + (secondary ? primary_rows : 0);
            }
            const bool valid = token >= 0;
            if (valid) {
                if (chosen >= 0 && chosen != token) incompatible = true;
                chosen = token;
            }
            const unsigned bits = __ballot_sync(0xffffffff, valid);
            if ((tid & 31) == 0 && col < total)
                masks[((long)batch * qpr + q) * (total / 32) + col / 32] = bits;
        }
        if (col < total) united[(long)batch * total + col] = chosen;
        if (__syncthreads_or(incompatible)) {
            if (tid == 0) errors[batch] = 1;
            return;
        }
    }
    if (tid == 0) errors[batch] = 0;
}

void run_native_fp8_attention_window(const flashmla::sparse_native_fp8::AttentionParams &params) {
    native_window_union_kernel<<<params.batch, 256, 0, params.stream>>>(
        params.idx, params.lengths, params.extra_idx, params.extra_lengths, params.united,
        params.masks, params.errors, params.qpr, params.width, params.padded_width,
        params.extra_width, params.kvrows, params.extra_rows);
    run_native_fp8_window_ws(params);
}

// Standard CTA sparse attention launchers for decode and verification.

// H16 uses complete head groups; no runtime head-tail predicates are needed.
// Keep the device-property cache local when multiple extension builds are loaded;
// an externally linked template can coalesce its GNU-unique thread-local symbols.
template <int Heads, int Prefix, bool GuardRequests = false, int BlockN = 64,
          bool SharedRequests = false>
static void launch_native_d512_small(const flashmla::sparse_native_fp8::AttentionParams &params) {
    using namespace flashmla_fp8_std;
    constexpr int Shared =
        (16 + 128) * 512 + 16 * BlockN + (BlockN == 128 ? 128 : 2176) * 4 + 11 * 16 * 4 + 256;
    auto kernel = flash_fwd_sparse_fp8_std_kernel<16, 512, false, 256, Prefix, GuardRequests, Heads,
                                                  BlockN, false, 8, SharedRequests>;
    int launch_shared = Shared;
    if constexpr (Heads == 16 && GuardRequests && !SharedRequests) {
        int device = 0;
        hggcGetDevice(&device);
        static thread_local int previous_device = -1, shared_per_cu = 0, cache_bytes = 0;
        if (previous_device != device) {
            hggcDeviceGetAttribute(&shared_per_cu, hggcDevAttrMaxSharedMemoryPerMultiprocessor,
                                   device);
            hggcDeviceGetAttribute(&cache_bytes, hggcDevAttrL2CacheSize, device);
            previous_device = device;
        }
        // One padded key tile per request must not change residency at the
        // reuse crossover. Larger pools need repeated full-pool passes to
        // offset contention from a third compact CTA.
        constexpr int MinPoolPasses = 3;
        const int64_t requested_rows = int64_t(params.queries) * (params.width / 64) * 64;
        const int64_t accessed_rows = std::min<int64_t>(params.kvrows, requested_rows);
        const int64_t padding_rows =
            int64_t(params.queries / params.qpr) * flashmla::sparse_native_fp8::TileKeys;
        const int64_t reuse_rows = std::max<int64_t>(0, accessed_rows - padding_rows);
        const bool cache_reuse =
            flashmla::sparse_native_fp8::prefer_kv_reuse(reuse_rows * 512, cache_bytes);
        const int64_t pool_rows = std::max<int64_t>(1, int64_t(params.kvrows) - padding_rows);
        constexpr int MaxRepeatedPoolCacheFootprints = 32;
        const bool repeated_pool =
            requested_rows >= pool_rows * MinPoolPasses &&
            pool_rows * 512 <= int64_t(cache_bytes) * MaxRepeatedPoolCacheFootprints;
        if (!cache_reuse && !repeated_pool)
            launch_shared = max(Shared, ((shared_per_cu / 3) / 256 + 1) * 256);
    }
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, launch_shared);
    kernel<<<params.queries, 256, launch_shared, params.stream>>>(
        params.q, static_cast<const E4M3 *>(params.kv), params.idx,
        static_cast<cutlass::bfloat16_t *>(params.out), params.scale,
        (params.width / 64) * 64 / BlockN, params.kvrows, params.os, Heads, params.maxima,
        params.lse, params.sink, params.lengths, params.errors, params.lse_queries, nullptr,
        nullptr, nullptr, 0, 0, params.qpr);
}

// Full head groups share KV; long lists trade more per-thread work for fewer warps.
void run_native_d512_compact(const flashmla::sparse_native_fp8::AttentionParams &params) {
    using namespace flashmla_fp8_std;
    if ((params.width / 64) == 2 && params.heads % 64 == 0) {
        constexpr int Threads = 512;
        constexpr int Shared = (64 + 128) * 512 + 64 * 128 + 128 * 4 + 11 * 64 * 4 + 256;
        auto kernel = flash_fwd_sparse_fp8_std_kernel<64, 512, false, Threads, 0, false, 64, 128>;
        hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Shared);
        kernel<<<dim3(int64_t(params.queries) * (params.heads / 64)), Threads, Shared,
                 params.stream>>>(params.q, static_cast<const E4M3 *>(params.kv), params.idx,
                                  static_cast<cutlass::bfloat16_t *>(params.out), params.scale, 1,
                                  params.kvrows, params.os, params.heads, params.maxima, params.lse,
                                  params.sink, params.lengths, nullptr, params.lse_queries, nullptr,
                                  nullptr, nullptr, 0, 0, 1);
        return;
    }
    if ((params.width / 64) == 2 && (params.heads == 8 || params.heads == 16)) {
        auto launch = params.heads == 8 ? launch_native_d512_small<8, 0, false, 128>
                                        : launch_native_d512_small<16, 0, false, 128>;
        launch(params);
        return;
    }
    if (params.heads == 64) {
        if ((params.width / 64) <= 4)
            dispatch<512>(params);
        else
            dispatch<256, 4>(params);
    } else if (params.heads == 8) {
        launch_native_d512_small<8, 4>(params);
    } else {
        if ((params.width / 64) <= 4)
            launch_native_d512_small<16, 0>(params);
        else
            launch_native_d512_small<16, 8>(params);
    }
}

// Only requests rejected by the union classifier execute this independent path.
void run_native_d512_h16_fallback(const flashmla::sparse_native_fp8::AttentionParams &params) {
    launch_native_d512_small<16, 8, true>(params);
}

// Shared queries retain three resident CTAs even when their allocation is large:
// adjacent queries repeatedly gather the same rows from that pool.
void run_native_d512_h16_shared(const flashmla::sparse_native_fp8::AttentionParams &params) {
    launch_native_d512_small<16, 8, true, 64, true>(params);
}

void run_native_d512_h8_fallback(const flashmla::sparse_native_fp8::AttentionParams &params) {
    launch_native_d512_small<8, 4, true>(params);
}

// Compact single-query pipeline for two independent KV pools.
template <int Heads, bool GuardRequests>
void launch_native_d512_dual_compact(const flashmla::sparse_native_fp8::AttentionParams &params) {
    using namespace flashmla_fp8_std;
    constexpr int BM = Heads == 64 ? 64 : 16;
    constexpr int Prefix = Heads == 16 ? 8 : 4;
    constexpr int Shared = (BM + 128) * 512 + BM * 64 + 2176 * 4 + 11 * BM * 4 + 256;
    // Full head tiles favor one gather pass over all 64 keys: four lanes
    // copy each row in 16-byte segments. Compact tiles retain their tuned
    // eight-lane gather; changing the load grouping does not change QK/PV.
    constexpr int GatherLanes = Heads == 64 ? 4 : 8;
    auto kernel = flash_fwd_sparse_fp8_std_kernel<BM, 512, false, 256, Prefix, GuardRequests, Heads,
                                                  64, true, GatherLanes>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Shared);
    kernel<<<dim3(int64_t(params.queries) * (Heads == 64 ? params.heads / 64 : 1)), 256, Shared,
             params.stream>>>(
        params.q, static_cast<const E4M3 *>(params.kv), params.idx,
        static_cast<cutlass::bfloat16_t *>(params.out), params.scale,
        (params.width + params.extra_width) / 64, params.kvrows + params.extra_rows, 1.f,
        params.heads, nullptr, params.lse, params.sink, params.lengths, params.errors,
        params.lse_queries, static_cast<const E4M3 *>(params.extra), params.extra_idx,
        params.extra_lengths, params.width, params.kvrows, params.qpr);
}

void run_native_d512_dual_compact(const flashmla::sparse_native_fp8::AttentionParams &params) {
    auto launch = params.heads == 8 ? (params.errors ? launch_native_d512_dual_compact<8, true>
                                                     : launch_native_d512_dual_compact<8, false>)
                  : params.heads == 16
                      ? (params.errors ? launch_native_d512_dual_compact<16, true>
                                       : launch_native_d512_dual_compact<16, false>)
                      : launch_native_d512_dual_compact<64, false>;
    launch(params);
}

// Native FP8 sparse decode/verify warp-specialized kernels and launchers.


void run_native_d512_h8_fallback(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_d512_h16_fallback(const flashmla::sparse_native_fp8::AttentionParams &params);

void run_native_d512_dual_compact(const flashmla::sparse_native_fp8::AttentionParams &params);

namespace flashmla_fp8_attention {
using namespace flashmla::sparse_native_fp8::ws;
using Pipeline = flashmla::sparse_native_fp8::ws::Pipeline<DecodePolicy>;
// WI = Warp-Interleave: overlap producer/consumer work using the shared-slot
// handoff schedule implemented by produce() and consume() in the header.
template <int D, int Stages, int BM, int Rows, bool Late = false, int Prefix = 0,
    bool Dual = false, bool Gated = false, int GatherLanes = 0, bool PairedV = false,
    bool ConsumerPrefetch = false>
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
    using AttentionPipeline = flashmla::sparse_native_fp8::ws::Pipeline<DecodePolicy, PairedV, ConsumerPrefetch>;
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
        AttentionPipeline::template consume<D, Stages, BM, Rows, Late, 0, GatherLanes>(
            s, out, sink, length, outscale, active, kv, kvrows);
}

template <int D, int ST, int BM, int Rows = 1, bool Late = false, int Prefix = 0, bool Dual = false,
          bool Gated = false, int GatherLanes = 0, bool PairedV = false, bool ConsumerPrefetch = false>
void launch_attention(const flashmla::sparse_native_fp8::AttentionParams &params) {
    constexpr int Smem =
        flashmla::sparse_native_fp8::ws::Pipeline<DecodePolicy, PairedV, ConsumerPrefetch>::template
        smem_bytes<D, ST, BM, true>() + (Prefix ? BM * (D - Prefix * 32) - 24 : 0);
    auto kernel = native_fp8_attention_kernel<D, ST, BM, Rows, Late, Prefix, Dual, Gated,
                                              GatherLanes, PairedV, ConsumerPrefetch>;
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

// Narrow gathers reduce row/address work while the KV pool benefits from
// cache reuse. Larger multi-request pools keep the streaming gather policy.
inline bool prefer_narrow_consumer_gather(const flashmla::sparse_native_fp8::AttentionParams &params) {
    constexpr int CacheBudgetMultiplier = 2;
    return params.cache_bytes > 0 &&
        int64_t(params.kvrows) * params.dim <= int64_t(CacheBudgetMultiplier) * params.cache_bytes;
}

// Only the validated Warp-Interleave (WI) double/triple-buffer paths
// are production choices.
template <int D, int BM, bool Dual = false, bool Gated = false>
void dispatch_attention(const flashmla::sparse_native_fp8::AttentionParams &params) {
    if constexpr (D == 576 && BM == 32 && !Dual) {
        // Partially occupied head tiles favor fewer gather row/address passes.
        // Retain fourteen Q fragments while keeping two three-stage CTAs per CU.
        // This also applies to rejected verify groups: each query still uses
        // its original indices, and the gated kernel preserves that contract.
        const bool partial_tile = params.heads > 16 && params.heads < 64 &&
                                  params.heads % BM != 0;
        const int groups = (params.heads + BM - 1) / BM;
        constexpr int MinPartitionTiles = 8;
        if (!params.qbf && partial_tile && (!Gated || params.heads > BM) &&
            params.width == flashmla::sparse_native_fp8::MaxTopk &&
            params.width / params.splits >= MinPartitionTiles * flashmla::sparse_native_fp8::TileKeys &&
            int64_t(params.queries) * params.splits * groups >= params.compute_units) {
            auto compact = prefer_narrow_consumer_gather(params)
                ? launch_attention<D, 3, BM, 1, false, 14, false, Gated, 4, true>
                : launch_attention<D, 3, BM, 1, false, 14, false, Gated, 8, true>;
            compact(params);
            return;
        }
    }
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
    // BF16-Q partial head groups also benefit from the eight-lane gather
    // and paired V loads on a full unsplit sparse list. Q conversion remains
    // in the existing loader; ready-Q8 keeps its measured cache-aware policy.
    const bool wide_bf16_query = D == 576 && params.qbf && params.heads != 16 &&
        params.heads % 64 != 0 && params.splits == 1 &&
        params.width == flashmla::sparse_native_fp8::MaxTopk;
    if (((!params.qbf && !narrow_cached) || wide_bf16_query) &&
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
    if constexpr (D == 576 && BM == 16 && !Dual && !Gated) {
        // The consumer owns the next KV gather after its V fragments are
        // resident, overlapping that copy with the remaining PV instructions.
        if (!params.qbf && params.heads < 16 && params.heads != 8 && params.two_stage &&
            params.splits == 1 && params.width == flashmla::sparse_native_fp8::MaxTopk &&
            prefer_narrow_consumer_gather(params))
            launch = launch_attention<D, 2, BM, 1, false, 0, false, false, 4, true, true>;
    }
    launch(params);
}
} // namespace flashmla_fp8_attention

namespace flashmla_fp8_attention {
template <int D, int ST, int H, bool Dual = false, bool Window = false, int BM = 64, int Prefix = 0>
__global__ __launch_bounds__(2 * (BM / 16) * 4 * 32, BM <= 32 ? 2 : 1) void native_verify_stream_kernel(const void *qraw,
    const F8 *kv, const int *ids, const unsigned *masks, cutlass::bfloat16_t *out, float *maxima,
    float *lse, const float *sink, const int *errors, float scale, int width, int kvrows, int qpr,
    const F8 *extra, int primary_rows, int lse_queries, int runtime_heads) {
    // H=0 shares the same pipeline across arbitrary logical head groups.
    const int logical_heads = H ? H : runtime_heads;
    // Paired V loads apply to the native single-KV folded path; short windows
    // and dual-KV verification retain their original fragment layout.
    using VerifyPipeline = flashmla::sparse_native_fp8::ws::Pipeline<DecodePolicy,
        !Dual && !Window && (BM >= 32)>;
    constexpr int G = (BM / 16) * 4 * 32, Threads = 2 * G;
    // Keep adjacent folded tiles of one request close in launch order so their
    // KV gathers can reuse cache lines even when the batch exceeds L2 capacity.
    const int heads = qpr * logical_heads, tiles = (heads + BM - 1) / BM;
    const int batch = H == 0 ? blockIdx.x / tiles : blockIdx.x;
    const int head = (H == 0 ? blockIdx.x % tiles : blockIdx.y) * BM,
              active = min(BM, heads - head), tid = threadIdx.x;
    if (errors[batch])
        return;
    extern __shared__ char bytes[];
    State state = VerifyPipeline::template shared_state<D, ST, BM, true>(bytes);
    int mask_stride = width / 32, first_mask_query = 0;
    const unsigned *query_masks = masks + (long)batch * qpr * mask_stride;
    if constexpr (H == 0) {
        constexpr int BaseSmem = Pipeline::smem_bytes<D, ST, BM, true>() +
            (Prefix ? BM * (D - Prefix * 32) - 24 : 0);
        auto *cached_masks = reinterpret_cast<unsigned *>(bytes + ((BaseSmem + 15) & ~15));
        // Cache only the queries touched by this head tile. Prefix14 leaves
        // room for these masks without exceeding the two-CTA shared budget.
        first_mask_query = head / logical_heads;
        const int cached_queries = (head + active - 1) / logical_heads - first_mask_query + 1;
        query_masks += first_mask_query * mask_stride;
        for (int i = tid; i < cached_queries * mask_stride; i += Threads)
            cached_masks[(i / mask_stride) * (mask_stride + 1) + i % mask_stride] = query_masks[i];
        query_masks = cached_masks;
        ++mask_stride;
    }
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
        VerifyPipeline::template produce<D, ST, BM, true, Prefix, Dual, true, H == 0 ? 8 : 0, H == 0>(state, kv, width, kvrows,
            scale, extra, primary_rows, query_masks, mask_stride, logical_heads, head, heads,
            first_mask_query);
        __ppu_barrier_sync(0, Threads, 15u);
        const int gt = role_lane<BM>();
        if (gt < active) {
            long row = (long)batch * heads + head + gt;
            if (maxima)
                maxima[row] = state.max[gt];
            const float value = state.sum[gt] > 0 ? state.max[gt] + log2f(state.sum[gt] / 448.f) : -INFINITY;
            flashmla::sparse_native_fp8::store_lse(lse, batch * qpr + (head + gt) / logical_heads,
                (head + gt) % logical_heads, logical_heads, lse_queries, value);
        }
    } else
        VerifyPipeline::template consume<D, ST, BM, 1, true, Window ? H : 0>(
            state, out + ((long)batch * heads + head) * 512,
            sink ? sink + (Window ? 0 : head) : nullptr, width, 1.f, active);
}
template <int D, int H, bool Dual = false, bool Window = false, int BM = 64, int Prefix = 0>
void launch_verify_stream(const flashmla::sparse_native_fp8::AttentionParams &params) {
    const int logical_heads = H ? H : params.heads;
    if constexpr (!Window && BM == 64) {
        const int folded_heads = params.qpr * logical_heads;
        // Avoid a mostly padded 64-row tile for short verify groups.
        if (folded_heads <= 32) {
            auto launch = folded_heads <= 16 ? launch_verify_stream<D, H, Dual, Window, 16>
                                             : launch_verify_stream<D, H, Dual, Window, 32>;
            launch(params);
            return;
        }
    }
    constexpr int ST = 3, Smem = Pipeline::smem_bytes<D, ST, BM, true>() +
        (Prefix ? BM * (D - Prefix * 32) - 24 : 0);
    auto kernel = native_verify_stream_kernel<D, ST, H, Dual, Window, BM, Prefix>;
    const int mask_queries = min(params.qpr, 1 + (BM + logical_heads - 2) / logical_heads);
    const int shared = H == 0 ? ((Smem + 15) & ~15) + mask_queries * (params.width / 32 + 1) * 4 : Smem;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, shared);
    const int tiles = (params.qpr * logical_heads + BM - 1) / BM;
    const dim3 grid = H == 0 ? dim3(tiles * params.batch) : dim3(params.batch, tiles);
    kernel<<<grid, 2 * (BM / 16) * 4 * 32, shared, params.stream>>>(
        params.q, static_cast<const F8 *>(params.kv), params.idx, params.masks,
        static_cast<cutlass::bfloat16_t *>(params.out), params.maxima, params.lse, params.sink,
        params.errors, params.scale, params.width, params.kvrows, params.qpr,
        static_cast<const F8 *>(params.extra), params.primary_rows, params.lse_queries, params.heads);
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
    // Three full 32-row head tiles favor the narrow gather with a reusable KV
    // pool; keep the wide streaming path for large decode batches.
    if (params.dim == 576 && params.heads == 96 && !params.qbf && !params.extra &&
        params.splits == 1 && params.width == flashmla::sparse_native_fp8::MaxTopk &&
        params.queries >= 2 * params.compute_units && prefer_narrow_consumer_gather(params)) {
        launch_attention<576, 3, 32, 1, false, 14, false, false, 4, true>(params);
        return;
    }
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
    if (params.dim == 576 && params.heads != 8 && params.heads != 16) {
        // A mostly occupied 64-row tile reuses each gathered KV tile across
        // more queries. Keep 32-row tiles when the wider tail wastes half
        // the work; the two layouts have matching rows per resident wave.
        const int tail = (params.qpr * params.heads) % 64;
        auto folded = params.heads > 32 && params.heads < 64 && (tail == 0 || tail > 32)
            ? launch_verify_stream<576, 0, false, false, 64>
            : launch_verify_stream<576, 0, false, false, 32, 14>;
        folded(folded_params);
        // Rejected groups retain each query's original list. Tail-masked
        // ordinary WS kernels support the same logical head count.
        auto independent = params.heads <= 16 ? dispatch_attention<576, 16, false, true>
                                              : dispatch_attention<576, 32, false, true>;
        independent(params);
        return;
    }
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
