// Native FP8 sparse decode/verify routing and auxiliary kernels.
// Warp-specialized attention kernels are implemented in native_fp8_decode_ws.cu.

#include "ppu/sparse_native_fp8_attention.cuh"

// WS launchers are implemented in native_fp8_decode_ws.cu.
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
         (params.heads <= 16 || (params.heads == 64 && params.queries >= 64))) ||
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

    native_verify_union_kernel<<<params.batch, 256, 0, params.stream>>>(
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
