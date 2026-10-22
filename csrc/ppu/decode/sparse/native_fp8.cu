// Native FP8 sparse decode/verify routing and auxiliary kernels.
// Warp-specialized attention kernels are implemented in native_fp8_decode_ws.cu.

#include "ppu/sparse_native_fp8_attention.cuh"

// WS launchers are implemented in native_fp8_decode_ws.cu.
void run_native_fp8_attention_ws(int dim, bool dual, bool two_stage,
    const void *q, const void *kv, const int *idx, void *out, float *maxima, float *lse,
    const float *sink, const int *lengths, float scale, float out_scale,
    int queries, int width, int kv_rows, int heads, bool q_bf16, int q_stride_t, int q_stride_h,
    hggcStream_t stream, int splits, const void *extra, const int *extra_indices,
    const int *extra_lengths, int main_width, int extra_width, int main_rows,
    int lse_queries, const int *errors, int qpr, bool cache_resident);

void run_native_fp8_verify_ws(const void *q, const void *kv, const int *idx, void *out,
    float *lse, const float *sink, const float *fold_sink, const int *lengths, float *maxima,
    int *united, unsigned *masks, int *errors, float scale, int dim, int batch, int qpr, int heads,
    int width, int kvrows, bool qbf, bool two_stage, int cache_bytes,
    hggcStream_t stream, int lse_queries);

void run_native_fp8_window_ws(const void *q, const void *kv, const int *ids,
    const void *extra, const int *extra_ids, void *out, float *lse, const float *sink,
    const int *lengths, const int *extra_lengths, int *united, unsigned *masks,
    int *errors, float scale, int batch, int qpr, int heads, int width,
    int padded_width, int extra_width, int primary_rows, int extra_rows, hggcStream_t stream, int lse_queries);

void run_native_d512_compact(const void *, const void *, const int *, void *, float *, float *,
    const float *, const int *, float, float, int, int, int, int, hggcStream_t, int = 0);

void run_native_d512_h16_fallback(const void *, const void *, const int *, void *, float *, float *,
    const float *, const int *, const int *, float, int, int, int, hggcStream_t, int, int);

void run_native_d512_h16_shared(const void *, const void *, const int *, void *, float *, float *,
    const float *, const int *, const int *, float, int, int, int, hggcStream_t, int, int);

bool try_sparse_fp8_std(const void *, const void *, const int *, void *,
    float *, float *, const float *, const int *, float, float, int, int, int, int, int, bool,
    hggcStream_t, int = 0);

void run_native_d512_dual_compact(const void *, const void *, const int *, const void *,
    const int *, void *, float *, const float *, const int *, const int *, const int *,
    float, int, int, int, int, int, int, hggcStream_t, int, int);

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

void run_native_fp8_attention_merge(const void *part, const float *plse, void *out, float *lse,
    const float *sink, int tokens, int splits, int heads, hggcStream_t stream, int lse_queries = 0,
    const float *partial_maxima = nullptr, float *maxima = nullptr) {
    const int blocks = tokens * heads;
    if (blocks) {
        auto kernel = maxima ? native_attention_merge_kernel<true, true>
            : lse ? native_attention_merge_kernel<false, true>
                  : native_attention_merge_kernel<false, false>;
        kernel<<<blocks, 64, 0, stream>>>(static_cast<const cutlass::bfloat16_t *>(part), plse,
            static_cast<cutlass::bfloat16_t *>(out), lse, sink, tokens, splits, heads,
            lse_queries, partial_maxima, maxima);
    }
}

void run_native_fp8_attention_decode(const void *q, const void *kv, const int *idx, void *out,
    float *lse, const float *sink, const int *lengths, float scale, float os, int dim, int queries,
    int width, int kvrows, int heads, bool qbf, int qst, int qsh, int splits, void *part,
    float *plse, float *maxima, bool two_stage, int compute_units, int cache_bytes,
    hggcStream_t stream, int lse_queries, float *partial_maxima) {
    if (splits == 1 && !qbf && heads % 64 == 0 &&
        try_sparse_fp8_std(q, kv, idx, out, maxima, lse, sink, lengths, scale,
            os, dim, queries, width / 64, kvrows, heads, false, stream, lse_queries))
        return;

    // Tiled attention requires at least eight N64 tiles, so it cannot consume
    // the short full-head cases. Keep long-list compact fallbacks after it.
    const bool short_full_heads = width == 128 && heads % 64 == 0 &&
                                  int64_t(queries) * (heads / 64) > compute_units;
    const bool compact_fallback =
        (width <= 128 && (heads <= 16 || (heads == 64 && queries >= 64))) ||
        (width >= 512 && ((queries >= 96 && (heads == 16 || (heads == 8 && width == 2048))) ||
                            (queries >= 192 && heads == 64)));
    if (dim == 512 && splits == 1 && !qbf && (short_full_heads || compact_fallback)) {
        run_native_d512_compact(q, kv, idx, out, maxima, lse, sink, lengths, scale, os, queries,
            width / 64, kvrows, heads, stream, lse_queries);
        return;
    }

    void *target = splits > 1 ? part : out;
    float *target_lse = splits > 1 ? plse : lse;
    const float *target_sink = splits > 1 ? nullptr : sink;
    run_native_fp8_attention_ws(dim, false, two_stage, q, kv, idx, target, splits > 1 ? partial_maxima : maxima,
        target_lse, target_sink, lengths, scale, os,
        queries, width, kvrows, heads, qbf, qst, qsh, stream, splits, nullptr, nullptr, nullptr, 0,
        0, 0, splits > 1 ? 0 : lse_queries, nullptr, 1,
        int64_t(kvrows) * dim <= cache_bytes);
    if (splits > 1)
        run_native_fp8_attention_merge(part, plse, out, lse, sink, queries, splits, heads,
            stream, lse_queries, partial_maxima, maxima);
}

__global__ void native_lse_natural_kernel(
    const float *src, float *dst, int batch, int queries, int heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * queries * heads)
        return;
    const int h = i % heads, q = (i / heads) % queries, b = i / (heads * queries);
    const float x = src[i];
    dst[(b * heads + h) * queries + q] = x == -INFINITY ? INFINITY : x * 0.6931471805599453f;
}
void run_native_lse_natural(
    const float *src, float *dst, int batch, int queries, int heads, hggcStream_t stream) {
    const int blocks = ((long)batch * queries * heads + 255) / 256;
    if (blocks)
        native_lse_natural_kernel<<<blocks, 256, 0, stream>>>(src, dst, batch, queries, heads);
}

// Two independent persistent KV pools, one online softmax and one PV output.
void run_native_fp8_attention_extra(const void *q, const void *kv, const int *idx, const void *extra,
    const int *extra_idx, void *out, float *lse, const float *sink, const int *lengths,
    const int *extra_lengths, float scale, int queries, int main_width, int extra_width,
    int main_rows, int extra_rows, int heads, bool qbf, int qst, int qsh, int splits, void *part,
    float *plse, float *maxima, bool two_stage, hggcStream_t stream, int lse_queries, int compute_units) {
    const bool small_heads = (heads == 8 || heads == 16) && queries > 2 * compute_units;
    const bool full_heads = heads % 64 == 0 && (int64_t)queries * (heads / 64) > compute_units;
    if (splits == 1 && !qbf && (small_heads || full_heads)) {
        run_native_d512_dual_compact(q, kv, idx, extra, extra_idx, out, lse, sink,
            lengths, extra_lengths, nullptr, scale, queries, heads, main_width,
            extra_width, main_rows, extra_rows, stream, lse_queries, 1);
        return;
    }
    void *target = splits > 1 ? part : out;
    float *target_lse = splits > 1 ? plse : lse;
    const float *target_sink = splits > 1 ? nullptr : sink;

    run_native_fp8_attention_ws(512, true, two_stage, q, kv, idx, target, maxima, target_lse, target_sink, lengths, scale, 1.f,
        queries, main_width + extra_width, main_rows + extra_rows, heads, qbf, qst, qsh, stream,
        splits, extra, extra_idx, extra_lengths, main_width, extra_width, main_rows,
        splits > 1 ? 0 : lse_queries, nullptr, 1, false);
    if (splits > 1)
        run_native_fp8_attention_merge(part, plse, out, lse, sink, queries, splits, heads, stream, lse_queries);
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

void run_native_fp8_attention_verify(const void *q, const void *kv, const int *idx, void *out,
    float *lse, const float *sink, const float *fold_sink, const int *lengths, float *maxima,
    int *united, unsigned *masks, int *errors, float scale, int dim, int batch, int qpr, int heads,
    int width, int kvrows, bool qbf, bool two_stage, int cache_bytes,
    hggcStream_t stream, int lse_queries) {
    const int folded_heads = qpr * heads, tail = folded_heads % 64;
    // A nearly empty final fold adds a full KV pass to a streaming pool.
    // Compact queries avoid that padding. A cheap reuse hint selects residency;
    // both branches preserve each query's original indices and lengths.
    if (dim == 512 && heads == 16 && !qbf && folded_heads > 64 &&
        tail > 0 && tail <= heads &&
        !flashmla::sparse_native_fp8::prefer_kv_reuse(int64_t(kvrows) * dim, cache_bytes)) {
        native_kv_reuse_hint_kernel<<<(batch + 255) / 256, 256, 0, stream>>>(
            idx, lengths, errors, batch, qpr, width, kvrows);
        run_native_d512_h16_shared(q, kv, idx, out, maxima, lse, sink, lengths, errors,
            scale, batch * qpr, width / 64, kvrows, stream, lse_queries, qpr);
        run_native_d512_h16_fallback(q, kv, idx, out, maxima, lse, sink, lengths, errors,
            scale, batch * qpr, width / 64, kvrows, stream, lse_queries, qpr);
        return;
    }

    native_verify_union_kernel<<<batch, 256, 0, stream>>>(
        idx, lengths, united, masks, errors, qpr, width, kvrows);
    run_native_fp8_verify_ws(q, kv, idx, out, lse, sink, fold_sink, lengths, maxima,
        united, masks, errors, scale, dim, batch, qpr, heads, width, kvrows, qbf,
        two_stage, cache_bytes, stream, lse_queries);
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

void run_native_fp8_attention_window(const void *q, const void *kv, const int *ids,
    const void *extra, const int *extra_ids, void *out, float *lse, const float *sink,
    const int *lengths, const int *extra_lengths, int *united, unsigned *masks,
    int *errors, float scale, int batch, int qpr, int heads, int width,
    int padded_width, int extra_width, int primary_rows, int extra_rows, hggcStream_t stream, int lse_queries) {
    native_window_union_kernel<<<batch, 256, 0, stream>>>(ids, lengths, extra_ids,
        extra_lengths, united, masks, errors, qpr, width, padded_width, extra_width,
        primary_rows, extra_rows);
    run_native_fp8_window_ws(q, kv, ids, extra, extra_ids, out, lse, sink,
        lengths, extra_lengths, united, masks, errors, scale, batch, qpr, heads,
        width, padded_width, extra_width, primary_rows, extra_rows, stream, lse_queries);
}
