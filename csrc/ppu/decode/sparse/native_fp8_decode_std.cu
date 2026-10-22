// Standard CTA sparse attention launchers for decode and verification.
#include "ppu/sparse_native_fp8_attention.cuh"

// H16 uses complete head groups; no runtime head-tail predicates are needed.
// Keep the device-property cache local when multiple extension builds are loaded;
// an externally linked template can coalesce its GNU-unique thread-local symbols.
template <int Heads, int Prefix, bool GuardRequests = false, int BlockN = 64,
    bool SharedRequests = false>
static void launch_native_d512_small(const void *q, const void *kv, const int *idx, void *out,
    float *maxima, float *lse, const float *sink, const int *lengths, float scale, float os,
    int queries, int tiles, int kvrows, hggcStream_t stream, const int *errors = nullptr, int lse_queries = 0,
    int request_queries = 1) {
    using namespace flashmla_fp8_std;
    constexpr int Shared = (16 + 128) * 512 + 16 * BlockN + (BlockN == 128 ? 128 : 2176) * 4 + 11 * 16 * 4 + 256;
    auto kernel = flash_fwd_sparse_fp8_std_kernel<16, 512, false, 256, Prefix, GuardRequests, Heads, BlockN, false, 8, SharedRequests>;
    int launch_shared = Shared;
    if constexpr (Heads == 16 && GuardRequests && !SharedRequests) {
        int device = 0;
        hggcGetDevice(&device);
        static thread_local int previous_device = -1, shared_per_cu = 0, cache_bytes = 0;
        if (previous_device != device) {
            hggcDeviceGetAttribute(&shared_per_cu, hggcDevAttrMaxSharedMemoryPerMultiprocessor, device);
            hggcDeviceGetAttribute(&cache_bytes, hggcDevAttrL2CacheSize, device);
            previous_device = device;
        }
        // One padded key tile per request must not change residency at the
        // reuse crossover. Larger pools need repeated full-pool passes to
        // offset contention from a third compact CTA.
        constexpr int MinPoolPasses = 3;
        const int64_t requested_rows = int64_t(queries) * tiles * 64;
        const int64_t accessed_rows = std::min<int64_t>(kvrows, requested_rows);
        const int64_t padding_rows = int64_t(queries / request_queries) *
                                     flashmla::sparse_native_fp8::TileKeys;
        const int64_t reuse_rows = std::max<int64_t>(0, accessed_rows - padding_rows);
        const bool cache_reuse = flashmla::sparse_native_fp8::prefer_kv_reuse(
            reuse_rows * 512, cache_bytes);
        const int64_t pool_rows = std::max<int64_t>(1, int64_t(kvrows) - padding_rows);
        constexpr int MaxRepeatedPoolCacheFootprints = 32;
        const bool repeated_pool = requested_rows >= pool_rows * MinPoolPasses &&
            pool_rows * 512 <= int64_t(cache_bytes) * MaxRepeatedPoolCacheFootprints;
        if (!cache_reuse && !repeated_pool)
            launch_shared = max(Shared, ((shared_per_cu / 3) / 256 + 1) * 256);
    }
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, launch_shared);
    kernel<<<queries, 256, launch_shared, stream>>>(q, static_cast<const E4M3 *>(kv), idx,
        static_cast<cutlass::bfloat16_t *>(out), scale, tiles * 64 / BlockN, kvrows, os, Heads, maxima,
        lse, sink, lengths, errors, lse_queries, nullptr, nullptr, nullptr, 0, 0, request_queries);
}

// Full head groups share KV; long lists trade more per-thread work for fewer warps.
void run_native_d512_compact(const void *q, const void *kv, const int *idx, void *out,
    float *maxima, float *lse, const float *sink, const int *lengths, float scale, float os,
    int queries, int tiles, int kvrows, int heads, hggcStream_t stream, int lse_queries) {
    using namespace flashmla_fp8_std;
    if (tiles == 2 && heads % 64 == 0) {
        constexpr int Threads = 512;
        constexpr int Shared = (64 + 128) * 512 + 64 * 128 + 128 * 4 + 11 * 64 * 4 + 256;
        auto kernel = flash_fwd_sparse_fp8_std_kernel<64, 512, false,
            Threads, 0, false, 64, 128>;
        hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Shared);
        kernel<<<dim3(int64_t(queries) * (heads / 64)), Threads, Shared, stream>>>(q,
            static_cast<const E4M3 *>(kv), idx, static_cast<cutlass::bfloat16_t *>(out),
            scale, 1, kvrows, os, heads, maxima, lse, sink, lengths, nullptr, lse_queries,
            nullptr, nullptr, nullptr, 0, 0, 1);
        return;
    }
    if (tiles == 2 && (heads == 8 || heads == 16)) {
        auto launch = heads == 8 ? launch_native_d512_small<8, 0, false, 128>
                                 : launch_native_d512_small<16, 0, false, 128>;
        launch(q, kv, idx, out, maxima, lse, sink, lengths, scale, os,
            queries, tiles, kvrows, stream, nullptr, lse_queries, 1);
        return;
    }
    if (heads == 64) {
        if (tiles <= 4)
            dispatch<512>(q, kv, idx, out, maxima, lse, sink, lengths, scale, os, 512,
                queries, tiles, kvrows, false, stream, heads, lse_queries);
        else
            dispatch<256, 4>(q, kv, idx, out, maxima, lse, sink, lengths, scale, os, 512,
                queries, tiles, kvrows, false, stream, heads, lse_queries);
    } else if (heads == 8) {
        launch_native_d512_small<8, 4>(
            q, kv, idx, out, maxima, lse, sink, lengths, scale, os, queries, tiles, kvrows, stream, nullptr, lse_queries);
    } else {
        if (tiles <= 4)
            launch_native_d512_small<16, 0>(q, kv, idx, out, maxima, lse, sink, lengths, scale, os,
                queries, tiles, kvrows, stream, nullptr, lse_queries);
        else
            launch_native_d512_small<16, 8>(q, kv, idx, out, maxima, lse, sink, lengths, scale, os,
                queries, tiles, kvrows, stream, nullptr, lse_queries);
    }
}

// Only requests rejected by the union classifier execute this independent path.
void run_native_d512_h16_fallback(const void *q, const void *kv, const int *idx, void *out,
    float *maxima, float *lse, const float *sink, const int *lengths, const int *errors,
    float scale, int queries, int tiles, int kvrows, hggcStream_t stream, int lse_queries,
    int request_queries) {
    launch_native_d512_small<16, 8, true>(q, kv, idx, out, maxima, lse, sink, lengths, scale, 1.f,
        queries, tiles, kvrows, stream, errors, lse_queries, request_queries);
}

// Shared queries retain three resident CTAs even when their allocation is large:
// adjacent queries repeatedly gather the same rows from that pool.
void run_native_d512_h16_shared(const void *q, const void *kv, const int *idx, void *out,
    float *maxima, float *lse, const float *sink, const int *lengths, const int *errors,
    float scale, int queries, int tiles, int kvrows, hggcStream_t stream, int lse_queries,
    int request_queries) {
    launch_native_d512_small<16, 8, true, 64, true>(q, kv, idx, out, maxima, lse, sink,
        lengths, scale, 1.f, queries, tiles, kvrows, stream, errors, lse_queries, request_queries);
}

void run_native_d512_h8_fallback(const void *q, const void *kv, const int *idx, void *out,
    float *maxima, float *lse, const float *sink, const int *lengths, const int *errors,
    float scale, int queries, int tiles, int kvrows, hggcStream_t stream, int lse_queries,
    int request_queries) {
    launch_native_d512_small<8, 4, true>(q, kv, idx, out, maxima, lse, sink, lengths, scale, 1.f,
        queries, tiles, kvrows, stream, errors, lse_queries, request_queries);
}

// Compact single-query pipeline for two independent KV pools.
template <int Heads, bool GuardRequests>
void launch_native_d512_dual_compact(const void *q, const void *kv, const int *idx,
    const void *extra, const int *extra_idx, void *out, float *lse, const float *sink,
    const int *lengths, const int *extra_lengths, const int *errors, float scale,
    int queries, int width, int extra_width, int primary_rows, int extra_rows,
    hggcStream_t stream, int lse_queries, int qpr, int runtime_heads) {
    using namespace flashmla_fp8_std;
    constexpr int BM = Heads == 64 ? 64 : 16;
    constexpr int Prefix = Heads == 16 ? 8 : 4;
    constexpr int Shared = (BM + 128) * 512 + BM * 64 + 2176 * 4 + 11 * BM * 4 + 256;
    // Full head tiles favor one gather pass over all 64 keys: four lanes
    // copy each row in 16-byte segments. Compact tiles retain their tuned
    // eight-lane gather; changing the load grouping does not change QK/PV.
    constexpr int GatherLanes = Heads == 64 ? 4 : 8;
    auto kernel = flash_fwd_sparse_fp8_std_kernel<BM, 512, false, 256, Prefix,
        GuardRequests, Heads, 64, true, GatherLanes>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Shared);
    kernel<<<dim3(int64_t(queries) * (Heads == 64 ? runtime_heads / 64 : 1)), 256, Shared, stream>>>(q, static_cast<const E4M3 *>(kv), idx,
        static_cast<cutlass::bfloat16_t *>(out), scale, (width + extra_width) / 64, primary_rows + extra_rows,
        1.f, runtime_heads, nullptr, lse, sink, lengths,
        errors, lse_queries, static_cast<const E4M3 *>(extra), extra_idx, extra_lengths,
        width, primary_rows, qpr);
}

void run_native_d512_dual_compact(const void *q, const void *kv, const int *idx,
    const void *extra, const int *extra_idx, void *out, float *lse, const float *sink,
    const int *lengths, const int *extra_lengths, const int *errors, float scale,
    int queries, int heads, int width, int extra_width, int primary_rows, int extra_rows,
    hggcStream_t stream, int lse_queries, int qpr) {
    auto launch = heads == 8 ? (errors ? launch_native_d512_dual_compact<8, true>
                                    : launch_native_d512_dual_compact<8, false>)
                            : heads == 16 ? (errors ? launch_native_d512_dual_compact<16, true>
                                                     : launch_native_d512_dual_compact<16, false>)
                                          : launch_native_d512_dual_compact<64, false>;
    launch(q, kv, idx, extra, extra_idx, out, lse, sink, lengths, extra_lengths, errors,
        scale, queries, width, extra_width, primary_rows, extra_rows, stream, lse_queries, qpr, heads);
}
