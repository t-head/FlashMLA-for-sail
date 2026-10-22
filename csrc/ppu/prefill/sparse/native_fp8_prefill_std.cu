// Standard CTA sparse attention selection, shared with the decode dispatcher.
#include "ppu/sparse_native_fp8_attention.cuh"

bool try_sparse_fp8_std(const void *q, const void *kv, const int *idx, void *out,
    float *maxima, float *lse, const float *sink, const int *lengths, float scale, float os,
    int dim, int queries, int tiles, int kvrows, int heads, bool trusted, hggcStream_t stream, int lse_queries) {
    if (queries <= 0 || tiles <= 0 || tiles > 34 || (dim != 512 && dim != 576))
        return false;
    // Low-query grids retain the producer/consumer Warp-Interleave (WI) path.
    // A full-query CTA uses a larger per-thread register budget at high grids.
    int device = -1;
    if (hggcGetDevice(&device) != hggcSuccess)
        return false;
    struct CachedDevice {
        int id = -1, cus = 0, max_threads = 0, registers = 0, cache_bytes = 0;
        size_t shared = 0;
    };
    static thread_local CachedDevice cached;
    if (cached.id != device) {
        hggcDeviceProp props;
        if (hggcGetDeviceProperties(&props, device) != hggcSuccess)
            return false;
        if (props.major != 8 || props.minor != 9)
            return false;
        cached.id = device;
        cached.cus = props.multiProcessorCount;
        cached.cache_bytes = props.l2CacheSize;
        cached.max_threads = props.maxThreadsPerBlock;
        cached.registers = props.regsPerMultiprocessor;
        cached.shared = props.sharedMemPerBlockOptin;
    }
    constexpr size_t Shared = 3 * 64 * 576 + 64 * 64 + 2176 * 4 + 11 * 64 * 4 + 256;
    // Prefix residency has an initialization cost. The short-list full-grid
    // matrix favors native, so retain it for fewer than eight N64 tiles.
    if (tiles < 8 || (long)queries * (heads / 64) <= cached.cus || cached.max_threads < 256 ||
        cached.registers < 131072 || cached.shared < Shared)
        return false;
    if (dim == 512)
        flashmla_fp8_std::dispatch<256, 4>(q, kv, idx, out, maxima, lse, sink,
            lengths, scale, os, dim, queries, tiles, kvrows, trusted, stream, heads, lse_queries);
    else {
        // Output-only prefill favors four-lane gathers. For statistics-producing
        // calls, wide gathers repay their extra row passes only with enough CTAs
        // and a sparse scan of the backing pool; repeated verify queries favor
        // the narrower gather even when their allocation is large.
        constexpr int MinPoolToScanRatio = 20, MinGridWaves = 4;
        const int64_t selected_rows = int64_t(queries) * tiles * 64;
        // Discount a trailing key tile so page padding cannot flip this crossover.
        const int64_t pool_rows = std::max<int64_t>(0, int64_t(kvrows) -
            int64_t(queries) * flashmla::sparse_native_fp8::TileKeys);
        const bool streaming = pool_rows > selected_rows * MinPoolToScanRatio &&
            int64_t(queries) * (heads / 64) >= MinGridWaves * cached.cus;
        // Narrow gathers favor cache reuse; wide gathers hide scattered-load
        // latency once the pool outgrows aggregate or per-request cache reuse.
        // Guarded static kernels avoid carrying both gather address sets through
        // the attention loop. Their identical per-request hints are complementary;
        // neither kernel substitutes one query's indices for another's.
        const int requests = lse_queries > 1 ? queries / lse_queries : queries;
        const int64_t pool_bytes = pool_rows * dim;
        constexpr int MinScanCacheFootprints = 3, LargePoolCacheFootprints = 24;
        // With less selected work, the extra row passes of wide gathers cost
        // more than the latency they hide, even for a large backing allocation.
        const bool enough_scan = selected_rows * dim >=
            int64_t(cached.cache_bytes) * MinScanCacheFootprints;
        const bool scattered_pool = pool_bytes >
                int64_t(cached.cache_bytes) * LargePoolCacheFootprints ||
            pool_bytes * 2 > int64_t(requests) * cached.cache_bytes * 3;
        // Dense selections repeatedly touch the same small per-request pool;
        // a large batch alone must not promote them to scattered wide gathers.
        constexpr int MinPoolToHistoryRatio = 2;
        const bool sparse_history = pool_rows > int64_t(requests) * tiles *
            flashmla::sparse_native_fp8::TileKeys * MinPoolToHistoryRatio;
        if ((maxima || lse) && lse_queries > 1 && queries >= 2 &&
            enough_scan && scattered_pool && sparse_history) {
            flashmla_fp8_std::dispatch<256, 8, 8, true>(q, kv, idx, out,
                maxima, lse, sink, lengths, scale, os, dim, queries, tiles, kvrows,
                trusted, stream, heads, lse_queries);
            flashmla_fp8_std::dispatch<256, 8, 16, true>(q, kv, idx, out,
                maxima, lse, sink, lengths, scale, os, dim, queries, tiles, kvrows,
                trusted, stream, heads, lse_queries);
            return true;
        }
        auto launch = !maxima && !lse ? flashmla_fp8_std::dispatch<256, 8, 4>
                    : streaming ? flashmla_fp8_std::dispatch<256, 8, 16>
                                : flashmla_fp8_std::dispatch<256, 8, 8>;
        launch(q, kv, idx, out, maxima, lse, sink, lengths, scale, os, dim, queries,
            tiles, kvrows, trusted, stream, heads, lse_queries);
    }
    return true;
}
