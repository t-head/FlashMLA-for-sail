// Standard CTA sparse attention launchers for decode and verification.
#include "ppu/sparse_native_fp8_attention.cuh"

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
