// Native FP8 sparse prefill warp-specialized kernel and launch selection.
// Producer/consumer pipeline components are shared with decode.

#include "ppu/sparse_native_fp8_attention.cuh"

namespace flashmla_fp8_prefill_ws {
using namespace flashmla::sparse_native_fp8::ws;
template <int D, int BM>
using PrefillPipeline = flashmla::sparse_native_fp8::ws::Pipeline<PrefillPolicy, true, D == 512 && BM == 16>;
template <int D, int Stages, int BM, bool ReuseQ, int GatherLanes,
          bool Split = false, bool PagedLse = false, bool QBF16 = false>
__global__ __launch_bounds__(2 * (BM / 16) * 4 * 32,
    BM == 32 ? 2 : 1) void flash_sparse_fp8_prefill_ws_kernel(const void *__restrict__ q_data,
    const F8 *__restrict__ kv, const int *__restrict__ ids, cutlass::bfloat16_t *__restrict__ out,
    float *maxima, float *lse, const float *sink, const int *lengths, float scale, float outscale,
    int width, int kvrows, int heads, int splits, int lse_queries) {
    using Pipeline = PrefillPipeline<D, BM>;
    using Query = std::conditional_t<QBF16, cutlass::bfloat16_t, F8>;
    const Query *q = static_cast<const Query *>(q_data);
    constexpr int M = BM, G = (BM / 16) * 4 * 32, Threads = 2 * G;
    const int tid = threadIdx.x, output_query = blockIdx.x, head = blockIdx.y * BM;
    const int query = Split ? output_query / splits : output_query;
    const int part = Split ? output_query % splits : 0;
    q += ((long)query * heads + head) * D;
    ids += (long)query * width;
    out += ((long)output_query * heads + head) * V;
    if (sink)
        sink += head;
    if (maxima)
        maxima += (long)output_query * heads + head;
    if constexpr (!PagedLse)
        if (lse)
            lse += (long)output_query * heads + head;
    const int total_length = lengths ? max(0, min(width, lengths[query])) : width;
    int length = total_length;
    if constexpr (Split) {
        const int first_tile = (width / N) * part / splits;
        const int last_tile = (width / N) * (part + 1) / splits;
        ids += first_tile * N;
        width = (last_tile - first_tile) * N;
        length = max(0, min(width, total_length - first_tile * N));
    }
    length = COOP_UNIF(length);
    extern __shared__ char bytes[];
    State s = Pipeline::template shared_state<D, Stages, BM, ReuseQ>(bytes);
    auto sq = make_tensor(make_smem_ptr(s.q), LQ<D, BM>{});
    for (int i = tid; i < M * (D / 16); i += Threads) {
        const int r = i / (D / 16), c = i % (D / 16) * 16;
        if constexpr (QBF16)
            *reinterpret_cast<uint128_t *>(&sq(r, c)) =
                flashmla::sparse_native_fp8::load_bf16_as_fp8_16(q + r * D + c);
        else if constexpr (Pipeline::AsyncInputStaging)
            PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
                *reinterpret_cast<const uint128_t *>(q + r * D + c),
                *reinterpret_cast<uint128_t *>(&sq(r, c)), true);
        else {
            *reinterpret_cast<uint128_t *>(&sq(r, c)) =
                *reinterpret_cast<const uint128_t *>(q + r * D + c);
        }
    }
    if constexpr (Pipeline::AsyncInputStaging)
        cp_async_fence();
    if (tid < M) {
        s.max[tid] = Neg;
        s.sum[tid] = 0;
    }
    if constexpr (Pipeline::AsyncInputStaging) {
        // Sparse widths are multiples of 64. A contiguous sliced index tensor
        // can still have an unaligned base, so retain scalar staging for it.
        const bool vector_indices = (reinterpret_cast<uintptr_t>(ids) & 15u) == 0;
        if (vector_indices) {
            for (int i = tid * 4; i < width; i += Threads * 4)
                PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
                    *reinterpret_cast<const uint128_t *>(ids + i),
                    *reinterpret_cast<uint128_t *>(s.idx + i), true);
            cp_async_fence();
        } else {
            for (int i = tid; i < width; i += Threads)
                s.idx[i] = i < length ? ids[i] : -1;
        }
        cp_async_wait<0>();
        if (vector_indices) {
            // Each copy owner masks its own words before the CTA publishes them.
            for (int i = tid * 4; i < width; i += Threads * 4) {
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j)
                    if (i + j >= length)
                        s.idx[i + j] = -1;
            }
        }
    } else {
        for (int i = tid; i < width; i += Threads)
            s.idx[i] = i < length ? ids[i] : -1;
    }
    __ppu_barrier_sync(0, Threads, 15u);
    if (COOP_UNIF((role<BM>())) == 0) {
        Pipeline::template produce<D, Stages, BM, ReuseQ, 0, false, false, GatherLanes>(
            s, kv, length, kvrows, scale);
        __ppu_barrier_sync(0, Threads, 15u);
        const int gt = role_lane<BM>();
        if (gt < M) {
            if (maxima)
                maxima[gt] = s.max[gt];
            if (lse) {
                const float value = s.sum[gt] > 0 ? s.max[gt] + log2f(s.sum[gt] / 448.f) : -INFINITY;
                if constexpr (PagedLse)
                    flashmla::sparse_native_fp8::store_lse(lse, query, head + gt, heads, lse_queries, value);
                else
                    lse[gt] = value;
            }
        }
    } else
        Pipeline::template consume<D, Stages, BM, 1, false, 0, GatherLanes>(
            s, out, sink, length, outscale, BM, kv, kvrows);
}
template <int D, int Stages, int BM = 64, bool ReuseQ = false, int GatherLanes = 0,
          bool Split = false, bool PagedLse = false, bool QBF16 = false>
void launch(const flashmla::sparse_native_fp8::AttentionParams &params) {
    using Pipeline = PrefillPipeline<D, BM>;
    constexpr int Shared = Pipeline::template smem_bytes<D, Stages, BM, ReuseQ>();
    auto kernel = flash_sparse_fp8_prefill_ws_kernel<D, Stages, BM, ReuseQ, GatherLanes, Split, PagedLse, QBF16>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Shared);
    kernel<<<dim3(int64_t(params.queries) * params.splits, params.heads / BM), 2 * (BM / 16) * 4 * 32, Shared,
             params.stream>>>(params.q, static_cast<const F8 *>(params.kv),
                              params.idx, static_cast<cutlass::bfloat16_t *>(params.out),
                              params.maxima, params.lse, params.sink, params.lengths, params.scale,
                              params.os, params.width, params.kvrows, params.heads,
                              params.splits, params.lse_queries);
}
template <int GatherLanes, bool QBF16 = false>
void launch_h16(const flashmla::sparse_native_fp8::AttentionParams &params) {
    if (params.splits > 1)
        launch<512, 2, 16, true, GatherLanes, true, false, QBF16>(params);
    else if (params.lse_queries)
        launch<512, 2, 16, true, GatherLanes, false, true, QBF16>(params);
    else
        launch<512, 2, 16, true, GatherLanes, false, false, QBF16>(params);
}
} // namespace flashmla_fp8_prefill_ws

bool try_sparse_prefill_fp8_ws(const flashmla::sparse_native_fp8::AttentionParams &params,
                               bool wide_gather) {
    if (params.queries <= 0 || (params.width / 64) <= 0 || (params.width / 64) > 34 ||
        (params.dim != 512 && params.dim != 576))
        return false;
    const auto *device = flashmla::sparse_native_fp8::device_properties();
    if (!device)
        return false;
    if (params.heads == 16) {
        // Full H16 tiles retain Q in registers and reuse its shared storage for KV.
        // Two stages preserve more resident CTAs than three on large prefill grids.
        constexpr int HeadTile = 16, Stages = 2;
        constexpr int Shared =
            flashmla_fp8_prefill_ws::PrefillPipeline<512, HeadTile>::smem_bytes<512, Stages, HeadTile, true>();
        if (params.dim != 512 || params.width <= 2 * flashmla::sparse_native_fp8::TileKeys ||
            device->max_threads < 256 || device->shared_bytes < size_t(Shared))
            return false;
        if (params.qbf)
            flashmla_fp8_prefill_ws::launch_h16<4, true>(params);
        else if (wide_gather)
            flashmla_fp8_prefill_ws::launch_h16<8>(params);
        else
            flashmla_fp8_prefill_ws::launch_h16<4>(params);
        return true;
    }
    // BF16 Q for full-head grids currently uses std or the generic loader.
    if (params.qbf)
        return false;
    const long groups = (long)params.queries * (params.heads / 64);
    if (groups > device->compute_units)
        return false;
    const bool half_heads = groups <= device->compute_units / 2;
    if (half_heads) {
        if (device->max_threads < 512 ||
            device->shared_bytes <
                size_t(flashmla_fp8_prefill_ws::PrefillPipeline<576, 32>::smem_bytes<576, 3, 32, true>()))
            return false;
        if (params.dim == 512)
            flashmla_fp8_prefill_ws::launch<512, 3, 32, true>(params);
        else
            flashmla_fp8_prefill_ws::launch<576, 3, 32, true>(params);
    } else {
        if (device->max_threads < 1024 ||
            device->shared_bytes <
                size_t(flashmla_fp8_prefill_ws::PrefillPipeline<576, 64>::smem_bytes<576, 3, 64, false>()))
            return false;
        if (params.dim == 512)
            flashmla_fp8_prefill_ws::launch<512, 3, 64, false>(params);
        else
            flashmla_fp8_prefill_ws::launch<576, 3, 64, false>(params);
    }
    return true;
}

bool try_sparse_prefill_fp8_ws(const flashmla::sparse_native_fp8::AttentionParams &params) {
    return try_sparse_prefill_fp8_ws(params, false);
}
