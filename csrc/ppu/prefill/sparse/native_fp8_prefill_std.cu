// Standard CTA sparse attention selection, shared with the decode dispatcher.
#include "ppu/sparse_native_fp8_attention.cuh"

// H16/D512 prefill whose complete sparse list fits one 128-key tile.
namespace flashmla_fp8_prefill_single_tile {
using namespace cute;
using namespace flashmla::sparse_native_fp8::ws;
using ScoreRows = flashmla_fp8_std::ScoreRows;
constexpr int Heads = 16, Keys = 128, Dim = 512, Warps = 8, Threads = Warps * 32;
constexpr int SharedBytes = Keys * Dim;
static_assert(Heads * Dim == Keys * 64, "Q staging must fit the final 64-column KV group");
static_assert(2 * Warps * Heads * sizeof(float) + Heads * Keys <= SharedBytes);
static_assert(Heads * Dim * sizeof(cutlass::bfloat16_t) <= SharedBytes);

__global__ __launch_bounds__(Threads, 1) void single_tile_kernel(
    const F8 *__restrict__ q, const F8 *__restrict__ kv, const int *__restrict__ indices,
    cutlass::bfloat16_t *__restrict__ out, float *maxima, float *lse, const int *lengths,
    const float *sink, int width, int kvrows, float scale, float output_scale) {
    using KVLayout = decltype(tile_to_shape(Atom{}, Shape<Int<Keys>, Int<Dim>>{}));
    using PLayout = decltype(tile_to_shape(Atom{}, Shape<Int<Heads>, Int<Keys>>{}));
    using QK = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<_1, Int<Warps>, _1>>, Tile<Int<Heads>, Int<Keys>, _32>>;
    using PV = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<_1, Int<Warps>, _1>>,
        Tile<Int<Heads>, Layout<Shape<_16, Int<Warps>, _2>, Stride<_1, _32, _16>>, _32>>;
    const int tid = threadIdx.x, lane = tid % 32, warp = COOP_UNIF(tid / 32);
    const int row0 = lane / 4, row1 = row0 + 8, query = blockIdx.x;
    q += int64_t(query) * Heads * Dim;
    indices += int64_t(query) * width;
    out += int64_t(query) * Heads * Dim;
    const int length = COOP_UNIF(lengths ? max(0, min(width, lengths[query])) : width);
    extern __shared__ char bytes[];
    F8 *storage = reinterpret_cast<F8 *>(bytes);
    auto sk = make_tensor(make_smem_ptr(storage), KVLayout{});
    QK qk;
    auto tq = qk.get_slice(tid);
    auto coordinates = tq.partition_C(make_identity_tensor(Shape<Int<Heads>, Int<Keys>>{}));
    auto scores = tq.partition_fragment_C(
        make_tensor(make_smem_ptr(storage), Shape<Int<Heads>, Int<Keys>>{}, GenRowMajor{}));
    constexpr int ScoreValues = decltype(size(scores))::value;
    // Both startup copy phases gather the same rows. Load each index once;
    // retaining these two words avoids serial global loads between the phases.
    constexpr int GatherLanes = 4, GatherRows = Threads / GatherLanes;
    constexpr int GatherWidth = GatherLanes * 16, GatherRepeats = Keys / GatherRows;
    int tokens[GatherRepeats];
    CUTE_UNROLL
    for (int r = 0; r < GatherRepeats; ++r) {
        const int index = tid / GatherLanes + r * GatherRows;
        tokens[r] = index < length ? indices[index] : -1;
    }
    {
        // Stage Q in the last 64 KV columns. The remaining columns can be
        // copied concurrently; only overwrite Q after every warp has read it.
        constexpr int SharedQueryOffset = (Keys - Heads) * Dim;
        F8 *query_storage = storage + SharedQueryOffset;
        auto sq = make_tensor(make_smem_ptr(query_storage), LQ<Dim, Heads>{});
        for (int i = tid; i < Heads * Dim / 16; i += Threads)
            PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
                *reinterpret_cast<const uint128_t *>(q + i * 16),
                *reinterpret_cast<uint128_t *>(&sq(i / (Dim / 16), i % (Dim / 16) * 16)), true);
        cp_async_fence();
        auto gather_columns = [&](int first, int last) {
            CUTE_UNROLL
            for (int r = 0; r < GatherRepeats; ++r) {
                const int col = tid % GatherLanes * 16, index = tid / GatherLanes + r * GatherRows;
                const int token = tokens[r];
                const bool valid = (unsigned)token < (unsigned)kvrows;
                const F8 *src = kv + int64_t(valid ? token : 0) * Dim + col;
                F8 *dst = &sk(index, col);
                CUTE_UNROLL
                for (int j = first; j < last; ++j)
                    PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
                        *reinterpret_cast<const uint128_t *>(src + j * GatherWidth),
                        *reinterpret_cast<uint128_t *>(dst + j * GatherWidth * Keys), valid);
            }
            cp_async_fence();
        };
        constexpr int ColumnGroups = Dim / GatherWidth;
        gather_columns(0, ColumnGroups - 1);
        cp_async_wait<1>();
        __syncthreads();
        auto rq = tq.partition_fragment_A(
            make_tensor(make_smem_ptr(storage), Shape<Int<Heads>, Int<Dim>>{}, GenRowMajor{}));
        auto rk = tq.partition_fragment_B(
            make_tensor(make_smem_ptr(storage), Shape<Int<Keys>, _192>{}, GenRowMajor{}));
        const int qt = COOP_UNIF(int(reinterpret_cast<uintptr_t>(query_storage) / 16));
        CUTE_UNROLL
        for (int k = 0; k < Dim / 32; ++k)
            COOP_LDM(reinterpret_cast<uint32_t *>(&rq(0, 0, k)),
                qt + (k >> 1) * Heads * 4 + (k & 1) * 2, 1, 32);
        __syncthreads();
        gather_columns(ColumnGroups - 1, ColumnGroups);
        cp_async_wait<0>();
        __syncthreads();
        auto scores1 = tq.partition_fragment_C(
            make_tensor(make_smem_ptr(storage), Shape<Int<Heads>, Int<Keys>>{}, GenRowMajor{}));
        clear(scores);
        clear(scores1);
        const int kt = COOP_UNIF(int(reinterpret_cast<uintptr_t>(storage) / 16) + warp * 64);
        auto load = [&](int k) {
            COOP_LDM(reinterpret_cast<uint32_t *>(&rk(0, 0, k % 6)),
                kt + (k >> 1) * Keys * 4 + (k & 1) * 2, 32, 1);
        };
        load(0);
        load(1);
        load(2);
        CUTE_UNROLL
        for (int k = 0; k < Dim / 32; k += 3) {
            CUTE_UNROLL
            for (int j = 0; j < 3; ++j)
                if (k + 3 + j < Dim / 32)
                    load(k + 3 + j);
            CUTE_UNROLL
            for (int j = 0; j < 3; ++j)
                if (k + j < Dim / 32) {
                    if (j == 1)
                        gemm(qk, rq(_, _, k + j), rk(_, _, (k + j) % 6), scores1);
                    else
                        gemm(qk, rq(_, _, k + j), rk(_, _, (k + j) % 6), scores);
                }
        }
        CUTE_UNROLL
        for (int e = 0; e < size(scores); ++e) {
            const int col = int(get<1>(coordinates(e)));
            const int token = col < length ? indices[col] : -1;
            scores(e) = (unsigned)token < (unsigned)kvrows ? scores(e) + scores1(e) : -INFINITY;
        }
    }
    // Capture all V fragments before reusing KV storage for softmax and P.
    PV pv;
    auto tp = pv.get_slice(tid);
    auto rv = tp.partition_fragment_B(make_tensor(
        make_smem_ptr(storage), Layout<Shape<Int<Dim>, Int<Keys>>, Stride<_1, Int<Dim>>>{}));
    CUTE_UNROLL
    for (int k = 0; k < Keys / 32; ++k) {
        CUTE_UNROLL
        for (int n = 0; n < Dim / (Warps * 16); n += 2)
            PairedVLoad::load<Keys>(reinterpret_cast<uint32_t *>(&rv(0, n, k)),
                reinterpret_cast<uint32_t *>(&rv(0, n + 1, k)), storage,
                warp * 32 + (n / 2) * Warps * 32, k);
    }
    __syncthreads();
    float *partial_max = reinterpret_cast<float *>(storage);
    float *partial_sum = partial_max + Warps * Heads;
    F8 *probabilities = reinterpret_cast<F8 *>(partial_sum + Warps * Heads);
    auto sp = make_tensor(make_smem_ptr(probabilities), PLayout{});
    float maximum[2] = {Neg, Neg};
    flashmla_fp8_std::accumulate_score_max<ScoreRows, ScoreValues>(scores, maximum);
    CUTE_UNROLL
    for (int d = 1; d <= 2; d *= 2) {
        maximum[0] = fmaxf(maximum[0], __shfl_xor_sync(0xffffffff, maximum[0], d));
        maximum[1] = fmaxf(maximum[1], __shfl_xor_sync(0xffffffff, maximum[1], d));
    }
    if ((lane & 3) == 0) {
        partial_max[warp * Heads + row0] = maximum[0];
        partial_max[warp * Heads + row1] = maximum[1];
    }
    __syncthreads();
    maximum[0] = maximum[1] = Neg;
    CUTE_UNROLL
    for (int w = 0; w < Warps; ++w) {
        maximum[0] = fmaxf(maximum[0], partial_max[w * Heads + row0]);
        maximum[1] = fmaxf(maximum[1], partial_max[w * Heads + row1]);
    }
    maximum[0] = ceilf(maximum[0] * scale);
    maximum[1] = ceilf(maximum[1] * scale);
    float sums[2] = {0, 0};
    auto store_pair = [&](int e, uint16_t packed) {
        *reinterpret_cast<uint16_t *>(&sp(int(get<0>(coordinates(e))),
            PairedVLoad::column(int(get<1>(coordinates(e)))))) = packed;
    };
    flashmla_fp8_std::quantize_probability<ScoreRows, ScoreValues>(
        scores, sums, maximum[0], maximum[1], scale, store_pair);
    CUTE_UNROLL
    for (int d = 1; d <= 2; d *= 2) {
        sums[0] += __shfl_xor_sync(0xffffffff, sums[0], d);
        sums[1] += __shfl_xor_sync(0xffffffff, sums[1], d);
    }
    if ((lane & 3) == 0) {
        partial_sum[warp * Heads + row0] = sums[0];
        partial_sum[warp * Heads + row1] = sums[1];
    }
    __syncthreads();
    sums[0] = sums[1] = 0;
    CUTE_UNROLL
    for (int w = 0; w < Warps; ++w) {
        sums[0] += partial_sum[w * Heads + row0];
        sums[1] += partial_sum[w * Heads + row1];
    }
    if (warp == 0 && (lane & 3) == 0) {
        if (maxima) {
            maxima[query * Heads + row0] = maximum[0];
            maxima[query * Heads + row1] = maximum[1];
        }
        if (lse) {
            lse[query * Heads + row0] = sums[0] > 0 ? maximum[0] + log2f(sums[0] / 448.f) : -INFINITY;
            lse[query * Heads + row1] = sums[1] > 0 ? maximum[1] + log2f(sums[1] / 448.f) : -INFINITY;
        }
    }
    if (sink) {
        sums[0] += exp2f(sink[row0] * 1.4426950408889634f - maximum[0] + LogP);
        sums[1] += exp2f(sink[row1] * 1.4426950408889634f - maximum[1] + LogP);
    }
    float inv[2];
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i)
        inv[i] = sums[i] > 0 && sums[i] == sums[i] ? output_scale / sums[i] : 0.f;
    auto acc = tp.partition_fragment_C(
        make_tensor(make_smem_ptr(storage), Shape<Int<Heads>, Int<Dim>>{}, GenRowMajor{}));
    auto rp = tp.partition_fragment_A(sp);
    clear(acc);
    const int pt = COOP_UNIF(int(reinterpret_cast<uintptr_t>(probabilities) / 16));
    CUTE_UNROLL
    for (int k = 0; k < Keys / 32; ++k) {
        COOP_LDM(reinterpret_cast<uint32_t *>(&rp(0, 0, k)),
            pt + (k >> 1) * Heads * 4 + (k & 1) * 2, 1, 32);
        gemm(pv, rp(_, _, k), rv(_, _, k), acc);
    }
    PairedVLoad::restore_output(acc);
    // P must be consumed by every warp before output overwrites the same bytes.
    __syncthreads();
    auto output_coordinates = tp.partition_C(make_identity_tensor(Shape<Int<Heads>, Int<Dim>>{}));
    auto shared_output = make_tensor(make_smem_ptr(reinterpret_cast<cutlass::bfloat16_t *>(storage)),
        Layout<Shape<Int<Heads>, Int<Dim>>, Stride<Int<Dim>, _1>>{});
    CUTE_UNROLL
    for (int e = 0; e < size(acc); e += 2) {
        cutlass::bfloat16_t a(acc(e) * inv[(e >> 1) & 1]), b(acc(e + 1) * inv[(e >> 1) & 1]);
        const uint32_t packed = uint32_t(reinterpret_cast<uint16_t &>(a)) |
                                (uint32_t(reinterpret_cast<uint16_t &>(b)) << 16);
        *reinterpret_cast<uint32_t *>(&shared_output(int(get<0>(output_coordinates(e))),
            PairedVLoad::column(int(get<1>(output_coordinates(e)))))) = packed;
    }
    __syncthreads();
    for (int i = tid; i < Heads * Dim / 8; i += Threads)
        *reinterpret_cast<uint128_t *>(out + i * 8) =
            *reinterpret_cast<const uint128_t *>(&shared_output(i / (Dim / 8), i % (Dim / 8) * 8));
}

bool try_launch(const flashmla::sparse_native_fp8::AttentionParams &params,
    const flashmla::sparse_native_fp8::DeviceProperties &device) {
    // This kernel writes flat statistics; paged H16 decode has its own path.
    if (params.dim != Dim || params.width > Keys || params.lse_queries != 0 ||
        device.max_threads < Threads || device.shared_bytes < size_t(SharedBytes))
        return false;
    hggcFuncSetAttribute(single_tile_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, SharedBytes);
    single_tile_kernel<<<params.queries, Threads, SharedBytes, params.stream>>>(
        static_cast<const F8 *>(params.q), static_cast<const F8 *>(params.kv), params.idx,
        static_cast<cutlass::bfloat16_t *>(params.out), params.maxima, params.lse, params.lengths,
        params.sink, params.width, params.kvrows, params.scale, params.os);
    return true;
}
} // namespace flashmla_fp8_prefill_single_tile

bool try_sparse_fp8_std(const flashmla::sparse_native_fp8::AttentionParams &params) {
    if (params.queries <= 0 || (params.width / 64) <= 0 || (params.width / 64) > 34 ||
        (params.dim != 512 && params.dim != 576) || params.qbf ||
        params.qst != params.heads * params.dim || params.qsh != params.dim ||
        (params.heads != 16 && params.heads % 64 != 0))
        return false;
    const auto *device = flashmla::sparse_native_fp8::device_properties();
    if (!device)
        return false;
    if (params.heads == flashmla_fp8_prefill_single_tile::Heads)
        return flashmla_fp8_prefill_single_tile::try_launch(params, *device);

    // Low-query full-head grids retain the Warp-Interleave (WI) path.
    // A full-query CTA uses a larger per-thread register budget at high grids.
    constexpr size_t Shared = 3 * 64 * 576 + 64 * 64 + 2176 * 4 + 11 * 64 * 4 + 256;
    // Prefix residency has an initialization cost. The short-list full-grid
    // matrix favors native, so retain it for fewer than eight N64 tiles.
    if ((params.width / 64) < 8 || (long)params.queries * (params.heads / 64) <= device->compute_units ||
        device->max_threads < 256 || device->registers < 131072 || device->shared_bytes < Shared)
        return false;
    if (params.dim == 512)
        flashmla_fp8_std::dispatch<256, 4>(params);
    else {
        // Output-only prefill favors four-lane gathers. For statistics-producing
        // calls, wide gathers repay their extra row passes only with enough CTAs
        // and a sparse scan of the backing pool; repeated verify queries favor
        // the narrower gather even when their allocation is large.
        constexpr int MinPoolToScanRatio = 20, MinGridWaves = 4;
        const int64_t selected_rows = int64_t(params.queries) * (params.width / 64) * 64;
        // Discount a trailing key tile so page padding cannot flip this crossover.
        const int64_t pool_rows = std::max<int64_t>(
            0, int64_t(params.kvrows) -
                   int64_t(params.queries) * flashmla::sparse_native_fp8::TileKeys);
        const bool streaming =
            pool_rows > selected_rows * MinPoolToScanRatio &&
            int64_t(params.queries) * (params.heads / 64) >= MinGridWaves * device->compute_units;
        // Narrow gathers favor cache reuse; wide gathers hide scattered-load
        // latency once the pool outgrows aggregate or per-request cache reuse.
        // Guarded static kernels avoid carrying both gather address sets through
        // the attention loop. Their identical per-request hints are complementary;
        // neither kernel substitutes one query's indices for another's.
        const int requests =
            params.lse_queries > 1 ? params.queries / params.lse_queries : params.queries;
        const int64_t pool_bytes = pool_rows * params.dim;
        constexpr int MinScanCacheFootprints = 3, LargePoolCacheFootprints = 24;
        // With less selected work, the extra row passes of wide gathers cost
        // more than the latency they hide, even for a large backing allocation.
        const bool enough_scan =
            selected_rows * params.dim >= int64_t(device->cache_bytes) * MinScanCacheFootprints;
        const bool scattered_pool =
            pool_bytes > int64_t(device->cache_bytes) * LargePoolCacheFootprints ||
            pool_bytes * 2 > int64_t(requests) * device->cache_bytes * 3;
        // Dense selections repeatedly touch the same small per-request pool;
        // a large batch alone must not promote them to scattered wide gathers.
        constexpr int MinPoolToHistoryRatio = 2;
        const bool sparse_history = pool_rows > int64_t(requests) * (params.width / 64) *
                                                    flashmla::sparse_native_fp8::TileKeys *
                                                    MinPoolToHistoryRatio;
        if ((params.maxima || params.lse) && params.lse_queries > 1 && params.queries >= 2 &&
            enough_scan && scattered_pool && sparse_history) {
            flashmla_fp8_std::dispatch<256, 8, 8, true>(params);
            flashmla_fp8_std::dispatch<256, 8, 16, true>(params);
            return true;
        }
        auto launch = !params.maxima && !params.lse ? flashmla_fp8_std::dispatch<256, 8, 4>
                      : streaming                   ? flashmla_fp8_std::dispatch<256, 8, 16>
                                                    : flashmla_fp8_std::dispatch<256, 8, 8>;
        launch(params);
    }
    return true;
}
