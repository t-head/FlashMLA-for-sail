// Native FP8 sparse prefill warp-specialized kernel and launch selection.
// Producer/consumer pipeline components are shared with decode.

#include "ppu/sparse_native_fp8_attention.cuh"

namespace flashmla_fp8_prefill_ws {
using namespace flashmla::sparse_native_fp8::ws;
using Pipeline = flashmla::sparse_native_fp8::ws::Pipeline<PrefillPolicy, true>;
template <int D, int Stages, int BM, bool ReuseQ>
__global__ __launch_bounds__(2 * (BM / 16) * 4 * 32,
    BM == 32 ? 2 : 1) void flash_sparse_fp8_prefill_ws_kernel(const F8 *__restrict__ q,
    const F8 *__restrict__ kv, const int *__restrict__ ids, cutlass::bfloat16_t *__restrict__ out,
    float *maxima, float *lse, const float *sink, const int *lengths, float scale, float outscale,
    int width, int kvrows, int heads) {
    constexpr int M = BM, G = (BM / 16) * 4 * 32, Threads = 2 * G;
    const int tid = threadIdx.x, query = blockIdx.x, head = blockIdx.y * BM;
    q += ((long)query * heads + head) * D;
    ids += (long)query * width;
    out += ((long)query * heads + head) * V;
    if (sink)
        sink += head;
    if (maxima)
        maxima += (long)query * heads + head;
    if (lse)
        lse += (long)query * heads + head;
    const int length = COOP_UNIF(lengths ? max(0, min(width, lengths[query])) : width);
    extern __shared__ char bytes[];
    State s = Pipeline::shared_state<D, Stages, BM, ReuseQ>(bytes);
    auto sq = make_tensor(make_smem_ptr(s.q), LQ<D, BM>{});
    for (int i = tid; i < M * (D / 16); i += Threads) {
        const int r = i / (D / 16), c = i % (D / 16) * 16;
        *reinterpret_cast<uint128_t *>(&sq(r, c)) =
            *reinterpret_cast<const uint128_t *>(q + r * D + c);
    }
    if (tid < M) {
        s.max[tid] = Neg;
        s.sum[tid] = 0;
    }
    for (int i = tid; i < width; i += Threads)
        s.idx[i] = i < length ? ids[i] : -1;
    __ppu_barrier_sync(0, Threads, 15u);
    if (COOP_UNIF((role<BM>())) == 0) {
        Pipeline::produce<D, Stages, BM, ReuseQ>(s, kv, length, kvrows, scale);
        __ppu_barrier_sync(0, Threads, 15u);
        const int gt = role_lane<BM>();
        if (gt < M) {
            if (maxima)
                maxima[gt] = s.max[gt];
            if (lse)
                lse[gt] = s.sum[gt] > 0 ? s.max[gt] + log2f(s.sum[gt] / 448.f) : -INFINITY;
        }
    } else
        Pipeline::consume<D, Stages, BM>(s, out, sink, length, outscale);
}
template <int D, int Stages, int BM = 64, bool ReuseQ = false>
void launch(const void *q, const void *kv, const int *ids, void *out, float *maxima, float *lse,
    const float *sink, const int *lengths, float scale, float outscale, int queries, int width,
    int kvrows, hggcStream_t stream, int heads = 64) {
    constexpr int Shared = Pipeline::smem_bytes<D, Stages, BM, ReuseQ>();
    auto kernel = flash_sparse_fp8_prefill_ws_kernel<D, Stages, BM, ReuseQ>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Shared);
    kernel<<<dim3(queries, heads / BM), 2 * (BM / 16) * 4 * 32, Shared, stream>>>(
        static_cast<const F8 *>(q), static_cast<const F8 *>(kv), ids,
        static_cast<cutlass::bfloat16_t *>(out), maxima, lse, sink, lengths, scale, outscale, width,
        kvrows, heads);
}
} // namespace flashmla_fp8_prefill_ws

bool try_sparse_prefill_fp8_ws(const void *q, const void *kv, const int *ids, void *out,
    float *maxima, float *lse, const float *sink, const int *lengths, float scale, float outscale,
    int dim, int queries, int tiles, int kvrows, int heads, hggcStream_t stream) {
    if (queries <= 0 || tiles <= 0 || tiles > 34 || (dim != 512 && dim != 576))
        return false;
    int device = -1;
    if (hggcGetDevice(&device) != hggcSuccess)
        return false;
    struct CachedDevice {
        int id = -1, cus = 0, max_threads = 0;
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
        cached.max_threads = props.maxThreadsPerBlock;
        cached.shared = props.sharedMemPerBlockOptin;
    }
    const long groups = (long)queries * (heads / 64);
    if (groups > cached.cus)
        return false;
    const bool half_heads = groups <= cached.cus / 2;
    if (half_heads) {
        if (cached.max_threads < 512 ||
            cached.shared < size_t(flashmla_fp8_prefill_ws::Pipeline::smem_bytes<576, 3, 32, true>()))
            return false;
        if (dim == 512)
            flashmla_fp8_prefill_ws::launch<512, 3, 32, true>(q, kv, ids, out, maxima, lse,
                sink, lengths, scale, outscale, queries, tiles * 64, kvrows, stream, heads);
        else
            flashmla_fp8_prefill_ws::launch<576, 3, 32, true>(q, kv, ids, out, maxima, lse,
                sink, lengths, scale, outscale, queries, tiles * 64, kvrows, stream, heads);
    } else {
        if (cached.max_threads < 1024 ||
            cached.shared < size_t(flashmla_fp8_prefill_ws::Pipeline::smem_bytes<576, 3, 64, false>()))
            return false;
        if (dim == 512)
            flashmla_fp8_prefill_ws::launch<512, 3, 64, false>(q, kv, ids, out, maxima, lse,
                sink, lengths, scale, outscale, queries, tiles * 64, kvrows, stream, heads);
        else
            flashmla_fp8_prefill_ws::launch<576, 3, 64, false>(q, kv, ids, out, maxima, lse,
                sink, lengths, scale, outscale, queries, tiles * 64, kvrows, stream, heads);
    }
    return true;
}
