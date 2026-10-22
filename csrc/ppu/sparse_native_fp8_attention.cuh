#pragma once

// Shared native FP8 sparse attention for prefill, decode and verification:
// E4M3 conversion, LSE storage, standard CTA kernels and WS pipeline components.

#include "ppu/sparse_native_fp8_arch.h"
#include "sparse_native_fp8_config.h"
#include <hggc_runtime.h>
#include <hgrt/hggc_extend_device_functions.h>
#include <hggc_bf16.h>
#include <hggc_fp8.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cstring>
#include <type_traits>
#ifdef USE_PPU
#include "ppu/ppu_include.hpp"
#endif

namespace flashmla::sparse_native_fp8 {

// Round to nearest even and saturate finite overflow. The actlize pair
// converter packs adjacent P8 or FP8 output elements into one 16-bit value.
__device__ __forceinline__ uint16_t pack_e4m3_pair(float2 value) {
    cutlass::Array<float, 2> input;
    input[0] = value.x;
    input[1] = value.y;
    const auto packed = cutlass::NumericArrayConverter<cutlass::float_e4m3_t, float, 2>{}(input);
    uint16_t result;
    static_assert(sizeof(packed) == sizeof(result));
    memcpy(&result, &packed, sizeof(result));
    return result;
}

// Preserve the SDK's scalar packing for BF16 Q/KV compatibility inputs;
// changing this sequence also changes code generation in decode kernels.
__device__ __forceinline__ uint16_t pack_e4m3_pair(__ppu_bfloat162_raw value) {
    return __hg_fp8x2_e4m3(__ppu_bfloat162(value)).__x;
}

}  // namespace flashmla::sparse_native_fp8

namespace flashmla::sparse_native_fp8 {
// Flat APIs expose base-2 LSE. Paged APIs expose natural LSE as [B,H,S],
// with +inf for an empty row. Apply that contract at the final store.
__device__ __forceinline__ void store_lse(float *dst, int token, int head,
    int heads, int queries_per_request, float value) {
    if (queries_per_request) {
        const int request = token / queries_per_request;
        const int query = token % queries_per_request;
        dst[(static_cast<long>(request) * heads + head) * queries_per_request + query] =
            value == -INFINITY ? INFINITY : value * 0.6931471805599453f;
    } else {
        dst[static_cast<long>(token) * heads + head] = value;
    }
}
} // namespace flashmla::sparse_native_fp8

// Native FP8 standard CTA sparse attention shared by prefill, decode, and verify.
// QK, online softmax, and PV execute as CTA-synchronized stages.
// Full/compact head tiles, cached QK prefixes, and independent-query fallback
// share the QK/online-softmax/PV pipeline, including dual-KV variants.

extern "C" {
__device__ unsigned ppu_mat_trans_b8(unsigned) __asm("llvm.ppu.mat.trans.b8");
}

// Raw ldmatrix.swzl. Validated bit-exact against cute's PPU_TSM_LD_SWZL copy
// For CUBE_W = 64 e4m3: row_stride = 8*64/16 = 32,
// swzl_mode = 1; A is (lbo,sbo) = (1,32), B is (32,1).
#define PPU_LDM(v, tsm, LBO, SBO)                                                                  \
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 "                            \
                 "{%0, %1, %2, %3}, [%4], %5, %6, %7;"                                             \
        : "=r"((v)[0]), "=r"((v)[1]), "=r"((v)[2]), "=r"((v)[3])                                   \
        : "l"(tsm), "r"(LBO), "r"(SBO), "n"(1))
using namespace cute;
using E4M3 = cutlass::float_e4m3_t;

#define UNIF(x) ((int)__ppu_read_firstlane((unsigned)(x)))
namespace flashmla_fp8_std {
static constexpr int DV = 512;
static constexpr float NEG_INF = -1e30f;
static constexpr float P_SCALE = 448.f;
static constexpr float LOG2_P_SCALE = 8.807354922f;
using AtomS8 = decltype(composition(Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

template <int BlockM, int DenseThreads, int BlockN = 64> struct Fp8SparsePrefillTraits {
    static constexpr int BN = BlockN;
    static_assert(BlockM == 16 || BlockM == 64, "unsupported head tile");
    static constexpr bool Compact = BlockM == 16;
    static constexpr int Threads = Compact ? 256 : DenseThreads;
    static constexpr int QKWarpN = Compact ? 4 : (DenseThreads == 256 ? 2 : 4);
    static constexpr int QKThreads = (BlockM / 16) * QKWarpN * 32;
    static constexpr int MaxTopk = BlockN == 128 ? 128 : 2176;
    using MmaQK = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<Int<BlockM / 16>, Int<QKWarpN>, _1>>,
        Tile<Int<BlockM>, Int<16 * QKWarpN>, _32>>;
    using MmaPV = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<_1, Int<Threads / 32>, _1>>, Tile<_16, Int<Threads / 2>, _32>>;
    using LayoutP = decltype(tile_to_shape(AtomS8{}, Shape<Int<BlockM>, Int<BN>>{}));

    CUTE_HOST_DEVICE static constexpr int score_row_offset(int e) {
        return ((e >> 1) & 1) * 8;
    }

    CUTE_HOST_DEVICE static constexpr int output_row(int e) {
        return (Compact ? 0 : ((e >> 3) & 3) * 16) + ((e >> 1) & 1) * 8;
    }
};

template <int BM, int DQK, bool TRUSTED_FULL_TOPK,
    int DenseThreads = 512,
    int Prefix = 0, bool GuardRequests = false, int LogicalHeads = BM, int BlockN = 64, bool Dual = false, int GatherLanes = 8, bool SharedRequests = false, bool GuardGatherReuse = false>
__global__ __launch_bounds__(Fp8SparsePrefillTraits<BM, DenseThreads, BlockN>::Threads,
    2) void flash_fwd_sparse_fp8_std_kernel(const void *__restrict__ gQRaw,
    const E4M3 *__restrict__ gKV, const int *__restrict__ gIdx, cutlass::bfloat16_t *__restrict__ gO,
    float qk_scale_log2, int n_tiles, int s_kv, float out_scale, int h_q, float *__restrict__ gMaxLogits, float *__restrict__ gLse,
    const float *__restrict__ gAttnSink, const int *__restrict__ gTopkLength,
    const int *request_errors, int lse_queries = 0, const E4M3 *extra_kv = nullptr,
    const int *extra_ids = nullptr, const int *extra_lengths = nullptr,
    int primary_width = 0, int primary_rows = 0, int request_queries = 1) {
    using Traits = Fp8SparsePrefillTraits<BM, DenseThreads, BlockN>;
    constexpr int BN = BlockN, NK_PV = BN / 32;
    constexpr bool SingleTile = BlockN == 128;
    constexpr bool COMPACT = Traits::Compact;
    static_assert(LogicalHeads == BM || (BM == 16 && LogicalHeads == 8),
        "supported partial head tile");
    constexpr int NTHR = Traits::Threads, QK_THREADS = Traits::QKThreads;
    using MmaQK = typename Traits::MmaQK;
    using MmaPV = typename Traits::MmaPV;
    using LayoutP8 = typename Traits::LayoutP;
    static_assert(DQK == 512 || DQK == 576, "supported q8 sparse head dims are 512 and 576");
    static constexpr int NK_QK = DQK / 32;
    static constexpr int NFrag = (BN / 16) / Traits::QKWarpN, ScoreValues = 8 * NFrag;
    static constexpr int QTILE = BM * DQK, KTILE = BN * DQK;
    using LayoutQ8 = decltype(tile_to_shape(AtomS8{}, Shape<Int<BM>, Int<DQK>>{}));
    using LayoutK8 = decltype(tile_to_shape(AtomS8{}, Shape<Int<BN>, Int<DQK>>{}));

    extern __shared__ char smem[];
    E4M3 *pQ = reinterpret_cast<E4M3 *>(smem);
    E4M3 *pK = pQ + QTILE; // Two alternating swizzled KV stages.
    E4M3 *pP = pK + (SingleTile ? 1 : 2) * KTILE;
    int *sIdx = reinterpret_cast<int *>(pP + BM * BN);
    float *sf = reinterpret_cast<float *>(sIdx + Traits::MaxTopk);
    float *sPmax = sf, *sPsum = sf + 4 * BM, *sRunMax = sf + 8 * BM;
    float *sRunSum = sRunMax + BM, *sScale = sRunSum + BM;
    int *sFlag = reinterpret_cast<int *>(sScale + BM);

    const int tid = threadIdx.x, wp = tid / 32, lane = tid % 32;
    // Keep head groups adjacent to reuse KV, without the grid.y size limit.
    const int head_groups = COMPACT ? 1 : h_q / BM;
    const int q_token = blockIdx.x / head_groups;
    const int head_group = blockIdx.x % head_groups;

    if constexpr (GuardRequests) {
        if (bool(request_errors[q_token / request_queries]) == SharedRequests)
            return;
    }
    if constexpr (GuardGatherReuse) {
        static_assert(!Dual && (GatherLanes == 8 || GatherLanes == 16));
        // This agreement only selects gather width. Both kernels consume every
        // query's own indices, even when the sampled prefix agrees by chance.
        // The launcher supplies at least two queries per request. Keep the
        // choice per request so a batch may mix shared and independent histories.
        const int first_query = (q_token / lse_queries) * lse_queries;
        const int width = n_tiles * BN;
        const int first = !gTopkLength || gTopkLength[first_query] > 0
            ? gIdx[int64_t(first_query) * width] : -1;
        const int next = !gTopkLength || gTopkLength[first_query + 1] > 0
            ? gIdx[int64_t(first_query + 1) * width] : -1;
        const bool reused = (unsigned)first < (unsigned)s_kv && first == next;
        if (reused != (GatherLanes == 8))
            return;
    }
    const long q_head_offset = ((long)q_token * h_q + head_group * BM) * DQK;
    const E4M3 *gQ8 = reinterpret_cast<const E4M3 *>(gQRaw) + q_head_offset;
    const int extra_width = Dual ? n_tiles * BN - primary_width : 0;
    gIdx += (long)q_token * (Dual ? primary_width : n_tiles * BN);
    if constexpr (Dual) extra_ids += (long)q_token * extra_width;
    gO += ((long)q_token * h_q + head_group * BM) * DV;
    if (gMaxLogits)
        gMaxLogits += (long)q_token * h_q + head_group * BM;

    if (gAttnSink)
        gAttnSink += head_group * BM;
    int main_length = 0, secondary_length = 0;
    int topk_length;
    if constexpr (Dual) {
        main_length = gTopkLength ? max(0, min(primary_width, gTopkLength[q_token])) : primary_width;
        secondary_length = extra_lengths ? max(0, min(extra_width, extra_lengths[q_token])) : extra_width;
        topk_length = secondary_length > 0 ? primary_width + secondary_length : main_length;
    } else {
        topk_length = gTopkLength ? max(0, min(n_tiles * BN, gTopkLength[q_token])) : n_tiles * BN;
    }
    // Match the BF16 kernel's dynamic work bound. Keep one all-invalid tile for
    // length zero so output/auxiliary behavior remains identical to the old path.
    const int active_tiles = SingleTile ? 1 : (topk_length > 0 ? (topk_length + BN - 1) / BN : 1);
    const int wm = wp % (BM / 16), wn = (wp / (BM / 16)) % Traits::QKWarpN, mi = lane / 4;
    const int r0 = wm * 16 + mi, r1 = r0 + 8;

    Tensor sQt = make_tensor(make_smem_ptr(pQ), LayoutQ8{});
    Tensor sKt0 = make_tensor(make_smem_ptr(pK), LayoutK8{});
    Tensor sPt = make_tensor(make_smem_ptr(pP), LayoutP8{});

    MmaQK mqk;
    auto tqk = mqk.get_slice(tid % QK_THREADS);
    MmaPV mpv;
    auto tpv = mpv.get_slice(tid);
    Tensor gOt = make_tensor(make_gmem_ptr(gO), Shape<Int<BM>, Int<DV>>{}, GenRowMajor{});
    Tensor acc_o = tpv.partition_fragment_C(gOt);
    clear(acc_o);

    // tile_to_shape(8x64 atom, (rows, cols)) puts cube (r/8, c/64) at
    //   ((c/64)*(rows/8) + r/8)*512 + (r%8)*64 + (c%64)
    // and for a 16-aligned row r%8 == 0, so the swizzle XOR key is 0.
    const int qk_lbo_a = 1, qk_sbo_a = 32;
    const int qk_lbo_b = 32, qk_sbo_b = 1;
    // Tile addresses are uniform within each warp; keep them in scalar registers.
    const int qtsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pQ) / 16) + wm * 64);
    const int ktsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pK) / 16) + wn * 64);
    const int ptsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pP) / 16));
    const int kstage = KTILE / 16;

    Tensor rQ = tqk.partition_fragment_A(sQt);
    Tensor rK = tqk.partition_fragment_B(sKt0);
    Tensor rP = tpv.partition_fragment_A(sPt);

    // Stage Q once, retain a short fragment prefix, and stream the remainder
    // through the QK loop to keep register use within two CTAs per CU.
    {
        Tensor gQt = make_tensor(make_gmem_ptr(gQ8), Shape<Int<BM>, Int<DQK>>{}, GenRowMajor{});
        for (int i = tid; i < BM * (DQK / 16); i += NTHR) {
            int r = i / (DQK / 16), g = i % (DQK / 16);
            *reinterpret_cast<cute::uint128_t *>(&sQt(r, g * 16)) =
                r < LogicalHeads ? *reinterpret_cast<const cute::uint128_t *>(&gQt(r, g * 16))
                                 : cute::uint128_t{};
        }

    }
    if (tid < BM) {
        sRunMax[tid] = NEG_INF;
        sRunSum[tid] = 0.f;
    }
    // The per-tile index load was a *dependent* gmem round-trip: the token id had
    // to arrive before any cp.async address could be formed, so nothing hid it.
    // Hoist only the prefix whose tiles can contain a valid index into smem once.
    for (int i = tid; i < active_tiles * BN; i += NTHR) {
        if constexpr (Dual) {
            int token = -1;
            if (i < primary_width) {
                if (i < main_length) {
                    const int value = gIdx[i];
                    if ((unsigned)value < (unsigned)primary_rows) token = value;
                }
            } else if (i - primary_width < secondary_length) {
                const int value = extra_ids[i - primary_width];
                if ((unsigned)value < (unsigned)(s_kv - primary_rows)) token = value + primary_rows;
            }
            sIdx[i] = token;
        } else if constexpr (TRUSTED_FULL_TOPK) {
            // The final partial tile uses slot bounds rather than index validity.
            sIdx[i] = i < topk_length ? gIdx[i] : (COMPACT ? -1 : 0);
        } else {
            sIdx[i] = i < topk_length ? gIdx[i] : -1;
        }
    }
    __syncthreads();

    // Per-thread gather offsets are tile-invariant: compute the swizzled
    // destinations once instead of re-deriving the XOR on every granule.
    // Gather whole 16-byte granules cooperatively; wider groups amortize
    // sparse row lookups when the grid has enough parallel work.
    static constexpr int GATHER_LANES = GatherLanes;
    static constexpr int GATHER_ROWS = NTHR / GATHER_LANES;
    static constexpr int GATHER_REPEATS = BN / GATHER_ROWS;
    static constexpr int GATHER_WIDTH = GATHER_LANES * 16;
    static constexpr int KSEGS = (DQK + GATHER_WIDTH - 1) / GATHER_WIDTH;
    const int grow = tid / GATHER_LANES, glane = tid % GATHER_LANES, col0 = glane * 16;
    // Both gather widths are multiples of the 64-byte cube width.
    // Keep one lane offset; later cubes add a constant physical stride.
    int gather_offsets[GATHER_REPEATS];
    CUTE_UNROLL
    for (int r = 0; r < GATHER_REPEATS; ++r)
        gather_offsets[r] = (int)(reinterpret_cast<const uint8_t *>(
            &sKt0(grow + r * GATHER_ROWS, col0)) - reinterpret_cast<const uint8_t *>(pK));

    auto issue_K = [&](int t) {
        uint8_t *kdst = reinterpret_cast<uint8_t *>(pK + (t & 1) * KTILE);
        CUTE_UNROLL
        for (int r = 0; r < GATHER_REPEATS; ++r) {
            const int row = grow + r * GATHER_ROWS;
            const int kdst0 = gather_offsets[r];
            const int tok = sIdx[t * BN + row];
            if constexpr (TRUSTED_FULL_TOPK) {
                const bool full_tile = (t + 1) * BN <= topk_length;
                if (full_tile) {
                    const uint8_t *src =
                        reinterpret_cast<const uint8_t *>(gKV) + (long)tok * DQK + col0;
                    CUTE_UNROLL
                    for (int j = 0; j < KSEGS; ++j)
                        if (col0 + j * GATHER_WIDTH < DQK)
                            cute::PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                                *reinterpret_cast<const cute::uint128_t *>(src + j * GATHER_WIDTH),
                                *reinterpret_cast<cute::uint128_t *>(
                                    kdst + kdst0 + j * GATHER_WIDTH * BN));
                } else {
                    const bool ok = t * BN + row < topk_length;
                    const uint8_t *src =
                        reinterpret_cast<const uint8_t *>(gKV) + (long)(ok ? tok : 0) * DQK + col0;
                    CUTE_UNROLL
                    for (int j = 0; j < KSEGS; ++j)
                        if (col0 + j * GATHER_WIDTH < DQK)
                            cute::PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                                *reinterpret_cast<const cute::uint128_t *>(src + j * GATHER_WIDTH),
                                *reinterpret_cast<cute::uint128_t *>(
                                    kdst + kdst0 + j * GATHER_WIDTH * BN),
                                ok);
                }
            } else {
                // one unsigned compare: every invalid encoding production uses
                // (-1, INT_MIN, s_kv, INT_MAX) maps to an unsigned value >= s_kv.
                const bool ok = (unsigned)tok < (unsigned)s_kv;
                const bool secondary = Dual && ok && tok >= primary_rows;
                const E4M3 *base = secondary ? extra_kv : gKV;
                const int row = ok ? tok - (secondary ? primary_rows : 0) : 0;
                const uint8_t *src = reinterpret_cast<const uint8_t *>(base) + (long)row * DQK + col0;
                CUTE_UNROLL
                for (int j = 0; j < KSEGS; ++j)
                    if (col0 + j * GATHER_WIDTH < DQK)
                        cute::PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                            *reinterpret_cast<const cute::uint128_t *>(src + j * GATHER_WIDTH),
                            *reinterpret_cast<cute::uint128_t *>(kdst + kdst0 + j * GATHER_WIDTH * BN),
                            ok);
            }
        }
    };
    // PV reads the same swizzled KV stage as QK for both head tiles.
    const int vrow = (lane & 3) * 4 + ((lane >> 2) & 3);
    const int vcol = wp * 16 + (lane >> 4) * 4;
    const int voff = (int)(reinterpret_cast<const uint8_t *>(&sKt0(vrow, vcol)) -
                           reinterpret_cast<const uint8_t *>(pK));
    if constexpr (Prefix > 0) {
        CUTE_UNROLL
        for (int k = 0; k < Prefix; ++k)
            PPU_LDM((uint32_t *)&rQ(0, 0, k), qtsm0 + (k >> 1) * (BM * 4) + (k & 1) * 2, qk_lbo_a,
                qk_sbo_a);
    }
    issue_K(0);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    for (int t = 0; t < active_tiles; ++t) {
        if (t + 1 < active_tiles) {
            issue_K(t + 1);
            cute::cp_async_fence();
        }

        Tensor acc_s = tqk.partition_fragment_C(sPt);
        clear(acc_s);
        if (tid < QK_THREADS) {
// Streamed QK: only a small window of Q/K fragments is live at once, so the
// per-thread register footprint drops far enough to run 2 CTAs/CU.
#define QADDR(kk) (qtsm0 + ((kk) >> 1) * (BM * 4) + ((kk) & 1) * 2)
#define KADDR(kk) (ktsm0 + ((t & 1) * kstage) + ((kk) >> 1) * (BN * 4) + ((kk) & 1) * 2)
            if constexpr (Prefix > 0) {
                auto load = [&](int k) {
                    if (k >= Prefix)
                        PPU_LDM((uint32_t *)&rQ(0, 0, k), QADDR(k), qk_lbo_a, qk_sbo_a);
                    CUTE_UNROLL
                    for (int n = 0; n < NFrag; ++n)
                        PPU_LDM((uint32_t *)&rK(0, n, k), KADDR(k) + n * Traits::QKWarpN * 64,
                            qk_lbo_b, qk_sbo_b);
                };
                load(0);
                load(1);
                CUTE_UNROLL
                for (int k = 0; k < NK_QK; ++k) {
                    if (k + 2 < NK_QK)
                        load(k + 2);
                    cute::gemm(mqk, rQ(_, _, k), rK(_, _, k), acc_s);
                }
            } else {
                auto load = [&](int k) {
                    PPU_LDM((uint32_t *)&rQ(0, 0, k), QADDR(k), qk_lbo_a, qk_sbo_a);
                    CUTE_UNROLL
                    for (int n = 0; n < NFrag; ++n)
                        PPU_LDM((uint32_t *)&rK(0, n, k), KADDR(k) + n * Traits::QKWarpN * 64,
                            qk_lbo_b, qk_sbo_b);
                };
                load(0);
                load(1);
                load(2);
                CUTE_UNROLL
                for (int k = 0; k + 2 < NK_QK; k += 3) {
                    if (k + 3 < NK_QK)
                        load(k + 3);
                    if (k + 4 < NK_QK)
                        load(k + 4);
                    if (k + 5 < NK_QK)
                        load(k + 5);
                    cute::gemm(mqk, rQ(_, _, k), rK(_, _, k), acc_s);
                    cute::gemm(mqk, rQ(_, _, k + 1), rK(_, _, k + 1), acc_s);
                    cute::gemm(mqk, rQ(_, _, k + 2), rK(_, _, k + 2), acc_s);
                }
                constexpr int Tail = (NK_QK / 3) * 3;
                if constexpr (NK_QK % 3 >= 1)
                    cute::gemm(mqk, rQ(_, _, Tail), rK(_, _, Tail), acc_s);
                if constexpr (NK_QK % 3 == 2)
                    cute::gemm(mqk, rQ(_, _, Tail + 1), rK(_, _, Tail + 1), acc_s);
            }

#undef QADDR
#undef KADDR
            CUTE_UNROLL
            for (int e = 0; e < ScoreValues; ++e) {
                const int col = wn * 16 + (e / 8) * Traits::QKWarpN * 16 + (lane % 4) * 2 +
                                ((e >> 2) & 1) * 8 + (e & 1);
                const bool valid = TRUSTED_FULL_TOPK
                                       ? t * BN + col < topk_length
                                       : (unsigned)sIdx[t * BN + col] < (unsigned)s_kv;

                acc_s(e) = valid ? acc_s(e) : -INFINITY;
            }
            if (tid == 0)
                *sFlag = 0;

            float mx[2] = {NEG_INF, NEG_INF};
            CUTE_UNROLL
            for (int e = 0; e < ScoreValues; ++e) {
                int s = Traits::score_row_offset(e) ? 1 : 0;
                mx[s] = fmaxf(mx[s], acc_s(e)); // raw; scaled once in the combine
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                mx[s] = fmaxf(mx[s], __shfl_xor_sync(0xffffffff, mx[s], 1));
                mx[s] = fmaxf(mx[s], __shfl_xor_sync(0xffffffff, mx[s], 2));
            }
            if ((lane & 3) == 0) {
                sPmax[wn * BM + r0] = mx[0];
                sPmax[wn * BM + r1] = mx[1];
            }
        } // All participating QK warps reconverge before the block barrier.
        __syncthreads();
        if constexpr (!SingleTile) {
        if (tid < BM) {
            float m = fmaxf(sPmax[tid], sPmax[BM + tid]);
            if constexpr (Traits::QKWarpN == 4)
                m = fmaxf(m, fmaxf(sPmax[2 * BM + tid], sPmax[3 * BM + tid]));
            m *= qk_scale_log2;
            float mprev = sRunMax[tid];
            // Quantise the running max to whole exp2 steps: the correction factor
            // becomes an exact power of two and, more importantly, stays at 1 for
            // most tiles, so the 64-multiply acc_o rescale is skipped.
            float mnew = fmaxf(mprev, ceilf(m));
            sRunMax[tid] = mnew;
            float sc = (mprev == NEG_INF) ? 1.f : exp2f(mprev - mnew);
            sScale[tid] = sc;
            if (sc != 1.f)
                *sFlag = 1;
        }
        __syncthreads();

        }
        if (tid < QK_THREADS) {
            float mx0, mx1;
            if constexpr (SingleTile) {
                float m0 = fmaxf(sPmax[r0], sPmax[BM + r0]);
                float m1 = fmaxf(sPmax[r1], sPmax[BM + r1]);
                if constexpr (Traits::QKWarpN == 4) {
                    m0 = fmaxf(m0, fmaxf(sPmax[2 * BM + r0], sPmax[3 * BM + r0]));
                    m1 = fmaxf(m1, fmaxf(sPmax[2 * BM + r1], sPmax[3 * BM + r1]));
                }
                mx0 = fmaxf(NEG_INF, ceilf(m0 * qk_scale_log2));
                mx1 = fmaxf(NEG_INF, ceilf(m1 * qk_scale_log2));
                if (wn == 0 && (lane & 3) == 0) {
                    sRunMax[r0] = mx0;
                    sRunMax[r1] = mx1;
                }
            } else {
                mx0 = sRunMax[r0];
                mx1 = sRunMax[r1];
            }
            float sm[2] = {0.f, 0.f};
            Tensor cP = tqk.partition_C(sPt);
            // cP(e) and cP(e+1) are n and n+1 of the same row, hence physically
            // adjacent inside one 16-byte swizzle granule: pack them into one 2-byte
            // store to halve the store count and the store bank conflicts.
            CUTE_UNROLL
            for (int e = 0; e < ScoreValues; e += 2) {

                int s = Traits::score_row_offset(e) ? 1 : 0;
                float mxs = s ? mx1 : mx0;
                const float bias = LOG2_P_SCALE - mxs;
                float v0 = exp2f(fmaf(acc_s(e), qk_scale_log2, bias));
                float v1 = exp2f(fmaf(acc_s(e + 1), qk_scale_log2, bias));
                sm[s] += v0 + v1; // sums already carry P_SCALE
                const __hg_fp8x2_storage_t packed =
                    flashmla::sparse_native_fp8::pack_e4m3_pair(make_float2(v0, v1));
                *reinterpret_cast<__hg_fp8x2_storage_t *>(&cP(e)) = packed;
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                sm[s] += __shfl_xor_sync(0xffffffff, sm[s], 1);
                sm[s] += __shfl_xor_sync(0xffffffff, sm[s], 2);
            }
            if ((lane & 3) == 0) {
                sPsum[wn * BM + r0] = sm[0];
                sPsum[wn * BM + r1] = sm[1];
            }
        }

        if (!SingleTile && *sFlag) {
            float os[BM / 8];
#pragma unroll
            for (int j = 0; j < BM / 8; ++j)
                os[j] = sScale[j * 8 + mi];
            CUTE_UNROLL
            for (int e = 0; e < size(acc_o); ++e)
                acc_o(e) *= os[Traits::output_row(e) >> 3];
        }
        __syncthreads();

        if (tid < BM) {
            float tile_sum = sPsum[tid] + sPsum[BM + tid];
            if constexpr (Traits::QKWarpN == 4)
                tile_sum = tile_sum + sPsum[2 * BM + tid] + sPsum[3 * BM + tid];
            sRunSum[tid] = SingleTile ? tile_sum : sRunSum[tid] * sScale[tid] + tile_sum;
        }

        // Accumulate PV directly in the running output fragment.
        // Fragment storage follows the logical PV operand shape. Actual loads
        // transpose four-byte pieces from the swizzled KV stage into that shape.
        Tensor rV = tpv.partition_fragment_B(
            make_tensor(make_smem_ptr(pK), Layout<Shape<Int<DV>, Int<BN>>, Stride<_1, Int<DV>>>{}));
        CUTE_UNROLL
        for (int k = 0; k < NK_PV; ++k) {
            CUTE_UNROLL
            for (int n = 0; n < DV / (NTHR / 2); ++n) {
                const int ub = UNIF((t & 1) * KTILE + n * (NTHR / 2 * BN) + k * (4 * 512));
                const auto *vb = reinterpret_cast<const uint8_t *>(pK) + ub;
                uint32_t a = *(const uint32_t *)(vb + voff);
                uint32_t b = *(const uint32_t *)(vb + voff + 1024);
                uint32_t c = *(const uint32_t *)(vb + voff + 8);
                uint32_t d = *(const uint32_t *)(vb + voff + 1032);
                auto *rv = (uint32_t *)&rV(0, n, k);
                rv[0] = ppu_mat_trans_b8(a);
                rv[1] = ppu_mat_trans_b8(b);
                rv[2] = ppu_mat_trans_b8(c);
                rv[3] = ppu_mat_trans_b8(d);
            }
            CUTE_UNROLL
            for (int m2 = 0; m2 < BM / 16; ++m2)
                PPU_LDM((uint32_t *)&rP(0, m2, k), ptsm0 + (k >> 1) * (BM * 4) + m2 * 64 + (k & 1) * 2, qk_lbo_a, qk_sbo_a);
            __ppu_sched_bound();
            cute::gemm(mpv, rP(_, _, k), rV(_, _, k), acc_o);
        }

        if (t + 1 < active_tiles) {
            cute::cp_async_wait<0>(); // Next KV stage has landed.
            __syncthreads();          // Current PV readers have released the other stage.
        }
    }
    // Publish the final running sum before any PV warp consumes it.
    __syncthreads();
    float rs[BM / 8];
#pragma unroll
    for (int j = 0; j < BM / 8; ++j) {
        float s = sRunSum[j * 8 + mi];
        if (gAttnSink && j * 8 + mi < LogicalHeads) {
            // s is P_SCALE * sum(exp2(logit - running_max)). attn_sink is in
            // natural-log space, matching sparse_prefill_fwd's public contract.
            const float sink_log2 = gAttnSink[j * 8 + mi] * 1.4426950408889634f;
            s += exp2f(sink_log2 - sRunMax[j * 8 + mi] + LOG2_P_SCALE);
        }
        rs[j] = (s > 0.f && s == s) ? (out_scale / s) : 0.f;
    }

    // Optional statistics follow the FP8 interface: a ceil-quantized maximum
    // and base-2 LSE of the quantized QK scores, before attention-sink scaling.
    if (tid < LogicalHeads && (gMaxLogits || gLse)) {
        const float m = sRunMax[tid], s = sRunSum[tid];
        if (gMaxLogits)
            gMaxLogits[tid] = m;
        if (gLse)
            flashmla::sparse_native_fp8::store_lse(gLse, q_token, head_group * BM + tid, h_q,
                lse_queries, (s > 0.f) ? (m + log2f(s * (1.f / P_SCALE))) : -INFINITY);
    }
    // bf16 output, matching the production kernel: fp32 would double the store
    // traffic and thrash L2 against the KV cache at realistic grid sizes.
    Tensor gObf = make_tensor(make_gmem_ptr(gO),
        Shape<Int<BM>, Int<DV>>{}, GenRowMajor{});
    Tensor cO = tpv.partition_C(gObf);
    // Pack adjacent BF16 elements into a single global store.
    CUTE_UNROLL
    for (int e = 0; e < size(acc_o); e += 2) {

        if constexpr (LogicalHeads < BM) {
            if (Traits::output_row(e) >= LogicalHeads)
                continue;
        }
        const float sc = rs[Traits::output_row(e) >> 3];
        const cutlass::bfloat16_t b0(acc_o(e) * sc), b1(acc_o(e + 1) * sc);
        const uint32_t packed = (uint32_t)reinterpret_cast<const uint16_t &>(b0) |
                                ((uint32_t)reinterpret_cast<const uint16_t &>(b1) << 16);
        *reinterpret_cast<uint32_t *>(&cO(e)) = packed;
    }

}

// ---------------------------------------------------------------------------

template <int D, int Threads, bool Trusted, int Prefix = 0, int GatherLanes = 8, bool GuardGatherReuse = false>
void launch(const void *q, const void *kv, const int *idx, void *out, float *maxima, float *lse,
    const float *sink, const int *lengths, float scale, float os, int queries, int tiles,
    int kvrows, hggcStream_t stream, int heads = 64, int lse_queries = 0) {
    constexpr int Shared = 3 * 64 * D + 64 * 64 + 2176 * 4 + 11 * 64 * 4 + 256;
    auto kernel = flash_fwd_sparse_fp8_std_kernel<64, D, Trusted, Threads, Prefix, false, 64, 64, false, GatherLanes, false, GuardGatherReuse>;
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, Shared);
    kernel<<<dim3(int64_t(queries) * (heads / 64)), Threads, Shared, stream>>>(q, static_cast<const E4M3 *>(kv), idx,
        static_cast<cutlass::bfloat16_t *>(out), scale, tiles, kvrows, os, heads, maxima, lse, sink,
        lengths, nullptr, lse_queries, nullptr, nullptr, nullptr, 0, 0, 1);
}
template <int Threads, int Prefix = 0, int GatherLanes = 8, bool GuardGatherReuse = false>
void dispatch(const void *q, const void *kv, const int *idx, void *out, float *maxima, float *lse,
    const float *sink, const int *lengths, float scale, float os, int dim, int queries, int tiles,
    int kvrows, bool trusted, hggcStream_t stream, int heads = 64, int lse_queries = 0) {
    if (dim == 512) {
        if (trusted)
            launch<512, Threads, true, Prefix, GatherLanes>(q, kv, idx, out, maxima, lse, sink, lengths,
                scale, os, queries, tiles, kvrows, stream, heads, lse_queries);
        else
            launch<512, Threads, false, Prefix, GatherLanes>(q, kv, idx, out, maxima, lse, sink,
                lengths, scale, os, queries, tiles, kvrows, stream, heads, lse_queries);
    } else {
        if (trusted)
            launch<576, Threads, true, Prefix, GatherLanes, GuardGatherReuse>(q, kv, idx, out, maxima, lse, sink, lengths,
                scale, os, queries, tiles, kvrows, stream, heads, lse_queries);
        else
            launch<576, Threads, false, Prefix, GatherLanes, GuardGatherReuse>(q, kv, idx, out, maxima, lse, sink,
                lengths, scale, os, queries, tiles, kvrows, stream, heads, lse_queries);
    }
}
} // namespace flashmla_fp8_std

#define COOP_LDM(v, addr, lbo, sbo)                                                                \
    asm volatile(                                                                                  \
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0,%1,%2,%3}, [%4],%5,%6,%7;"        \
        : "=r"((v)[0]), "=r"((v)[1]), "=r"((v)[2]), "=r"((v)[3])                                   \
        : "l"(addr), "r"(lbo), "r"(sbo), "n"(1))
#define COOP_UNIF(x) ((int)__ppu_read_firstlane((unsigned)(x)))

// Shared native FP8 sparse WS pipeline for prefill, decode and verification.
// WI (Warp-Interleave) overlaps QK/softmax producers and PV consumers using
// two P/alpha slots and independently staged KV buffers.
// QK retains native Tensor Cell accumulation, without K32 FP32 promotion
// or a K16 repair pass, under the approximate Q8/KV8/P8 output contract.
namespace flashmla::sparse_native_fp8::ws {

using namespace cute;
using F8 = cutlass::float_e4m3_t;
constexpr int N = 64, V = 512;
constexpr float Neg = -1e30f, LogP = 8.807354922f;
using Atom = decltype(composition(Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
template <int D, int BM> using LQ = decltype(tile_to_shape(Atom{}, Shape<Int<BM>, Int<D>>{}));
template <int D> using LK = decltype(tile_to_shape(Atom{}, Shape<_64, Int<D>>{}));
template <int BM> using LP = decltype(tile_to_shape(Atom{}, Shape<Int<BM>, _64>{}));
template <int BM>
using MQ = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
    Layout<Shape<Int<BM / 16>, _4, _1>>, Tile<Int<BM>, _64, _32>>;
template <int BM, int Rows = 1, bool PairedV = false>
using MP = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
    Layout<Shape<Int<Rows>, Int<BM / (4 * Rows)>, _1>>,
    Tile<Int<16 * Rows>,
        std::conditional_t<PairedV,
            Layout<Shape<_16, Int<BM / (4 * Rows)>, _2>, Stride<_1, _32, _16>>,
            Int<BM * 4 / Rows>>, _32>>;

// A b16 transpose moves pairs of FP8 bytes, not individual FP8 elements.
// Keep its fragment order through PV instead of restoring the scalar-load
// layout with lane shuffles. P's reduction columns and the output columns
// must use the matching permutations below.
struct PairedVLoad {
    __device__ __forceinline__ static int column(int value) {
        // Rotate bits [3:1] within each 16-column group. Bit 0 is preserved
        // so adjacent FP8 probabilities and packed BF16 output pairs stay together.
        return (value & ~14) | ((value & 6) << 1) | ((value & 8) >> 2);
    }

    template <int Half>
    __device__ __forceinline__ static void unpack(uint32_t *dst,
        const uint32_t (&lo)[4], const uint32_t (&hi)[4]) {
        static_assert(Half == 0 || Half == 1);
        // Separate even/odd bytes from the two transposed row blocks.
        dst[0] = __byte_perm(lo[2 * Half], lo[2 * Half + 1], 0x6420u);
        dst[1] = __byte_perm(hi[2 * Half], hi[2 * Half + 1], 0x6420u);
        dst[2] = __byte_perm(lo[2 * Half], lo[2 * Half + 1], 0x7531u);
        dst[3] = __byte_perm(hi[2 * Half], hi[2 * Half + 1], 0x7531u);
    }

    __device__ __forceinline__ static void load(uint32_t *first, uint32_t *second,
        const F8 *base, int col, int k) {
        // A 64-row tile uses 8x64-byte swizzled atoms. Addresses are in
        // 16-byte units; the two loads cover K32 x N32, shared by two N16
        // fragments. col is a warp-uniform multiple of 32.
        const int address = COOP_UNIF(int(reinterpret_cast<uintptr_t>(base) / 16) +
            (col / 64) * 256 + (col % 64) / 16 + k * 128);
        uint32_t lo[4], hi[4];
        asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans "
                     "{%0,%1,%2,%3}, [%4], %5,%6,%7;"
            : "=r"(lo[0]), "=r"(lo[1]), "=r"(lo[2]), "=r"(lo[3])
            : "l"(address), "r"(1), "r"(32), "n"(1));
        asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans "
                     "{%0,%1,%2,%3}, [%4], %5,%6,%7;"
            : "=r"(hi[0]), "=r"(hi[1]), "=r"(hi[2]), "=r"(hi[3])
            : "l"(address + 64), "r"(1), "r"(32), "n"(1));
        unpack<0>(first, lo, hi);
        unpack<1>(second, lo, hi);
    }

    template <class Tensor>
    __device__ __forceinline__ static void restore_output(Tensor &acc) {
        // Restore adjacent output pairs before the packed BF16 stores.
        CUTE_UNROLL
        for (int i = 0; i < size(acc); i += 8) {
            const float a = acc(i + 1), b = acc(i + 3);
            acc(i + 1) = acc(i + 4);
            acc(i + 4) = a;
            acc(i + 3) = acc(i + 6);
            acc(i + 6) = b;
        }
    }
};
struct State {
    F8 *q, *k, *p;
    float *max, *sum, *alpha, *pmax, *psum;
    int *flag, *idx;
};
template <int BM> __device__ int role() {
    return threadIdx.x / ((BM / 16) * 4 * 32);
}
template <int BM> __device__ int role_lane() {
    return threadIdx.x % ((BM / 16) * 4 * 32);
}

template <int Group> __device__ void producer_sync() { __ppu_barrier_sync(12, Group, 15u); }

// Keep the validated arithmetic order, shared-memory layout and barrier lifecycle
// explicit. These policies are compile-time choices, not runtime branches.
struct PrefillPolicy {
    static constexpr bool SplitQk = true;
    static constexpr bool IndexBitmap = false;
    static constexpr bool DrainReleases = false;
    static constexpr bool FullHeadTile = true;
    static constexpr int IndexPadding = 0;
};
struct DecodePolicy {
    static constexpr bool SplitQk = false;
    static constexpr bool IndexBitmap = true;
    static constexpr bool DrainReleases = true;
    static constexpr bool FullHeadTile = false;
    static constexpr int IndexPadding = 4;
};

template <typename Policy, bool PairedV = false>
struct Pipeline {
    template <int D, int Stages, int BM, bool ReuseQ> static constexpr int smem_bytes() {
        return (ReuseQ ? 0 : BM * D) + Stages * N * D + 2 * BM * N +
            12 * BM * 4 + (2 + Policy::IndexPadding) * 4 + 2176 * 4;
    }
    template <int D, int Stages, int BM, bool ReuseQ> static __device__ State shared_state(char *bytes) {
        constexpr int M = BM;
        State s;
        s.q = reinterpret_cast<F8 *>(bytes);
        s.k = s.q + (ReuseQ ? 0 : M * D);
        s.p = s.k + Stages * N * D;
        s.max = reinterpret_cast<float *>(s.p + 2 * M * N);
        s.sum = s.max + M;
        s.alpha = s.sum + M;
        s.pmax = s.alpha + 2 * M;
        s.psum = s.pmax + 4 * M;
        s.flag = reinterpret_cast<int *>(s.psum + 4 * M);
        // Decode's four padding words align the cached Q tail at s.idx + 2170
        // to a 512-byte shared-memory cube; prefill retains its compact layout.
        s.idx = s.flag + 2 + Policy::IndexPadding;
        return s;
    }

    // WI (Warp-Interleave) overlaps the QK/softmax producer and PV consumer.
    // The producer prefetches the next KV tile before computing the current QK;
    // the consumer releases shared slots after loading P/V into registers, so
    // the producer can advance while PV runs.
    template <int D, int Stages, int BM, bool ReuseQ, int Prefix = 0, bool Dual = false,
        bool Fold = false, int GatherLanes = 0>
    static __device__ __forceinline__ void produce(State s, const F8 *kv, int length, int kvrows, float scale,
        const F8 *extra = nullptr, int primary_rows = 0, const unsigned *masks = nullptr,
        int mask_stride = 0, int logical_heads = 0, int head_offset = 0, int total_heads = 0) {
        constexpr int M = BM, G = (BM / 16) * 4 * 32, Threads = 2 * G;
        const int gt = role_lane<BM>(), lane = gt % 32, warp = COOP_UNIF(gt / 32);
        const int wm = warp % (M / 16), wn = warp / (M / 16), r0 = wm * 16 + lane / 4, r1 = r0 + 8;
        const int mask_shift = logical_heads == 8 ? 3 : 4;
        const int mask_row0 = (head_offset + r0) >> mask_shift,
                  mask_row1 = (head_offset + r1) >> mask_shift;
        const int tiles = max(1, (length + N - 1) / N);
        MQ<BM> mq;
        auto tq = mq.get_slice(gt);
        auto rq = tq.partition_fragment_A(make_tensor(make_smem_ptr(s.q),
            Shape<Int<BM>, Int<Prefix ? Prefix * 32 : D>>{}, GenRowMajor{}));
        auto rk =
            tq.partition_fragment_B(make_tensor(make_smem_ptr(s.k), Shape<_64, _192>{}, GenRowMajor{}));
        const int qt = COOP_UNIF(int(reinterpret_cast<uintptr_t>(s.q) / 16) + wm * 64);
        auto *tail = reinterpret_cast<F8 *>(s.idx + 2170);
        auto rt = tq.partition_fragment_A(
            make_tensor(make_smem_ptr(s.q), Shape<Int<BM>, _64>{}, GenRowMajor{}));
        const int tailt = COOP_UNIF(int(reinterpret_cast<uintptr_t>(tail) / 16) + wm * 64);
        auto sk = make_tensor(make_smem_ptr(s.k), LK<D>{});
        constexpr int GL = GatherLanes ? GatherLanes : G / N, GW = GL * 16;
        constexpr int GatherRows = G / GL, GatherRepeats = N / GatherRows;
        const int grow = gt / GL, col0 = (gt % GL) * 16;
        int dst0[GatherRepeats], dst1[GatherRepeats];
        CUTE_UNROLL
        for (int r = 0; r < GatherRepeats; ++r) {
            dst0[r] = int(reinterpret_cast<unsigned char *>(&sk(grow + r * GatherRows, col0)) -
                         reinterpret_cast<unsigned char *>(s.k));
            dst1[r] = int(reinterpret_cast<unsigned char *>(&sk(grow + r * GatherRows, col0 + 32)) -
                         reinterpret_cast<unsigned char *>(s.k));
        }
        auto gather = [&](int t) {
            CUTE_UNROLL
            for (int r = 0; r < GatherRepeats; ++r) {
                const int token = s.idx[t * N + grow + r * GatherRows];
                const bool valid = (unsigned)token < (unsigned)kvrows;
                const F8 *pool = kv;
                int row = valid ? token : 0;
                if constexpr (Dual) {
                    if (valid && token >= primary_rows) {
                        pool = extra;
                        row = token - primary_rows;
                    }
                }
                const auto *src = pool + (long)row * D + col0;
                auto *dst = s.k + (t % Stages) * N * D;
                // Retain the full-head gather's base-address hoist. Reassociating
                // it with each column changes HGGC's prefetch register schedule.
                if constexpr (Policy::FullHeadTile)
                    dst += dst0[r];
                CUTE_UNROLL
                for (int j = 0; j < (D + GW - 1) / GW; j++)
                    if (col0 + j * GW < D)
                        PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
                            *reinterpret_cast<const uint128_t *>(src + j * GW),
                            *reinterpret_cast<uint128_t *>(
                                dst + (Policy::FullHeadTile ? j * GW * N
                                    : GW == 32 ? ((j & 1) ? dst1[r] : dst0[r]) + (j >> 1) * 64 * N
                                               : dst0[r] + j * GW * N)),
                            valid);
            }
        };
        CUTE_UNROLL
        for (int k = 0; k < (Prefix ? Prefix : D / 32); k++)
            COOP_LDM(reinterpret_cast<uint32_t *>(&rq(0, 0, k)),
                qt + (k >> 1) * M * 4 + (k & 1) * 2, 1, 32);
        if constexpr (Prefix) {
            static_assert(BM == 32 && (Prefix == 12 || Prefix == 8),
                "512-aligned compact tail fits two stage3 CTAs only for this prefix");
            auto fullq = make_tensor(make_smem_ptr(s.q), LQ<D, BM>{});
            auto tailq = make_tensor(make_smem_ptr(tail), LQ<D - Prefix * 32, BM>{});
            for (int i = gt; i < BM * ((D - Prefix * 32) / 16); i += G) {
                const int r = i / ((D - Prefix * 32) / 16), c = i % ((D - Prefix * 32) / 16) * 16;
                *reinterpret_cast<uint128_t *>(&tailq(r, c)) =
                    *reinterpret_cast<const uint128_t *>(&fullq(r, c + Prefix * 32));
            }
        }
        if constexpr (ReuseQ) {
            producer_sync<G>();
        }
        gather(0);
        cp_async_fence();
        cp_async_wait<0>();
        producer_sync<G>();
        // All N warps observe the same online maximum. Keep each warp's share of
        // the normalizer local and reduce it only after the final KV tile.
        float running_max0 = Neg, running_max1 = Neg, normalizer0 = 0.f, normalizer1 = 0.f;
        for (int t = 0; t < tiles; t++) {
            const int slot = t & 1;
            if constexpr (Stages == 3) {
                if (t >= 2) {
                    __ppu_barrier_sync(8 + slot, Threads, 15u);
                }
            } else {
                if (t > 0) {
                    __ppu_barrier_sync(8 + ((t - 1) & 1), Threads, 15u);
                }
            }
            if (t + 1 < tiles) {
                gather(t + 1);
                cp_async_fence();
            }
            auto sp = make_tensor(make_smem_ptr(s.p + slot * M * N), LP<BM>{});
            auto scores = tq.partition_fragment_C(sp);
            auto scores1 = tq.partition_fragment_C(sp);
            clear(scores);
            if constexpr (Policy::SplitQk)
                clear(scores1);
            const int kt =
                COOP_UNIF(int(reinterpret_cast<uintptr_t>(s.k + (t % Stages) * N * D) / 16) + wn * 64);
            auto load = [&](int k) {
                COOP_LDM(reinterpret_cast<uint32_t *>(&rk(0, 0, k % 6)),
                    kt + (k >> 1) * N * 4 + (k & 1) * 2, 32, 1);
            };
            load(0);
            load(1);
            load(2);
            CUTE_UNROLL
            for (int k = 0; k < D / 32; k += 3) {
                CUTE_UNROLL
                for (int j = 0; j < 3; j++)
                    if (k + 3 + j < D / 32)
                        load(k + 3 + j);
                CUTE_UNROLL
                for (int j = 0; j < 3; j++)
                    if (k + j < D / 32) {
                        if constexpr (Policy::SplitQk) {
                            static_assert(Prefix == 0, "Split QK requires the full Q fragment");
                            if (j == 1)
                                gemm(mq, rq(_, _, k + j), rk(_, _, (k + j) % 6), scores1);
                            else
                                gemm(mq, rq(_, _, k + j), rk(_, _, (k + j) % 6), scores);
                        } else if constexpr (Prefix) {
                            if (k + j >= Prefix) {
                                const int local = k + j - Prefix;
                                COOP_LDM(reinterpret_cast<uint32_t *>(&rt(0, 0, (k + j) & 1)),
                                    tailt + (local >> 1) * M * 4 + (local & 1) * 2, 1, 32);
                                gemm(mq, rt(_, _, (k + j) & 1), rk(_, _, (k + j) % 6), scores);
                            } else
                                gemm(mq, rq(_, _, k + j), rk(_, _, (k + j) % 6), scores);
                        } else
                            gemm(mq, rq(_, _, k + j), rk(_, _, (k + j) % 6),
                                scores);
                    }
            }
            float mx0 = Neg, mx1 = Neg;
            if constexpr (Policy::IndexBitmap) {
                unsigned valid0, valid1;
                if constexpr (Fold) {
                    const int word = t * 2 + (wn >> 1);
                    valid0 = COOP_UNIF(head_offset + r0 < total_heads
                                           ? __ldg(masks + mask_row0 * mask_stride + word)
                                           : 0u);
                    valid1 = COOP_UNIF(head_offset + r1 < total_heads
                                           ? __ldg(masks + mask_row1 * mask_stride + word)
                                           : 0u);
                } else
                    valid0 = valid1 = unsigned(COOP_UNIF(unsigned(s.idx[2048 + t * 2 + (wn >> 1)])));
                if (valid0 != 0xffffffffu || valid1 != 0xffffffffu) {
                    CUTE_UNROLL
                    for (int e = 0; e < 8; e++) {
                        const int bit = (wn & 1) * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
                        const unsigned valid = ((e >> 1) & 1) ? valid1 : valid0;
                        scores(e) = (valid & (1u << bit)) ? scores(e) : -INFINITY;
                    }
                }
            }
            CUTE_UNROLL
            for (int e = 0; e < 8; e++) {
                if constexpr (!Policy::IndexBitmap) {
                    const int col = wn * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
                    scores(e) = (unsigned)s.idx[t * N + col] < (unsigned)kvrows
                                    ? scores(e) + scores1(e) : -INFINITY;
                }
                if ((e >> 1) & 1)
                    mx1 = fmaxf(mx1, scores(e));
                else
                    mx0 = fmaxf(mx0, scores(e));
            }
            CUTE_UNROLL
            for (int delta = 1; delta <= 2; delta *= 2) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, delta));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, delta));
            }
            if (gt == 0)
                s.flag[slot] = 0;
            if ((lane & 3) == 0) {
                s.pmax[wn * M + r0] = mx0;
                s.pmax[wn * M + r1] = mx1;
            }
            producer_sync<G>();
            const float tile0 = fmaxf(fmaxf(s.pmax[r0], s.pmax[M + r0]),
                                    fmaxf(s.pmax[2 * M + r0], s.pmax[3 * M + r0])) *
                                scale;
            const float tile1 = fmaxf(fmaxf(s.pmax[r1], s.pmax[M + r1]),
                                    fmaxf(s.pmax[2 * M + r1], s.pmax[3 * M + r1])) *
                                scale;
            const float next0 = fmaxf(running_max0, ceilf(tile0)),
                        next1 = fmaxf(running_max1, ceilf(tile1));
            const float alpha0 = running_max0 == Neg ? 1.f : exp2f(running_max0 - next0);
            const float alpha1 = running_max1 == Neg ? 1.f : exp2f(running_max1 - next1);
            running_max0 = next0;
            running_max1 = next1;
            if (wn == 0 && (lane & 3) == 0) {
                s.alpha[slot * M + r0] = alpha0;
                s.alpha[slot * M + r1] = alpha1;
                if (alpha0 != 1.f || alpha1 != 1.f)
                    atomicOr(s.flag + slot, 1);
            }
            float sums[2] = {0, 0};
            auto cp = tq.partition_C(sp);
            auto pxy = tq.partition_C(make_identity_tensor(Shape<Int<BM>, _64>{}));
            CUTE_UNROLL
            for (int e = 0; e < 8; e += 2) {
                const int which = (e >> 1) & 1;
                const float bias = LogP - (which ? running_max1 : running_max0);
                const float a = exp2f(fmaf(scores(e), scale, bias)),
                            b = exp2f(fmaf(scores(e + 1), scale, bias));
                sums[which] += a + b;
                auto *dst = &cp(e);
                if constexpr (PairedV)
                    dst = &sp(get<0>(pxy(e)), PairedVLoad::column(int(get<1>(pxy(e)))));
                *reinterpret_cast<__hg_fp8x2_storage_t *>(dst) =
                    flashmla::sparse_native_fp8::pack_e4m3_pair(make_float2(a, b));
            }
            CUTE_UNROLL
            for (int delta = 1; delta <= 2; delta *= 2) {
                sums[0] += __shfl_xor_sync(0xffffffff, sums[0], delta);
                sums[1] += __shfl_xor_sync(0xffffffff, sums[1], delta);
            }
            normalizer0 = normalizer0 * alpha0 + sums[0];
            normalizer1 = normalizer1 * alpha1 + sums[1];
            // The ready barrier counts every producer and consumer warp. Its completion
            // already orders the P/alpha stores before PV reads; no local group wait is
            // needed now that the per-tile denominator reduction is gone.
            __syncwarp();
            __ppu_barrier_arrive(4 + slot, Threads, 15u);
            if (t + 1 < tiles) {
                cp_async_wait<0>();
                producer_sync<G>();
            }
        }
        // Preserve each schedule's release-barrier lifecycle at the end of the loop.
        if constexpr (Policy::DrainReleases) {
            constexpr int Pending = (Stages == 3) ? 2 : 1;
            for (int t = max(0, tiles - Pending); t < tiles; ++t)
                __ppu_barrier_sync(8 + (t & 1), Threads, 15u);
        }
        if ((lane & 3) == 0) {
            s.psum[wn * M + r0] = normalizer0;
            s.psum[wn * M + r1] = normalizer1;
            if (wn == 0) {
                s.max[r0] = running_max0;
                s.max[r1] = running_max1;
            }
        }
        producer_sync<G>();
        if (gt < M)
            s.sum[gt] = s.psum[gt] + s.psum[M + gt] + s.psum[2 * M + gt] + s.psum[3 * M + gt];
    }

    template <int D, int Stages, int BM, int Rows = 1, bool Late = false, int SinkHeads = 0>
    static __device__ __forceinline__ void consume(State s, cutlass::bfloat16_t *out, const float *sink,
        int length, float outscale, int active_heads = BM) {
        constexpr int M = BM, G = (BM / 16) * 4 * 32, Threads = 2 * G;
        const int gt = role_lane<BM>(), lane = gt % 32, warp = COOP_UNIF(gt / 32),
                  mi = lane / 4;
        const int tiles = max(1, (length + N - 1) / N);
        MP<BM, Rows, PairedV> mp;
        auto tp = mp.get_slice(gt);
        auto shape = make_tensor(make_smem_ptr(s.k), Shape<Int<BM>, _512>{}, GenRowMajor{});
        auto acc = tp.partition_fragment_C(shape);
        clear(acc);
        auto xy = tp.partition_C(make_identity_tensor(Shape<Int<BM>, _512>{}));
        auto row_group = [&](int e) {
            if constexpr (Policy::FullHeadTile) {
                static_assert(Rows == 1, "Full-head schedule uses the original PV row layout");
                return ((e >> 3) % (M / 16)) * 2 + ((e >> 1) & 1);
            } else
                return int(get<0>(xy(e))) / 8;
        };
        auto rp = tp.partition_fragment_A(
            make_tensor(make_smem_ptr(s.p), Shape<Int<BM>, Int<Late ? 32 : 64>>{}, GenRowMajor{}));
        auto rv = tp.partition_fragment_B(make_tensor(
            make_smem_ptr(s.k), Layout<Shape<_512, Int<Late ? 32 : 64>>, Stride<_1, _512>>{}));
        auto sk = make_tensor(make_smem_ptr(s.k), LK<D>{});
        const int vr = (lane & 3) * 4 + ((lane >> 2) & 3), vc = (warp / Rows) * 16 + (lane >> 4) * 4;
        const int voff = int(
            reinterpret_cast<unsigned char *>(&sk(vr, vc)) - reinterpret_cast<unsigned char *>(s.k));
        for (int t = 0; t < tiles; t++) {
            const int slot = t & 1;
            __ppu_barrier_sync(4 + slot, Threads, 15u);
            if (COOP_UNIF(s.flag[slot])) {
                float factors[M / 8];
                CUTE_UNROLL
                for (int j = 0; j < M / 8; j++)
                    factors[j] = s.alpha[slot * M + j * 8 + mi];
                CUTE_UNROLL
                for (int e = 0; e < size(acc); e++)
                    acc(e) *= factors[row_group(e)];
            }
            const int pt = COOP_UNIF(
                int(reinterpret_cast<uintptr_t>(s.p + slot * M * N) / 16) + (warp % Rows) * 64);
            auto load = [&](int k) {
                if constexpr (PairedV) {
                    CUTE_UNROLL
                    for (int n = 0; n < V / (G / (2 * Rows)); n += 2) {
                        PairedVLoad::load(
                            reinterpret_cast<uint32_t *>(&rv(0, n, Late ? 0 : k)),
                            reinterpret_cast<uint32_t *>(&rv(0, n + 1, Late ? 0 : k)),
                            s.k + (t % Stages) * N * D,
                            (warp / Rows) * 32 + (n / 2) * (G / Rows), k);
                    }
                } else {
                    CUTE_UNROLL
                    for (int n = 0; n < V / (G / (2 * Rows)); n++) {
                        const auto *vp =
                            reinterpret_cast<const unsigned char *>(s.k + (t % Stages) * N * D) +
                            n * (G / (2 * Rows)) * N + k * 2048 + voff;
                        auto *r = reinterpret_cast<uint32_t *>(&rv(0, n, Late ? 0 : k));
                        r[0] = ppu_mat_trans_b8(*reinterpret_cast<const uint32_t *>(vp));
                        r[1] = ppu_mat_trans_b8(*reinterpret_cast<const uint32_t *>(vp + 1024));
                        r[2] = ppu_mat_trans_b8(*reinterpret_cast<const uint32_t *>(vp + 8));
                        r[3] = ppu_mat_trans_b8(*reinterpret_cast<const uint32_t *>(vp + 1032));
                    }
                }
                CUTE_UNROLL
                for (int m = 0; m < M / (16 * Rows); m++)
                    COOP_LDM(reinterpret_cast<uint32_t *>(&rp(0, m, Late ? 0 : k)),
                        pt + m * Rows * 64 + k * 2, 1, 32);
            };
            if constexpr (Late) {
                CUTE_UNROLL
                for (int k = 0; k < 2; k++) {
                    load(k);
                    __ppu_sched_bound();
                    if (k == 1)
                        __ppu_barrier_arrive(8 + slot, Threads, 15u);
                    gemm(mp, rp(_, _, 0), rv(_, _, 0), acc);
                    __ppu_sched_bound();
                }
            } else {
                load(0);
                load(1);
                __ppu_sched_bound();
                __ppu_barrier_arrive(8 + slot, Threads, 15u);
                CUTE_UNROLL
                for (int k = 0; k < 2; k++)
                    gemm(mp, rp(_, _, k), rv(_, _, k), acc);
            }
        }
        __ppu_barrier_sync(0, Threads, 15u);
        float inv[M / 8];
        CUTE_UNROLL
        for (int j = 0; j < M / 8; j++) {
            const int row = j * 8 + mi;
            float sum = s.sum[row];
            if constexpr (Policy::FullHeadTile) {
                if (sink)
                    sum += exp2f(sink[row] * 1.4426950408889634f - s.max[row] + LogP);
            } else {
                if (sink && row < active_heads)
                    sum += exp2f(sink[SinkHeads ? row % SinkHeads : row] * 1.4426950408889634f - s.max[row] + LogP);
            }
            inv[j] = (sum > 0 && sum == sum) ? outscale / sum : 0.f;
        }
        if constexpr (PairedV)
            PairedVLoad::restore_output(acc);
        CUTE_UNROLL
        for (int e = 0; e < size(acc); e += 2) {
            if constexpr (!Policy::FullHeadTile) {
                if (get<0>(xy(e)) >= active_heads)
                    continue;
            }
            const float f = inv[row_group(e)];
            cutlass::bfloat16_t a(acc(e) * f), b(acc(e + 1) * f);
            uint32_t packed = uint32_t(reinterpret_cast<uint16_t &>(a)) |
                              (uint32_t(reinterpret_cast<uint16_t &>(b)) << 16);
            const int col = int(get<1>(xy(e)));
            *reinterpret_cast<uint32_t *>(out + get<0>(xy(e)) * V +
                (PairedV ? PairedVLoad::column(col) : col)) = packed;
        }
    }
};

} // namespace flashmla::sparse_native_fp8::ws
