// PPU0015 FP8 sparse prefill. Both head tiles share Q conversion, QK,
// online softmax and the output epilogue. Two swizzled KV stages serve
// both QK and PV, with the next gather overlapping the current tile.

#include "ppu/sparse_native_fp8_attention.cuh"

// Shared preprocessing for BF16 KV callers of the existing FP8 interface.
// Quantization has unit scale, matching fused BF16 Q conversion. Per-tensor
// dequantization scales remain in the attention launcher's exponent/epilogue.
__global__ void sparse_prefill_quantize_kv(
    const __ppu_bfloat16_raw *input, E4M3 *output, int rows, int dim, int64_t row_stride) {
    const int64_t pair = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t offset = pair * 2;
    if (offset >= (int64_t)rows * dim)
        return;
    const int row = offset / dim, column = offset % dim;
    const auto value =
        *reinterpret_cast<const __ppu_bfloat162_raw *>(input + (int64_t)row * row_stride + column);
    *reinterpret_cast<__hg_fp8x2_storage_t *>(output + offset) =
        flashmla::sparse_native_fp8::pack_e4m3_pair(value);
}

template <int Dim>
__global__ void sparse_prefill_quantize_kv_vectorized(
    const __ppu_bfloat16_raw *input, E4M3 *output, int rows, int64_t row_stride) {
    constexpr int Elements = 16;
    const int64_t vector = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= (int64_t)rows * (Dim / Elements))
        return;
    const int row = vector / (Dim / Elements);
    const int column = (vector % (Dim / Elements)) * Elements;
    const auto *source = input + (int64_t)row * row_stride + column;
    const cute::uint128_t low = *reinterpret_cast<const cute::uint128_t *>(source);
    const cute::uint128_t high = *reinterpret_cast<const cute::uint128_t *>(source + 8);
    cute::uint128_t packed;
    auto *pairs = reinterpret_cast<__hg_fp8x2_storage_t *>(&packed);
    CUTE_UNROLL
    for (int j = 0; j < 4; ++j) {
        pairs[j] = flashmla::sparse_native_fp8::pack_e4m3_pair(
            reinterpret_cast<const __ppu_bfloat162_raw *>(&low)[j]);
        pairs[j + 4] = flashmla::sparse_native_fp8::pack_e4m3_pair(
            reinterpret_cast<const __ppu_bfloat162_raw *>(&high)[j]);
    }
    reinterpret_cast<cute::uint128_t *>(output)[vector] = packed;
}

__global__ void sparse_prefill_index_bounds(
    const int *indices, int *lengths, int width, int kv_rows) {
    int bound = 0;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const int value = indices[(int64_t)blockIdx.x * width + col];
        if ((unsigned)value < (unsigned)kv_rows)
            bound = col + 1;
    }
    for (int step = 16; step; step >>= 1)
        bound = max(bound, __shfl_down_sync(0xffffffff, bound, step));
    __shared__ int warp_bounds[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0)
        warp_bounds[warp] = bound;
    __syncthreads();
    if (warp == 0) {
        bound = lane < 8 ? warp_bounds[lane] : 0;
        for (int step = 16; step; step >>= 1)
            bound = max(bound, __shfl_down_sync(0xffffffff, bound, step));
        if (lane == 0)
            lengths[blockIdx.x] = bound;
    }
}

void prepare_sparse_prefill_fp8_kv(
    const void *input, void *output, int rows, int dim, int64_t row_stride, hggcStream_t stream) {
    if (rows == 0)
        return;
    if (reinterpret_cast<uintptr_t>(input) % 16 == 0 && row_stride % 8 == 0) {
        const int blocks = ((int64_t)rows * dim / 16 + 255) / 256;
        if (dim == 512) {
            sparse_prefill_quantize_kv_vectorized<512>
                <<<blocks, 256, 0, stream>>>(reinterpret_cast<const __ppu_bfloat16_raw *>(input),
                    reinterpret_cast<E4M3 *>(output), rows, row_stride);
        } else {
            sparse_prefill_quantize_kv_vectorized<576>
                <<<blocks, 256, 0, stream>>>(reinterpret_cast<const __ppu_bfloat16_raw *>(input),
                    reinterpret_cast<E4M3 *>(output), rows, row_stride);
        }
        return;
    }
    const int blocks = ((int64_t)rows * dim / 2 + 255) / 256;
    sparse_prefill_quantize_kv<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __ppu_bfloat16_raw *>(input), reinterpret_cast<E4M3 *>(output), rows,
        dim, row_stride);
}

void prepare_sparse_prefill_index_bounds(
    const int *indices, int *lengths, int queries, int width, int kv_rows, hggcStream_t stream) {
    if (queries == 0)
        return;
    sparse_prefill_index_bounds<<<queries, 256, 0, stream>>>(indices, lengths, width, kv_rows);
}


static constexpr int BN = 64, DV = 512;
static constexpr float NEG_INF = -1e30f;
static constexpr float P_SCALE = 448.f;
static constexpr float LOG2_P_SCALE = 8.807354922f;
using AtomS8 = decltype(composition(Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

template <int BlockM> struct Fp8SparsePrefillTraits : flashmla_fp8_std::ScoreRows {
    static_assert(BlockM == 16 || BlockM == 32 || BlockM == 64, "unsupported head tile");
    static constexpr bool Compact = BlockM < 64;
    static constexpr int Threads = Compact ? 256 : 512;
    static constexpr int QKWarpN = 4;
    static constexpr int QKThreads = (BlockM / 16) * QKWarpN * 32;
    static constexpr int MaxTopk = 2176;
    using MmaQK = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<Int<BlockM / 16>, _4, _1>>, Tile<Int<BlockM>, Int<BN>, _32>>;
    using MmaPV = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<_1, Int<Threads / 32>, _1>>, Tile<_16, Int<Threads / 2>, _32>>;
    using LayoutP = decltype(tile_to_shape(AtomS8{}, Shape<Int<BlockM>, Int<BN>>{}));
    using CopyA = Copy_Atom<PPU_TSM_LD_SWZL<E4M3, BlockM, 64, false, false, 1>, E4M3>;
    using CopyB = Copy_Atom<PPU_TSM_LD_SWZL<E4M3, BN, 64, true, false, 1>, E4M3>;

    CUTE_HOST_DEVICE static constexpr int output_row(int e) {
        return ((e >> 3) % (BlockM / 16) * 16) + ((e >> 1) & 1) * 8;
    }
};

template <int BM, int DQK, bool Q_IS_BF16, bool TRUSTED_FULL_TOPK, bool TWO_GROUP_ORDER,
    bool CHECK_Q_BOUNDS>
__global__ __launch_bounds__(Fp8SparsePrefillTraits<BM>::Threads,
    2) void flash_fwd_sparse_prefill_fp8_kernel(const void *__restrict__ gQRaw,
    const E4M3 *__restrict__ gKV, const int *__restrict__ gIdx, cutlass::bfloat16_t *__restrict__ gO,
    float qk_scale_log2, int n_tiles, int s_kv, float out_scale, int h_q, int q_stride_t,
    int s_q, float *__restrict__ gMaxLogits, float *__restrict__ gLse,
    const float *__restrict__ gAttnSink, const int *__restrict__ gTopkLength, int q_stride_h) {
    using Traits = Fp8SparsePrefillTraits<BM>;
    constexpr bool COMPACT = Traits::Compact;
    constexpr int NTHR = Traits::Threads, QK_THREADS = Traits::QKThreads;
    using MmaQK = typename Traits::MmaQK;
    using MmaPV = typename Traits::MmaPV;
    using LayoutP8 = typename Traits::LayoutP;
    using CopyA8 = typename Traits::CopyA;
    using CopyB8 = typename Traits::CopyB;
    static_assert(DQK == 512 || DQK == 576, "supported q8 sparse head dims are 512 and 576");
    static constexpr int NK_QK = DQK / 32;
    static constexpr int QTILE = BM * DQK, KTILE = BN * DQK;
    using LayoutQ8 = decltype(tile_to_shape(AtomS8{}, Shape<Int<BM>, Int<DQK>>{}));
    using LayoutK8 = decltype(tile_to_shape(AtomS8{}, Shape<Int<BN>, Int<DQK>>{}));

    extern __shared__ char smem[];
    E4M3 *pQ = reinterpret_cast<E4M3 *>(smem);
    E4M3 *pK = pQ + QTILE; // Two alternating swizzled KV stages.
    E4M3 *pP = pK + 2 * KTILE;
    int *sIdx = reinterpret_cast<int *>(pP + BM * BN);
    float *sf = reinterpret_cast<float *>(sIdx + Traits::MaxTopk);
    float *sPmax = sf, *sPsum = sf + 4 * BM, *sRunMax = sf + 8 * BM;
    float *sRunSum = sRunMax + BM, *sScale = sRunSum + BM;
    int *sFlag = reinterpret_cast<int *>(sScale + BM);

    const int tid = threadIdx.x, wp = tid / 32, lane = tid % 32;
    int q_token, head_group;
    if constexpr (TWO_GROUP_ORDER) {
        // Gray-code x: q0g0, q1g0, q1g1, q0g1.
        // This changes only CTA work order; tensor indexing and math are unchanged.
        const int dispatch_x = blockIdx.x ^ (blockIdx.x >> 1);
        q_token = blockIdx.y * 2 + (dispatch_x & 1);
        head_group = dispatch_x >> 1;
        if constexpr (CHECK_Q_BOUNDS) {
            if (q_token >= s_q)
                return;
        }
    } else {
        q_token = blockIdx.x;
        head_group = blockIdx.y;
    }
    const int active_heads = min(BM, h_q - head_group * BM);
    const long q_head_offset = ((long)q_token * h_q + head_group * BM) * DQK;
    const E4M3 *gQ8 = reinterpret_cast<const E4M3 *>(gQRaw) + q_head_offset;
    const __ppu_bfloat16_raw *gQbf16 = reinterpret_cast<const __ppu_bfloat16_raw *>(gQRaw) +
                                      (long)q_token * q_stride_t +
                                      (long)head_group * BM * q_stride_h;
    gIdx += (long)q_token * n_tiles * BN;
    gO += ((long)q_token * h_q + head_group * BM) * DV;
    if (gMaxLogits)
        gMaxLogits += (long)q_token * h_q + head_group * BM;
    if (gLse)
        gLse += (long)q_token * h_q + head_group * BM;
    if (gAttnSink)
        gAttnSink += head_group * BM;
    int topk_length = gTopkLength ? gTopkLength[q_token] : n_tiles * BN;
    topk_length = topk_length < 0 ? 0 : topk_length;
    topk_length = topk_length > n_tiles * BN ? n_tiles * BN : topk_length;
    // Match the BF16 kernel's dynamic work bound. Keep one all-invalid tile for
    // length zero so output/auxiliary behavior remains identical to the old path.
    const int active_tiles = topk_length > 0 ? (topk_length + BN - 1) / BN : 1;
    const int wm = wp % (BM / 16), wn = (wp / (BM / 16)) % 4, mi = lane / 4;
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

    auto tcQ = make_tiled_copy_A(CopyA8{}, mqk);
    auto thQ = tcQ.get_thread_slice(tid % QK_THREADS);
    auto tcK = make_tiled_copy_B(CopyB8{}, mqk);
    auto thK = tcK.get_thread_slice(tid % QK_THREADS);
    auto tcP = make_tiled_copy_A(CopyA8{}, mpv);
    auto thP = tcP.get_thread_slice(tid);

    Tensor srcQ = thQ.partition_S(make_mix_tensor_like(sQt));
    Tensor srcP = thP.partition_S(make_mix_tensor_like(sPt));

    // tile_to_shape(8x64 atom, (rows, cols)) puts cube (r/8, c/64) at
    //   ((c/64)*(rows/8) + r/8)*512 + (r%8)*64 + (c%64)
    // and for a 16-aligned row r%8 == 0, so the swizzle XOR key is 0.
    const int qk_lbo_a = 1, qk_sbo_a = 32;
    const int qk_lbo_b = 32, qk_sbo_b = 1;
    // Matrix addresses are warp-uniform. Keep their address arithmetic in
    // scalar registers to limit vector-register pressure in the QK/PV loops.
    const int qtsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pQ) / 16) + wm * 64);
    const int ktsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pK) / 16) + wn * 64);
    const int ptsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pP) / 16));
    const int kstage = KTILE / 16;

    Tensor rQ = tqk.partition_fragment_A(sQt);
    Tensor rK = tqk.partition_fragment_B(sKt0);
    Tensor rP = tpv.partition_fragment_A(sPt);
    Tensor dQ = thQ.retile_D(rQ);
    Tensor dK = thK.retile_D(rK);
    Tensor dP = thP.retile_D(rP);

    // Q -> swizzled smem once; fragments are streamed (reloaded per k-step) in the
    // QK loop rather than held in 72 resident registers, to fit 2 CTAs/CU.
    {
        if constexpr (!Q_IS_BF16) {
            Tensor gQt = make_tensor(make_gmem_ptr(gQ8), Shape<Int<BM>, Int<DQK>>{}, GenRowMajor{});
            for (int i = tid; i < BM * (DQK / 16); i += NTHR) {
                int r = i / (DQK / 16), g = i % (DQK / 16);
                *reinterpret_cast<cute::uint128_t *>(&sQt(r, g * 16)) =
                    (r < active_heads) ? *reinterpret_cast<const cute::uint128_t *>(&gQt(r, g * 16))
                                       : cute::uint128_t{};
            }
        } else {
            // Fuse the one-time BF16 -> E4M3 conversion with the Q gmem -> smem
            // load. This removes the materialized q_fp8 tensor and, for H128,
            // both strided H64 contiguous copies from the production path.
            for (int i = tid; i < BM * (DQK / 16); i += NTHR) {
                const int r = i / (DQK / 16), g = i % (DQK / 16);
                cute::uint128_t packed;
                auto *pairs = reinterpret_cast<__hg_fp8x2_storage_t *>(&packed);
                CUTE_UNROLL
                for (int j = 0; j < 8; ++j) {
                    const auto bf16x2 = (r < active_heads)
                                            ? *reinterpret_cast<const __ppu_bfloat162_raw *>(
                                                  gQbf16 + (long)r * q_stride_h + g * 16 + j * 2)
                                            : __ppu_bfloat162_raw{};
                    pairs[j] = flashmla::sparse_native_fp8::pack_e4m3_pair(bf16x2);
                }

                *reinterpret_cast<cute::uint128_t *>(&sQt(r, g * 16)) = packed;
            }
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
        if constexpr (TRUSTED_FULL_TOPK) {
            // The final partial tile uses slot bounds rather than index validity.
            sIdx[i] = i < topk_length ? gIdx[i] : (COMPACT ? -1 : 0);
        } else {
            sIdx[i] = i < topk_length ? gIdx[i] : -1;
        }
    }
    __syncthreads();

    // Per-thread gather offsets are tile-invariant: compute the swizzled
    // destinations once instead of re-deriving the XOR on every granule.
    static constexpr int GATHER_LANES = COMPACT ? 4 : 8;
    static constexpr int GATHER_WIDTH = GATHER_LANES * 16;
    static constexpr int KSEGS = (DQK + GATHER_WIDTH - 1) / GATHER_WIDTH;
    const int grow = tid / GATHER_LANES, glane = tid % GATHER_LANES, col0 = glane * 16;
    int kdst_off[KSEGS];
    {
        Tensor sK0 = make_tensor(make_smem_ptr(pK), LayoutK8{});
        const uint8_t *kbase0 = reinterpret_cast<const uint8_t *>(pK);
        CUTE_UNROLL
        for (int j = 0; j < KSEGS; ++j) {
            const int col = col0 + j * GATHER_WIDTH;
            const int gc = col < DQK ? col : 0;
            kdst_off[j] =
                COMPACT ? (int)(reinterpret_cast<const uint8_t *>(&sK0(grow, col0)) - kbase0) +
                              j * (BN * 64)
                        : (int)(reinterpret_cast<const uint8_t *>(&sK0(grow, gc)) - kbase0);
        }
    }

    auto issue_K = [&](int t) {
        uint8_t *kdst = reinterpret_cast<uint8_t *>(pK + (t & 1) * KTILE);
        const int tok = sIdx[t * BN + grow];
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
                            *reinterpret_cast<cute::uint128_t *>(kdst + kdst_off[j]));
            } else {
                const bool ok = t * BN + grow < topk_length;
                const uint8_t *src =
                    reinterpret_cast<const uint8_t *>(gKV) + (long)(ok ? tok : 0) * DQK + col0;
                CUTE_UNROLL
                for (int j = 0; j < KSEGS; ++j)
                    if (col0 + j * GATHER_WIDTH < DQK)
                        cute::PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                            *reinterpret_cast<const cute::uint128_t *>(src + j * GATHER_WIDTH),
                            *reinterpret_cast<cute::uint128_t *>(kdst + kdst_off[j]), ok);
            }
        } else {
            // one unsigned compare: every invalid encoding production uses
            // (-1, INT_MIN, s_kv, INT_MAX) maps to an unsigned value >= s_kv.
            const bool ok = (unsigned)tok < (unsigned)s_kv;
            const uint8_t *src =
                reinterpret_cast<const uint8_t *>(gKV) + (long)(ok ? tok : 0) * DQK + col0;
            CUTE_UNROLL
            for (int j = 0; j < KSEGS; ++j)
                if (col0 + j * GATHER_WIDTH < DQK)
                    cute::PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                        *reinterpret_cast<const cute::uint128_t *>(src + j * GATHER_WIDTH),
                        *reinterpret_cast<cute::uint128_t *>(kdst + kdst_off[j]), ok);
        }
    };
    // PV reads the same swizzled KV stage as QK for both head tiles.
    const int vrow = (lane & 3) * 4 + ((lane >> 2) & 3);
    const int vcol = wp * 16 + (lane >> 4) * 4;
    const int voff = (int)(reinterpret_cast<const uint8_t *>(&sKt0(vrow, vcol)) -
                           reinterpret_cast<const uint8_t *>(pK));
    flashmla_fp8_std::prime_kv_pipeline(issue_K);

    for (int t = 0; t < active_tiles; ++t) {
        flashmla_fp8_std::prefetch_next_kv(issue_K, t, active_tiles);

        Tensor acc_s = tqk.partition_fragment_C(sPt);
        Tensor acc_s1 = tqk.partition_fragment_C(sPt);
        clear(acc_s);
        clear(acc_s1);
        if (tid < QK_THREADS) {
// Streamed QK: only a small window of Q/K fragments is live at once, so the
// per-thread register footprint drops far enough to run 2 CTAs/CU.
#define QADDR(kk) (qtsm0 + ((kk) >> 1) * (BM * 4) + ((kk) & 1) * 2)
#define KADDR(kk) (ktsm0 + ((t & 1) * kstage) + ((kk) >> 1) * 256 + ((kk) & 1) * 2)
            // 1-group-ahead: prefetch the next 3 Q/K fragments while MMAing the current 3.
            PPU_LDM((uint32_t *)&rQ(0, 0, 0), QADDR(0), qk_lbo_a, qk_sbo_a);
            PPU_LDM((uint32_t *)&rK(0, 0, 0), KADDR(0), qk_lbo_b, qk_sbo_b);
            PPU_LDM((uint32_t *)&rQ(0, 0, 1), QADDR(1), qk_lbo_a, qk_sbo_a);
            PPU_LDM((uint32_t *)&rK(0, 0, 1), KADDR(1), qk_lbo_b, qk_sbo_b);
            PPU_LDM((uint32_t *)&rQ(0, 0, 2), QADDR(2), qk_lbo_a, qk_sbo_a);
            PPU_LDM((uint32_t *)&rK(0, 0, 2), KADDR(2), qk_lbo_b, qk_sbo_b);
            CUTE_UNROLL
            for (int k = 0; k + 2 < NK_QK; k += 3) {
                if (k + 3 < NK_QK) {
                    PPU_LDM((uint32_t *)&rQ(0, 0, k + 3), QADDR(k + 3), qk_lbo_a, qk_sbo_a);
                    PPU_LDM((uint32_t *)&rK(0, 0, k + 3), KADDR(k + 3), qk_lbo_b, qk_sbo_b);
                }
                if (k + 4 < NK_QK) {
                    PPU_LDM((uint32_t *)&rQ(0, 0, k + 4), QADDR(k + 4), qk_lbo_a, qk_sbo_a);
                    PPU_LDM((uint32_t *)&rK(0, 0, k + 4), KADDR(k + 4), qk_lbo_b, qk_sbo_b);
                }
                if (k + 5 < NK_QK) {
                    PPU_LDM((uint32_t *)&rQ(0, 0, k + 5), QADDR(k + 5), qk_lbo_a, qk_sbo_a);
                    PPU_LDM((uint32_t *)&rK(0, 0, k + 5), KADDR(k + 5), qk_lbo_b, qk_sbo_b);
                }
                cute::gemm(mqk, rQ(_, _, k + 0), rK(_, _, k + 0), acc_s);
                cute::gemm(mqk, rQ(_, _, k + 1), rK(_, _, k + 1), acc_s1);
                cute::gemm(mqk, rQ(_, _, k + 2), rK(_, _, k + 2), acc_s);
            }
            static constexpr int K_TAIL = (NK_QK / 3) * 3;
            if constexpr (NK_QK % 3 >= 1) {
                cute::gemm(mqk, rQ(_, _, K_TAIL), rK(_, _, K_TAIL), acc_s);
            }
            if constexpr (NK_QK % 3 == 2) {
                cute::gemm(mqk, rQ(_, _, K_TAIL + 1), rK(_, _, K_TAIL + 1), acc_s1);
            }
#undef QADDR
#undef KADDR
            CUTE_UNROLL
            for (int e = 0; e < 8; ++e) {
                const int col = wn * 16 + (lane % 4) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
                const bool valid = TRUSTED_FULL_TOPK
                                       ? t * BN + col < topk_length
                                       : (unsigned)sIdx[t * BN + col] < (unsigned)s_kv;

                acc_s(e) = valid ? acc_s(e) + acc_s1(e) : -INFINITY;
            }
            if (tid == 0)
                *sFlag = 0;

            flashmla_fp8_std::reduce_score_max<Traits, BM, 8>(acc_s, sPmax, lane, wn, r0, r1);
        } // All participating QK warps reconverge before the block barrier.
        __syncthreads();
        if (tid < BM) {
            flashmla_fp8_std::update_running_max<Traits, BM>(sPmax, sRunMax, sScale, sFlag, tid, qk_scale_log2);
        }
        __syncthreads();

        if (tid < QK_THREADS) {
            const float mx0 = sRunMax[r0], mx1 = sRunMax[r1];
            flashmla_fp8_std::quantize_probability<Traits, BM, 8>(
                acc_s, tqk, sPt, sPsum, lane, wn, r0, r1, mx0, mx1, qk_scale_log2);
        }

        if (*sFlag) {
            flashmla_fp8_std::rescale_output<Traits, BM>(acc_o, sScale, mi);
        }
        __syncthreads();

        if (tid < BM) {
            flashmla_fp8_std::update_running_sum<Traits, BM, false>(sPsum, sRunSum, sScale, tid);
        }

        // Accumulate PV directly in the running output fragment.
        // Fragment storage follows the logical PV operand shape. Actual loads
        // transpose four-byte pieces from the swizzled KV stage into that shape.
        flashmla_fp8_std::accumulate_pv<Traits, BM, BN>(
            tpv, rP, mpv, acc_o, pK, ptsm0, voff, (t & 1) * KTILE);

        flashmla_fp8_std::wait_next_kv(t, active_tiles);
    }
    // Publish the final running sum before any PV warp consumes it.
    __syncthreads();
    float rs[BM / 8];
#pragma unroll
    for (int j = 0; j < BM / 8; ++j) {
        float s = sRunSum[j * 8 + mi];
        if (gAttnSink && j * 8 + mi < active_heads) {
            // s is P_SCALE * sum(exp2(logit - running_max)). attn_sink is in
            // natural-log space, matching sparse_prefill_fwd's public contract.
            const float sink_log2 = gAttnSink[j * 8 + mi] * 1.4426950408889634f;
            s += exp2f(sink_log2 - sRunMax[j * 8 + mi] + LOG2_P_SCALE);
        }
        rs[j] = (s > 0.f && s == s) ? (out_scale / s) : 0.f;
    }

    // Optional statistics follow the FP8 interface: a ceil-quantized maximum
    // and base-2 LSE of the quantized QK scores, before attention-sink scaling.
    if (tid < active_heads && (gMaxLogits || gLse)) {
        const float m = sRunMax[tid], s = sRunSum[tid];
        if (gMaxLogits)
            gMaxLogits[tid] = m;
        if (gLse)
            gLse[tid] = (s > 0.f) ? (m + log2f(s * (1.f / P_SCALE))) : -INFINITY;
    }
    // bf16 output, matching the production kernel: fp32 would double the store
    // traffic and thrash L2 against the KV cache at realistic grid sizes.
    Tensor gObf = make_tensor(make_gmem_ptr(gO), Shape<Int<BM>, Int<DV>>{}, GenRowMajor{});
    Tensor cO = tpv.partition_C(gObf);
    // Pack adjacent BF16 elements into a single global store.
    CUTE_UNROLL
    for (int e = 0; e < size(acc_o); e += 2) {
        if (Traits::output_row(e) + mi >= active_heads)
            continue;
        const float sc = rs[Traits::output_row(e) >> 3];
        const cutlass::bfloat16_t b0(acc_o(e) * sc), b1(acc_o(e + 1) * sc);
        const uint32_t packed = (uint32_t)reinterpret_cast<const uint16_t &>(b0) |
                                ((uint32_t)reinterpret_cast<const uint16_t &>(b1) << 16);
        *reinterpret_cast<uint32_t *>(&cO(e)) = packed;
    }
}

// ---------------------------------------------------------------------------
bool try_sparse_fp8_std(const flashmla::sparse_native_fp8::AttentionParams &params);
bool try_sparse_prefill_fp8_ws(const flashmla::sparse_native_fp8::AttentionParams &params);
// Host launcher: head count, Q layout and sparse width select a tile.
// Both variants share the same public binding, stream and output contract.
// ---------------------------------------------------------------------------
template <int BM, int DQK> static constexpr int fp8_sparse_prefill_smem() {
    using Traits = Fp8SparsePrefillTraits<BM>;
    return BM * DQK + (2 * BN * DQK) + BM * BN + Traits::MaxTopk * 4 + 11 * BM * 4 + 256;
}

template <int BM, int DQK, bool Q_IS_BF16, bool TRUSTED_FULL_TOPK>
void launch_sparse_prefill_fp8_fwd(const flashmla::sparse_native_fp8::AttentionParams &params) {
    constexpr int NTHR = Fp8SparsePrefillTraits<BM>::Threads;
    constexpr int smem = fp8_sparse_prefill_smem<BM, DQK>();
    const int n_head_groups = (params.heads + BM - 1) / BM;
#define LAUNCH_ORDER(TWO_GROUPS, CHECK_BOUNDS, GRID)                                            \
    do {                                                                                        \
        hggcFuncSetAttribute(                                                                   \
            flash_fwd_sparse_prefill_fp8_kernel<BM, DQK, Q_IS_BF16, TRUSTED_FULL_TOPK,          \
                                                TWO_GROUPS, CHECK_BOUNDS>,                      \
            hggcFuncAttributeMaxDynamicSharedMemorySize, smem);                                 \
        flash_fwd_sparse_prefill_fp8_kernel<BM, DQK, Q_IS_BF16, TRUSTED_FULL_TOPK,              \
                                            TWO_GROUPS, CHECK_BOUNDS>                           \
            <<<GRID, NTHR, smem, params.stream>>>(                                              \
                params.q, reinterpret_cast<const E4M3 *>(params.kv), params.idx,                \
                reinterpret_cast<cutlass::bfloat16_t *>(params.out), params.scale,              \
                params.width / BN, params.kvrows, params.os, params.heads, params.qst,          \
                params.queries, params.maxima, params.lse, params.sink, params.lengths,         \
                params.qsh);                                                                    \
    } while (0)
    if (n_head_groups == 2) {
        const dim3 grid(4, (params.queries + 1) / 2);
        if (params.queries & 1)
            LAUNCH_ORDER(true, true, grid);
        else
            LAUNCH_ORDER(true, false, grid);
    } else {
        // Preserve the original H64 grid shape and blockIdx mapping exactly.
        const dim3 grid(params.queries, n_head_groups);
        LAUNCH_ORDER(false, false, grid);
    }
#undef LAUNCH_ORDER
}

template <int BM>
static void
dispatch_sparse_prefill_fp8_fwd(const flashmla::sparse_native_fp8::AttentionParams &params) {
#define LAUNCH(D, QBF16)                                                                        \
    do {                                                                                        \
        if (params.trusted_full_topk)                                                           \
            launch_sparse_prefill_fp8_fwd<BM, D, QBF16, true>(params);                          \
        else                                                                                    \
            launch_sparse_prefill_fp8_fwd<BM, D, QBF16, false>(params);                         \
    } while (0)
    if (params.dim == 512 && params.qbf) {
        LAUNCH(512, true);
    } else if (params.dim == 512) {
        LAUNCH(512, false);
    } else if (params.qbf) {
        LAUNCH(576, true);
    } else {
        LAUNCH(576, false);
    }
#undef LAUNCH
}

void run_sparse_prefill_fp8_fwd(const flashmla::sparse_native_fp8::AttentionParams &params) {
    if (params.queries == 0)
        return;
    if ((params.heads == 16 || params.heads % 64 == 0) && !params.qbf &&
        params.qst == params.heads * params.dim && params.qsh == params.dim) {
        if (try_sparse_fp8_std(params) || try_sparse_prefill_fp8_ws(params))
            return;
    }
    // Complete 64-head groups reuse each gathered KV tile across
    // all heads, including sparse lists with a partially filled final tile.
    auto dispatch = params.heads <= 16   ? dispatch_sparse_prefill_fp8_fwd<16>
                    : params.heads <= 32 ? dispatch_sparse_prefill_fp8_fwd<32>
                                         : dispatch_sparse_prefill_fp8_fwd<64>;
    dispatch(params);
}
