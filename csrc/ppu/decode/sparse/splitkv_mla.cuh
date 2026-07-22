/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "splitkv_mla.h"

#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <c10/cuda/CUDAException.h>

#include "kernel_traits.h"
#include "kerutils/common/static_switch.h"
#include "kerutils/device/ppu/softmax.cuh"
#include "kerutils/device/ppu/mask.cuh"
#include "kerutils/device/ppu/dequant.cuh"
#include "utils.h"
#include "ppuxx/decode/combine/combine.cuh"

#include <hggc_ad.h>

namespace flash {
using namespace cute;

template<typename Kernel_traits, typename Params>
__forceinline__ __device__ void compute_attn_fp8_sparse_splitkv(
    const Params &params, const int batch_id, const int bidh, const int m_block,
    const int n_split_idx, const int n_block_min, const int n_block_max, const bool NoSplit,
    const int ori_klen, const int ori_block_max, const int ext_klen) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps0 = Kernel_traits::kNWarps0;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int AtomLayoutQ = Kernel_traits::AtomLayoutQ;
    constexpr int AtomLayoutP = Kernel_traits::AtomLayoutP;
    constexpr bool USE_MMA_M8 = Kernel_traits::USE_MMA_M8;
    constexpr int MMA_ATOM_M = USE_MMA_M8 ? 8 : 16;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int h_k_idx = m_block % cute::ceil_div(params.ngroups, kBlockM); // s_q = s_q_ori * h_q
    const int s_q_idx = m_block / cute::ceil_div(params.ngroups, kBlockM);
    const int row_base = h_k_idx * kBlockM + s_q_idx * params.ngroups;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    if (row_base >= params.seqlen_q) return;

    using KVCacheGmem = KVCacheG2SFP8<kBlockN, Kernel_traits::kNThreads, kHeadDim>;

    using ElementKVCache = typename KVCacheGmem::ElementKVCache;
    using SmemLayoutKNoAiu = typename KVCacheGmem::SmemLayoutK;
    using SmemLayoutVtransposedNoAiu = typename KVCacheGmem::SmemLayoutVtransposed;
    using FragScale = typename KVCacheGmem::FragScale;

    ElementKVCache *gK_base = reinterpret_cast<ElementKVCache *>(params.k_ptr) +
        (bidh / params.h_h_k_ratio) * params.k_head_stride;
    ElementKVCache *gExK_base = reinterpret_cast<ElementKVCache *>(params.extra_k_ptr) +
        (bidh / params.h_h_k_ratio) * params.extra_k_head_stride;

    const int load_col_idx = tidx / 8; // + v * RowsPerGmem; // tidx/8,  0~64
    // One warps:[0 1 2 3], [5 6 7 8], ...., -> [0 16 32 48], [1 17 33 49]
    // col = col_x * 16 + col_y * 4 + col_z; -> col_in_indices = col_z * 16 + col_x * 4 + col_y;
    // (col_x, col_y, col_z) = (col_load / 16, (col_load % 16) / 4, col_load % 4)
    // -> (warp_idx/4 + v * RowsPerGmem/16, warp_idx%4, lane_idx/8)
    // convert to: col_in_indices = lane_idx/8 * 16 + warp_idx + v * RowsPerGmem/4
#if ACOMPUTE_VERSION ==10000
    const int lane_idx = tidx % 32;
    // const int col_in_indices = ((load_col_idx)% 8)*8 + (load_col_idx)/8;
    const int col_in_indices = (lane_idx/8) * 16 + warp_idx; // + v * RowsPerGmem/4
    // const int col_in_indices1 = (load_col_idx % 4) * 16 + (load_col_idx / 16) * 4 + (load_col_idx % 16) / 4;
    // const int col_in_indices = load_col_idx;
#else
    const int col_in_indices = load_col_idx; // + v * RowsPerGmem;
#endif
    index_t batch_stride = params.k_batch_stride;
    index_t row_stride = params.k_row_stride;
    int* __restrict__ gIndices = params.indices_ptr + batch_id * params.indices_batch_stride + s_q_idx * params.indices_row_stride + col_in_indices;

    // extra info need only extra instance enabled.
    int extra_block_size = params.extra_page_block_size;
    index_t extra_batch_stride = params.extra_k_batch_stride;
    index_t extra_row_stride = params.extra_k_row_stride;
    int* __restrict__ gExIndices = params.extra_indices_ptr + batch_id * params.extra_indices_batch_stride + s_q_idx * params.extra_indices_row_stride + col_in_indices;

    // only supposed tile single column load.
    int* __restrict__ gIndices_ptr = gIndices;
    // FragScale tKrScale0, tKrScale1;
    FragScale tKrScale;
    int block_index = 0, rel_idx_in_block = 0;
    bool valid = true;
    int nxt_token_idx;
    int topk_len = ori_klen;
    int page_block_size = params.page_block_size;
    ElementKVCache* k_ptr = gK_base;
    int real_block = n_block_min;

    if constexpr (kHeadDim == 512) {
        if (n_block_min >= ori_block_max) {
            gIndices_ptr = gExIndices;
            page_block_size = extra_block_size;
            topk_len = ext_klen;
            batch_stride = extra_batch_stride;
            row_stride = extra_row_stride;
            k_ptr = gExK_base;
            real_block = n_block_min - ori_block_max;
        }
    }

    auto load_token = [&](const int block_idx) {
        // #pragma unroll
        // for (int v = 0; v < KVCacheGmem::kColPerGmemNope; v++)
        nxt_token_idx = __ldg(gIndices_ptr + block_idx * kBlockN);
    };

    auto update_block = [&](const int block_idx) {
        if constexpr (Kernel_traits::kPagePow2) {
            int kLog2 = __ffs(page_block_size) - 1;
            int kMask = page_block_size - 1;
            block_index      = nxt_token_idx >> kLog2;
            rel_idx_in_block = nxt_token_idx & kMask;
        } else {
            block_index      = nxt_token_idx / page_block_size;
            rel_idx_in_block = nxt_token_idx - block_index * page_block_size;
        }
        valid = (nxt_token_idx >= 0)
            & ((col_in_indices + block_idx * kBlockN) < topk_len);
    };

    auto load_scale = [&](FragScale& scale) {
        if constexpr (kHeadDim == 512) {
            fp8_e8m0* gK_scale = (fp8_e8m0*)(
                k_ptr + block_index * batch_stride + page_block_size * KVCacheGmem::BytesPerToken
                    + rel_idx_in_block * 8);
            scale(0) = valid ? __ldg((int64_t*)gK_scale) : int64_t(0);
        } else {
            float4* gK_scale = (float4*)(k_ptr + block_index * batch_stride + rel_idx_in_block * row_stride + kHeadDimV);
            if (valid) {
                float4 val = __ldg(gK_scale);
                scale(0) = val.x;
                scale(1) = val.y;
                scale(2) = val.z;
                scale(3) = val.w;
            } else {
                #pragma unroll
                for (int k = 0; k < 4; k ++) {
                    scale(k) = float(0.f);
                }
            }
        }

    };

    load_token(real_block);

    // never has n_block_min >= n_block_max in tile scheduler mode
    assert(n_block_min < n_block_max);

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const int row_offset_q = batch_id * params.q_batch_stride + bidh * params.q_head_stride + row_base * params.q_row_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), SmemLayoutKNoAiu{});
    Tensor sVt = make_tensor(sK.data(), SmemLayoutVtransposedNoAiu{});
    Tensor sVtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // double shared memory for k/v cache.
    Tensor sK_double = make_tensor(sK.data() + size(sK), SmemLayoutKNoAiu{});
    Tensor sP = make_tensor(sK_double.data() + size(sK_double), typename Kernel_traits::SmemLayoutP{});

    Tensor smem_row_scale = make_tensor(make_smem_ptr(reinterpret_cast<float *>((sP.data() + size(sP)).get())),
        Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor smem_row_via_warp = make_tensor(smem_row_scale.data() + size(smem_row_scale),
        Shape<Int<kBlockM>, Int<kNWarps0/AtomLayoutQ>>{}, Stride<Int<kNWarps0/AtomLayoutQ>, _1>{});
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(
        reinterpret_cast<int*>((smem_row_via_warp.data() + ((kNWarps0==AtomLayoutQ) ? 0: size(smem_row_via_warp))).get())),
        Shape<_2, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{});

    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;

    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMma tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrP  = thr_mma_o.partition_fragment_A(sP);                           // (MMA,MMA_M,MMA_N)
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    // Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, kBlockM, params.q_row_stride, kBlockM, Kernel_traits::kBlockKSmem, 0};
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, kBlockM, params.d, params.q_row_stride);
#endif
    const int tid_thread_slice = warp_idx * 32;
#else
    const int tid_thread_slice = tidx;
#endif

    // auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    // auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid_thread_slice);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));

    // PREDICATES
    //
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    flash::copy<false, true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                params.ngroups - h_k_idx * kBlockM);

    if constexpr (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if constexpr (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    auto smem_tiled_copy_S = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomS{}, tiled_mma_s);
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tidx);
    Tensor tSsS = smem_thr_copy_S.partition_D(sP);

#if ACOMPUTE_VERSION == 10000
    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomP{}, tiled_mma_o);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tidx);
    Tensor tOsP = smem_thr_copy_P.partition_S(sP);
#else
    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomP_TLS{}, tiled_mma_o);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tid_thread_slice);
    Tensor tOsP = smem_thr_copy_P.partition_S(make_mix_tensor_like(sP));
#endif

#if ACOMPUTE_VERSION == 10000
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_V.partition_S(sVt);
#else
    auto smem_tiled_copy_K = make_tiled_copy_B(typename KVCacheGmem::SmemCopyAtomK{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid_thread_slice);
    auto tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK));

    auto smem_tiled_copy_V = make_tiled_copy_B(typename KVCacheGmem::SmemCopyAtomV{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid_thread_slice);
    auto tOsVt = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt));
#endif

    KVCacheGmem kvload_gmem(tidx);

    update_block(real_block);
    load_scale(tKrScale);

    kvload_gmem.template load_g2s_async(k_ptr, sK, block_index, rel_idx_in_block, batch_stride, row_stride, valid);


    real_block = n_block_min + 1;
    if (real_block < n_block_max) {
        if (kHeadDim == 512) {
            if (real_block >= ori_block_max) {
                gIndices_ptr = gExIndices;
                real_block = n_block_min + 1 - ori_block_max;
                topk_len = ext_klen;
                page_block_size = extra_block_size;

            }
        }
        load_token(real_block);
    }

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);
    smem_valid_indices(0, col_in_indices) = valid;

    constexpr bool ForceUseTsm = (kNWarps0 != kNWarps) || (AtomLayoutQ != AtomLayoutP);
    flash::SoftmaxBetweenWarps<USE_MMA_M8, kBlockM, AtomLayoutQ, AtomLayoutP, kNWarps0/AtomLayoutQ, ForceUseTsm> softmax;

    Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    // update_block(n_block_min + 1);

    struct IsOrigBlock {};
    struct IsExtraBlock {};
    struct IsFirstBlock {};
    struct IsMidBlock {};
    struct IsLastBlock {};
    struct IsIsoBlock {};

    auto  process_one_block = [&](int block_idx, auto nxt_is_extra_block_t, auto block_type_t) {
        static constexpr bool NXT_IS_EXTRA_BLOCK = std::is_same_v<decltype(nxt_is_extra_block_t), IsExtraBlock>;
        static constexpr bool IS_FIRST_BLOCK = std::is_same_v<decltype(block_type_t), IsFirstBlock> || std::is_same_v<decltype(block_type_t), IsIsoBlock>;
        static constexpr bool IS_LAST_BLOCK = std::is_same_v<decltype(block_type_t), IsLastBlock> || std::is_same_v<decltype(block_type_t), IsIsoBlock>;

        clear(acc_s);
        int buf_idx = (block_idx - n_block_min) % 2;
        int nxt_buf_idx = buf_idx ^ 0x1;
        int blk_update_idx = real_block;

        FragScale scale_load;
        real_block++;

        if constexpr (!IS_LAST_BLOCK) {
            k_ptr = gK_base;
            if constexpr (NXT_IS_EXTRA_BLOCK) {
                k_ptr = gExK_base;
                batch_stride = extra_batch_stride;
                row_stride = extra_row_stride;
                gIndices_ptr = gExIndices;
                page_block_size = extra_block_size;
                topk_len = ext_klen;
                real_block = block_idx + 2 - ori_block_max;
            } else {
                if (kHeadDim == 512) {
                    if ((block_idx + 2) >= ori_block_max) {
                        gIndices_ptr = gExIndices;
                        real_block = block_idx + 2 - ori_block_max;
                    }
                }
            }

            update_block(blk_update_idx);

            auto sK_load = make_tensor(sK.data() + nxt_buf_idx * size(sK), layout(sK));
            kvload_gmem.template load_g2s_async(k_ptr, sK_load, block_index, rel_idx_in_block, batch_stride, row_stride, valid);

            if (block_idx < n_block_max - 2) {
                load_token(real_block);
            }

            flash::cp_async_wait<1>();
        } else {
            flash::cp_async_wait<0>();
            __syncthreads();
        }

        auto sK_cvt = make_tensor(sK.data() + buf_idx * size(sK), layout(sK));
        kvload_gmem.template cvt_fp8_store(sK_cvt, sK_cvt, tKrScale);
        __syncthreads();

        auto qk_softmax_fused = [&]() {
            auto tSsk_curr = make_tensor(tSsK.data() + buf_idx * size(sK), layout(tSsK));
            if constexpr (Kernel_traits::Is_Q_in_regs) {
                flash::gemm_rss<Kernel_traits::KEEP_Q_NUM>(
                        acc_s, tSrQ, tSrK, tSsQ, tSsk_curr, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
                        smem_thr_copy_Q, smem_thr_copy_K
                );
            } else {
                flash::gemm<false>(
                        acc_s, tSrQ, tSrK, tSsQ, tSsk_curr, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
                        smem_thr_copy_Q, smem_thr_copy_K
                );
            }

            // scale load n+1
            if constexpr (!IS_LAST_BLOCK & kBlockM >= 64) {
                load_scale(scale_load);
            }

            constexpr int MMA_N_S = kBlockN / decltype(typename Kernel_traits::TiledMmaS{}.template tile_size_mnk<1>())::value;
            const int warpN_idx = (warp_idx / AtomLayoutQ) * MMA_N_S;
            flash::apply_indices_mask(acc_s, smem_valid_indices, warpN_idx, buf_idx);

            if constexpr (IS_FIRST_BLOCK) {
                softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2);
            } else {
                softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2);
            }

            Tensor rS = flash::convert_type<Element>(acc_s);
            Tensor tSaS = smem_thr_copy_S.retile_S(rS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_S, tSaS, tSsS);
        };

        if (warp_idx < kNWarps0) {
            qk_softmax_fused();
        }

        __syncthreads();
        if constexpr (!IS_LAST_BLOCK & kBlockM < 64) {
            load_scale(scale_load);
        }
        if constexpr (!IS_FIRST_BLOCK) {
            softmax.template softmax_rescale_o(acc_o, smem_row_scale);
        }

        auto tOsVt_curr = make_tensor(tOsVt.data() + buf_idx * size(sK), layout(tOsVt));
        flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt_curr, tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V,
                smem_thr_copy_P, smem_thr_copy_V);

        if constexpr (!IS_LAST_BLOCK) {
            smem_valid_indices(nxt_buf_idx, col_in_indices) = valid;
        }

        if constexpr (!IS_LAST_BLOCK) {
            tKrScale = scale_load;
        }
        __syncthreads();
    };


    if constexpr (kHeadDim == 512) {
        if ((n_block_max - n_block_min) > 1) {
            if (n_block_min < ori_block_max - 1) {
                process_one_block(n_block_min, IsOrigBlock{}, IsFirstBlock{});
            } else {
                process_one_block(n_block_min, IsExtraBlock{}, IsFirstBlock{});
            }

            int cur_ori_max = std::min(n_block_max, ori_block_max);
            for (int i = n_block_min + 1; i < cur_ori_max - 1; ++i) {
                process_one_block(i, IsOrigBlock{}, IsMidBlock{});
            }
            int cur_ext_min = std::max(cur_ori_max - 1, n_block_min + 1);
            for (int i = cur_ext_min; i < n_block_max - 1; ++i) {
                process_one_block(i, IsExtraBlock{}, IsMidBlock{});
            }
            process_one_block(n_block_max - 1, IsExtraBlock{}, IsLastBlock{});
        } else {
            process_one_block(n_block_min, IsOrigBlock{}, IsIsoBlock{});
        }
    } else {
        if ((n_block_max - n_block_min) == 1) {
            process_one_block(n_block_min, IsOrigBlock{}, IsIsoBlock{});
        } else {
            process_one_block(n_block_min, IsOrigBlock{}, IsFirstBlock{});
            for (int i = n_block_min + 1; i < n_block_max - 1; ++i) {
                process_one_block(i, IsOrigBlock{}, IsMidBlock{});
            }
            process_one_block(n_block_max - 1, IsOrigBlock{}, IsLastBlock{});
        }
    }

    if (warp_idx < kNWarps0) {
        if (NoSplit) {
            softmax.template normalize_softmax_lse_per_warp<false, /*UseTbSync=*/false>(smem_row_via_warp, smem_row_scale, params.scale_softmax, params.attn_sink_ptr, params.scale_softmax_log2, h_k_idx);
        } else {
            softmax.template normalize_softmax_lse_per_warp<true, /*UseTbSync=*/false>(smem_row_via_warp, smem_row_scale, params.scale_softmax);
        }
    }
    __syncthreads();
    softmax.template softmax_rescale_o(acc_o, smem_row_scale);

    // Epilogue
    if (NoSplit) {
        flash::store<Kernel_traits, false, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    } else {
        flash::store<Kernel_traits, true, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }
}

template<typename Kernel_traits, typename Params>
__forceinline__ __device__ void compute_attn_bf16_sparse_splitkv(
    const Params &params, const int batch_id, const int bidh, const int m_block,
    const int n_split_idx, const int n_block_min, int n_block_max, const bool NoSplit,
    const int ori_klen, const int ori_block_max, const int ext_klen) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int AtomLayoutQ = Kernel_traits::AtomLayoutQ;
    constexpr int AtomLayoutP = Kernel_traits::AtomLayoutP;
    constexpr bool USE_MMA_M8 = Kernel_traits::USE_MMA_M8;
    constexpr int MMA_ATOM_M = USE_MMA_M8 ? 8 : 16;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int lane_idx = tidx % 32;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int h_k_idx = m_block % cute::ceil_div(params.ngroups, kBlockM); // s_q = s_q_ori * h_q
    const int s_q_idx = m_block / cute::ceil_div(params.ngroups, kBlockM);
    const int row_base = h_k_idx * kBlockM + s_q_idx * params.ngroups;

#if DSA_SIM_AIU
    using KVCacheGmem = KVCacheGmemBf16SimAIU<Element, kBlockN, Kernel_traits::kNThreads, kHeadDim, kHeadDimV>;
    using SmemLayoutKSim = typename KVCacheGmem::SmemLayoutKSim;
#else
    using KVCacheGmem = KVCacheGmemBf16<Element, kBlockN, Kernel_traits::kNThreads, kHeadDim, kHeadDimV>;
#endif
    using SmemLayoutKNoAiu = typename KVCacheGmem::SmemLayoutK;
    using GmemTiledCopyKNoAiu = typename KVCacheGmem::GmemTiledCopy;
    using SmemLayoutVtNoAiu = typename KVCacheGmem::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename KVCacheGmem::SmemLayoutVtransposedNoSwizzle;
    constexpr int RowsPerGmem = Kernel_traits::kNThreads / KVCacheGmem::kGmemThreadsPerRow;
    static_assert(RowsPerGmem % 16 == 0 && kBlockN % RowsPerGmem == 0);
    // 4warps: 16; 8warps: 32; 16 warps: 64
    constexpr int indices_per_load = kBlockN / RowsPerGmem;
    // kBlockN64: 1; kBlockN32: 2; kBlockN16: 16

    if (row_base >= params.seqlen_q) return;
    // never has n_block_min >= n_block_max in tile scheduler mode
    assert(n_block_min < n_block_max);

    const int load_col_idx = tidx/8; // + v * RowsPerGmem; // tidx/8,  0~64
    // One warps:[0 1 2 3], [5 6 7 8], ...., -> [0 16 32 48], [1 17 33 49]
    // col = col_x * 16 + col_y * 4 + col_z; -> col_in_indices = col_z * 16 + col_x * 4 + col_y;
    // (col_x, col_y, col_z) = (col_load / 16, (col_load % 16) / 4, col_load % 4)
    // -> (warp_idx/4 + v * RowsPerGmem/16, warp_idx%4, lane_idx/8)
    // convert to: col_in_indices = lane_idx/8 * 16 + warp_idx + v * RowsPerGmem/4
#if ACOMPUTE_VERSION ==10000
    const int col_in_indices = (lane_idx/8) * 16 + warp_idx; // + v * RowsPerGmem/4
#else
    const int col_in_indices = load_col_idx; // + v * RowsPerGmem;
#endif
    int* gIndices_ptr = params.indices_ptr + batch_id * params.indices_batch_stride
                      + s_q_idx * params.indices_row_stride;
    int* gExIndices_ptr = params.extra_indices_ptr + batch_id * params.extra_indices_batch_stride
                      + s_q_idx * params.extra_indices_row_stride;
    int nxt_token_idx1[indices_per_load];

    auto token_idx_update = [&](int* indices_ptr, const int block_idx_load, const int topk_len){
        #pragma unroll
        for (int v = 0; v < indices_per_load; v++) {
            index_t row_offset_indices = block_idx_load * kBlockN + v * RowsPerGmem + load_col_idx;
            nxt_token_idx1[v] = row_offset_indices < topk_len ? __ldg(indices_ptr + row_offset_indices) : -1;
        }
    };

    if constexpr (kHeadDim == 512) {
        int real_block = n_block_min;
        int topk_len = ori_klen;
        int* working_gIndices = gIndices_ptr;
        if (n_block_min >= ori_block_max) {
            real_block = n_block_min - ori_block_max;
            topk_len = ext_klen;
            working_gIndices = gExIndices_ptr;
        }
        token_idx_update(working_gIndices, real_block, topk_len);
    } else {
        token_idx_update(gIndices_ptr, n_block_min, ori_klen);
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const int row_offset_q = batch_id * params.q_batch_stride + bidh * params.q_head_stride + row_base * params.q_row_stride;
    // q = q.view({batch_size, seqlen_q_ori, 1, ngroups, head_size}).transpose(2, 3)
    //     .reshape({batch_size,      seqlen_q_ori * ngroups, 1,             head_size});
    //               q_batch_stride,  q_row_stride,           q_head_stride, 1

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), SmemLayoutKNoAiu{});
#if DSA_SIM_AIU
    Tensor sKSim = make_tensor(sK.data(), SmemLayoutKSim{});
#endif
    Tensor sVt = make_tensor(sK.data(), SmemLayoutVtNoAiu{});
    Tensor sVtNoSwizzle = make_tensor(sK.data(), SmemLayoutVtNoSwizzle{});

    // double shared memory for k/v cache.
    Tensor sK_double = make_tensor(sK.data() + size(sK), SmemLayoutKNoAiu{});
#if DSA_SIM_AIU
    Tensor sKSim_double = make_tensor(sK_double.data(), SmemLayoutKSim{});
#endif
    Tensor sVt_double = make_tensor(sK_double.data(), SmemLayoutVtNoAiu{});

    Tensor sP = make_tensor(sK_double.data() + size(sK_double), typename Kernel_traits::SmemLayoutP{});

    Tensor smem_row_scale = make_tensor(make_smem_ptr(reinterpret_cast<float *>((sP.data() + size(sP)).get())),
        Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor smem_row_via_warp = make_tensor(smem_row_scale.data() + size(smem_row_scale),
        Shape<Int<kBlockM>, Int<kNWarps/AtomLayoutQ>>{}, Stride<Int<kNWarps/AtomLayoutQ>, _1>{});
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(
        reinterpret_cast<int*>((smem_row_via_warp.data() + ((kNWarps==AtomLayoutQ) ? 0: size(smem_row_via_warp))).get())),
        Shape<_2, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{});

    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;

    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMma tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrP  = thr_mma_o.partition_fragment_A(sP);                           // (MMA,MMA_M,MMA_N)
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

#if USE_AIU
#if ACOMPUTE_VERSION == 10000
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, kBlockM, params.q_row_stride, kBlockM, Kernel_traits::kBlockKSmem, 0};
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, kBlockM, params.d, params.q_row_stride);
#endif
    const int tid_thread_slice = warp_idx * 32;
#else
    const int tid_thread_slice = tidx;
#endif

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid_thread_slice);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));

    // PREDICATES
    //
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    flash::copy<false, true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                params.ngroups - h_k_idx * kBlockM);

    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    auto smem_tiled_copy_S = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomS{}, tiled_mma_s);
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tidx);
    Tensor tSsS = smem_thr_copy_S.partition_D(sP);

#if ACOMPUTE_VERSION == 10000
    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomP{}, tiled_mma_o);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tidx);
    Tensor tOsP = smem_thr_copy_P.partition_S(sP);
#else
    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomP_TLS{}, tiled_mma_o);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tid_thread_slice);
    Tensor tOsP = smem_thr_copy_P.partition_S(make_mix_tensor_like(sP));
#endif

    // KV not use AIU copy
#if DSA_SIM_AIU
    auto smem_tiled_copy_K = make_tiled_copy_B(typename KVCacheGmem::SmemCopyAtomK{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(warp_idx * 32);
    auto tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK));

    auto smem_tiled_copy_V = make_tiled_copy_B(typename KVCacheGmem::SmemCopyAtomV{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(warp_idx * 32);
    auto tOsVt = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt));
#else
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_V.partition_S(sVt);
#endif

    GmemTiledCopyKNoAiu gmem_tiled_copy_K;
#if DSA_SIM_AIU && (ACOMPUTE_VERSION ==10000)
    int cross_tid_h = (tidx & 0xFFFFFFF8) >> 3;
    int cross_tid_l = tidx & 0x7;
    int cross_bias = (cross_tid_l / 2 == 1) ? 2 : ((cross_tid_l / 2 == 2) ? 1 : cross_tid_l / 2);
    cross_tid_h = (cross_tid_h & 0xFFFFFFFC) | (((cross_tid_h & 0x3) + cross_bias) & 0x3);
    int sim_cross_tid = (cross_tid_h << 3) | cross_tid_l;
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(sim_cross_tid);
#else
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);
#endif
    Tensor tKgK = gmem_thr_copy_K.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
#if DSA_SIM_AIU
    Tensor tKsK = gmem_thr_copy_K.partition_D(sKSim);
    Tensor tKsK_double = gmem_thr_copy_K.partition_D(sKSim_double);
#else
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
    Tensor tKsK_double = gmem_thr_copy_K.partition_D(sK_double);
#endif

    Element *gK_base = reinterpret_cast<Element *>(params.k_ptr) +
        (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Element *gExK_base = reinterpret_cast<Element *>(params.extra_k_ptr) +
        (bidh / params.h_h_k_ratio) * params.extra_k_head_stride;

    static_assert(indices_per_load == size<1>(tKgK));
    auto KV_load = [&](int buf_idx, Element* gK_ptr, int block_size, index_t batch_stride,
        index_t row_stride) {
        #pragma unroll
        for (int v = 0; v < indices_per_load; v++) {
            auto tKgK_current = tKgK(_, v, _);
            auto tKsK_buf = make_tensor(tKsK.data() + buf_idx * size(sK), layout(tKsK));
            auto tKsK_current = tKsK_buf(_, v, _);
            int token_index = nxt_token_idx1[v];
            bool is_token_valid = token_index >= 0;
            int block_index = token_index / block_size;
            int rel_idx_in_block = (token_index + block_size) % block_size;
            tKgK_current.data() = gK_ptr + (int64_t) block_index * batch_stride
                        + rel_idx_in_block * row_stride + (tidx % 8) * 8;
            gmem_tiled_copy_K.pred = is_token_valid;
            cute::copy(gmem_tiled_copy_K, tKgK_current, tKsK_current);
#if ACOMPUTE_VERSION ==10000
            smem_valid_indices(buf_idx, col_in_indices + v * RowsPerGmem / 4) = is_token_valid;
# else
            smem_valid_indices(buf_idx, col_in_indices + v * RowsPerGmem) = is_token_valid;
#endif
        }
        cute::cp_async_fence();
    };

    if constexpr (kHeadDim == 512) {
        Element* working_gK = gK_base;
        int block_size = params.page_block_size;
        index_t batch_stride = params.k_batch_stride;
        index_t row_stride = params.k_row_stride;
        if (n_block_min >= ori_block_max) {
            working_gK = gExK_base;
            block_size = params.extra_page_block_size;
            batch_stride = params.extra_k_batch_stride;
            row_stride = params.extra_k_row_stride;
        }
        KV_load(0, working_gK, block_size, batch_stride, row_stride);
        if ((n_block_min + 1) < std::min(n_block_max, ori_block_max)) {
            token_idx_update(gIndices_ptr, n_block_min + 1, ori_klen);
        } else if ((n_block_min + 1) < n_block_max) {
            token_idx_update(gExIndices_ptr, n_block_min + 1 - ori_block_max, ext_klen);
        }
    } else {
        KV_load(0, gK_base, params.page_block_size, params.k_batch_stride, params.k_row_stride);
        if (n_block_min < n_block_max - 1) {
            token_idx_update(gIndices_ptr, n_block_min + 1, ori_klen);
        }
    }

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        #pragma unroll
        for (int i = 0; i < Kernel_traits::KEEP_Q_NUM; ++i) {
            cute::copy(smem_tiled_copy_Q, tSsQ(_, _, i), tSrQ_copy_view(_, _, i));
        }
    }

    clear(acc_o);

    flash::SoftmaxBetweenWarps<USE_MMA_M8, kBlockM, AtomLayoutQ, AtomLayoutP, kNWarps/AtomLayoutQ, 1/*ForceUseTsm*/> softmax;

    struct IsOrigBlock {};
    struct IsExtraBlock {};
    struct IsMidBlock {};
    struct IsLastBlock {};

    auto process_one_block = [&](int block_idx, auto nxt_is_extra_block_t, auto block_type_t) {
        static constexpr bool NXT_IS_EXTRA_BLOCK = std::is_same_v<decltype(nxt_is_extra_block_t), IsExtraBlock>;
        static constexpr bool IS_LAST_BLOCK = std::is_same_v<decltype(block_type_t), IsLastBlock> ;

        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        int buf_idx = (block_idx - n_block_min) % 2;

        if constexpr (!IS_LAST_BLOCK) {
            __syncthreads();
            if constexpr (NXT_IS_EXTRA_BLOCK) {
                KV_load(buf_idx ^ 0x1, gExK_base, params.extra_page_block_size, params.extra_k_batch_stride, params.extra_k_row_stride);
            } else {
                KV_load(buf_idx ^ 0x1, gK_base, params.page_block_size, params.k_batch_stride, params.k_row_stride);
            }
            if constexpr (kHeadDim == 512) {
                if ((block_idx + 2) < std::min(n_block_max, ori_block_max)) {
                    token_idx_update(gIndices_ptr, block_idx + 2, ori_klen);
                } else if ((block_idx + 2) < n_block_max) {
                    token_idx_update(gExIndices_ptr, block_idx + 2 - ori_block_max, ext_klen);
                }
            } else {
                if ((block_idx + 2) < n_block_max) {
                    token_idx_update(gIndices_ptr, block_idx + 2, ori_klen);
                }
            }
            flash::cp_async_wait<1>();
        } else {
            flash::cp_async_wait<0>();
        }
        __syncthreads();

        auto tSsk_curr = make_tensor(tSsK.data() + buf_idx * size(sK), layout(tSsK));
        if constexpr (Kernel_traits::Is_Q_in_regs) {
            flash::gemm_rss<Kernel_traits::KEEP_Q_NUM>(
                acc_s, tSrQ, tSrK, tSsQ, tSsk_curr, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);
        } else {
            flash::gemm<false>(
                acc_s, tSrQ, tSrK, tSsQ, tSsk_curr, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
        }

        constexpr int MMA_N_S = kBlockN / decltype(typename Kernel_traits::TiledMmaS{}.template tile_size_mnk<1>())::value;
        const int warpN_idx = (warp_idx / AtomLayoutQ) * MMA_N_S;
        flash::apply_indices_mask(acc_s, smem_valid_indices, warpN_idx, buf_idx);

        block_idx == n_block_min
            ? softmax.template softmax_rescale_per_warp</*Is_first=*/true,  /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2)
            : softmax.template softmax_rescale_per_warp</*Is_first=*/false, /*Check_inf=*/true>(acc_s, smem_row_via_warp, smem_row_scale, params.scale_softmax_log2);

        Tensor rS = flash::convert_type<Element>(acc_s);
        Tensor tSaS = smem_thr_copy_S.retile_S(rS);
        cute::copy(smem_tiled_copy_S, tSaS, tSsS);

        __syncthreads();
        if (block_idx > n_block_min) {
            softmax.template softmax_rescale_o(acc_o, smem_row_scale);
        }

        auto tOsVt_curr = make_tensor(tOsVt.data() + buf_idx * size(sK), layout(tOsVt));
        flash::gemm(acc_o, tOrP, tOrVt, tOsP, tOsVt_curr, tiled_mma_o, smem_tiled_copy_P, smem_tiled_copy_V,
            smem_thr_copy_P, smem_thr_copy_V);
    };

    if constexpr (kHeadDim == 512) {
        int cur_ori_max = std::min(n_block_max, ori_block_max);
        for (int i = n_block_min; i < cur_ori_max - 1; ++i) {
            process_one_block(i, IsOrigBlock{}, IsMidBlock{});
        }
        int cur_ext_min = std::max(cur_ori_max - 1, n_block_min);
        for (int i = cur_ext_min; i < n_block_max - 1; ++i) {
            process_one_block(i, IsExtraBlock{}, IsMidBlock{});
        }
        process_one_block(n_block_max - 1, IsExtraBlock{}, IsLastBlock{});
    } else {
        for (int i = n_block_min; i < n_block_max - 1; ++i) {
            process_one_block(i, IsOrigBlock{}, IsMidBlock{});
        }
        process_one_block(n_block_max - 1, IsOrigBlock{}, IsLastBlock{});
    }

    if (NoSplit) {
        softmax.template normalize_softmax_lse_per_warp<false>(smem_row_via_warp, smem_row_scale, params.scale_softmax, params.attn_sink_ptr, params.scale_softmax_log2, h_k_idx);
    } else {
        softmax.template normalize_softmax_lse_per_warp<true>(smem_row_via_warp, smem_row_scale, params.scale_softmax);
    }

    __syncthreads();

    softmax.template softmax_rescale_o(acc_o, smem_row_scale);

    // Epilogue
    if (NoSplit) {
        flash::store<Kernel_traits, false, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    } else {
        flash::store<Kernel_traits, true, true, true>(params, batch_id, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }
}

template<typename Kernel_traits, bool IsFP8 = false>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_sparse_decode_fwd_kernel(__grid_constant__ const Flash_fwd_params params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    static_assert(Kernel_traits::CrossCut);
    static_assert(!Kernel_traits::USE_MMA_M8);

    //extern __shared__ char shared_memory[];
    //auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y; // sched_begin_block_idx*kBlockN
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;
    if (begin_idx >= params.b) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

    #pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        if constexpr (kHeadDim == 512)  {
            const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
            int seqlen_k = params.topk_len_ptr ? params.topk_len_ptr[batch_id] : params.topk;
            int seqlen_kpad = std::max(seqlen_k, 1);
            // int seqlen_kpad = seqlen_k;

            int extra_seqlen_k = 0;
            if (params.extra_topk >= 0) {
                seqlen_kpad = cute::round_up(seqlen_kpad, kBlockN);
                extra_seqlen_k = params.extra_topk_len_ptr ? params.extra_topk_len_ptr[batch_id] : params.extra_topk;
            }
            const int total_k = seqlen_kpad + extra_seqlen_k;
            int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
            int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(total_k, kBlockN);
            bool NoSplit = n_block_min == 0 && n_block_max == cute::ceil_div(total_k, kBlockN);
            int ori_block_max = cute::ceil_div(seqlen_kpad, kBlockN);

            if constexpr (IsFP8) {
                compute_attn_fp8_sparse_splitkv<Kernel_traits>(
                    params, batch_id, bidh, m_block, n_split_idx,
                    n_block_min, n_block_max, NoSplit, seqlen_k,
                    ori_block_max, extra_seqlen_k);
            } else {
                compute_attn_bf16_sparse_splitkv<Kernel_traits>(
                    params, batch_id, bidh, m_block, n_split_idx,
                    n_block_min, n_block_max, NoSplit, seqlen_k,
                    ori_block_max, extra_seqlen_k);
            }

            __syncthreads();  // Barrier between two tiles.
        } else {
            const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
            const int seqlen_k = params.topk;
            int ori_block_max = cute::ceil_div(seqlen_k, kBlockN);
            int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
            int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : ori_block_max;
            bool NoSplit = n_block_min == 0 && n_block_max == ori_block_max;

            if constexpr (IsFP8) {
                compute_attn_fp8_sparse_splitkv<Kernel_traits>(
                    params, batch_id, bidh, m_block, n_split_idx,
                    n_block_min, n_block_max, NoSplit, seqlen_k,
                    ori_block_max, 0);

            } else {
                compute_attn_bf16_sparse_splitkv<Kernel_traits>(
                    params, batch_id, bidh, m_block, n_split_idx,
                    n_block_min, n_block_max, NoSplit, seqlen_k,
                    ori_block_max, 0);
            }

            __syncthreads();  // Barrier between two tiles.
        }
    }
}

} // namespace flash

////
template<typename Kernel_traits, bool IsFP8>
void run_flash_sparse_decode_fwd(Flash_fwd_params &params, hggcStream_t stream) {
    // TODO.
    constexpr size_t smem_size = Kernel_traits::kSmemSizeAccum + Kernel_traits::kBlockN * 2 * sizeof(int);
    const int num_m_block = (params.seqlen_q / params.ngroups) * cute::ceil_div(params.ngroups, Kernel_traits::kBlockM);

        auto kernel = &flash::flash_sparse_decode_fwd_kernel<Kernel_traits, IsFP8>;
        flash::printf_show_log<Kernel_traits>(reinterpret_cast<const void*>(kernel), params, smem_size, false, true, IsFP8);
        //CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        if (smem_size >= 48 * 1024) {
            hggcFuncSetAttribute(
                kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
#ifdef __HGGCCC__
       //TODO
        const void *flash_func = reinterpret_cast<const void*>(kernel);
        HGfunction func = static_cast<HGfunction>(NULL);
        hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        HGlaunchAttributeAD LaunchAttr = {HGAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        HGlaunchConfigAD LaunchCfg = {num_m_block, params.h,
        params.num_sm_parts, Kernel_traits::kNThreads, 1, 1, smem_size, stream, &LaunchAttr, 0};
        CUDA_DRIVER_CHECK(hgLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
#endif
    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<Kernel_traits, kMaxSplits>;
#ifdef __HGGCCC__
        const void *flash_func = reinterpret_cast<const void*>(combine_kernel);
        HGfunction func = static_cast<HGfunction>(NULL);
        hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&func), flash_func);

        void* kernel_args[] = {&params};
        HGlaunchAttributeAD LaunchAttr = {HGAD_LAUNCH_ATTRIBUTE_IGNORE}; //HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE
        HGlaunchConfigAD LaunchCfg = {grid_combine.x, grid_combine.y, grid_combine.z, 128, 1, 1, 0, stream, &LaunchAttr, 0};
        // LaunchAttr.value.schedPreference.blocksPerMultiprocessor = 1;//schedule.bits.tb_per_cu;
        // LaunchAttr.value.schedPreference.gridStepX = 2;
        // LaunchAttr.value.schedPreference.gridStepY = 2;
        // LaunchAttr.value.schedPreference.flags = 2;
        CUDA_DRIVER_CHECK(hgLaunchKernelExAD(&LaunchCfg, func, kernel_args, nullptr));
#else
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
#endif
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, bool IsFP8, int Headdim, int Headdim_V>
void run_sparse_decode_fwd_dispatch(Flash_fwd_params& params, hggcStream_t stream) {
    constexpr int TOPK_BLOCK_SIZE = 64;    // kBlockN
    // constexpr int NUM_THREADS = 128*4; // 16*32
    // static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)
    FLASH_ASSERT(params.h == 1);
    FLASH_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);

    if constexpr (IsFP8) {
        constexpr bool USE_MMA_M8 = 0;
        constexpr bool KeepQ = true;
        constexpr static int kBlockN = 64;
        SEQLENG_SWITCH_ALIGN(params.seqlen_q, [&] {
            IS_PAGE_POWER2(params.page_block_size, params.extra_page_block_size, [&] {
                constexpr int AtomLayoutQ = kBlockM / 16;
                constexpr int kNwarps0 = AtomLayoutQ * (kBlockN / 16);
                constexpr int kNwarps = 16;
                constexpr int AtomLayoutP = kBlockM == 64 ? 2 : 1; // to save regs(sum/max in softmax)
                run_flash_sparse_decode_fwd<Flash_fwd_kernel_traits<
                    Headdim, kBlockM, kBlockN, kNwarps, KeepQ/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
                    T, Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP,
                    kBlockN/*kBlockNPagedPerAiuLoad*/, 2/*kStages*/, kNwarps0, kPagePow2
                    >, IsFP8>(params, stream);
            });
        });
    } else {
        constexpr bool USE_MMA_M8 = 0;
        constexpr bool KeepQ = true;
        constexpr static int kBlockN = 64;
        SEQLENG_SWITCH_ALIGN(params.seqlen_q, [&] {
            constexpr int AtomLayoutQ = kBlockM / 16;
            constexpr int kNwarps = AtomLayoutQ * (kBlockN / 16);
            constexpr int AtomLayoutP = kBlockM == 64 ? 2 : 1;
            run_flash_sparse_decode_fwd<Flash_fwd_kernel_traits<
                Headdim, kBlockM, kBlockN, kNwarps, KeepQ/*Is_Q_in_regs*/, USE_MMA_M8/*Share_Q_K_smem*/,
                T, Headdim_V, 1/*CrossCut*/, USE_MMA_M8/*USE_MMA_M8*/, AtomLayoutQ, AtomLayoutP
                >, IsFP8>(params, stream);
        });
    }
}
