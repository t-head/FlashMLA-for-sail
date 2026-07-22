/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include "params.h"
#include "kerutils/common/common.h"
#include "utils.h"

__global__ void __launch_bounds__(32, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const Mla_metadata_params params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.batch_size;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    extern __shared__ int shared_mem[];
    int* num_blocks_shared = shared_mem; // [batch_size]
    int* num_splits_shared = shared_mem + batch_size; // [batch_size+1]
    int* seqlens_k_shared = shared_mem + batch_size*2+1; // [batch_size]

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int cur_s_k;
        if (params.topk == -1) {
            // Dense model, cur_s_k = actual s_k
            cur_s_k = __ldg(seqlens_k_ptr + i);
        } else {
            // Sparse model, cur_s_k = topk (+ extra topk)
            cur_s_k = params.topk_length ? __ldg(params.topk_length + i) : params.topk;
            // FIXME.
            if (cur_s_k == 0) cur_s_k = 1;  // Ensure the main loop will never be empty
            if (params.extra_topk) {
                cur_s_k = cutlass::round_up(cur_s_k, block_size_n);
                cur_s_k += params.extra_topk_length ? __ldg(params.extra_topk_length + i) : params.extra_topk;
            }
        }
        // NOTE if seqlens_k is 0, the sequence will have 1 block. We will correct this later in this kernel.
        seqlens_k_shared[i] = cur_s_k;
        int num_blocks = cutlass::ceil_div(max(1, cur_s_k), block_size_n);
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
    }
    for (int offset = 16; offset >= 1; offset /= 2) {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;

        int now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
            tile_scheduler_metadata0[0] = now_idx;
            tile_scheduler_metadata0[1] = now_block * block_size_n;
            tile_scheduler_metadata1 = now_n_split_idx;
            int remain_payload = payload;
            while (now_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_idx;
                    now_block = 0;
                    now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx;
                        remain_payload = 0;
                    }
                    break;
                }
            }
            tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
            tile_scheduler_metadata0[3] = now_block > 0 ? now_block * block_size_n : seqlens_k_shared[now_idx - 1];
            *reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize) = *reinterpret_cast<int4 *>(tile_scheduler_metadata0);
            tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
        }
        FLASH_DEVICE_ASSERT(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32) {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}

void get_mla_metadata_func(Mla_metadata_params &params, hggcStream_t stream) {
    int smem_size = sizeof(int) * (params.batch_size*3+1);
    CHECK_CUDA(hggcFuncSetAttribute(get_mla_metadata_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    get_mla_metadata_kernel<<<1, 32, smem_size, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
