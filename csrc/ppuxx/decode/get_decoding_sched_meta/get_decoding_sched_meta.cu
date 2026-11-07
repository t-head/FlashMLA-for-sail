/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <limits>
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

__global__ void __launch_bounds__(32, 1, 1)
get_mla_uniform_sparse_metadata_kernel(__grid_constant__ const Mla_metadata_params params,
                                       int total_k, int blocks_per_row, int payload) {
    const int lane = threadIdx.x;
    const int overhead = params.fixed_overhead_num_blocks;
    const int row_cost = blocks_per_row + overhead;
    const int rows_per_part = payload / row_cost;
    const int full_row_payload = rows_per_part * row_cost;
    int row = 0;
    int row_block = 0;
    int row_splits = 0;
    int split_prefix = 0;
    if (lane == 0) params.num_splits_ptr[0] = 0;

    for (int part = 0; part < params.num_sm_parts; ++part) {
        const int begin_row = row;
        const int begin_block = row_block;
        const int begin_split = row_splits;
        int remaining = payload;

        // Finish a carried row before advancing through a run of whole rows.
        const int carried_cost = blocks_per_row - row_block + overhead;
        if (row_block != 0 && remaining >= carried_cost) {
            remaining -= carried_cost;
            split_prefix += row_splits + 1;
            ++row;
            if (lane == 0) params.num_splits_ptr[row] = split_prefix;
            row_block = 0;
            row_splits = 0;
        }
        if (row_block == 0 && row < params.batch_size) {
            // Finishing a carried row costs less than one full row.
            const int affordable_rows = rows_per_part - (remaining < full_row_payload);
            const int rows_left = params.batch_size - row;
            const int whole_rows = affordable_rows < rows_left ? affordable_rows : rows_left;
            for (int offset = lane; offset < whole_rows; offset += 32) {
                params.num_splits_ptr[row + offset + 1] = split_prefix + offset + 1;
            }
            row += whole_rows;
            split_prefix += whole_rows;
            remaining -= whole_rows * row_cost;
        }
        if (row < params.batch_size && remaining > overhead) {
            row_block += remaining - overhead;
            ++row_splits;
        }

        if (lane == 0) {
            int* metadata = params.tile_scheduler_metadata_ptr + part * TileSchedulerMetaDataSize;
            const int4 tile{begin_row, begin_block * params.block_size_n,
                            row_block != 0 ? row : row - 1,
                            row_block != 0 ? row_block * params.block_size_n : total_k};
            *reinterpret_cast<int4*>(metadata) = tile;
            metadata[4] = begin_split;
        }
    }
    FLASH_DEVICE_ASSERT(row == params.batch_size && row_block == 0 && row_splits == 0);
}

void get_mla_uniform_sparse_metadata_func(Mla_metadata_params &params, hggcStream_t stream) {
    if (params.topk <= 0 || params.extra_topk < 0 || params.topk_length || params.extra_topk_length
        || params.batch_size <= 0 || params.num_sm_parts <= 0 || params.block_size_n <= 0
        || params.fixed_overhead_num_blocks < 0) {
        get_mla_metadata_func(params, stream);
        return;
    }

    constexpr int64_t max_int = std::numeric_limits<int>::max();
    const int64_t block_size = params.block_size_n;
    const int64_t total_k = params.extra_topk > 0
        ? (params.topk + block_size - 1) / block_size * block_size + params.extra_topk
        : params.topk;
    const int64_t blocks_per_row = (total_k + block_size - 1) / block_size;
    const int64_t row_cost = blocks_per_row + params.fixed_overhead_num_blocks;
    if (total_k > max_int || row_cost > max_int || params.batch_size > max_int - 32
        || params.num_sm_parts > max_int / TileSchedulerMetaDataSize
        || int64_t(params.batch_size) + params.num_sm_parts > max_int) {
        get_mla_metadata_func(params, stream);
        return;
    }

    const int64_t total_blocks = params.batch_size * row_cost;
    const int64_t payload = (total_blocks + params.num_sm_parts - 1) / params.num_sm_parts
        + params.fixed_overhead_num_blocks;
    if (total_blocks > max_int || payload > max_int) {
        get_mla_metadata_func(params, stream);
        return;
    }

    get_mla_uniform_sparse_metadata_kernel<<<1, 32, 0, stream>>>(
        params, static_cast<int>(total_k), static_cast<int>(blocks_per_row), static_cast<int>(payload));
    CHECK_CUDA_KERNEL_LAUNCH();
}
