# MLA Triton kernel is from: https://github.com/monellz/vllm/commit/feebaa7c063be6bfb590a876741aeef1c5f58cf8#diff-7b2e1c9032522f7266051b9887246a65753871dfb3625a258fee40109fe6e87a
import math
import random
import re

import torch
try:
    import flashinfer
except ImportError:
    print("Import flashinfer failed, please install if need!")
try:
    import triton
    import triton.language as tl
except ImportError:
    print("Import triton failed, please install if need!")
import argparse

# pip install flashinfer-python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache, flash_mla_sparse_fwd
import json

device_name = torch.cuda.get_device_name()
USE_PPU = (device_name.lower().find("ppu") != -1) or (device_name.lower().find("zw") != -1)
if not any(k in device_name.lower() for k in ['ppu','zw','nvidia']):
    print("Warning: Unrecognized device name: "+ device_name)

FLASHINFER_BACKEND = "fa2" if USE_PPU else "fa3"


def quantize_k_cache(
    input_k_cache: torch.Tensor,    # (num_blocks, block_size, h_k, d)
    dv: int,
    tile_size: int = 128,
) -> torch.Tensor:
    """
    Quantize the k-cache
    Return a tensor with shape (num_blocks, block_size, h_k, dv + 4(dv/tile_size) + t(d-dv)) of dtype uint8_t, where t = input_k_cache.element_size()
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py or README.md
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    result = torch.empty((num_blocks, block_size, dv + num_tiles * 4 + input_elem_size * (d - dv)), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
    result_k_nope_part = result[..., :dv]
    result_k_scale_factor = result[..., dv:dv + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., dv + num_tiles * 4:].view(input_k_cache.dtype)
    result_k_rope_part[:] = input_k_cache[..., dv:]

    for tile_idx in range(0, num_tiles):
        cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx * tile_size:(tile_idx + 1) * tile_size]).max(dim=-1).values / 448.0  # [num_blocks, block_size]
        result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

        cur_scale_factors_inv.unsqueeze_(-1)    # [num_blocks, block_size, 1]
        cur_quantized_nope = (input_k_cache[..., tile_idx * tile_size:(tile_idx + 1) * tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
        result_k_nope_part[..., tile_idx * tile_size:(tile_idx + 1) * tile_size] = cur_quantized_nope

    result = result.view(num_blocks, block_size, 1, -1)
    return result


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,    # (num_blocks, block_size, 1, bytes_per_token)
    dv: int = 512,
    tile_size: int = 128,
    d: int = 576
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty((num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device)

    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

    input_nope = quant_k_cache[..., :dv]
    input_scale = quant_k_cache[..., dv:dv + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[..., dv + num_tiles * 4:].view(torch.bfloat16)
    result[..., dv:] = input_rope

    for tile_idx in range(0, num_tiles):
        cur_nope = input_nope[..., tile_idx * tile_size:(tile_idx + 1) * tile_size].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
        result[..., tile_idx * tile_size:(tile_idx + 1) * tile_size] = cur_nope * cur_scales

    result = result.view(num_blocks, block_size, 1, d)
    return result


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q, h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_torch, lse_torch = ref_mla()
    t = triton.testing.do_bench(ref_mla)
    return out_torch, lse_torch, t

@torch.inference_mode()
def run_flash_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    # blocked_k = blocked_k.to('cuda')
    blocked_v = blocked_k[..., :dv]

    # q.to('cuda')
    cache_seqlens = cache_seqlens.to('cuda')

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def flash_mla():
        return flash_mla_with_kvcache(
            q.to('cuda'), blocked_k.to('cuda'), block_table.to('cuda'), cache_seqlens, dv,
            tile_scheduler_metadata, num_splits, causal=causal,
        )

    out_flash, lse_flash = flash_mla()
    # t = triton.testing.do_bench(flash_mla)
    return out_flash, lse_flash


@torch.inference_mode()
def run_flash_infer(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):

    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")

    # blocked_k.to('cuda')
    # cache_seqlens = cache_seqlens.to('cuda')

    assert d > dv, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv].contiguous(), q[..., dv:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[..., dv:].contiguous()

    kv_indptr = [0]
    kv_indices = []
    for i in range(b):
        seq_len = cache_seqlens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_table[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
    for seq_len in cache_seqlens[1:]:
        kv_indptr.append((seq_len + block_size - 1) // block_size + kv_indptr[-1])

    q_indptr = torch.arange(0, b + 1, device='cpu').int() * s_q
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)

    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8),
        backend=FLASHINFER_BACKEND
    )

    mla_wrapper.plan(
        q_indptr.to('cuda'),
        kv_indptr.to('cuda'),
        kv_indices.to('cuda'),
        cache_seqlens.to('cuda'),
        h_q,
        dv,
        d-dv,
        block_size,
        causal,
        1 / math.sqrt(d),
        q.dtype,
        blocked_k.dtype,
    )

    def flash_infer():
        output, lse = mla_wrapper.run(q_nope.view(-1, h_q, dv).to('cuda'), q_pe.view(-1, h_q, d-dv).to('cuda'), blocked_k_nope.to('cuda'), blocked_k_pe.to('cuda'), return_lse=True)
        return output.view(b, -1, h_q, dv), lse.view(b, h_q, 1)

    out_flash, lse_flash = flash_infer()
    # # t = triton.testing.do_bench(flash_infer)
    return out_flash, lse_flash

@triton.jit
def _mla_attn_kernel(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    sm_scale,
    stride_q_nope_bs,
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
):
    cur_batch = tl.program_id(1)
    cur_head_id = tl.program_id(0)
    split_kv_id = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_seq_len + cur_batch)

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_q_nope = cur_batch * stride_q_nope_bs + cur_head[:, None] * stride_q_nope_h + offs_d_ckv[None, :]
    q_nope = tl.load(Q_nope + offs_q_nope)

    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)
    offs_q_pe = cur_batch * stride_q_pe_bs + cur_head[:, None] * stride_q_pe_h + offs_d_kpe[None, :]
    q_pe = tl.load(Q_pe + offs_q_pe)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=tl.float32)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    offs_d_ckv_i64 = offs_d_ckv.cast(tl.int64)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_page_number = tl.load(
            Req_to_tokens + stride_req_to_tokens_bs * cur_batch + offs_n // PAGE_SIZE,
            mask=offs_n < split_kv_end,
            other=0,
        )
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        kv_loc_i64 = kv_loc.cast(tl.int64)
        stride_kv_c_bs_i64 = stride_kv_c_bs.cast(tl.int64)
        offs_k_c = kv_loc_i64[None, :] * stride_kv_c_bs_i64 + offs_d_ckv_i64[:, None]
        k_c = tl.load(Kv_c_cache + offs_k_c, mask=offs_n[None, :] < split_kv_end, other=0.0)

        qk = tl.dot(q_nope, k_c.to(q_nope.dtype))

        offs_k_pe = kv_loc[None, :] * stride_k_pe_bs + offs_d_kpe[:, None]
        k_pe = tl.load(K_pe_cache + offs_k_pe, mask=offs_n[None, :] < split_kv_end, other=0.0)

        qk += tl.dot(q_pe, k_pe.to(q_pe.dtype))
        qk *= sm_scale

        qk = tl.where(offs_n[None, :] < split_kv_end, qk, float("-inf"))

        v_c = tl.trans(k_c)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v_c.dtype), v_c)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
    offs_o = cur_batch * stride_o_b + cur_head[:, None] * stride_o_h + split_kv_id * stride_o_s + offs_d_ckv[None, :]
    tl.store(O + offs_o, acc / e_sum[:, None])
    offs_o_1 = cur_batch * stride_o_b + cur_head * stride_o_h + split_kv_id * stride_o_s + HEAD_DIM_CKV
    tl.store(O + offs_o_1, e_max + tl.log(e_sum))


def _mla_attn(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    attn_logits,
    req_to_tokens,
    b_seq_len,
    num_kv_splits,
    sm_scale,
    page_size,
):
    batch_size, head_num = q_nope.shape[0], q_nope.shape[1]
    head_dim_ckv = q_nope.shape[-1]
    head_dim_kpe = q_pe.shape[-1]

    BLOCK_H = 16
    BLOCK_N = 64
    grid = (
        triton.cdiv(head_num, BLOCK_H),
        batch_size,
        num_kv_splits,
    )
    _mla_attn_kernel[grid](
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        req_to_tokens,
        b_seq_len,
        attn_logits,
        sm_scale,
        # stride
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kv_c_cache.stride(-2),
        k_pe_cache.stride(-2),
        req_to_tokens.stride(0),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
    )

@triton.jit
def _mla_softmax_reducev_kernel(
    Logits,
    B_seq_len,
    O,
    stride_l_b,
    stride_l_h,
    stride_l_s,
    stride_o_b,
    stride_o_h,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_batch_seq_len = tl.load(B_seq_len + cur_batch)

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    offs_l = cur_batch * stride_l_b + cur_head * stride_l_h + offs_d_ckv
    offs_l_1 = cur_batch * stride_l_b + cur_head * stride_l_h + HEAD_DIM_CKV

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            logits = tl.load(Logits + offs_l + split_kv_id * stride_l_s)
            logits_1 = tl.load(Logits + offs_l_1 + split_kv_id * stride_l_s)

            n_e_max = tl.maximum(logits_1, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(logits_1 - n_e_max)
            acc += exp_logic * logits

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d_ckv,
        acc / e_sum,
    )


def _mla_softmax_reducev(
    logits,
    o,
    b_seq_len,
    num_kv_splits,
):
    batch_size, head_num, head_dim_ckv = o.shape[0], o.shape[1], o.shape[2]
    grid = (batch_size, head_num)
    _mla_softmax_reducev_kernel[grid](
        logits,
        b_seq_len,
        o,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_DIM_CKV=head_dim_ckv,
        num_warps=4,
        num_stages=2,
    )

def mla_decode_triton(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    o,
    req_to_tokens,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
):
    assert num_kv_splits == attn_logits.shape[2]
    _mla_attn(
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        attn_logits,
        req_to_tokens,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
    )
    _mla_softmax_reducev(
        attn_logits,
        o,
        b_seq_len,
        num_kv_splits,
    )


@torch.inference_mode()
def run_flash_mla_triton(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):

    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")

    blocked_v = blocked_k[..., :dv]

    assert d > dv, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv].contiguous(), q[..., dv:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[..., dv:].contiguous()

    # blocked_k = blocked_k.to('cuda')
    # blocked_v = blocked_v.to('cuda')
    # cache_seqlens = cache_seqlens.to('cuda')

    # blocked_k_nope.to('cuda')
    # blocked_k_pe.to('cuda')
    # q_nope.to('cuda')
    # q_pe.to('cuda')

    def flash_mla_triton():
        num_kv_splits = 32
        o = torch.empty([b * s_q, h_q, dv])
        attn_logits = torch.empty([b * s_q, h_q, num_kv_splits, dv + 1])
        mla_decode_triton(q_nope.view(-1, h_q, dv).to('cuda'),
                        q_pe.view(-1, h_q, d-dv).to('cuda'),
                        blocked_k_nope.view(-1, dv).to('cuda'),
                        blocked_k_pe.view(-1, d-dv).to('cuda'), o.to('cuda'),
                        block_table.to('cuda'), cache_seqlens.to('cuda'), attn_logits, num_kv_splits, 1 / math.sqrt(d), block_size)
        return o.view([b, s_q, h_q, dv])

    out_flash = flash_mla_triton()
    # # t = triton.testing.do_bench(flash_mla_triton)
    return out_flash, None


FUNC_TABLE = {
    "torch": run_torch_mla,
    "flash_mla": run_flash_mla,
    "flash_infer": run_flash_infer,
    "flash_mla_triton": run_flash_mla_triton,
}

def compare_a(target, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype, _block_size):
    print(f"{target}: {b=}, {s_q=}, mean_seqlens={cache_seqlens.float().mean()}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {dtype=}")

    torch.set_default_dtype(dtype)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    assert target in FUNC_TABLE
    target_func = FUNC_TABLE[target]

    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    # q = torch.randn(b, s_q, h_q, d, device='cpu')
    q = torch.randn(b, s_q, h_q, d, device='cpu')
    # q = torch.randn(b, s_q, h_q, d)
    block_size = _block_size
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32,
        device='cpu').view(b, max_seqlen_pad // block_size)
    # block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, device='cpu')
    # blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)

    out_b, lse_b = target_func(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype)

    # FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    # bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    # print(f"perf {target}: {perf_b:.3f} ms, {FLOPS / 10 ** 9 / perf_b:.0f} TFLOPS, {bytes / 10 ** 6 / perf_b:.0f} GB/s")
    # return bytes / 10 ** 6 / perf_b
    return 1

def run_dsa_prefill(s_q, s_kv, h_q, h_kv, d, dv, topk, dtype):
    print(f"dsa_prefill: {s_q=}, {s_kv=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {topk=}, {dtype=}")
    torch.set_default_dtype(torch.bfloat16)

    device = torch.device("cpu")
    torch.set_default_device(torch.device("cpu"))
    torch.manual_seed(0)
    random.seed(0)

    q = torch.randn((s_q, h_q, d), device='cpu')
    kv = torch.randn((s_kv, h_kv, d), device='cpu')
    indices = torch.full((s_q, h_kv, topk), s_kv, dtype=torch.int32, device='cpu')
    for s in range(s_q):
        for h in range(h_kv):
            # NOTE We use the following method to generate indices so that most indices lies within [s_kv-20000, s_kv), which is more realistic for sparse attention
            near_mask = torch.randint(0, 32, (min(topk, s_kv),)) < 31
            cur_indices = torch.randperm(s_kv)[:topk]
            cur_indices[near_mask] = torch.randint(max(0, s_kv - 20000), s_kv - 1, (near_mask.sum().item(),))
            if len(cur_indices) < topk:
                cur_indices = torch.cat([cur_indices, torch.full((topk - len(cur_indices),), 2147480000)])
            cur_indices = cur_indices[torch.randperm(topk)]
            indices[s, h] = cur_indices


    sm_scale = 1 / math.sqrt(d)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.cuda.synchronize()
    out, max_logits, lse = flash_mla_sparse_fwd(q.to('cuda'), kv.to('cuda'), indices.to('cuda'), sm_scale=sm_scale)
    torch.cuda.synchronize()
    return 1

def run_dsa_decode(b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, topk, is_fp8, dtype, block_size):
    print(f"dsa_decode: {b=}, {s_q=}, topk={topk}, {h_q=}, {h_kv=}, {d=}, {dv=}, is_fp8={is_fp8}, {dtype=}")

    torch.set_default_dtype(dtype)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    # total_seqlens = cache_seqlens.sum().item()
    # mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    # q = torch.randn(b, s_q, h_q, d, device='cpu')
    q = torch.randn(b, s_q, h_q, d, device='cpu')
    # q = torch.randn(b, s_q, h_q, d)
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32,
        device='cpu').view(b, max_seqlen_pad // block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel(), device='cpu')].view(b, -1)

    # block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, device='cpu')
    # blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)

    # abs_indices = torch.empty(b, s_q, topk, dtype=torch.int32, device="cpu")
    indices_in_kvcache = torch.empty(b, s_q, topk, dtype=torch.int32, device="cpu")
    for i in range(b):
        # Generate indices
        for j in range(s_q):
            cur_abs_indices = torch.randperm(int(cache_seqlens[i].item()), device="cpu")[:topk]
            cur_blocked_indices = block_table[i, cur_abs_indices // block_size] * block_size + (cur_abs_indices % block_size)
            if len(cur_abs_indices) < topk:
                pad_len = topk - len(cur_abs_indices)
                cur_abs_indices = torch.cat([cur_abs_indices, torch.full((pad_len,), -1, device='cpu')])
                cur_blocked_indices = torch.cat([cur_blocked_indices, torch.full((pad_len,), -1, device='cpu')])

            # Mask KV
            perm = torch.randperm(topk, device='cpu')
            # cur_abs_indices = cur_abs_indices[perm]
            cur_blocked_indices = cur_blocked_indices[perm]

            # abs_indices[i, j, :] = cur_abs_indices
            indices_in_kvcache[i, j, :] = cur_blocked_indices

    if is_fp8:
        blocked_k_quantized = quantize_k_cache(blocked_k, dv, 128)
        blocked_k_dequantized = dequantize_k_cache(blocked_k_quantized)
        blocked_k = blocked_k_dequantized

    torch.cuda.synchronize()
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens.to('cuda'),
        s_q * h_q // h_kv,
        h_kv,
        h_q,
        is_fp8,
        topk
    )

    torch.cuda.synchronize()
    def flash_mla_decode_sparse():
        return flash_mla_with_kvcache(
            q.to('cuda'),
            blocked_k.cuda() if not is_fp8 else blocked_k_quantized.cuda(),
            block_table.to('cuda'),
            cache_seqlens.to('cuda'),
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=causal,
            is_fp8_kvcache=is_fp8,
            indices=indices_in_kvcache.to('cuda'),
        )
    out_flash, lse_flash = flash_mla_decode_sparse()
    return 1

available_targets = [
    "flash_mla",
    "flash_infer",
    "flash_mla_triton",
]

# shape_configs = [
#     {"b": batch, "s_q": 1, "cache_seqlens": torch.tensor([seqlen + 2 * i for i in range(batch)], dtype=torch.int32, device="cuda"), "h_q": head, "h_kv": 1, "d": 512+64, "dv": 512, "causal": True, "dtype": torch.bfloat16}
#     for batch in [128] for seqlen in [1024, 2048, 4096, 8192, 8192*2, 8192*4] for head in [128]
# ]

def convert_value(value):
    try:
        return int(value)
    except ValueError:
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        # 其他情况保持字符串
        return value

def get_params(input_str):
    input_str = re.sub(r'^.*?format=', '', input_str)
    input_str = re.sub(r'^.*?flash_mla,MLA:', '', input_str).rstrip(".")
    pattern = r'(\w+):(\[.*?\]|[^,]+?)(?=,\w+:|$|,)'
    matches = re.findall(pattern, input_str)
    config_dict = {key: value for key, value in matches}

    config_dict = {k: convert_value(v) for k, v in config_dict.items()}

    if "sparse" not in config_dict:
        config_dict["sparse"] = None
    if config_dict["sparse"] == "prefill":
        return config_dict

    config_dict["seq_q"] = int(config_dict["seqlen_q"])
    torch.manual_seed(0)
    random.seed(0)
    # rnd = max(random.normalvariate(config_dict["seqlen_k"], config_dict["seqlen_k"] / 2), config_dict["seq_q"])

    if type(config_dict["seqlen_k"]) is int:
        # varlen
        config_dict["cache_seqlens"] = torch.tensor([max(random.normalvariate(config_dict["seqlen_k"], config_dict["seqlen_k"] / 2), config_dict["seq_q"]) + i for i in range(config_dict["batch_size"])], dtype=torch.int32, device="cpu")
        # fixlen
        # config_dict["cache_seqlens"] = torch.full((config_dict["batch_size"],), config_dict["seqlen_k"], dtype=torch.int32, device="cpu")
    else:
        config_dict["cache_seqlens"] = torch.tensor(json.loads(config_dict["seqlen_k"]), dtype=torch.int32, device="cpu")

    config_dict["dtype"] = torch.bfloat16 if config_dict["dtype"] == "bf16" else torch.half

    return config_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, required=True, default="--format=flash_mla:flash_mla,batch_size:1,seqlen:200,num_heads:128,num_heads_kv:1,head_dim:576,head_dim_v:512,causal:True,dtype:bf16",
                        help="use this option to pass fmha_params string.")
    parser.add_argument('--backend', default="flash_mla", type=str, required=False, help='specify backend, flash_mla, flash_infer, flash_mla_triton')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    config = get_params(args.format)
    if config["sparse"] == "prefill":
        assert args.backend=="flash_mla", "DSA perf only support flash_mla"
        perf = run_dsa_prefill(config["seqlen_q"], config["seqlen_k"], config["num_heads"], config["num_heads_kv"], config["head_dim"], config["head_dim_v"], config["topk"], config["dtype"])
    elif config["sparse"] == "decode":
        assert args.backend=="flash_mla", "DSA perf only support flash_mla"
        if "block_size" not in config.keys():
            config["block_size"] = 64

        # FIXME: bf16 is not supported for CUDA. fp8 is not optimized for PPU.
        if not USE_PPU:
            config["is_fp8"] = 1
        perf = run_dsa_decode(config["batch_size"], config["seq_q"], config["cache_seqlens"],
                              config["num_heads"], config["num_heads_kv"], config["head_dim"], config["head_dim_v"],
                              config["causal"], config["topk"], config["is_fp8"], config["dtype"], config["block_size"])
    else:
        config["mla"] = args.backend
        if "block_size" not in config.keys():
            config["block_size"] = 64
        # exit(0)
        perf = compare_a(config["mla"], config["batch_size"], config["seq_q"], config["cache_seqlens"], config["num_heads"], config["num_heads_kv"], config["head_dim"], config["head_dim_v"], config["causal"], config["dtype"], config["block_size"])