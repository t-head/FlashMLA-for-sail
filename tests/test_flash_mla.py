import argparse
import math
import random

import torch
import triton

from flash_mla import flash_mla_with_kvcache, get_mla_metadata


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


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    assert cos_diff < 1e-5


@torch.inference_mode()
def test_flash_mla(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen, paged_block_size):

    case_name = f"{varlen=},b{b},sq{s_q},sk{mean_sk},hq{h_q},hkv{h_kv},d{d},dv{dv},causal{causal},page{paged_block_size}"
    print(case_name)

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    # print(cache_seqlens)

    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")
    if args.ref:
        dev = 'cuda'
    else:
        dev = 'cpu'
    q = torch.randn(b, s_q, h_q, d, device=dev)
    # q = torch.ones(b, s_q, h_q, d, device="cpu")

    block_size = paged_block_size
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    # blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, device=dev)
    blocked_base = torch.randn(block_table.numel(), block_size+16, h_kv, d, device=dev)
    blocked_k = blocked_base[:,:block_size, ...]
    # blocked_k = torch.ones(block_table.numel(), block_size, h_kv, d, device="cpu")

    for i in range(b):
        cur_len = cache_seqlens[i].item()
        cur_num_blocks = triton.cdiv(cur_len, block_size)
        blocked_k[block_table[i][cur_num_blocks:]] = float("nan")
        if cur_len % block_size != 0:
            blocked_k[block_table[i][cur_num_blocks - 1]][cur_len % block_size:] = float("nan")
        # blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
        #     float("nan")
        # )
        block_table[i][cur_num_blocks:] = 2147480000
    blocked_v = blocked_k[..., :dv]

    torch.cuda.nvtx.range_push(case_name)
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv
    )

    def flash_mla():
        return flash_mla_with_kvcache(
            q.to('cuda'),
            blocked_k.to('cuda'),
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=causal,
        )

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1).to('cuda'),
                # blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1).to('cuda'),
                # blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1).to('cuda'),
                blocked_k.reshape(-1, h_kv, d)[begin:end].transpose(0, 1).to('cuda'),
                blocked_v.reshape(-1, h_kv, dv)[begin:end].transpose(0, 1).to('cuda'),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_flash_, lse_flash = flash_mla()
    torch.cuda.nvtx.range_pop()
    if args.ref:
        out_flash = out_flash_[..., :dv]
        out_torch, lse_torch = ref_mla()
        diff = out_flash - out_torch

        # print(f'diff.max = {diff.max()}, diff.min = {diff.min()}')
        cal_diff(out_flash, out_torch, "out")
        cal_diff(lse_flash, lse_torch, "lse")

    # t = triton.testing.do_bench(flash_mla)
    # FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    # bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (
    #     torch.finfo(q.dtype).bits // 8
    # )
    # print(
    #     f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s"
    # )


def main(torch_dtype, loops = 1):
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    h_kv = 1
    d, dv = 576, 512
    for i in range(loops):
        print(f'LOOP ROUND {i} START:')
        if 1:
            if 0:
                causal = True
                b = 32
                s = 8192
                s_q = 1
                h_q = 128
                varlen = False
                paged_block_size = 64
                test_flash_mla(b, s_q, s, h_q, h_kv, d, dv, causal, varlen, paged_block_size)
            else:
                for b in [1, 8, 32, 128]:
                    for s in [256, 512, 1024, 4096, 8192]:
                        for h_q in [16, 32, 64, 128]:  # TP = 8, 4, 2, 1
                            for s_q in [1, 2]:  # MTP = 1, 2
                                for varlen in [False, True]:
                                    for paged_block_size in [16, 64, 256]:
                                        for causal in [True, False]:
                                            test_flash_mla(b, s_q, s, h_q, h_kv, d, dv, causal, varlen, paged_block_size)
        else:
            for b in [1, 8, 128]:
                for s in [4096, 8192]:
                    for h_q in [16, 32]:  # TP = 8, 4, 2, 1
                        for s_q in [1]:  # MTP = 1, 2
                            for varlen in [True, False]:
                                for paged_block_size in [64, 16, 256]:
                                    for causal in [True, False]:
                                        test_flash_mla(b, s_q, s, h_q, h_kv, d, dv, causal, varlen, paged_block_size)
        print(f'LOOP ROUND {i} FINISH')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type to use for testing (bf16 or fp16)",
    )

    parser.add_argument(
        "--ref",
        type=int,
        default=1,
        help="Data type to use for testing (bf16 or fp16)",
    )

    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="loop round num for precision test",
    )

    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    if args.dtype == "fp16":
        torch_dtype = torch.float16

    main(torch_dtype, int(args.loops))
