import pytest
import torch

import kernelkit as kk
import flash_mla

from test_flash_mla_dense_decoding import TestParam, generate_test_data, reference_torch


CASE = TestParam(
    b=8,
    s_q=1,
    s_k=1024,
    is_varlen=False,
    is_causal=False,
    test_performance=False,
    block_size=64,
    h_q=128,
    h_kv=1,
)


@pytest.fixture(autouse=True)
def cuda_defaults():
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    yield
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)


def run_decode(t, q, blocked_k, block_table, cache_seqlens, sched_meta, num_splits=None):
    return flash_mla.flash_mla_with_kvcache(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        t.dv,
        sched_meta,
        num_splits,
        causal=t.is_causal,
    )


def warmup_on_side_stream(fn, rounds=3):
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(rounds):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)


def check_against_reference(t, out_ans, lse_ans, cache_seqlens, block_table, q, blocked_k):
    out_ref, lse_ref = reference_torch(cache_seqlens, block_table, q, blocked_k, t.dv, t.is_causal)
    is_correct = kk.check_is_allclose("out", out_ans, out_ref, abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
    is_correct &= kk.check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536)
    assert is_correct


@torch.inference_mode()
def test_metadata_lazy_gen_and_runtime_refresh_inside_graph_capture():
    t = CASE
    cache_seqlens, q, block_table, blocked_k = generate_test_data(t)

    # Warm up through a throwaway sched_meta so the kernel and the allocator are hot
    # while the metadata path of the sched_meta under test is still cold.
    throwaway_meta, _ = flash_mla.get_mla_metadata()
    warmup_on_side_stream(lambda: run_decode(t, q, blocked_k, block_table, cache_seqlens, throwaway_meta))

    sched_meta, num_splits = flash_mla.get_mla_metadata()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_ans, lse_ans = run_decode(t, q, blocked_k, block_table, cache_seqlens, sched_meta, num_splits)

    assert sched_meta.tile_scheduler_metadata is not None
    assert sched_meta.num_splits is not None
    metadata_ptr = sched_meta.tile_scheduler_metadata.data_ptr()
    num_splits_ptr = sched_meta.num_splits.data_ptr()

    # A -> B -> A proves that the metadata-generation kernel captured during the
    # first invocation refreshes device metadata on every replay. Warming up the
    # sched_meta under test outside the graph would bypass this supported path.
    short_seqlen = max(t.s_q, t.s_k // 4 + 1)
    for replay_seqlen in (t.s_k, short_seqlen, t.s_k):
        cache_seqlens.fill_(replay_seqlen)
        graph.replay()
        torch.cuda.synchronize()

        assert sched_meta.tile_scheduler_metadata.data_ptr() == metadata_ptr
        assert sched_meta.num_splits.data_ptr() == num_splits_ptr
        check_against_reference(
            t, out_ans, lse_ans, cache_seqlens, block_table, q, blocked_k
        )
