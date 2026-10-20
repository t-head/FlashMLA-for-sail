"""H128 Sq1 length scheduler: short -> split -> short, with fresh Graph metadata.

--list-cases is CPU only. Execution requires the retained Prefix entries and a
39-partition schedule. All reference tolerances are the original decode values.
--continue-on-numerical-failure records every stage, but still exits nonzero.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


FORMATS = ("fp8_v41", "fp4_v41_extra")
TOLERANCES = dict(out=dict(abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6),
                  lse=dict(abs_tol=1e-6, rel_tol=8.01 / 65536))


def canonical_metadata(extra_lengths, parts=39):
    """Canonical N64 / overhead5 producer; only its five written fields matter."""
    lengths = [128 + n for n in extra_lengths]
    blocks = [(n + 63) // 64 for n in lengths]
    payload = (sum(n + 5 for n in blocks) + parts - 1) // parts + 5
    batch = block = split = total = 0
    prefix, metadata = [0], []
    for _ in range(parts):
        first, start, first_split, remaining = batch, block * 64, split, payload
        while batch < len(blocks):
            left = blocks[batch] - block
            if remaining >= left + 5:
                total += split + 1
                prefix.append(total)
                remaining -= left + 5
                batch, block, split = batch + 1, 0, 0
            else:
                if remaining > 5:
                    block += remaining - 5
                    split += 1
                break
        last = batch if block else batch - 1
        end = block * 64 if block else lengths[batch - 1]
        metadata.append([first, start, last, end, first_split])
    assert batch == len(blocks) and block == split == 0
    return metadata, prefix


def cases():
    result = []
    for fmt in FORMATS:
        stages = []
        for name in ("short", "full", "restored_short"):
            extra = [1024] * 300 if name == "full" else [13, 39] * 150
            _, prefix = canonical_metadata(extra)
            stages.append(dict(id=name, total_tasks=prefix[-1], task_slots=338,
                               ghost_slots=338 - prefix[-1],
                               split_rows=sum(b - a > 1 for a, b in zip(prefix, prefix[1:]))))
        result.append(dict(id=fmt, seed=0, b=300, sq=1, heads=128, main=128,
                           extra=1024, page=64, metadata_parts=39, stages=stages))
    return result


def emit(kind, **record):
    print(json.dumps(dict(kind=kind, **record), sort_keys=True), flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(case, expected_entry, keep_going, torch, kk, lib, quant, ref, flash_mla):
    mixed = case["id"] == "fp4_v41_extra"
    raw = lib.RawTestParamForDecode(b=300, h_q=128, h_kv=1, s_q=1, s_kv=256,
        topk=128, extra_topk=1024, extra_s_k=2048, block_size=64,
        extra_block_size=64, d_qk=512, d_v=512, is_varlen=False, seed=0,
        num_runs=0, enable_attn_sink=True, have_topk_length=False,
        have_extra_topk_length=False, is_fp8=True,
        fp8_kvcache_layout=quant.FP8KVCacheLayout.V41_FP8Sparse,
        extra_fp8_kvcache_layout=(quant.FP8KVCacheLayout.V41_FP4Sparse if mixed
                                 else quant.FP8KVCacheLayout.V41_FP8Sparse))
    params = raw.to_test_param()
    tensors = lib.generate_testcase_for_decode(params)
    # Generate without length masking first: later full rows must not expose NaNs
    # that a fixture initialized under a shorter length could have introduced.
    tensors.kv_scope.topk_length = torch.full((300,), 128, dtype=torch.int32)
    tensors.extra_kv_scope.topk_length = torch.tensor([13, 39] * 150, dtype=torch.int32)
    params.have_topk_length = params.decode.have_extra_topk_length = True
    items = dict(q=tensors.q, sink=tensors.attn_sink)
    for label, scope in (("main", tensors.kv_scope), ("extra", tensors.extra_kv_scope)):
        for key in ("blocked_k", "blocked_k_quantized", "indices_in_kvcache",
                    "abs_indices", "block_table", "cache_seqlens", "topk_length"):
            items[label + "_" + key] = getattr(scope, key)

    def fingerprint(value):
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()),
                    sha256=hashlib.sha256(memoryview(data).cast("B")).hexdigest())

    def outputs(answer):
        return {key: fingerprint(value) for key, value in zip(("out", "lse"), answer)}

    pointers = {key: value.data_ptr() for key, value in items.items()}
    before = {key: fingerprint(value) for key, value in items.items()}
    emit("fixture", case=case, inputs=before)
    failures, retained_metadata, stage_records = [], [], []
    first_outputs = first_cache = None
    trait = ("QuantDecodeDirectPairedTraits<" + ("true" if mixed else "false") + ">"
             if expected_entry == "original" else
             "QuantDecodeSq1LengthsPrefix" + ("FP4Traits" if mixed else "FP8Traits"))
    qualified = "flashmla::v41prefill::" + trait
    expected_main = f"void flashmla::v41prefill::sparse_quant_wi_kernel<{qualified}, true>({qualified}::KernelParams)"

    def trace(callback, metadata_count):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            answer = callback()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        mains = [n for n in names if "sparse_quant_wi_kernel" in n or "flash_sparse_decode" in n]
        assert len(mains) == 1 and "".join(mains[0].split()) == "".join(expected_main.split()), names
        assert sum("get_mla_metadata_kernel" in n for n in names) == metadata_count, names
        return answer, names

    for stage in case["stages"]:
        lengths = [1024] * 300 if stage["id"] == "full" else [13, 39] * 150
        tensors.extra_kv_scope.topk_length.copy_(torch.tensor(lengths, dtype=torch.int32))
        metadata, _ = flash_mla.get_mla_metadata()
        assert not metadata.have_initialized
        retained_metadata.append(metadata)  # Prevent allocator reuse across stages.
        reference = ref.ref_sparse_attn_decode(params, tensors)

        def call():
            return lib.run_flash_mla_decode(params, tensors, metadata, None)

        eager, names = trace(call, 1)
        assert metadata.num_splits.shape == (301,) and metadata.tile_scheduler_metadata.shape == (39, 8)
        expected_meta, expected_prefix = canonical_metadata(lengths)
        assert metadata.num_splits.tolist() == expected_prefix
        assert metadata.tile_scheduler_metadata[:, :5].tolist() == expected_meta
        assert stage["total_tasks"] == expected_prefix[-1] <= stage["task_slots"]
        assert (expected_prefix[-1] > 300) == (stage["id"] == "full")
        cache = dict(prefix=fingerprint(metadata.num_splits),
                     metadata=fingerprint(metadata.tile_scheduler_metadata[:, :5]))
        cached_pointers = (metadata.num_splits.data_ptr(), metadata.tile_scheduler_metadata.data_ptr())
        assert all(cached_pointers[0] != old.num_splits.data_ptr() and
                   cached_pointers[1] != old.tile_scheduler_metadata.data_ptr()
                   for old in retained_metadata[:-1])
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        checks = []
        for index in range(3):
            answer = eager
            if index:
                trace(graph.replay, 0)
                answer = captured
                assert all(torch.equal(a, b) for a, b in zip(eager, answer)), stage
            label = f"{case['id']}/{stage['id']}/" + ("eager" if index == 0 else f"graph{index - 1}")
            result = {key: bool(kk.check_is_allclose(label + "/" + key, value, expected,
                      **TOLERANCES[key])) for key, value, expected in zip(("out", "lse"), answer, reference)}
            checks.append(result)
            emit("comparison", comparison_id=label, checks=result)
            if not all(result.values()):
                failures.append(dict(comparison_id=label, checks=result))
                assert keep_going, failures[-1]
        assert cache == dict(prefix=fingerprint(metadata.num_splits),
                             metadata=fingerprint(metadata.tile_scheduler_metadata[:, :5]))
        assert cached_pointers == (metadata.num_splits.data_ptr(), metadata.tile_scheduler_metadata.data_ptr())
        assert pointers == {key: value.data_ptr() for key, value in items.items()}
        answer_hashes = outputs(eager)
        if stage["id"] == "short":
            first_outputs, first_cache = answer_hashes, cache
        elif stage["id"] == "restored_short":
            assert answer_hashes == first_outputs and cache == first_cache
        record = dict(case=case["id"], stage=stage, kernels=names, metadata_hashes=cache,
                      outputs=answer_hashes, reference=outputs(reference), checks=checks,
                      main_lengths=fingerprint(tensors.kv_scope.topk_length),
                      extra_lengths=fingerprint(tensors.extra_kv_scope.topk_length),
                      fresh_metadata=True, graph_replays=2, graph_eager_bitwise_equal=True)
        emit("stage", **record)
        stage_records.append(record)
    assert before == {key: fingerprint(value) for key, value in items.items()}
    return failures, stage_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--module-dir", type=Path)
    parser.add_argument("--expect-module-sha")
    parser.add_argument("--expected-entry", choices=("original", "prefix"), default="prefix")
    parser.add_argument("--continue-on-numerical-failure", action="store_true")
    args = parser.parse_args()
    selected = cases()
    if args.list_cases:
        print(json.dumps(selected, indent=2))
        return
    source = args.source_dir.resolve()
    sys.path[:0] = ([str(args.module_dir.resolve())] if args.module_dir else []) + [str(source / "tests"), str(source)]
    os.environ.update(FLASHMLA_V41_WI="1", FLASHMLA_V41_HS64="0")
    import torch
    import kernelkit as kk
    import lib
    import quant
    import ref
    import flash_mla
    import flash_mla_cuda
    module = Path(flash_mla_cuda.__file__).resolve()
    assert not args.module_dir or module.parent == args.module_dir.resolve(), module
    assert not args.expect_module_sha or sha(module) == args.expect_module_sha, module
    torch.cuda.set_device(args.device)
    torch.set_default_device(f"cuda:{args.device}")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(16)
    checker = Path(kk.check_is_allclose.__code__.co_filename).resolve()
    process_stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    device_properties = torch.cuda.get_device_properties(args.device)
    device_uuid = getattr(device_properties, "uuid", None)
    emit("binding", module=str(module), module_sha256=sha(module), tolerances=TOLERANCES,
         pid=os.getpid(), start_ticks=process_stat[19], logical_device=args.device,
         CU=device_properties.multi_processor_count,
         device_uuid=None if device_uuid is None else str(device_uuid),
         CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
         test_sha256=sha(__file__), expected_entry=args.expected_entry,
         continue_on_numerical_failure=args.continue_on_numerical_failure,
         checker=dict(path=str(checker), sha256=sha(checker)),
         imports={name: dict(path=str(Path(mod.__file__).resolve()), sha256=sha(mod.__file__))
                  for name, mod in (("lib", lib), ("quant", quant), ("ref", ref), ("flash_mla", flash_mla))})
    failures, stages = [], []
    with torch.inference_mode():
        for case in selected:
            case_failures, records = run_case(case, args.expected_entry, args.continue_on_numerical_failure,
                                             torch, kk, lib, quant, ref, flash_mla)
            failures.extend(case_failures)
            stages.extend(records)
    emit("summary", cases=len(selected), stages=len(stages), reference_comparisons=3 * len(stages),
         failures=failures, status="FAIL" if failures else "PASS", numerical_qualification=not failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
