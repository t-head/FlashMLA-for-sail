"""End-to-end FlashMLA interface tests driven by a live SGLang server.

Unlike the kernel-level tests in this directory, these cases launch
``sglang serve`` with a real model that routes attention to FlashMLA,
send one simple generation request, and verify the response.

Covered FlashMLA entry points (on PPU):

- ``dense_decode_fwd``      — DeepSeek-V2-Lite and Kimi-K2.5:
  ``--decode-attention-backend flashmla``;
  GLM-5.1: NSA decode via ``flashmla_kv`` (gathered top-k KV, DSA attention)
- ``sparse_decode_fwd``     — DeepSeek-V4-Flash: DSA sparse decode (EAGLE,
  CUDA graph, dp-attention), i.e. the production deployment config
- ``flash_mla_sparse_fwd``  — DeepSeek-V4-Flash sparse prefill path;
  GLM-5.1: NSA sparse prefill (``--nsa-prefill-backend flashmla_sparse``)

Usage:

    # Run all cases with pytest (each case boots its own server):
    python -m pytest tests/test_sglang_e2e_flashmla.py -v -s

    # Or run directly as a script (exit code 0 = pass, non-zero = fail):
    python tests/test_sglang_e2e_flashmla.py                    # all cases
    python tests/test_sglang_e2e_flashmla.py --case kimi-k2.5   # single case

    # Lightweight single-GPU dense decode case:
    python tests/test_sglang_e2e_flashmla.py --case deepseek-v2-lite

Environment overrides:

    FLASHMLA_E2E_DSV2_PATH   model path for the deepseek-v2-lite case
    FLASHMLA_E2E_KIMI_PATH   model path for the kimi-k2.5 case
    FLASHMLA_E2E_DSV4_PATH   model path for the deepseek-v4-flash case
    FLASHMLA_E2E_DSV4_INT8_PATH  model path for the deepseek-v4-flash-int8 case
    FLASHMLA_E2E_GLM_PATH    model path for the glm-5.1 case
    FLASHMLA_E2E_PORT        server port (default 8999)
    FLASHMLA_E2E_TIMEOUT     server startup timeout in seconds (default 1800)
    FLASHMLA_E2E_LOG_DIR     directory for server logs (default tests/logs)

DeepSeek-V2-Lite defaults to models/DeepSeek-V2-Lite under the repository
root. Set FLASHMLA_E2E_DSV2_PATH to an existing local checkpoint directory.

Cases may also carry case-specific env vars for the server process (glm-5.1
sets ``SGLANG_NSA_DUAL_STREAM=0`` and
``SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8=0``); they are merged on
top of the ambient environment and take precedence.

Any failure prints the reason plus the tail of the server log and exits
non-zero (or fails the pytest case).
"""

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_LOG_DIR = Path(os.environ.get("FLASHMLA_E2E_LOG_DIR", _TEST_DIR / "logs"))

_DEFAULT_DSV2_PATH = "/ppusw/datasets/checkpoints/LLM/DeepSeek/V2/DeepSeek-V2-Lite"
_DEFAULT_KIMI_PATH = "/ppusw/datasets/checkpoints/LLM/kimi/v2.5/Kimi-K2.5"
_DEFAULT_DSV4_PATH = (
    "/ppusw/datasets/checkpoints/LLM/deepseek-ai/v1.0/DeepSeek-V4-Flash-0731-w8a8"
)
_DEFAULT_DSV4_INT8_PATH = (
    "/ppusw/datasets/checkpoints/LLM/deepseek/v4/DeepSeek-V4-Flash-W8A8-INT8"
)
_DEFAULT_GLM_PATH = "/ppusw/datasets/checkpoints/LLM/zhipu/v5.1/GLM-5.1-W8A8-INT8"

_STARTUP_TIMEOUT = int(os.environ.get("FLASHMLA_E2E_TIMEOUT", "1800"))
_REQUEST_TIMEOUT = 120
_LOG_TAIL_LINES = 100


@dataclasses.dataclass(frozen=True)
class E2ECase:
    """One server-launch test case."""

    name: str
    model_path: str
    # Extra CLI args appended after the common ones.
    extra_args: tuple
    # Human-readable note on which FlashMLA interfaces this case exercises.
    covers: str = ""
    # Common launch args whose value differs between cases.
    tp_size: int = 8
    mem_fraction_static: str = "0.8"
    # Extra env vars for the server process, merged over os.environ
    # (case values take precedence).
    env: dict = dataclasses.field(default_factory=dict)


_CASES = {
    "deepseek-v2-lite": E2ECase(
        name="deepseek-v2-lite",
        model_path=os.environ.get("FLASHMLA_E2E_DSV2_PATH", _DEFAULT_DSV2_PATH),
        extra_args=(
            "--served-model-name",
            "DeepSeek-V2-Lite",
            "--dtype",
            "bfloat16",
            "--decode-attention-backend",
            "flashmla",
            "--prefill-attention-backend",
            "fa3",
            # Keep startup and graph capture small for the smoke request.
            "--context-length",
            "4096",
            "--max-running-requests",
            "8",
            "--cuda-graph-max-bs-decode",
            "8",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--max-total-tokens",
            "16384",
        ),
        covers="dense_decode_fwd (MLA dense decode)",
        tp_size=2,
    ),
    "kimi-k2.5": E2ECase(
        name="kimi-k2.5",
        model_path=os.environ.get("FLASHMLA_E2E_KIMI_PATH", _DEFAULT_KIMI_PATH),
        extra_args=(
            "--served-model-name",
            "Kimi-K2.5",
            "--decode-attention-backend",
            "flashmla",
            "--prefill-attention-backend",
            "fa3",
        ),
        covers="dense_decode_fwd (MLA dense decode)",
    ),
    "deepseek-v4-flash": E2ECase(
        name="deepseek-v4-flash",
        model_path=os.environ.get("FLASHMLA_E2E_DSV4_PATH", _DEFAULT_DSV4_PATH),
        extra_args=(
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
            "--deepep-mode",
            "auto",
            "--enable-dp-attention",
            "--dp-size",
            "8",
            "--quantization",
            "w8a8_int8",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
            "--served-model-name",
            "DeepSeek-V4",
            "--reasoning-parser",
            "deepseek-v4",
            "--tool-call-parser",
            "deepseekv4",
            "--speculative-attention-mode",
            "decode",
            "--moe-a2a-backend",
            "deepep",
            "--moe-dense-tp-size",
            "1",
            "--cuda-graph-max-bs",
            "64",
            "--disable-custom-all-reduce",
            "--enable-dp-lm-head",
            "--disable-piecewise-cuda-graph",
            "--disable-shared-experts-fusion",
        ),
        covers=(
            "sparse_decode_fwd (DSA sparse decode + EAGLE target_verify / "
            "draft_extend_v2, CUDA graph), flash_mla_sparse_fwd (sparse prefill)"
        ),
    ),
    "deepseek-v4-flash-int8": E2ECase(
        name="deepseek-v4-flash-int8",
        model_path=os.environ.get(
            "FLASHMLA_E2E_DSV4_INT8_PATH", _DEFAULT_DSV4_INT8_PATH
        ),
        extra_args=(
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
            "--deepep-mode",
            "auto",
            "--enable-dp-attention",
            "--dp-size",
            "4",
            "--quantization",
            "w8a8_int8",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
            "--served-model-name",
            "DeepSeek-V4",
            "--reasoning-parser",
            "deepseek-v4",
            "--tool-call-parser",
            "deepseekv4",
            "--speculative-attention-mode",
            "decode",
            "--moe-a2a-backend",
            "deepep",
            "--moe-dense-tp-size",
            "1",
            "--cuda-graph-max-bs",
            "64",
            "--disable-custom-all-reduce",
            "--enable-dp-lm-head",
            "--disable-piecewise-cuda-graph",
            "--disable-shared-experts-fusion",
        ),
        covers=(
            "sparse_decode_fwd (DSA sparse decode + EAGLE target_verify / "
            "draft_extend_v2, CUDA graph), flash_mla_sparse_fwd (sparse prefill)"
        ),
        tp_size=4,
        # Leave room for both target and EAGLE draft CUDA graphs on 96 GiB PPU.
        mem_fraction_static="0.85",
        # Target verification captures 64 requests * 3 draft tokens = 192.
        env={"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256"},
    ),
    "glm-5.1": E2ECase(
        name="glm-5.1",
        model_path=os.environ.get("FLASHMLA_E2E_GLM_PATH", _DEFAULT_GLM_PATH),
        extra_args=(
            "--page-size",
            "64",
            "--attention-backend",
            "dsa",
            "--nsa-prefill-backend",
            "flashmla_sparse",
            "--nsa-decode-backend",
            "flashmla_kv",
            "--disable-radix-cache",
            "--quantization",
            "w8a8_int8",
            "--max-running-requests",
            "128",
            "--log-level",
            "info",
            "--disable-custom-all-reduce",
            "--disable-shared-experts-fusion",
            "--watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
            "--chunked-prefill-size",
            "8192",
            "--disable-piecewise-cuda-graph",
        ),
        covers=(
            "flash_mla_sparse_fwd (NSA sparse prefill via flashmla_sparse), "
            "dense_decode_fwd (NSA decode via flashmla_kv on gathered top-k KV)"
        ),
        tp_size=16,
        mem_fraction_static="0.90",
        env={
            "SGLANG_NSA_DUAL_STREAM": "0",
            "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8": "0",
        },
    ),
}


class E2ETestError(RuntimeError):
    """Raised on any e2e failure; carries the full diagnostic message."""


def _build_cmd(case: E2ECase, port: int) -> list:
    return [
        "sglang",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--model-path",
        case.model_path,
        "--tp-size",
        str(case.tp_size),
        "--mem-fraction-static",
        case.mem_fraction_static,
        "--trust-remote-code",
        *case.extra_args,
    ]


def _read_log_tail(log_path: Path) -> str:
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"<cannot read log file {log_path}: {e}>"
    tail = "".join(lines[-_LOG_TAIL_LINES:])
    return tail if tail.strip() else "<log file is empty>"


def _fail(case: E2ECase, stage: str, reason: str, log_path: Path):
    message = (
        f"\n{'=' * 72}\n"
        f"[E2E FAIL] case={case.name} stage={stage}\n"
        f"reason: {reason}\n"
        f"server log: {log_path}\n"
        f"----- server log tail (last {_LOG_TAIL_LINES} lines) -----\n"
        f"{_read_log_tail(log_path)}\n"
        f"{'=' * 72}"
    )
    raise E2ETestError(message)


def _kill_server(proc: subprocess.Popen):
    """Kill the whole process group (sglang spawns scheduler/worker children)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=30)


def _http_get_status(url: str, timeout: float):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def _wait_ready(case: E2ECase, proc: subprocess.Popen, port: int,
                timeout: int, log_path: Path):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    print(f"[{case.name}] waiting for server on :{port} "
          f"(timeout={timeout}s, log={log_path})", flush=True)
    while time.time() < deadline:
        ret = proc.poll()
        if ret is not None:
            _fail(case, "startup",
                  f"sglang serve exited early with code {ret}", log_path)
        if _http_get_status(url, timeout=5) == 200:
            print(f"[{case.name}] server is ready", flush=True)
            return
        time.sleep(5)
    _fail(case, "startup", f"server not ready within {timeout}s", log_path)


def _send_generate(case: E2ECase, port: int, log_path: Path) -> str:
    payload = {
        "text": "The capital of France is",
        "sampling_params": {"max_new_tokens": 16, "temperature": 0},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    print(f"[{case.name}] sending request: {payload['text']!r}", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        _fail(case, "request", f"POST /generate failed: {e!r}", log_path)

    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str) or not text.strip():
        _fail(case, "validate",
              f"empty or malformed completion in response: {body!r}", log_path)
    print(f"[{case.name}] response: {text!r}", flush=True)
    return text


def run_case(case: E2ECase, port: int, startup_timeout: int):
    """Boot the server, send one request, verify, tear down.

    Raises E2ETestError on failure.
    """
    if not os.path.isdir(case.model_path):
        raise E2ETestError(
            f"[E2E FAIL] case={case.name} stage=preflight\n"
            f"reason: model path does not exist: {case.model_path}\n"
            f"set the FLASHMLA_E2E_*_PATH env var to override."
        )

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / f"sglang_e2e_{case.name}.log"
    cmd = _build_cmd(case, port)
    print(f"[{case.name}] covers: {case.covers}", flush=True)
    print(f"[{case.name}] launching: {' '.join(cmd)}", flush=True)
    if case.env:
        print(f"[{case.name}] env: {case.env}", flush=True)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            # Own process group so we can kill scheduler/worker children too.
            start_new_session=True,
            # Case env vars merged over the ambient environment; the
            # scheduler/worker children inherit them as well.
            env={**os.environ, **case.env},
        )
    try:
        _wait_ready(case, proc, port, startup_timeout, log_path)
        _send_generate(case, port, log_path)
    finally:
        _kill_server(proc)

    print(f"[{case.name}] PASS", flush=True)


# ---------------------------------------------------------------------------
# pytest entry
# ---------------------------------------------------------------------------

try:
    import pytest

    @pytest.mark.parametrize(
        "case_name", [pytest.param(n, id=n) for n in _CASES]
    )
    def test_sglang_e2e_flashmla(case_name):
        port = int(os.environ.get("FLASHMLA_E2E_PORT", "8999"))
        run_case(_CASES[case_name], port=port, startup_timeout=_STARTUP_TIMEOUT)

except ImportError:  # pragma: no cover - script mode without pytest
    pass


# ---------------------------------------------------------------------------
# script entry
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*_CASES, "all"], default="all",
                        help="which case to run (default: all, sequentially)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLASHMLA_E2E_PORT", "8999")),
    )
    parser.add_argument("--timeout", type=int, default=_STARTUP_TIMEOUT,
                        help="server startup timeout in seconds")
    args = parser.parse_args()

    names = list(_CASES) if args.case == "all" else [args.case]
    failures = []
    for name in names:
        case = _CASES[name]
        try:
            run_case(case, port=args.port, startup_timeout=args.timeout)
        except E2ETestError as e:
            print(str(e), file=sys.stderr, flush=True)
            failures.append(name)
        except Exception as e:  # unexpected harness error
            print(f"[E2E FAIL] case={name} unexpected error: {e!r}",
                  file=sys.stderr, flush=True)
            failures.append(name)
        # Give the port a moment to be released before the next case.
        time.sleep(5)

    if failures:
        print(f"\n[E2E] FAILED cases: {failures}", file=sys.stderr)
        return 1
    print("\n[E2E] all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
