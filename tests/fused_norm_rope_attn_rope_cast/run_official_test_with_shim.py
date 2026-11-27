#!/usr/bin/env python3
"""Runner that lets the official fused unit test run without ``tile_kernels``.

WHY
---
``test-fused-norm-rope-attn-rope-cast.py`` does ``import tile_kernels`` at module
scope, and that package cannot be installed on PPU (see ``_compat/tile_kernels.py``
for the full explanation).  Rather than editing the official test — which would
make future upstream merges messy and hide what we changed — this runner
installs the local shim into ``sys.modules`` *before* the test is imported, so
the test's own source stays byte-for-byte upstream.

USAGE
-----
    python run_official_test_with_shim.py [--h-q 64 128] [--output-scale block32] ...

Every argument is forwarded verbatim to the official test, so anything that
works there works here.  ``--`` is not needed.

WHAT IT CHANGES
---------------
  * inserts ``tests/`` and ``tests/_compat`` on ``sys.path``
  * if the real ``tile_kernels`` is importable, it is used and nothing is faked
  * otherwise the shim is registered and a loud notice is printed
"""
from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parent                      # tests/
COMPAT_DIR = TESTS_DIR / "_compat"           # tests/_compat
OFFICIAL = HERE / "test-fused-norm-rope-attn-rope-cast.py"


def _install_shim() -> str:
    """Register the local tile_kernels shim if the real package is missing.

    The test accesses ``tile_kernels.quant``, ``tile_kernels.utils`` and does
    ``from tile_kernels.torch import cast_back``.  The latter needs the
    submodules registered under dotted names, so a single flat module is not
    enough -- register each piece in ``sys.modules`` explicitly.
    """
    shim_path = COMPAT_DIR / "tile_kernels.py"

    # Is a *real* package importable?  Probe without leaving the shim's own
    # directory on sys.path, otherwise we would find ourselves and call it real.
    probe_path = [p for p in sys.path if p != str(COMPAT_DIR)]
    real = importlib.util.find_spec("tile_kernels") if str(COMPAT_DIR) not in sys.path \
        else None
    if real is not None and real.origin and Path(real.origin) != shim_path:
        import tile_kernels as _tk  # noqa: F401
        return f"real tile_kernels {getattr(_tk, '__file__', '?')}"
    del probe_path

    spec = importlib.util.spec_from_file_location("tile_kernels", shim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shim from {shim_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tile_kernels"] = module
    spec.loader.exec_module(module)

    # Make `from tile_kernels.quant import per_token_cast` and
    # `from tile_kernels.torch import cast_back` work: register the three
    # submodules under their dotted names.
    for sub in ("quant", "utils", "torch"):
        sys.modules[f"tile_kernels.{sub}"] = getattr(module, sub)
    module.__path__ = []          # mark it package-like for import machinery
    return f"LOCAL SHIM {shim_path}"


def main() -> int:
    for p in (str(TESTS_DIR), str(COMPAT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    origin = _install_shim()
    print("=" * 72)
    print("tile_kernels source :", origin)
    if origin.startswith("LOCAL SHIM"):
        print()
        print("!!  NOTICE: the real `tile_kernels` is unavailable on this system.")
        print("!!  A LOCAL, pure-PyTorch equivalent is being used for the test's")
        print("!!  reference-side FP8 quantization only.  It is NOT the upstream")
        print("!!  package and is NOT bit-exact with it.  Verify it with")
        print("!!  `_compat/selfcheck_tile_kernels_shim.py` before trusting any")
        print("!!  correctness verdict that depends on it.")
    print("official test       :", OFFICIAL)
    print("=" * 72)
    sys.stdout.flush()

    # Hand over to the official test exactly as if it had been run directly.
    sys.argv = [str(OFFICIAL)] + sys.argv[1:]
    runpy.run_path(str(OFFICIAL), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())