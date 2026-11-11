# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test setup: a private cache, the CUPTI import hook off (GPU tests turn it on)."""

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("PTXV_CACHE", tempfile.mkdtemp(prefix="ptxv-test-"))
GPU = os.environ.get("PTXV_TEST_GPU")
if GPU:
    excluded = set(filter(None, os.environ.get("PTXV_EXCLUDE_GPUS", "").split(",")))
    if excluded & set(GPU.split(",")):
        raise SystemExit(f"PTXV_TEST_GPU={GPU}: PTXV_EXCLUDE_GPUS={sorted(excluded)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # before torch initializes CUDA
    # Triton's bundled ptxas does not know sm_103a.
    for tool in ("ptxas", "cuobjdump", "nvdisasm"):
        os.environ.setdefault(
            f"TRITON_{tool.upper()}_PATH", f"/usr/local/cuda-13.0/bin/{tool}"
        )
else:
    os.environ["PTXV_NO_IMPORT_HOOK"] = "1"

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures"


def have_binary() -> bool:
    from ptxv import _binary

    try:
        _binary.find(build=False)
        return True
    except FileNotFoundError:
        return False


needs_binary = pytest.mark.skipif(
    not have_binary(), reason="ptxv binary not found (PTXV_BIN)"
)


def vendor_skip(*names: str) -> None:
    """Skip when a third-party kernel file of these fixtures is absent (not distributed; see
    fixtures/VENDOR_FIXTURES.toml and tools/regen_fixture.py)."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        return
    man = FIX / "VENDOR_FIXTURES.toml"
    if not man.exists():
        return
    with open(man, "rb") as f:
        entries = tomllib.load(f)["fixture"]
    missing = [
        f"{e['name']}/{f}"
        for e in entries
        if e["name"] in names
        for f in e["files"]
        if not (FIX / e["name"] / f).is_file()
    ]
    if missing:
        pytest.skip(
            "vendor fixture files not present: "
            + ", ".join(missing)
            + " (python3 tools/regen_fixture.py --missing)"
        )
