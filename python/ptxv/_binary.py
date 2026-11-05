# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Locating (or building) the `ptxv` binary.

Order: `$PTXV_BIN`; `bin = "..."` in the config file (`$PTXV_CONFIG`, else
`~/.config/ptxv/config.toml`); `$CARGO_TARGET_DIR/release/ptxv`; the checkout's
`target/release/ptxv`. If none exists and the package sits in a ptxv checkout, `cargo build
--release` runs there once (through `env.sh`).
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from ._util import repo_root, sha256_file

try:  # Python >= 3.11
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

_lock = threading.Lock()
_hash_cache: dict[tuple[str, int, int], str] = {}


def config() -> dict:
    p = Path(
        os.environ.get("PTXV_CONFIG", os.path.expanduser("~/.config/ptxv/config.toml"))
    )
    if tomllib and p.exists():
        with open(p, "rb") as f:
            return tomllib.load(f)
    return {}


def candidates() -> list[Path]:
    out = []
    if os.environ.get("PTXV_BIN"):
        out.append(Path(os.environ["PTXV_BIN"]))
    if config().get("bin"):
        out.append(Path(os.path.expanduser(config()["bin"])))
    if os.environ.get("CARGO_TARGET_DIR"):
        out.append(Path(os.environ["CARGO_TARGET_DIR"]) / "release" / "ptxv")
    root = repo_root()
    if root:
        out.append(root / "target" / "release" / "ptxv")
    return out


def find(build: bool = True) -> Path:
    """Path of a usable `ptxv` binary; builds it in the checkout when `build` and none exists."""
    for c in candidates():
        if c.is_file() and os.access(c, os.X_OK):
            return c
    root = repo_root()
    if not build or root is None:
        raise FileNotFoundError(
            "ptxv binary not found: set PTXV_BIN, or `bin` in ~/.config/ptxv/config.toml"
        )
    return build_binary(root)


def build_binary(root: Path) -> Path:
    """`cargo build --release` in the checkout, with env.sh's toolchain settings."""
    with _lock:
        target = os.environ.get("CARGO_TARGET_DIR") or str(root / "target")
        cmd = f'source "{root}/env.sh" && CARGO_TARGET_DIR="{target}" cargo build --release'
        subprocess.run(["bash", "-c", cmd], check=True, cwd=root)
        return Path(target) / "release" / "ptxv"


def binary_hash(path: str | os.PathLike) -> str:
    """Content hash of the binary (part of every cache key), memoized by mtime and size."""
    st = os.stat(path)
    key = (str(path), int(st.st_mtime_ns), st.st_size)
    if key not in _hash_cache:
        _hash_cache[key] = sha256_file(path)
    return _hash_cache[key]
