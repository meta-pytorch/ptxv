# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paths, hashing and small helpers shared by the package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def cache_dir(sub: str = "") -> Path:
    """`$PTXV_CACHE`, else `$TMPDIR/ptxv-<uid>` (home directories are often small)."""
    base = os.environ.get("PTXV_CACHE") or os.path.join(
        tempfile.gettempdir(), f"ptxv-{os.getuid()}"
    )
    p = Path(base) / sub if sub else Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def repo_root() -> Path | None:
    """The ptxv checkout this package lives in (python/ptxv/ -> repo), if any."""
    here = Path(__file__).resolve().parent.parent.parent
    if (here / "Cargo.toml").exists() and (here / "crates" / "cli").is_dir():
        return here
    env = os.environ.get("PTXV_ROOT")
    if env and (Path(env) / "Cargo.toml").exists():
        return Path(env)
    return None


def sha256_file(path: str | os.PathLike, h: "hashlib._Hash | None" = None) -> str:
    h = h or hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(root: str | os.PathLike, skip: Iterable[str] = ("out",)) -> str:
    """Content hash of a directory: every regular file's relative path and bytes, sorted."""
    root = Path(root)
    h = hashlib.sha256()
    skip = set(skip)
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not (set(p.relative_to(root).parts[:1]) & skip)
    )
    for p in files:
        rel = p.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(4, "little") + rel)
        h.update(p.stat().st_size.to_bytes(8, "little"))
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def canon_json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


def job_key(kind: str, args: dict, case_hashes: list[str], ptxv_hash: str) -> str:
    """Cache key of a job: its kind and arguments, the cases' content, the ptxv binary."""
    return hashlib.sha256(
        canon_json([kind, args, case_hashes, ptxv_hash]).encode()
    ).hexdigest()


def short(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."
