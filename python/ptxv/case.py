# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`ptxv.Case`: a case directory (case.toml + data + cubins), as the CLI consumes it."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

_NP = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "i32": np.int32,
    "i64": np.int64,
    "i16": np.int16,
    "i8": np.int8,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
}


def decode(raw: bytes, dtype: str, shape: list[int] | None = None) -> np.ndarray:
    """Bytes of a case buffer as numpy (bf16 widened to float32)."""
    if dtype == "bf16":
        a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        a = np.frombuffer(raw, _NP[dtype]).copy()
    if shape and int(np.prod(shape)) == a.size:
        a = a.reshape(shape)
    return a


class Case:
    """A ptxv case directory. `inputs()` / `expected()` give its buffers as numpy arrays."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path).resolve()
        if not (self.path / "case.toml").exists():
            raise FileNotFoundError(f"{self.path}: no case.toml")
        with open(self.path / "case.toml", "rb") as f:
            self.toml: dict = tomllib.loads(f.read().decode())  # type: ignore[union-attr]

    def case_dir(self) -> Path:
        return self.path

    @property
    def name(self) -> str:
        return self.toml.get("name", self.path.name)

    @property
    def buffers(self) -> list[dict]:
        return self.toml.get("buffers", [])

    @property
    def outputs(self) -> list[str]:
        return list(self.toml.get("outputs", []))

    @property
    def kernels(self) -> list[str]:
        return [self.toml["entry"]] + [
            la["entry"] for la in self.toml.get("launches", [])
        ]

    def buffer(self, name: str) -> dict:
        for b in self.buffers:
            if b["name"] == name:
                return b
        raise KeyError(name)

    def inputs(self) -> dict[str, np.ndarray]:
        """Initial contents of every buffer that has them (init = "file")."""
        out = {}
        for b in self.buffers:
            if b.get("init") == "file":
                out[b["name"]] = decode(
                    (self.path / b["file"]).read_bytes(), b["dtype"], b.get("shape")
                )
        return out

    def expected(self) -> dict[str, np.ndarray]:
        """The captured GPU bytes of each output (`expect`)."""
        out = {}
        for b in self.buffers:
            if b.get("expect"):
                raw = (self.path / b["expect"]).read_bytes()[
                    b.get("expect_offset", 0) :
                ]
                out[b["name"]] = decode(raw, b["dtype"], b.get("shape"))
        return out

    def check(self, **kw: Any):
        from .session import session

        return session().check(self, **kw)

    def __repr__(self) -> str:
        ks = ", ".join(k[:40] for k in self.kernels[:3]) + (
            " ..." if len(self.kernels) > 3 else ""
        )
        return f"Case({self.name!r}, kernels=[{ks}], outputs={self.outputs}, path={str(self.path)!r})"
