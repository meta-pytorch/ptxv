# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Owned portable inputs for strict check keys and asynchronous execution."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import weakref
from pathlib import Path

from .case import tomllib

STRICT_MEMORY_CONTRACT = 3


def _remove_snapshot(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


class _OwnedCase:
    def __init__(self, name: str) -> None:
        parent = tempfile.mkdtemp(prefix="ptxv-strict-input-")
        self.name = str(Path(parent) / name)
        Path(self.name).mkdir()
        self._finalizer = weakref.finalize(self, _remove_snapshot, parent)

    def cleanup(self) -> None:
        self._finalizer()


def _relative_input(root: Path, value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"strict memory check requires a nonempty {field} file path")
    path = Path(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[0] == "out"
    ):
        raise ValueError(
            f"strict memory check {field} must stay in the case input inventory, outside out/: {value!r}"
        )
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"strict memory check input cannot use a symlink: {value!r}"
            )
    if not current.is_file():
        raise ValueError(f"strict memory check input file does not exist: {value!r}")


def _validate(root: Path) -> None:
    if tomllib is None:
        raise RuntimeError("strict memory check snapshots require Python 3.11 or later")
    manifest = root / "case.toml"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(
            "strict memory check requires a regular case.toml in a case directory"
        )
    with manifest.open("rb") as stream:
        document = tomllib.load(stream)
    _relative_input(root, document.get("ptx"), "ptx")
    if document.get("sass"):
        _relative_input(root, document["sass"], "sass")
    for index, buffer in enumerate(document.get("buffers", [])):
        for field in ("file", "expect"):
            if buffer.get(field):
                _relative_input(root, buffer[field], f"buffers[{index}].{field}")


def copy_strict_case(case: str | os.PathLike) -> _OwnedCase:
    """Copy regular input bytes; the returned owner keeps the snapshot alive."""
    root = Path(case)
    if not root.is_dir():
        raise ValueError(
            "strict memory check requires a case directory, not a manifest path"
        )
    root = root.resolve()
    _validate(root)
    files = []
    for directory, children, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        if parent == root:
            children[:] = [name for name in children if name != "out"]
        for name in children + names:
            if parent == root and name == "out":
                continue
            path = parent / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"strict memory check input cannot use a symlink: {path.relative_to(root)}"
                )
            if stat.S_ISREG(mode):
                files.append(path.relative_to(root))
            elif not stat.S_ISDIR(mode):
                raise ValueError(
                    f"strict memory check input is not a regular file: {path.relative_to(root)}"
                )
    owner = _OwnedCase(root.name)
    snapshot = Path(owner.name)
    try:
        for relative in files:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                (root / relative).open("rb") as source,
                target.open("xb") as destination,
            ):
                shutil.copyfileobj(source, destination)
            target.chmod(0o400)
        _validate(snapshot)
    except BaseException:
        owner.cleanup()
        raise
    return owner


class _OwnedExecutable:
    def __init__(self) -> None:
        parent = tempfile.mkdtemp(prefix="ptxv-strict-binary-")
        self.name = str(Path(parent) / "ptxv")
        self.digest = ""
        self._finalizer = weakref.finalize(self, _remove_snapshot, parent)

    def cleanup(self) -> None:
        self._finalizer()


def resolve_executable(binary: str | os.PathLike) -> Path:
    """Resolve an executable command using the caller's current directory and PATH."""
    command = os.fspath(binary)
    if not os.path.dirname(command):
        selected = shutil.which(command)
        if selected is None:
            raise FileNotFoundError(f"ptxv executable not found on PATH: {command!r}")
        command = selected
    return Path(command).resolve(strict=True)


def copy_strict_executable(binary: str | os.PathLike) -> _OwnedExecutable:
    """Capture executable bytes and hash the exact private copy that will run."""
    path = resolve_executable(binary)
    if not os.access(path, os.X_OK):
        raise ValueError(f"strict memory check binary is not executable: {binary!r}")
    owner = _OwnedExecutable()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        with os.fdopen(descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError(
                    "strict memory check binary must be a regular executable: "
                    f"{binary!r}"
                )
            target = Path(owner.name)
            with target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            target.chmod(0o500)
            digest = hashlib.sha256()
            with target.open("rb") as captured:
                for chunk in iter(lambda: captured.read(1024 * 1024), b""):
                    digest.update(chunk)
            owner.digest = digest.hexdigest()
    except BaseException:
        owner.cleanup()
        raise
    return owner
