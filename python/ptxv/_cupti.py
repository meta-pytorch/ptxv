# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Building and loading the in-process capture library (`_native/ptxvcap.c`).

The library is compiled on first use with gcc against the CUDA toolkit's headers
(`$CUDA_HOME`, else /usr/local/cuda-13.0, else /usr/local/cuda) into
`$PTXV_CACHE/native/ptxvcap-<source hash>.so`. It links neither libcupti nor libcuda: the
libcupti already mapped in the process is used when there is one (torch's profiler loads it),
else `$PTXV_CUPTI`, else the first found next to Python, in nvidia pip wheels, or in the toolkit.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

from ._util import cache_dir

_SRC = Path(__file__).parent / "_native" / "ptxvcap.c"
_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_init_error: str | None = None
MODULE_DIR = None  # set by init()


class CaptureUnavailable(RuntimeError):
    """In-process capture cannot run here (no CUPTI, or another CUPTI subscriber)."""


def cuda_home() -> Path:
    for c in (os.environ.get("CUDA_HOME"), "/usr/local/cuda-13.0", "/usr/local/cuda"):
        if c and (Path(c) / "include" / "cuda.h").exists():
            return Path(c)
    raise CaptureUnavailable("CUDA toolkit headers not found (set CUDA_HOME)")


def _build() -> Path:
    src = _SRC.read_bytes()
    h = hashlib.sha256(src).hexdigest()[:16]
    out = cache_dir("native") / f"ptxvcap-{h}.so"
    if out.exists():
        return out
    cuda = cuda_home()
    incs = [cuda / "include"] + [
        Path(p) for p in glob.glob(str(cuda / "targets" / "*" / "include"))
    ]
    incs += [cuda / "extras" / "CUPTI" / "include"]
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-o",
        str(tmp),
        str(_SRC),
        "-ldl",
        "-lpthread",
    ]
    cmd += [f"-I{i}" for i in incs if i.is_dir()]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise CaptureUnavailable(
            f"building the capture library failed:\n{r.stderr[-2000:]}"
        )
    tmp.replace(out)
    return out


def cupti_path() -> str:
    """The libcupti to use (see the module docstring)."""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcupti.so" in line:
                    return line.split()[-1]
    except OSError:
        pass
    if os.environ.get("PTXV_CUPTI"):
        return os.environ["PTXV_CUPTI"]
    cands = [
        os.path.join(sys.prefix, "lib", "libcupti.so.13"),
        os.path.join(sys.prefix, "lib", "libcupti.so"),
    ]
    for sp in sys.path:
        cands += glob.glob(os.path.join(sp, "nvidia", "cu13", "lib", "libcupti.so*"))
        cands += glob.glob(
            os.path.join(sp, "nvidia", "cuda_cupti", "lib", "libcupti.so*")
        )
    try:
        cuda = cuda_home()
        cands += glob.glob(str(cuda / "extras" / "CUPTI" / "lib64" / "libcupti.so*"))
        cands += glob.glob(str(cuda / "targets" / "*" / "lib" / "libcupti.so*"))
    except CaptureUnavailable:
        pass
    for c in cands:
        if os.path.exists(c):
            return c
    raise CaptureUnavailable("libcupti not found (set PTXV_CUPTI)")


def init() -> ctypes.CDLL:
    """Load the library and subscribe to CUPTI (once per process); raises CaptureUnavailable."""
    global _lib, _init_error, MODULE_DIR
    with _lock:
        if _lib is not None:
            return _lib
        if _init_error:
            raise CaptureUnavailable(_init_error)
        try:
            lib = ctypes.CDLL(str(_build()))
            lib.ptxvcap_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.ptxvcap_start.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
            ]
            lib.ptxvcap_stop.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)] * 2
            lib.ptxvcap_set_tag.argtypes = [ctypes.c_char_p]
            lib.ptxvcap_modules.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
            lib.ptxvcap_modules.restype = ctypes.c_size_t
            MODULE_DIR = cache_dir("modules")
            rc = lib.ptxvcap_init(cupti_path().encode(), str(MODULE_DIR).encode())
            if rc != 0:
                why = {
                    -1: "libcupti could not be loaded",
                    -2: "libcuda could not be loaded",
                    39: "another CUPTI subscriber is active (a profiler?); run under "
                    "`python -m ptxv run` with ptxv imported first",
                }.get(rc, f"cuptiSubscribe failed ({rc})")
                raise CaptureUnavailable(why)
        except CaptureUnavailable as e:
            _init_error = str(e)
            raise
        _lib = lib
        return lib


def modules() -> list[str]:
    """Hashes of the module images this process loaded since init (files in MODULE_DIR)."""
    lib = init()
    n = lib.ptxvcap_modules(None, 0)
    buf = ctypes.create_string_buffer(n)
    lib.ptxvcap_modules(buf, n)
    return [h for h in buf.value.decode().split() if h]


def cuda_initialized() -> bool:
    """Whether torch (if imported) has initialized CUDA in this process."""
    t = sys.modules.get("torch")
    try:
        return bool(t is not None and t.cuda.is_initialized())
    except Exception:
        return False
