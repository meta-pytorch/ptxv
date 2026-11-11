# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Building and loading the in-process capture library (`_native/ptxvcap.c`).

The library is compiled on first use with gcc against the CUDA toolkit's headers
(`$CUDA_HOME`, else /usr/local/cuda-13.0, else /usr/local/cuda) into
`$PTXV_CACHE/native/ptxvcap-<source hash>.so`. It links neither libcupti nor libcuda: the
libcupti already mapped in the process is used when there is one (torch's profiler loads it),
else `$PTXV_CUPTI`, else a CUPTI whose CUDA major version is at least the driver's (closest
first; torch/lib, next to Python, nvidia pip wheels, the toolkit), else the others in that order.
A CUPTI of a lower major than the driver (torch 2.8 cu129's libcupti.so.12 under a CUDA 13
driver) subscribes and then delivers no launch callbacks. A wheel's libcupti can subscribe and still deliver no
launch callbacks when it does not match the CUDA runtime torch uses (torch 2.8 cu129 with
nvidia-cuda-cupti installed): `launch_selftest` detects that.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import re
import subprocess
import sys
import threading
import warnings
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


def _torch_lib_dir() -> str | None:
    """torch's lib directory, without importing torch."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    d = os.path.join(os.path.dirname(spec.origin), "lib")
    return d if os.path.isdir(d) else None


def driver_major() -> int | None:
    """The CUDA driver's major version (cuDriverGetVersion), or None."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int(0)
        if cu.cuDriverGetVersion(ctypes.byref(v)) == 0 and v.value:
            return v.value // 1000
    except OSError:
        pass
    return None


def cupti_major(path: str) -> int | None:
    """The CUDA major version of a libcupti (from its `.so.N` name, or its symlink target's)."""
    for p in (path, os.path.realpath(path)):
        m = re.search(r"libcupti\.so\.(\d+)", os.path.basename(p))
        if (
            m and int(m.group(1)) < 1000
        ):  # the real files are named by year (.so.2025.x)
            return int(m.group(1))
    return None


_MAPPED_WARNING: str | None = None


def cupti_path() -> str:
    """The libcupti to use (see the module docstring)."""
    global _MAPPED_WARNING
    drv = driver_major()
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcupti.so" in line:
                    p = line.split()[-1]
                    mj = cupti_major(p)
                    if drv and mj and mj < drv:
                        _MAPPED_WARNING = (
                            f"the process already mapped {p} (CUPTI {mj}) but the driver is CUDA "
                            f"{drv}: launch callbacks may not arrive and the loaded library cannot be "
                            "replaced; import ptxv before torch (or run under `python -m ptxv run`), "
                            "or set PTXV_CUPTI to a CUPTI of the driver's major version"
                        )
                        warnings.warn(_MAPPED_WARNING, stacklevel=2)
                    return p
    except OSError:
        pass
    if os.environ.get("PTXV_CUPTI"):
        return os.environ["PTXV_CUPTI"]
    cands = sorted(
        glob.glob(os.path.join(_torch_lib_dir() or "/nonexistent", "libcupti.so*"))
    )
    cands += [
        os.path.join(sys.prefix, "lib", "libcupti.so.13"),
        os.path.join(sys.prefix, "lib", "libcupti.so"),
    ]
    for sp in sys.path:
        cands += glob.glob(os.path.join(sp, "nvidia", "cu13", "lib", "libcupti.so*"))
        cands += glob.glob(
            os.path.join(sp, "nvidia", "cuda_cupti", "lib", "libcupti.so*")
        )
    homes = []
    try:
        homes.append(cuda_home())
    except CaptureUnavailable:
        pass
    for h in (
        os.environ.get("PTXV_CUDA_HOME"),
        "/usr/local/cuda-13.0",
        "/usr/local/cuda",
    ):
        if h and Path(h).is_dir() and Path(h) not in homes:
            homes.append(Path(h))
    for cuda in homes:
        cands += glob.glob(str(cuda / "extras" / "CUPTI" / "lib64" / "libcupti.so*"))
        cands += glob.glob(str(cuda / "targets" / "*" / "lib" / "libcupti.so*"))
    cands = [c for c in dict.fromkeys(cands) if os.path.exists(c)]
    if not cands:
        raise CaptureUnavailable("libcupti not found (set PTXV_CUPTI)")
    # Launch callbacks arrive only from a CUPTI whose major version matches the driver's (a
    # CUPTI 12 under a CUDA 13 driver subscribes and then reports nothing): a CUPTI of at least
    # the driver's major version first, the closest major first, then the others in order.
    if drv:

        def rank(c: str) -> tuple[int, int]:
            mj = cupti_major(c)
            if mj is not None and mj >= drv:
                return (0, mj - drv)
            return (1 if mj is None else 2, 0)

        cands.sort(key=rank)
    return cands[0]


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
            lib.ptxvcap_set_snap.argtypes = [ctypes.c_int]
            lib.ptxvcap_set_snap.restype = ctypes.c_int
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


def launch_selftest() -> str | None:
    """With CUDA initialized by torch: record one torch kernel launch in a scratch region. None if
    its callback arrived (or the test cannot run), else a warning naming the libcupti in use."""
    import tempfile

    t = sys.modules.get("torch")
    if not cuda_initialized():
        return None
    lib = init()
    with tempfile.TemporaryDirectory(prefix="ptxv-selftest-") as d:
        if lib.ptxvcap_start(d.encode(), 0, 0, 0) != 0:
            return None
        try:
            x = t.ones(1, device="cuda")
            x.add_(1)
            t.cuda.synchronize()
        finally:
            g, s = ctypes.c_ulonglong(), ctypes.c_ulonglong()
            lib.ptxvcap_stop(ctypes.byref(g), ctypes.byref(s))
        p = Path(d) / "launches.jsonl"
        if p.exists() and p.stat().st_size > 0:
            return None
    return (
        f"CUPTI subscribed ({cupti_path()}) but a torch kernel launch produced no callback: "
        "this libcupti probably does not match the CUDA runtime in use; set PTXV_CUPTI to the "
        "libcupti torch ships (torch/lib) or the toolkit's"
    )
