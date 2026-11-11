# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor metadata for CUDA-injection captures (tools/ptxvcap-run).

An injection capture sees launches and memory, not Python: every buffer of its cases would be
u8 bytes. This log records, at each CuTeDSL or Triton kernel call, the torch tensors passed to
it (directly, or as CuTe tensors made by `cutlass.cute.runtime.from_dlpack`): argument name,
storage start and size, data pointer, ptxv dtype, shape and whether the tensor covers its
storage. One JSON line per call, `{"t_us", "call", "tensors": [...]}`, in
`<PTXV_CAPTURE_DIR>/tensors.jsonl` (`%r` as ptxvcap.c expands it). The staging step
(ptxvcap_cases.py) gives a launch the tensors of the latest call before it, so buffers get the
tensors' dtype and shape (`ptxv num` then reads bf16 / f16 / f32 outputs as such).

Standard library only, loaded by python/ptxv/_site/sitecustomize.py without the ptxv package.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
import weakref

DT = {
    "torch.bfloat16": "bf16",
    "torch.float16": "f16",
    "torch.float32": "f32",
    "torch.float64": "f64",
    "torch.int32": "i32",
    "torch.int64": "i64",
    "torch.int16": "i16",
    "torch.int8": "i8",
    "torch.uint8": "u8",
    "torch.bool": "u8",
    "torch.uint16": "u16",
    "torch.uint32": "u32",
    "torch.uint64": "u64",
    "torch.float8_e4m3fn": "e4m3",
}

_lock = threading.Lock()
_path: str | None = None
_calls = 0
# CuTe tensors made by from_dlpack: id -> (weak reference, torch tensor info).
_cute: dict[int, tuple[object, dict]] = {}


def log_path(capture_dir: str) -> str:
    """`<capture dir>/tensors.jsonl`, `%r` as ptxvcap.c expands it ($RANK, else p<pid>)."""
    r = os.environ.get("RANK")
    d = capture_dir.replace("%r", r if r is not None else f"p{os.getpid()}")
    return os.path.join(d, "tensors.jsonl")


def tensor_info(t) -> dict | None:
    """A CUDA torch tensor's storage, dtype and shape; None for anything else."""
    try:
        if not getattr(t, "is_cuda", False):
            return None
        st = t.untyped_storage()
        p, n = st.data_ptr(), st.nbytes()
    except Exception:
        return None
    if not n:
        return None
    dt = DT.get(str(t.dtype))
    shape = [int(x) for x in t.shape]
    if dt is None:
        # Other dtypes (fp8 e5m2, fp4 packs): bytes, the last dimension in bytes.
        dt = "u8"
        if shape:
            shape[-1] *= t.element_size()
    whole = bool(
        t.is_contiguous()
        and t.storage_offset() == 0
        and t.numel() * t.element_size() == n
        and t.dim() > 0
    )
    return {
        "ptr": p,
        "nbytes": n,
        "data_ptr": int(t.data_ptr()),
        "dtype": dt,
        "shape": shape,
        "whole": whole,
    }


def _info_of(v) -> dict | None:
    i = tensor_info(v)
    if i is not None:
        return i
    e = _cute.get(id(v))
    if e is not None and e[0]() is v:
        return e[1]
    return None


def record(call: str, named) -> None:
    """One call's tensors: `named` is [(name, value)]; values that are not CUDA tensors (or
    CuTe tensors of them) are skipped."""
    global _calls
    if _path is None:
        return
    ts, seen = [], set()
    for n, v in named:
        i = _info_of(v)
        if i is None or (i["data_ptr"], i["dtype"]) in seen:
            continue
        seen.add((i["data_ptr"], i["dtype"]))
        ts.append(dict(i, name=n))
    if not ts:
        return
    with _lock:
        _calls += 1
        line = {
            "t_us": time.time_ns() // 1000,
            "call": call,
            "seq": _calls,
            "tensors": ts,
        }
        try:
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            with open(_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:
            pass


def _named_args(fn, args, kwargs):
    try:
        b = inspect.signature(fn).bind_partial(*args, **kwargs)
        return list(b.arguments.items())
    except (TypeError, ValueError):
        return [(f"arg{i}", a) for i, a in enumerate(args)] + list(kwargs.items())


# ------------------------------------------------------------------------- hooks
def _patch_runtime(mod) -> None:
    f = getattr(mod, "from_dlpack", None)
    if f is None or getattr(f, "_ptxv", False):
        return

    def from_dlpack(tensor, *a, **kw):
        r = f(tensor, *a, **kw)
        i = tensor_info(tensor)
        if i is not None:
            try:
                k = id(r)
                _cute[k] = (weakref.ref(r, lambda _, k=k: _cute.pop(k, None)), i)
            except TypeError:
                pass  # no weak reference: the CuTe tensor is not mapped (its buffer stays u8)
        return r

    from_dlpack._ptxv = True  # type: ignore[attr-defined]
    mod.from_dlpack = from_dlpack


def _patch_dsl(mod) -> None:
    cls = getattr(mod, "BaseDSL", None)
    f = getattr(cls, "_func", None)
    if f is None or getattr(f, "_ptxv", False):
        return

    def _func(self, funcBody, *args, **kwargs):
        record(
            getattr(funcBody, "__name__", "cute"), _named_args(funcBody, args, kwargs)
        )
        return f(self, funcBody, *args, **kwargs)

    _func._ptxv = True  # type: ignore[attr-defined]
    cls._func = _func


def _patch_executor(mod) -> None:
    for cname in ("JitCompiledFunction", "JitExecutor"):
        cls = getattr(mod, cname, None)
        f = getattr(cls, "__call__", None)
        if f is None or getattr(f, "_ptxv", False):
            continue

        def make(f, cname):
            def __call__(self, *args, **kwargs):
                name = getattr(self, "function_name", None) or cname
                record(
                    str(name),
                    [(f"arg{i}", a) for i, a in enumerate(args)] + list(kwargs.items()),
                )
                return f(self, *args, **kwargs)

            __call__._ptxv = True  # type: ignore[attr-defined]
            return __call__

        cls.__call__ = make(f, cname)


def _patch_triton(mod) -> None:
    cls = getattr(mod, "JITFunction", None)
    f = getattr(cls, "run", None)
    if f is None or getattr(f, "_ptxv", False):
        return

    def run(self, *args, **kwargs):
        names = list(getattr(self, "arg_names", []) or [])
        named = [
            (names[i] if i < len(names) else f"arg{i}", a) for i, a in enumerate(args)
        ]
        named += [(k, v) for k, v in kwargs.items() if k in names]
        record(getattr(self, "__name__", "triton"), named)
        return f(self, *args, **kwargs)

    run._ptxv = True  # type: ignore[attr-defined]
    cls.run = run


HOOKS = {
    "cutlass.cute.runtime": _patch_runtime,
    "cutlass.base_dsl.dsl": _patch_dsl,
    "cutlass.base_dsl.jit_executor": _patch_executor,
    "triton.runtime.jit": _patch_triton,
}


def install(capture_dir: str) -> None:
    """Record tensor metadata for the CuTeDSL / Triton calls of this process (hooks installed
    now for imported modules, at import for the others)."""
    global _path
    import importlib.abc
    import importlib.util

    _path = log_path(capture_dir)
    for name, patch in HOOKS.items():
        if name in sys.modules:
            patch(sys.modules[name])

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            patch = HOOKS.get(fullname)
            if patch is None or fullname in sys.modules:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def run(module):
                exec_module(module)
                try:
                    patch(module)
                except Exception as e:  # noqa: BLE001  (a hook must never break the target)
                    print(f"ptxv tensor log: {fullname}: {e}", file=sys.stderr)

            spec.loader.exec_module = run  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, Finder())


# ------------------------------------------------------------------------- staging side
def read_log(capture_dir) -> list[dict]:
    """The call records of a capture directory, in time order ([] without a log)."""
    p = os.path.join(str(capture_dir), "tensors.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return sorted(out, key=lambda r: (r["t_us"], r.get("seq", 0)))


def tensors_at(calls: list[dict], t_us: int) -> dict[str, dict]:
    """{name: tensor info} of the latest call recorded at or before a launch's time (the call
    that issued it); a name repeated with another tensor gets a `_<k>` suffix."""
    last = None
    for c in calls:
        if c["t_us"] > t_us:
            break
        last = c
    named: dict[str, dict] = {}
    for t in (last or {}).get("tensors", []):
        n, k = t["name"], 1
        while n in named:
            k += 1
            n = f"{t['name']}_{k}"
        named[n] = t
    return named
