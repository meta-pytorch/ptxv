# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`ptxv.capture()`: record the kernels a region of Python launches, in process.

    with ptxv.capture() as cap:
        y = torch.softmax(x, -1)
    cap.kernels                  # deduplicated by kernel and launch shape
    k = cap.kernels[0]           # or cap.kernels["softmax"] (regex), cap.select(op="aten::mm")
    k.check()                    # Verdict
    cap.all().case()             # every launch of the region as one case

Mechanism: CUPTI callbacks (`_native/ptxvcap.c`). Each launch records its function, grid,
block, cluster, PDL attribute and parameter bytes; the device allocations its pointer
parameters reach are copied before and after it (the context is synchronized around each
launch; results are unchanged). Module images come from CUPTI's module-load callback, which
`import ptxv` enables, so import ptxv before the program's first CUDA work. With torch, the
caching allocator's blocks (live before the region, allocated during it) give buffer
boundaries, and tensors bound to local variables of the calling frame give buffer names,
dtypes and shapes.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import _cupti
from ._util import cache_dir, short

_DT = {
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
}


def _size(s: str | int) -> int:
    if isinstance(s, int):
        return s
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)i?[bB]?", s.strip())
    if not m:
        raise ValueError(f"bad size {s!r}")
    return int(float(m.group(1)) * 1024 ** " kmgt".index(m.group(2).lower() or " "))


@dataclass
class TensorInfo:
    """A named tensor's storage at capture time (the tensor itself is not kept)."""

    name: str
    ptr: int  # storage start
    nbytes: int
    dtype: str  # ptxv dtype
    shape: list[int]
    whole: bool  # contiguous, offset 0, covering its storage
    data_ptr: int
    before: bool  # existed when the region started


def _tensor_info(name: str, t: Any, before: set[int]) -> TensorInfo | None:
    try:
        if not t.is_cuda:
            return None
        st = t.untyped_storage()
        p, n = st.data_ptr(), st.nbytes()
    except Exception:
        return None
    if not n:
        return None
    dt = _DT.get(str(t.dtype), "u8")
    whole = bool(
        t.is_contiguous()
        and t.storage_offset() == 0
        and t.numel() * t.element_size() == n
        and t.dim() > 0
    )
    return TensorInfo(name, p, n, dt, list(t.shape), whole, t.data_ptr(), p in before)


def _frame_tensors(frame: Any) -> dict[str, Any]:
    """Tensors bound to the frame's local variables (names starting with _ skipped)."""
    out = {}
    t = sys.modules.get("torch")
    if frame is None or t is None:
        return out
    for k, v in list(frame.f_locals.items()):
        if isinstance(v, t.Tensor) and not k.startswith("_"):
            out[k] = v
    return out


@dataclass(repr=False)
class Launch:
    """One recorded kernel launch."""

    index: int
    name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    cluster: tuple[int, int, int]
    smem: int
    pdl: bool
    stream: int
    tag: str
    capturing: bool
    snap: int | None
    record: dict
    capture: "Capture"

    @property
    def key(self) -> tuple:
        return (self.name, self.grid, self.block, self.cluster, self.smem)

    @property
    def short_name(self) -> str:
        return short_name(self.name)

    def __repr__(self) -> str:
        c = f", cluster={self.cluster}" if self.cluster != (1, 1, 1) else ""
        t = f", op={self.tag!r}" if self.tag else ""
        return f"Launch({self.index}: {short(self.short_name, 50)!r}, grid={self.grid}, block={self.block}{c}{t})"


def short_name(name: str) -> str:
    """A readable kernel name: the identifier of a mangled C++ name, else the name."""
    if not name.startswith("_Z"):
        return name

    # Itanium: _Z<len><name>... or _ZN<len><ns>...<len><name>[I...]E...: the function's name is
    # the first source name, or the last component of the nested name.
    def ident(i: int) -> tuple[str, int] | None:
        j = i
        while j < len(name) and name[j].isdigit():
            j += 1
        if j == i:
            return None
        n = int(name[i:j])
        return name[j : j + n], j + n

    i = 2
    if name[i : i + 1] == "N":
        i += 1
        while name[i : i + 1] in ("K", "V", "r"):
            i += 1
        last = None
        while (r := ident(i)) is not None:
            last, i = r
        return last or name
    r = ident(i)
    return r[0] if r else name


class Launches:
    """A sequence of launches that forms one case (one kernel, or a multi-kernel region).

    `case()` writes it (content-addressed, cached); `check()`, `compare()` and `errdist()`
    run ptxv on it through the current session.
    """

    def __init__(
        self, launches: Sequence[Launch], capture: "Capture", label: str = ""
    ) -> None:
        self.launches = list(launches)
        self.capture = capture
        self.label = label or (
            launches[0].short_name
            if len({la.name for la in launches}) == 1
            else f"{len(launches)} launches"
        )
        self._cases: dict[str, Any] = {}

    # -- case -----------------------------------------------------------------------
    def case(
        self,
        *,
        outputs: Sequence[str] | None = None,
        write_only: Sequence[str] = (),
        ptx: bool = False,
        prune: bool = True,
        rename: dict[str, str] | None = None,
    ):
        """Write (or reuse) this group as a ptxv case; returns a `ptxv.Case`.

        outputs: buffer names to compare (default: buffers the launches changed, and buffers
        allocated inside the region). write_only: outputs whose initial bytes the kernels do
        not read (they start as zeros instead of their pre-launch bytes). ptx: a PTX case
        when Triton's cache holds this kernel's PTX. prune: blank other kernels' code in
        module cubins. rename: buffer name -> name in the case.
        """
        from ._casewriter import write_case
        from .case import Case

        key = json.dumps(
            [outputs, list(write_only), ptx, prune, rename], sort_keys=True, default=str
        )
        if key not in self._cases:
            d = write_case(
                self.launches,
                self.capture,
                outputs=outputs,
                write_only=tuple(write_only) + tuple(self.capture.write_only),
                ptx=ptx,
                prune=prune,
                rename=rename,
                name=re.sub(r"[^\w.-]", "_", self.label)[:60],
            )
            self._cases[key] = Case(d)
        return self._cases[key]

    def case_dir(self) -> Path:
        return self.case().path

    def write(self, path: str | os.PathLike, **kw: Any) -> Path:
        """Copy the case to `path` (a directory the CLI and tools/ accept)."""
        src = self.case(**kw).path
        dst = Path(path)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return dst

    # -- verification ---------------------------------------------------------------
    def check(self, *, session: Any = None, wait: bool = True, **args: Any):
        """Execution verdict (`ptxv check --certify`); see `Session.check`."""
        from .session import session as cur

        return (session or cur()).check(self, wait=wait, **args)

    def compare(
        self, other: Any, *, session: Any = None, wait: bool = True, **args: Any
    ):
        """This kernel as the reference, `other` as the candidate; see `ptxv.compare`."""
        from .api import compare

        return compare(self, other, session=session, wait=wait, **args)

    def errdist(
        self, inputs: Any = None, *, ref: Any = None, session: Any = None, **args: Any
    ):
        """Error distribution over an input distribution; see `ptxv.errdist`."""
        from .api import errdist

        return errdist(self, inputs, ref=ref, session=session, **args)

    def __len__(self) -> int:
        return len(self.launches)

    def __iter__(self) -> Iterator[Launch]:
        return iter(self.launches)

    def __repr__(self) -> str:
        return f"Launches({self.label!r}, {len(self.launches)} launches)"


class Kernel(Launches):
    """A kernel with one launch shape; `launches` holds every recorded instance. Its case is
    the first instance that has snapshots."""

    def __init__(self, launches: Sequence[Launch], capture: "Capture") -> None:
        first = next((la for la in launches if la.snap is not None), launches[0])
        super().__init__([first], capture, first.short_name)
        self.instances = list(launches)
        self.name = first.name
        self.grid, self.block, self.cluster, self.smem = (
            first.grid,
            first.block,
            first.cluster,
            first.smem,
        )

    @property
    def ops(self) -> list[str]:
        return sorted({la.tag for la in self.instances if la.tag})

    def __repr__(self) -> str:
        c = f", cluster={self.cluster}" if self.cluster != (1, 1, 1) else ""
        n = f", launches={len(self.instances)}" if len(self.instances) > 1 else ""
        o = f", op={self.ops[0]!r}" if self.ops else ""
        return f"Kernel({short(self.label, 60)!r}, grid={self.grid}, block={self.block}{c}{n}{o})"


class KernelList(list):
    """`cap.kernels`: a list of Kernel with selection helpers."""

    def __getitem__(self, i):  # type: ignore[override]
        if isinstance(i, str):
            m = self.filter(name=i)
            if not m:
                raise KeyError(f"no kernel matches {i!r}")
            return m[0]
        r = super().__getitem__(i)
        return KernelList(r) if isinstance(i, slice) else r

    def filter(self, name: str | None = None, op: str | None = None) -> "KernelList":
        """Kernels whose (mangled or short) name matches the regex `name`, launched under an
        op whose name matches `op` (with `capture(ops=True)`)."""
        out = KernelList()
        for k in self:
            if name and not (re.search(name, k.name) or re.search(name, k.label)):
                continue
            if op and not any(re.search(op, o) for o in k.ops):
                continue
            out.append(k)
        return out


class Capture:
    """Context manager that records every kernel launched inside it (see module docstring).

    names: {name: tensor} naming buffers (default: the calling frame's tensor locals).
    snapshots: "all" (every launch), "first" (first launch of each kernel and shape) or
    "none". max_alloc: allocations larger than this are not copied. budget: total snapshot
    bytes. ops: tag launches with the torch op that issued them (a TorchDispatchMode; off by
    default because it changes how torch.compile traces). write_only: buffer names whose
    initial contents the kernels do not read.
    """

    _n = 0

    def __init__(
        self,
        names: dict[str, Any] | None = None,
        *,
        snapshots: str = "all",
        max_alloc: str | int = "1G",
        budget: str | int = "16G",
        ops: bool = False,
        write_only: Sequence[str] = (),
        frame_locals: bool = True,
    ) -> None:
        if snapshots not in ("all", "first", "none"):
            raise ValueError("snapshots: 'all', 'first' or 'none'")
        self.explicit = dict(names or {})
        self.snapshots = snapshots
        self.max_alloc = _size(max_alloc)
        self.budget = _size(budget)
        self.ops = ops
        self.write_only = tuple(write_only)
        self.frame_locals = frame_locals
        Capture._n += 1
        self.dir = (
            cache_dir("captures") / f"{os.getpid()}-{int(time.time())}-{Capture._n}"
        )
        self.launches: list[Launch] = []
        self.names: dict[str, TensorInfo] = {}
        self.blocks: list[tuple[int, int, str]] = []  # (addr, size, "pre" | "during")
        self.modules: list[str] = []
        self.warnings: list[str] = []
        self.graph_launches = 0
        self.skipped_snapshots = 0
        self._mode = None
        self._torch = None
        self._before_ptrs: set[int] = set()
        self._frame = None

    # -- context --------------------------------------------------------------------
    def __enter__(self) -> "Capture":
        lib = _cupti.init()
        self._torch = sys.modules.get("torch")
        self._frame = inspect.currentframe().f_back  # type: ignore[union-attr]
        t = self._torch
        if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
            t.cuda.synchronize()
            for seg in t.cuda.memory_snapshot():
                addr = seg["address"]
                for blk in seg["blocks"]:
                    if blk["state"] == "active_allocated":
                        self.blocks.append(
                            (blk.get("address", addr), blk["size"], "pre")
                        )
                    addr += blk["size"]
            try:
                t.cuda.memory._record_memory_history(
                    max_entries=1_000_000, context=None, stacks="python"
                )
                self._history = True
            except Exception:
                self._history = False
        else:
            self._history = False
        for v in list(_frame_tensors(self._frame).values()) + list(
            self.explicit.values()
        ):
            try:
                self._before_ptrs.add(v.untyped_storage().data_ptr())
            except Exception:
                pass
        self.dir.mkdir(parents=True, exist_ok=True)
        mode = {"none": 0, "all": 1, "first": 2}[self.snapshots]
        if (
            lib.ptxvcap_start(str(self.dir).encode(), mode, self.max_alloc, self.budget)
            != 0
        ):
            raise RuntimeError("ptxv.capture regions cannot nest")
        if self.ops and t is not None:
            self._mode = _op_tagger(t, lib)
            self._mode.__enter__()
        return self

    def __exit__(self, et: Any, ev: Any, tb: Any) -> None:  # noqa: C901 (teardown steps must run in order on the same state)
        import ctypes

        lib = _cupti.init()
        t = self._torch or sys.modules.get("torch")
        try:
            if self._mode is not None:
                self._mode.__exit__(et, ev, tb)
            if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
                t.cuda.synchronize()
        finally:
            g, s = ctypes.c_ulonglong(), ctypes.c_ulonglong()
            lib.ptxvcap_stop(ctypes.byref(g), ctypes.byref(s))
        self.graph_launches, self.skipped_snapshots = g.value, s.value
        if self._history and t is not None:
            try:
                snap = t.cuda.memory._snapshot()
                for tr in snap.get("device_traces", []):
                    for e in tr:
                        if e.get("action") == "alloc":
                            self.blocks.append((e["addr"], e["size"], "during"))
            finally:
                t.cuda.memory._record_memory_history(enabled=None)
        tensors = dict(_frame_tensors(self._frame)) if self.frame_locals else {}
        tensors.update(self.explicit)
        for n, v in tensors.items():
            ti = _tensor_info(n, v, self._before_ptrs)
            if ti and not any(o.ptr == ti.ptr and o.whole for o in self.names.values()):
                self.names[n] = ti
        self._frame = None
        self.modules = _cupti.modules()
        p = self.dir / "launches.jsonl"
        recs = [json.loads(la) for la in open(p)] if p.exists() else []
        for i, r in enumerate(recs):
            self.launches.append(
                Launch(
                    index=i,
                    name=r["name"],
                    grid=tuple(r["grid"]),
                    block=tuple(r["block"]),
                    cluster=tuple(r["cluster"]) if any(r["cluster"]) else (1, 1, 1),
                    smem=r["smem"],
                    pdl=bool(r.get("pdl")),
                    stream=r.get("stream", 0),
                    tag=r.get("tag", ""),
                    capturing=bool(r.get("capturing")),
                    snap=r.get("snap"),
                    record=r,
                    capture=self,
                )
            )
        if self.graph_launches:
            self.warnings.append(
                f"{self.graph_launches} CUDA graph launches: their kernels are not captured"
            )
        if self.skipped_snapshots:
            self.warnings.append(
                f"{self.skipped_snapshots} allocations or launches not snapshotted (max_alloc / budget)"
            )
        if len({la.stream for la in self.launches}) > 1:
            self.warnings.append(
                "launches on several streams: a case orders them as recorded"
            )
        if any(la.capturing for la in self.launches):
            self.warnings.append(
                "launches under CUDA graph capture: recorded without snapshots"
            )

    # -- views ----------------------------------------------------------------------
    @property
    def kernels(self) -> KernelList:
        """Distinct kernels (by name and launch shape), in order of first launch."""
        groups: dict[tuple, list[Launch]] = {}
        for la in self.launches:
            groups.setdefault(la.key, []).append(la)
        return KernelList(Kernel(v, self) for v in groups.values())

    def all(self) -> Launches:
        """Every launch of the region, in order, as one case."""
        return Launches(self.launches, self, f"region of {len(self.launches)} launches")

    def select(
        self,
        name: str | None = None,
        op: str | None = None,
        index: int | slice | Sequence[int] | None = None,
    ) -> Launches:
        """Launches by name regex, op regex, or index (an int, slice or list), in order."""
        ls = self.launches
        if index is not None:
            if isinstance(index, int):
                ls = [ls[index]]
            elif isinstance(index, slice):
                ls = ls[index]
            else:
                ls = [ls[i] for i in index]
        if name:
            ls = [
                la
                for la in ls
                if re.search(name, la.name) or re.search(name, la.short_name)
            ]
        if op:
            ls = [la for la in ls if la.tag and re.search(op, la.tag)]
        if not ls:
            raise KeyError("no launch matches")
        return Launches(ls, self)

    def __getitem__(self, i: int) -> Kernel:
        return self.kernels[i]

    def __len__(self) -> int:
        return len(self.launches)

    def __iter__(self) -> Iterator[Launch]:
        return iter(self.launches)

    def __repr__(self) -> str:
        w = f", warnings={len(self.warnings)}" if self.warnings else ""
        return f"Capture({len(self.launches)} launches, {len(self.kernels)} kernels{w})"


def capture(names: dict[str, Any] | None = None, **kw: Any) -> Capture:
    """Record the kernels launched inside a `with` block; see `Capture`."""
    return Capture(names, **kw)


def _op_tagger(torch: Any, lib: Any):
    """A TorchDispatchMode that sets the capture tag to the aten op being executed."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Tagger(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            lib.ptxvcap_set_tag(str(func.name()).encode())
            try:
                return func(*args, **(kwargs or {}))
            finally:
                lib.ptxvcap_set_tag(b"")

    return Tagger()
