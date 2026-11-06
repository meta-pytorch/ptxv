# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Top-level functions: `ptxv.check`, `ptxv.compare`, `ptxv.errdist`."""

from __future__ import annotations

import functools
import subprocess
from typing import Any, Callable, Mapping

from . import _binary
from .capture import Capture, Launches
from .session import session as _session


def _flat_tensors(x: Any, prefix: str = "out") -> list[tuple[str, Any]]:
    if hasattr(x, "untyped_storage"):
        return [(prefix, x)]
    if isinstance(x, dict):
        return [t for k, v in x.items() for t in _flat_tensors(v, f"{prefix}_{k}")]
    if isinstance(x, (tuple, list)):
        return [t for i, v in enumerate(x) for t in _flat_tensors(v, f"{prefix}{i}")]
    return []


def capture_call(
    fn: Callable | str,
    inputs: Mapping[str, Any] | None = None,
    *,
    warmup: bool = True,
    kernels: str | None = None,
    **capture_kw: Any,
) -> Launches:
    """Capture what `fn(**inputs)` launches (`fn` may be a torch expression string over the
    inputs). It runs once first (lazy module loading, autotuning; inputs it changes in place
    are restored), then under capture. Outputs are named out, out0, ...; inputs by their keys.
    `kernels`: keep only launches whose name matches this regex."""
    import torch

    inputs = dict(inputs or {})
    if isinstance(fn, str):
        expr = fn
        import torch.nn.functional as F

        def fn(**kw: Any) -> Any:  # type: ignore[no-redef]
            return eval(expr, {"torch": torch, "F": F, "math": __import__("math")}, kw)

    if warmup:
        saved = {
            n: t.clone()
            for n, t in inputs.items()
            if hasattr(t, "is_cuda") and t.is_cuda
        }
        fn(**inputs)
        torch.cuda.synchronize()
        for n, t in saved.items():
            inputs[n].copy_(t)
        del saved
    cap = Capture(names=inputs, frame_locals=False, **capture_kw)
    with cap:
        res = fn(**inputs)
        for n, t in _flat_tensors(res):
            cap.explicit.setdefault(n, t)
    if not cap.launches:
        raise RuntimeError("no kernel launches recorded")
    return cap.select(name=kernels) if kernels else cap.all()


def _as_group(x: Any, inputs: Mapping[str, Any] | None, **kw: Any) -> Any:
    if (
        callable(x)
        and not isinstance(x, Launches)
        or (isinstance(x, str) and inputs is not None and not _is_path(x))
    ):
        if inputs is None:
            raise ValueError("a callable side needs `inputs`")
        return capture_call(x, inputs, **kw)
    return x


def _is_path(s: str) -> bool:
    import os

    return os.path.exists(s)


def _aligned(ref: Any, cand: Any) -> tuple[Any, Any]:
    """Rename the candidate's buffers so names match the reference's: inputs holding the same
    bytes get the reference's name, and the first output gets the reference's first output's
    name (equiv pairs buffers by name)."""
    if not (isinstance(ref, Launches) and isinstance(cand, Launches)):
        return ref, cand
    rc, cc = ref.case(), cand.case()
    rin = rc.inputs()
    ren: dict[str, str] = {}
    for n, a in cc.inputs().items():
        if n in cc.outputs:
            continue
        for m, b in rin.items():
            if m in rc.outputs or m in ren.values():
                continue
            if a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes():
                if m != n:
                    ren[n] = m
                break
    if cc.outputs and rc.outputs and cc.outputs[0] != rc.outputs[0]:
        ren[cc.outputs[0]] = rc.outputs[0]
    if not ren:
        return ref, cand
    taken = set(ren.values())
    for b in cc.buffers:  # move buffers out of the way of the new names
        if b["name"] in taken and b["name"] not in ren:
            ren[b["name"]] = b["name"] + "_cand"
    cand.case(rename=ren)  # validates
    return ref, _Renamed(cand, ren)


class _Renamed:
    """A group whose case is written with renamed buffers."""

    def __init__(self, g: Launches, rename: dict[str, str]) -> None:
        self.g, self.rename = g, rename

    def case_dir(self):
        return self.g.case(rename=self.rename).path


def compare(
    ref: Any,
    cand: Any,
    inputs: Mapping[str, Any] | None = None,
    *,
    session: Any = None,
    wait: bool = True,
    align: bool = True,
    kernels: str | None = None,
    **args: Any,
):
    """Equivalence of `cand` to `ref` (`ptxv compare`): an `Equivalence`, or a Job.

    Each side is a captured kernel or group (`cap.kernels[0]`, `cap.all()`), a `ptxv.Case` or
    case directory, or a callable / torch expression string, captured on `inputs` (both
    sides read the same tensors). `align` renames the candidate's buffers to match the
    reference's. Other keyword arguments go to `ptxv compare` (max_states, trials, segments,
    function, timeout).
    """
    r = _as_group(ref, inputs, kernels=kernels)
    c = _as_group(cand, inputs, kernels=kernels)
    if align:
        r, c = _aligned(r, c)
    return (session or _session()).compare(r, c, wait=wait, **args)


def check(x: Any, *, session: Any = None, wait: bool = True, **args: Any):
    """Execution verdict of a captured kernel, `ptxv.Case` or case directory."""
    return (session or _session()).check(x, wait=wait, **args)


@functools.lru_cache(maxsize=4)
def _has_errdist(binary: str) -> bool:
    try:
        r = subprocess.run(
            [binary, "errdist", "--help"], capture_output=True, timeout=30
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def errdist(
    target: Any,
    inputs: Any = None,
    *,
    ref: Any = None,
    session: Any = None,
    wait: bool = True,
    metric: str = "abs",
    seed: int = 1,
    mc_inputs: int | None = None,
    sim_trials: int | None = None,
    gpu_trials: int | None = None,
    per_output: bool = False,
    **args: Any,
):
    """Error distribution of `target` over an input distribution (`ptxv errdist`).

    inputs: {buffer name: ptxv.dist distribution}, or one distribution for every input.
    Alone: C - exact (the kernel against its own algorithm in exact arithmetic). With `ref`:
    C - R. metric: "abs", "u" (units of u S(x)) or "ulp". Returns an `ErrDist` (or a Job).
    """
    from . import _errdist_adapter
    from .dist import Dist

    s = session or _session()
    if getattr(s.backend, "name", "") == "local":
        b = str(s.backend.binary or _binary.find())
        if not _has_errdist(b):
            raise NotImplementedError(
                f"{b} has no `errdist` subcommand yet (agent T's ptxv errdist). "
                "For the distribution of the proven bound, use Equivalence.bound.distribution()."
            )
    if isinstance(inputs, Dist):
        a = {"default": inputs.spec(), "inputs": {}}
    else:
        a = {"inputs": {n: d.spec() for n, d in (inputs or {}).items()}}
    a.update(
        seed=seed,
        mc_inputs=mc_inputs,
        sim_trials=sim_trials,
        gpu_trials=gpu_trials,
        per_output=per_output,
    )
    a.update(args)
    a = {k: v for k, v in a.items() if v is not None}
    cases = [ref, target] if ref is not None else [target]
    job = s.submit("errdist", cases, **a)
    conv = functools.partial(_errdist_adapter.to_errdist, metric=metric)
    if wait:
        return conv(job.result())
    from .session import _mapped

    return _mapped(job, conv)


class ErrdistModel:
    """A compiled `ptxv errdist` model: `m(inputs) -> ErrDist` in seconds.

    The compile stage (simulating the kernel(s) and extracting their numerics programs) runs
    once per case and is cached by the case contents; each query runs Tier 0, Tier 2 and a
    small Tier 3 cross-check (300 inputs; `mc_inputs` overrides, 0 skips it) on the model,
    Tier 1 only when `tier1`, and is memoized per input distribution. The headline is Tier 0
    when it applies, else Tier 2 where its applicability check and the cross-check pass,
    else Tier 3 (`ErrDist.tier`, `ErrDist.crosscheck`). Local backend only.
    """

    def __init__(self, target: Any, *, ref: Any = None, binary: str | None = None):
        import hashlib
        import json

        from . import _errdist_adapter
        from ._util import cache_dir, tree_hash
        from .session import case_dir

        self.binary = str(binary or _binary.find())
        if not _has_errdist(self.binary):
            raise NotImplementedError(f"{self.binary} has no `errdist` subcommand")
        cases = [
            str(case_dir(c)) for c in ([ref, target] if ref is not None else [target])
        ]
        h = hashlib.sha256(
            json.dumps(
                [tree_hash(c) for c in cases] + [_binary.binary_hash(self.binary)]
            ).encode()
        ).hexdigest()[:20]
        self.work = cache_dir("errdist-models") / h
        self.work.mkdir(parents=True, exist_ok=True)
        self.path = self.work / "model.json"
        self.pair = ref is not None
        self._memo: dict = {}
        if not self.path.exists():
            target_s = (
                str(_errdist_adapter.equiv_toml(cases[0], cases[1], self.work))
                if self.pair
                else cases[0]
            )
            tmp = self.work / "model.json.tmp"
            self._run(
                _errdist_adapter.compile_argv(target_s, tmp)
                + ["--out", str(self.work / "scratch")]
            )
            tmp.replace(self.path)

    def _run(self, argv: list[str]) -> str:
        r = subprocess.run([self.binary, *argv], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"ptxv {' '.join(argv)} failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}"
            )
        return r.stdout

    def __call__(
        self,
        inputs: Any,
        *,
        metric: str = "abs",
        tier1: bool = False,
        mc_inputs: int | None = None,
        seed: int = 1,
    ):
        import hashlib
        import json

        from . import _errdist_adapter
        from .dist import Dist

        if isinstance(inputs, Dist):
            a = {"default": inputs.spec(), "inputs": {}}
        else:
            a = {"inputs": {n: d.spec() for n, d in (inputs or {}).items()}}
        a.update(seed=seed, tier1=tier1, mc_inputs=mc_inputs)
        key = hashlib.sha256(
            json.dumps(a, sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        if key not in self._memo:
            q = self.work / "q" / key
            doc_path = q / _errdist_adapter.DOC_FILE
            if not doc_path.exists():
                self._run(_errdist_adapter.query_argv(self.path, a, q))
            self._memo[key] = json.loads(doc_path.read_text())
        return _errdist_adapter.to_errdist(self._memo[key], metric=metric)


def errdist_model(
    target: Any, *, ref: Any = None, binary: str | None = None
) -> ErrdistModel:
    """Compile `target` (and `ref`: C - R) once; returns `m` with `m(inputs) -> ErrDist`."""
    return ErrdistModel(target, ref=ref, binary=binary)
