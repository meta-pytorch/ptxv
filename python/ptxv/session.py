# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`ptxv.Session`: where verification jobs run, and the result cache.

    sess = ptxv.Session()                         # local subprocesses (default)
    sess = ptxv.Session("ptxvd://127.0.0.1:8765") # a ptxvd daemon (python -m ptxv.daemon)
    sess = ptxv.Session(ptxv.SlurmBackend(dry_run=True))

Results are cached by content: the job kind and arguments, the case directories' bytes
(cubins, launch, inputs) and the ptxv binary's hash. A repeated job returns the cached result
without running.
"""

from __future__ import annotations

import contextvars
import os
from pathlib import Path
from typing import Any, Iterable, Union

from . import _binary
from ._jobs import Job, JobSpec, LocalBackend, result_from_cache, store_cache
from ._util import cache_dir, job_key, tree_hash

_current: contextvars.ContextVar["Session | None"] = contextvars.ContextVar(
    "ptxv_session", default=None
)

CaseLike = Union[str, os.PathLike, Any]


def case_dir(x: CaseLike) -> Path:
    """A case directory for a path, a `ptxv.Case`, or captured launches (written on demand)."""
    if hasattr(x, "case_dir"):
        return Path(x.case_dir())
    p = Path(x)
    if not (p / "case.toml").exists() and not p.suffix == ".toml":
        raise FileNotFoundError(f"{p}: not a case directory (no case.toml)")
    return p


class Session:
    """Runs ptxv jobs on a backend and caches their results.

    backend: "local" (default), "ptxvd://HOST:PORT", or a backend object (`LocalBackend`,
    `DaemonBackend`, `SlurmBackend`). Local options (`workers`, `mem_cap`, `threads`) go to
    `LocalBackend`. `cache=False` disables the result cache.
    """

    def __init__(
        self,
        backend: Any = "local",
        *,
        cache: bool = True,
        workers: int = 2,
        mem_cap: str | None = "64G",
        threads: int | None = None,
    ) -> None:
        if isinstance(backend, str):
            if backend == "local":
                backend = LocalBackend(
                    workers=workers, mem_cap=mem_cap, threads=threads
                )
            elif backend.startswith("ptxvd://") or backend.startswith("http://"):
                from .daemon import DaemonBackend

                backend = DaemonBackend(backend)
            elif backend.startswith("slurm"):
                from .slurm import SlurmBackend

                backend = SlurmBackend.from_url(backend)
            else:
                raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.cache = cache
        self._token: contextvars.Token | None = None

    # -- context: `with Session(...):` makes it the default for Kernel.check() etc.
    def __enter__(self) -> "Session":
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    def ptxv_hash(self) -> str:
        h = getattr(self.backend, "ptxv_hash", None)
        if callable(h):
            return h()
        return _binary.binary_hash(_binary.find())

    def key(self, spec: JobSpec) -> str:
        hashes = [
            tree_hash(c) if Path(c).is_dir() else tree_hash(Path(c).parent)
            for c in spec.cases
        ]
        args = {k: v for k, v in spec.args.items() if k not in ("timeout",)}
        tag = getattr(self.backend, "cache_tag", "")
        if tag:
            args["_backend"] = tag
        return job_key(spec.kind, args, hashes, self.ptxv_hash())

    def submit(self, kind: str, cases: Iterable[CaseLike], **args: Any) -> Job:
        """Start a job; returns a `Job` (a `concurrent.futures.Future` of the JSON result)."""
        spec = JobSpec(kind, [str(case_dir(c)) for c in cases], args)
        cacheable = self.cache and getattr(self.backend, "cacheable", True)
        key = self.key(spec) if cacheable else None
        job = Job(spec, key)
        path = cache_dir("results") / f"{key}.json" if key else None
        if path is not None and path.exists():
            r = result_from_cache(path)
            if r is not None:
                job.cached = True
                job.set_running_or_notify_cancel()
                job.set_result(r)
                return job
        self.backend.submit(spec, job)
        if path is not None:

            def save(f: Job) -> None:
                if not f.cancelled() and f.exception() is None:
                    r = f.result()
                    if not r.get("dry_run"):
                        store_cache(path, r)

            job.add_done_callback(save)
        return job

    # -- typed front ends -------------------------------------------------------------
    def check(self, case: CaseLike, *, wait: bool = True, **args: Any):
        """Execution verdict of a case (`ptxv check --certify`): a `Verdict`, or a Job."""
        from .results import Verdict

        job = self.submit("check", [case], **args)
        return (
            Verdict.from_json(job.result()) if wait else _mapped(job, Verdict.from_json)
        )

    def compare(self, ref: CaseLike, cand: CaseLike, *, wait: bool = True, **args: Any):
        """Equivalence of two cases (`ptxv compare`): an `Equivalence`, or a Job."""
        from .results import Equivalence

        job = self.submit("compare", [ref, cand], **args)
        return (
            Equivalence.from_json(job.result())
            if wait
            else _mapped(job, Equivalence.from_json)
        )

    def close(self) -> None:
        sd = getattr(self.backend, "shutdown", None)
        if sd:
            sd()

    def __repr__(self) -> str:
        return f"Session({self.backend!r}, cache={self.cache})"


class _Mapped(Job):
    """A Job whose result is converted (JSON -> result object) when it completes."""


def _mapped(job: Job, fn) -> Job:
    out = _Mapped(job.spec, job.key)
    out.id = job.id
    out._kill = lambda: job.cancel()  # type: ignore[assignment]
    job.add_progress_callback(out._line)
    out.set_running_or_notify_cancel()

    def done(f: Job) -> None:
        if f.cancelled():
            out.set_exception(cf_cancelled())
        elif f.exception() is not None:
            out.set_exception(f.exception())
        else:
            try:
                out.set_result(fn(f.result()))
            except Exception as e:  # noqa: BLE001
                out.set_exception(e)
            except BaseException as e:
                # KeyboardInterrupt / SystemExit: still fail the job (waiters never hang), then propagate.
                out.set_exception(e)
                raise

    job.add_done_callback(done)
    out.cached = job.cached
    return out


def cf_cancelled() -> BaseException:
    from ._jobs import JobCancelled

    return JobCancelled("job cancelled")


def session() -> Session:
    """The current session (`with Session(...)`), else a process-wide local one."""
    s = _current.get()
    if s is None:
        global _default
        if _default is None:
            _default = Session()
        s = _default
    return s


_default: Session | None = None
