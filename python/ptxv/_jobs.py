# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Jobs: what the CLI runs for each kind, the future that tracks it, the local backend.

A job is (kind, case directories, arguments). `run_job` turns it into one `ptxv` command
with `--json`, runs it under a memory cap (`systemd-run --user --scope -p MemoryMax=...`),
streams its stderr as progress, and returns the parsed JSON document. Child processes are
tracked by PID (their own session); cancelling kills that process group, nothing else.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import _binary
from ._strict_inputs import (
    copy_strict_case,
    copy_strict_executable,
    resolve_executable,
    STRICT_MEMORY_CONTRACT,
)
from ._util import cache_dir

KINDS = ("check", "compare", "equiv", "errdist")


class PtxvError(RuntimeError):
    """A ptxv command failed (its stderr tail is in the message)."""


class JobCancelled(cf.CancelledError):
    pass


@dataclass
class JobSpec:
    kind: str
    cases: list[str]
    args: dict = field(default_factory=dict)
    _input_snapshots: list = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    _binary_snapshot: Any = field(default=None, init=False, repr=False, compare=False)
    _input_users: int = field(default=0, init=False, repr=False, compare=False)
    _input_lock: Any = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )
    _original_cases: list[str] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown job kind {self.kind!r} (one of {KINDS})")
        if "strict" in self.args:
            self.args = dict(self.args)
            if self.args.pop("strict"):
                if self.kind != "check":
                    raise ValueError("strict=True is supported only for check jobs")
                self.args.update(strict_memory_safety=True, nonfinite="strict")
        self.cases = [str(Path(c).resolve()) for c in self.cases]
        self._original_cases = list(self.cases)
        if self.kind == "check" and self.args.get("strict_memory_safety"):
            # Previous Python versions ignored this request even in cached jobs.
            self.args = {
                **self.args,
                "_strict_memory_safety_contract": STRICT_MEMORY_CONTRACT,
            }

    def own_strict_inputs(self) -> None:
        with self._input_lock:
            if (
                self.kind != "check"
                or not self.args.get("strict_memory_safety")
                or self._input_snapshots
            ):
                return
            owners = [copy_strict_case(case) for case in self._original_cases]
            self.cases = [owner.name for owner in owners]
            self._input_snapshots = owners

    def bind_strict_binary(self, binary: str | os.PathLike) -> None:
        with self._input_lock:
            if (
                self.kind == "check"
                and self.args.get("strict_memory_safety")
                and self._binary_snapshot is None
            ):
                self._binary_snapshot = copy_strict_executable(binary)

    @property
    def bound_binary(self) -> str | None:
        with self._input_lock:
            return self._binary_snapshot.name if self._binary_snapshot else None

    @property
    def bound_binary_hash(self) -> str | None:
        with self._input_lock:
            return self._binary_snapshot.digest if self._binary_snapshot else None

    def retain_inputs(self) -> None:
        with self._input_lock:
            self.own_strict_inputs()
            if self._input_snapshots or self._binary_snapshot is not None:
                self._input_users += 1

    def release_inputs(self) -> None:
        with self._input_lock:
            if not self._input_users:
                return
            self._input_users -= 1
            if self._input_users:
                return
            owners, self._input_snapshots = self._input_snapshots, []
            if self._binary_snapshot is not None:
                owners.append(self._binary_snapshot)
                self._binary_snapshot = None
            self.cases = list(self._original_cases)
        for owner in owners:
            owner.cleanup()


class Job(cf.Future):
    """A `concurrent.futures.Future` for one ptxv job, plus progress and cancellation.

    `progress` is the last progress line ptxv printed; `log` keeps every stderr line.
    `cancel()` also stops a running job (its process group is killed).
    """

    def __init__(
        self, spec: JobSpec, key: str | None = None, *, _own_inputs: bool = True
    ) -> None:
        super().__init__()
        self.spec = spec
        self._input_spec: JobSpec | None = None
        self._input_lock = threading.Lock()
        if _own_inputs:
            self.use_spec(spec)
        self.key = key
        self.id: str = key[:16] if key else hex(id(self))[2:]
        self.progress: str = ""
        self.log: list[str] = []
        self.cached = False
        self._kill: Callable[[], None] | None = None
        self._listeners: list[Callable[[str], None]] = []
        self._before_result: list[Callable[[Any], None]] = []

    def use_spec(self, spec: JobSpec) -> None:
        with self._input_lock:
            if self._input_spec is spec:
                return
            spec.retain_inputs()
            previous = self._input_spec
            self.spec = spec
            self._input_spec = (
                spec if spec._input_snapshots or spec.bound_binary is not None else None
            )
        if previous is not None:
            previous.release_inputs()

    def release_inputs(self) -> None:
        with self._input_lock:
            spec, self._input_spec = self._input_spec, None
        if spec is not None:
            spec.release_inputs()

    def set_result(self, result: Any) -> None:
        try:
            super().set_result(result)
        finally:
            if self.done():
                self.release_inputs()

    def set_exception(self, exception: BaseException) -> None:
        try:
            super().set_exception(exception)
        finally:
            if self.done():
                self.release_inputs()

    def add_progress_callback(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)

    def before_result(self, fn: Callable[[Any], None]) -> None:
        """Run `fn(result)` inside `set_result`, before any waiter wakes or done callback
        runs: what it does (the result cache write) is complete when `result()` returns.
        Done callbacks run only after the waiters are woken. Register before the backend
        can finish the job."""
        self._before_result.append(fn)

    def set_result(self, result: Any) -> None:
        if not self.done():
            for fn in self._before_result:
                try:
                    fn(result)
                except Exception:
                    pass
        super().set_result(result)

    def _line(self, line: str) -> None:
        self.progress = line
        self.log.append(line)
        if len(self.log) > 5000:
            del self.log[:1000]
        for fn in self._listeners:
            try:
                fn(line)
            except Exception:
                pass

    def cancel(self) -> bool:
        if super().cancel():
            self.release_inputs()
            return True
        if self.running() and self._kill:
            self._kill()
            return True
        return False

    def __repr__(self) -> str:
        st = (
            "cached"
            if self.cached
            else "done"
            if self.done()
            else "running"
            if self.running()
            else "pending"
        )
        p = (
            f", progress={self.progress[-60:]!r}"
            if self.progress and not self.done()
            else ""
        )
        return f"Job({self.spec.kind}, {self.id}, {st}{p})"


def argv(spec: JobSpec, work: Path) -> tuple[list[str], dict[str, Path]]:  # noqa: C901 (one flag mapping per job kind)
    """The ptxv arguments for a job, and the files it writes besides stdout."""
    a = spec.args
    files: dict[str, Path] = {}
    if spec.kind == "check":
        (case,) = spec.cases
        cmd = ["check", case, "--json", "--out", str(work / "out")]
        cmd += [
            "--variant",
            a.get("variant", "good"),
            "--front",
            a.get("front", "ptx" if a.get("strict_memory_safety") else "sass"),
        ]
        cmd += ["--max-states", str(a.get("max_states", 200_000))]
        if a.get("strict_memory_safety"):
            cmd.append("--strict-memory-safety")
        if a.get("certify", True):
            cmd.append("--certify")
        if a.get("slices"):
            cmd.append("--slices")
        # Refuted assumptions the case may rely on (e.g. "E-POLL-ACQ"; named in the verdict).
        for r in a.get("assume_refuted") or []:
            cmd += ["--assume-refuted", r]
        # gb300 (default True): False passes `--no-gb300`, which turns off both GB300 checks:
        # the hardware-model classification of races / stale reads and the GB300-profile
        # re-run of the two-level verdict (no `two_level`; the row says why in
        # `two_level_skipped`).
        if a.get("gb300", True) is False:
            cmd.append("--no-gb300")
        # NaN / Inf scan of the outputs (True, or "strict": a finding even with non-finite
        # inputs).
        nf = a.get("nonfinite")
        if nf == "strict":
            cmd.append("--nonfinite-strict")
        elif nf:
            cmd.append("--nonfinite")
        return cmd, files
    if spec.kind in ("compare", "equiv"):
        files["bound"] = work / "bound.json"
        if spec.kind == "compare":
            ref, cand = spec.cases
            cmd = ["compare", ref, cand, "--json", "--out", str(work / "compare")]
            cmd += ["--max-states", str(a.get("max_states", 200_000))]
            cmd += ["--trials", str(a.get("trials", 20))]
            for k in ("function", "ref_label", "cand_label"):
                if a.get(k):
                    cmd += [f"--{k.replace('_', '-')}", str(a[k])]
            if a.get("segments") is not None:
                cmd += ["--segments", "true" if a["segments"] else "false"]
            files["report_md"] = work / "compare" / "report.md"
        else:
            (toml,) = spec.cases
            cmd = ["equiv", toml, "--json", "--out", str(work / "equiv"), "--no-hw"]
            if a.get("trials") is not None:
                cmd += ["--trials", str(a["trials"])]
        cmd += ["--emit-bound", str(files["bound"])]
        return cmd, files
    if spec.kind == "errdist":
        from . import _errdist_adapter

        files["doc"] = work / _errdist_adapter.DOC_FILE
        return _errdist_adapter.argv(spec, work), files
    raise ValueError(spec.kind)


def mem_cap_prefix(mem_cap: str | None) -> list[str]:
    """`systemd-run --user --scope` with MemoryMax / no swap, when available."""
    if not mem_cap:
        return []
    if not shutil.which("systemd-run"):
        return []
    return [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        f"MemoryMax={mem_cap}",
        "-p",
        "MemorySwapMax=0",
        "--",
    ]


def run_job(
    spec: JobSpec,
    work: Path,
    *,
    binary: str | os.PathLike | None = None,
    mem_cap: str | None = "64G",
    threads: int | None = None,
    on_line: Callable[[str], None] | None = None,
    on_pid: Callable[[int | None], None] | None = None,
    timeout: float | None = None,
) -> dict:
    """Keep copied inputs owned until the subprocess and output readers have stopped."""
    spec.retain_inputs()
    try:
        if spec.kind == "check" and spec.args.get("strict_memory_safety"):
            if spec.bound_binary is None:
                selected = binary if binary is not None else _binary.find().resolve()
                spec.bind_strict_binary(selected)
            binary = spec.bound_binary
        document = _run_job(
            spec,
            work,
            binary=binary,
            mem_cap=mem_cap,
            threads=threads,
            on_line=on_line,
            on_pid=on_pid,
            timeout=timeout,
        )
        if spec.bound_binary_hash is not None:
            document["ptxv_binary_sha256"] = spec.bound_binary_hash
        return document
    finally:
        spec.release_inputs()


def _run_job(  # noqa: C901 (spawn/pump/timeout/kill/parse kept together for cancellation semantics)
    spec: JobSpec,
    work: Path,
    *,
    binary: str | os.PathLike | None = None,
    mem_cap: str | None = "64G",
    threads: int | None = None,
    on_line: Callable[[str], None] | None = None,
    on_pid: Callable[[int | None], None] | None = None,
    timeout: float | None = None,
) -> dict:
    """Run one job to completion in `work`; returns the JSON document (plus side files)."""
    spec.own_strict_inputs()
    work.mkdir(parents=True, exist_ok=True)
    bin_ = str(binary or _binary.find())
    args, files = argv(spec, work)
    cmd = mem_cap_prefix(mem_cap) + [bin_] + args
    env = dict(os.environ)
    env.setdefault("PTXV_ROOT", str(work))  # default out/ dirs land in the job dir
    if spec.kind in ("check", "compare", "equiv", "errdist") and not spec.args.get(
        "gpu"
    ):
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only verification
    if threads:
        env["PTXV_THREADS"] = str(threads)
    (work / "cmd.json").write_text(json.dumps(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=work,
        start_new_session=True,
    )
    try:
        if on_pid:
            on_pid(proc.pid)
        err_lines: list[str] = []

        def pump() -> None:
            buf = b""
            assert proc.stderr is not None
            with open(work / "stderr.log", "wb") as log:
                for chunk in iter(lambda: proc.stderr.read1(4096), b""):  # type: ignore[union-attr]
                    log.write(chunk)
                    buf += chunk
                    *lines, buf = buf.replace(b"\r", b"\n").split(b"\n")
                    for ln in lines:
                        s = ln.decode(errors="replace").rstrip()
                        if s:
                            err_lines.append(s)
                            if on_line:
                                on_line(s)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_group(proc.pid)
            proc.wait()
            raise PtxvError(f"ptxv {spec.kind} timed out after {timeout}s")
        finally:
            t.join(timeout=5)
            if on_pid:
                on_pid(None)
        (work / "stdout.json").write_bytes(out)
        if proc.returncode < 0 or proc.returncode == 137:
            raise JobCancelled(f"ptxv {spec.kind} killed (signal {-proc.returncode})")
        doc = None
        if "doc" in files and files["doc"].exists():
            doc = json.loads(files["doc"].read_text())
        for line in reversed(
            out.decode(errors="replace").splitlines() if doc is None else []
        ):
            if line.startswith("{"):
                try:
                    doc = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
        if doc is None:
            tail = "\n".join(err_lines[-15:])
            raise PtxvError(
                f"ptxv {spec.kind} failed (exit {proc.returncode}):\n{tail}"
            )
        if (
            "error" in doc and len(doc) <= 5
        ):  # {"error", "command", "schema", "ptxv", "ptxv_rev"}
            raise PtxvError(f"ptxv {spec.kind}: {doc['error']}")
        doc["exit_code"] = proc.returncode
        if "bound" in files and files["bound"].exists():
            doc["bound"] = json.loads(files["bound"].read_text())
        if "report_md" in files and files["report_md"].exists():
            doc["report_md"] = files["report_md"].read_text()
        return doc
    finally:
        if proc.poll() is None:
            kill_group(proc.pid)
            proc.wait()


def kill_group(pid: int, grace: float = 3.0) -> None:
    """SIGTERM, then SIGKILL, to the process group of a child we started (by PID)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        t0 = time.time()
        while time.time() - t0 < grace:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)


class LocalBackend:
    """Runs jobs as `ptxv` subprocesses on this machine, `workers` at a time.

    `mem_cap` (systemd MemoryMax, e.g. "64G"; None: no cap) applies to each job; `threads`
    sets PTXV_THREADS per job (default: cores / workers).
    """

    name = "local"

    def __init__(
        self,
        workers: int = 2,
        mem_cap: str | None = "64G",
        threads: int | None = None,
        binary: str | os.PathLike | None = None,
        root: str | os.PathLike | None = None,
    ) -> None:
        self.workers = workers
        self.mem_cap = mem_cap
        self.threads = threads or max(1, (os.cpu_count() or 2) // workers)
        self.binary = binary
        self.root = Path(root) if root else cache_dir("jobs")
        self._pool = cf.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ptxv-job"
        )
        self._pids: dict[str, int] = {}
        self._lock = threading.Lock()

    def prepare_spec(self, spec: JobSpec) -> None:
        spec.own_strict_inputs()
        if (
            spec.kind == "check"
            and spec.args.get("strict_memory_safety")
            and spec.bound_binary is None
        ):
            selected = (
                self.binary if self.binary is not None else _binary.find().resolve()
            )
            spec.bind_strict_binary(selected)

    def ptxv_hash(self) -> str:
        selected = (
            resolve_executable(self.binary)
            if self.binary is not None
            else _binary.find().resolve()
        )
        return _binary.binary_hash(selected)

    def ptxv_hash_for(self, spec: JobSpec) -> str:
        self.prepare_spec(spec)
        return spec.bound_binary_hash or self.ptxv_hash()

    def pids(self) -> dict[str, int]:
        """Running jobs' process IDs (job id -> pid)."""
        with self._lock:
            return dict(self._pids)

    def submit(self, spec: JobSpec, job: Job) -> Job:
        try:
            return self._submit(spec, job)
        except BaseException:
            job.release_inputs()
            raise

    def _submit(self, spec: JobSpec, job: Job) -> Job:
        self.prepare_spec(spec)
        job.use_spec(spec)
        binary = spec.bound_binary
        if spec._input_snapshots:
            self.root.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=f"{job.id}-", dir=self.root))
        else:
            work = self.root / f"{job.id}-{int(time.time() * 1000)}"

        def set_pid(pid: int | None) -> None:
            with self._lock:
                if pid is None:
                    self._pids.pop(job.id, None)
                else:
                    self._pids[job.id] = pid

        def kill() -> None:
            pid = self.pids().get(job.id)
            if pid:
                kill_group(pid)

        job._kill = kill

        def go() -> None:
            try:
                if not job.set_running_or_notify_cancel():
                    job.release_inputs()
                    return
                try:
                    r = run_job(
                        spec,
                        work,
                        binary=binary if binary is not None else self.binary,
                        mem_cap=self.mem_cap,
                        threads=self.threads,
                        on_line=job._line,
                        on_pid=set_pid,
                        timeout=spec.args.get("timeout"),
                    )
                    r["work_dir"] = str(work)
                    job.set_result(r)
                except Exception as e:  # noqa: BLE001
                    job.set_exception(e)
                except BaseException as e:
                    # KeyboardInterrupt / SystemExit: still fail the job (waiters never hang), then propagate.
                    job.set_exception(e)
                    raise
            finally:
                spec.release_inputs()

        spec.retain_inputs()
        try:
            execution = self._pool.submit(go)
        except BaseException:
            spec.release_inputs()
            raise
        if spec._input_snapshots or binary is not None:

            def canceled(future: cf.Future) -> None:
                if future.cancelled():
                    spec.release_inputs()
                    job.cancel()

            def cancel_execution(future: cf.Future) -> None:
                if future.cancelled():
                    execution.cancel()

            execution.add_done_callback(canceled)
            job.add_done_callback(cancel_execution)
        return job

    def shutdown(self, cancel: bool = True) -> None:
        if cancel:
            for pid in self.pids().values():
                kill_group(pid)
        self._pool.shutdown(wait=False, cancel_futures=cancel)

    def __repr__(self) -> str:
        return f"LocalBackend(workers={self.workers}, mem_cap={self.mem_cap!r}, threads={self.threads})"


def result_from_cache(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def store_cache(path: Path, result: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # One temporary name per process and thread: two jobs of one key finishing together
    # (a resubmitted job and its first run) must not rename each other's file away.
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
