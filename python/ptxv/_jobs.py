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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import _binary
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

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown job kind {self.kind!r} (one of {KINDS})")
        self.cases = [str(Path(c).resolve()) for c in self.cases]


class Job(cf.Future):
    """A `concurrent.futures.Future` for one ptxv job, plus progress and cancellation.

    `progress` is the last progress line ptxv printed; `log` keeps every stderr line.
    `cancel()` also stops a running job (its process group is killed).
    """

    def __init__(self, spec: JobSpec, key: str | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.key = key
        self.id: str = key[:16] if key else hex(id(self))[2:]
        self.progress: str = ""
        self.log: list[str] = []
        self.cached = False
        self._kill: Callable[[], None] | None = None
        self._listeners: list[Callable[[str], None]] = []

    def add_progress_callback(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)

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
            a.get("front", "sass"),
        ]
        cmd += ["--max-states", str(a.get("max_states", 200_000))]
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


def run_job(  # noqa: C901 (spawn/pump/timeout/kill/parse kept together for cancellation semantics)
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
        raise PtxvError(f"ptxv {spec.kind} failed (exit {proc.returncode}):\n{tail}")
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

    def pids(self) -> dict[str, int]:
        """Running jobs' process IDs (job id -> pid)."""
        with self._lock:
            return dict(self._pids)

    def submit(self, spec: JobSpec, job: Job) -> Job:
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
            if not job.set_running_or_notify_cancel():
                return
            try:
                r = run_job(
                    spec,
                    work,
                    binary=self.binary,
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

        self._pool.submit(go)
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
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
