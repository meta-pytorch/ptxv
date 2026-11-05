# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Slurm backend: run a job on a Slurm CPU cluster with `tools/ptxv.sbatch`.

    SlurmBackend(host="login.cluster", root="/shared/me", ntasks=8,
                 extra_sbatch=["--partition=cpu", "--account=myacct"])
    Session("slurm://login.cluster?ntasks=16&dry_run=1")

Configuration: constructor arguments, else the environment, else neutral defaults.
    PTXV_SLURM_HOST         ssh host of the login node (unset: run sbatch locally)
    PTXV_SLURM_ROOT         absolute path of a directory shared by the cluster's nodes
                            (required, except for dry runs)
    PTXV_SLURM_SBATCH_ARGS  extra sbatch options, shell-split (partition, account, QoS, ...)
    PTXV_SLURM_SCRIPT       the batch script (default: ROOT/ptxv/tools/ptxv.sbatch)

Submission: the case directories are copied (rsync over ssh) to ROOT/ptxv-runs/cases/<sha>,
the job's files to ROOT/ptxv-runs/py/<job id>, and `sbatch tools/ptxv.sbatch <ptxv args>`
runs on the login node. ptxv.sbatch runs every rank of the distributed explorer
(docs/multinode.md); rank 0's stdout, the JSON document, goes to ROOT/ptxv-runs/<slurm id>.out.
The backend then polls `sacct` and reads that file (and copies back side files).

`dry_run=True` runs nothing: the job's result is the list of commands it would run. Remote
prerequisites (docs/multinode.md): the source at ROOT/ptxv and the x86 build at
ROOT/ptxv-target-x86.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ._jobs import argv, Job, JobSpec, PtxvError
from ._util import cache_dir, tree_hash


_ENV: Any = object()  # "not given": take the host from $PTXV_SLURM_HOST


class SlurmBackend:
    name = "slurm"

    def __init__(
        self,
        host: str | None = _ENV,  # type: ignore[assignment]
        root: str | None = None,
        *,
        ntasks: int = 8,
        cpus_per_task: int = 48,
        mem_per_cpu: str = "4G",
        time_limit: str = "24:00:00",
        extra_sbatch: list[str] | None = None,
        script: str | None = None,
        dry_run: bool = False,
        poll: float = 30.0,
    ) -> None:
        if host is _ENV:
            host = os.environ.get("PTXV_SLURM_HOST") or None
        root = root or os.environ.get("PTXV_SLURM_ROOT")
        if not root:
            if not dry_run:
                raise ValueError(
                    "SlurmBackend needs a directory shared by the cluster's nodes: "
                    "root=... or $PTXV_SLURM_ROOT"
                )
            root = "/path/to/shared"  # dry run: placeholder in the printed commands
        if not root.startswith("/"):
            raise ValueError(f"SlurmBackend root must be an absolute path: {root!r}")
        self.host, self.root = host, root.rstrip("/")
        self.ntasks, self.cpus, self.mem, self.time = (
            ntasks,
            cpus_per_task,
            mem_per_cpu,
            time_limit,
        )
        if extra_sbatch is None:
            extra_sbatch = shlex.split(os.environ.get("PTXV_SLURM_SBATCH_ARGS", ""))
        self.extra = list(extra_sbatch)
        self.script = (
            script
            or os.environ.get("PTXV_SLURM_SCRIPT")
            or f"{self.root}/ptxv/tools/ptxv.sbatch"
        )
        self.dry_run = dry_run
        self.poll = poll

    @classmethod
    def from_url(cls, url: str) -> "SlurmBackend":
        u = urlparse(url)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        kw: dict[str, Any] = {}
        for k in ("ntasks", "cpus_per_task"):
            if k in q:
                kw[k] = int(q[k])
        for k in ("mem_per_cpu", "time_limit", "root", "script"):
            if k in q:
                kw[k] = q[k]
        kw["dry_run"] = q.get("dry_run", "0") not in ("0", "false", "")
        return cls(host=u.hostname or None, **kw)

    @property
    def cacheable(self) -> bool:
        return not self.dry_run

    @property
    def cache_tag(self) -> str:
        """Results come from the cluster's own (x86) build: a separate cache namespace."""
        return f"slurm:{self.host}:{self.root}"

    # ------------------------------------------------------------------------------
    def _remote(self, cmd: list[str]) -> list[str]:
        return ["ssh", self.host, shlex.join(cmd)] if self.host else cmd

    def _copy(self, src: str, dst: str) -> list[str]:
        """rsync src/ to dst/ (remote when a host is set), creating dst."""
        tgt = f"{self.host}:{dst}/" if self.host else f"{dst}/"
        return ["rsync", "-a", "--mkpath", f"{src.rstrip('/')}/", tgt]

    def plan(self, spec: JobSpec, job_id: str) -> tuple[list[list[str]], dict]:
        """The commands that submit a job (copy cases and job files, sbatch), and the remote
        paths the result is read from."""
        cmds: list[list[str]] = []
        rcases = []
        for c in spec.cases:
            p = Path(c)
            d = p if p.is_dir() else p.parent
            sha = tree_hash(d)
            rd = f"{self.root}/ptxv-runs/cases/{sha}"
            cmds.append(self._copy(str(d), rd))
            rcases.append(rd if p.is_dir() else f"{rd}/{p.name}")
        rwork = f"{self.root}/ptxv-runs/py/{job_id}"
        stage = cache_dir("slurm") / job_id
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        rspec = JobSpec(spec.kind, [], dict(spec.args))
        rspec.cases = rcases  # remote paths: no local resolution
        args, files = argv(rspec, stage)
        args = [a.replace(str(stage), rwork) for a in args]
        if any(stage.iterdir()):  # job files (errdist spec) to the remote job dir
            cmds.append(self._copy(str(stage), rwork))
        ntasks = self.ntasks if spec.kind == "check" else 1
        sb = [
            "sbatch",
            "--parsable",
            f"--ntasks={ntasks}",
            f"--cpus-per-task={self.cpus}",
            f"--mem-per-cpu={self.mem}",
            f"--time={self.time}",
            f"--output={self.root}/ptxv-runs/%j.out",
            f"--export=ALL,PTXV_SLURM_ROOT={self.root}",
            *self.extra,
            self.script,
            *args,
        ]
        cmds.append(self._remote(["mkdir", "-p", rwork]))
        cmds.append(self._remote(sb))
        paths = {k: str(v).replace(str(stage), rwork) for k, v in files.items()}
        paths["stdout"] = f"{self.root}/ptxv-runs/%j.out"
        return cmds, paths

    def submit(self, spec: JobSpec, job: Job) -> Job:  # noqa: C901 (submit + follow thread share the job's remote paths)
        cmds, paths = self.plan(spec, job.id)

        def go() -> None:
            if not job.set_running_or_notify_cancel():
                return
            try:
                if self.dry_run:
                    for c in cmds:
                        job._line("dry run: " + shlex.join(c))
                    job.set_result(
                        {
                            "dry_run": True,
                            "commands": [shlex.join(c) for c in cmds],
                            "paths": paths,
                        }
                    )
                    return
                for c in cmds[:-1]:
                    subprocess.run(c, check=True, capture_output=True)
                r = subprocess.run(cmds[-1], check=True, capture_output=True, text=True)
                sid = r.stdout.strip().split(";")[0]
                job._line(f"slurm job {sid} submitted")
                job._kill = lambda: subprocess.run(
                    self._remote(["scancel", sid]), capture_output=True
                )
                state = ""
                while True:
                    s = subprocess.run(
                        self._remote(["sacct", "-j", sid, "-n", "-X", "-o", "State"]),
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    if s != state:
                        state = s
                        job._line(f"slurm job {sid}: {s or 'pending'}")
                    if s.split() and s.split()[0] in (
                        "COMPLETED",
                        "FAILED",
                        "CANCELLED",
                        "TIMEOUT",
                        "OUT_OF_MEMORY",
                        "NODE_FAIL",
                    ):
                        break
                    time.sleep(self.poll)
                out = subprocess.run(
                    self._remote(["cat", paths["stdout"].replace("%j", sid)]),
                    capture_output=True,
                    text=True,
                ).stdout
                doc = next(
                    (
                        json.loads(ln)
                        for ln in reversed(out.splitlines())
                        if ln.startswith("{")
                    ),
                    None,
                )
                if "doc" in paths:
                    doc = json.loads(
                        subprocess.run(
                            self._remote(["cat", paths["doc"]]),
                            capture_output=True,
                            text=True,
                        ).stdout
                    )
                if doc is None:
                    raise PtxvError(
                        f"slurm job {sid} ({state}) produced no JSON document"
                    )
                if "bound" in paths:
                    b = subprocess.run(
                        self._remote(["cat", paths["bound"]]),
                        capture_output=True,
                        text=True,
                    )
                    if b.returncode == 0 and b.stdout.startswith("{"):
                        doc["bound"] = json.loads(b.stdout)
                doc["slurm_job"] = sid
                job.set_result(doc)
            except Exception as e:  # noqa: BLE001
                job.set_exception(e)
            except BaseException as e:
                # KeyboardInterrupt / SystemExit: still fail the job (waiters never hang), then propagate.
                job.set_exception(e)
                raise

        threading.Thread(target=go, daemon=True).start()
        return job

    def __repr__(self) -> str:
        return f"SlurmBackend(host={self.host!r}, root={self.root!r}, ntasks={self.ntasks}, dry_run={self.dry_run})"
