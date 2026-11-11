# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ptxvd: a small HTTP + JSON job daemon, and its client backend.

    python -m ptxv.daemon --port 8765 [--workers 4] [--mem-cap 64G] [--root DIR]
                          [--slurm-host LOGIN --slurm-root DIR --slurm-dry-run]

Binds 127.0.0.1 unless `--bind` says otherwise. Case bundles are uploaded once (keyed by
their content hash), jobs run on the local cores through `LocalBackend` (each under the
memory cap; child processes tracked by PID), results are cached by job key, and progress
streams as JSON lines. A job with `"backend": "slurm"` goes to the Slurm backend.

API (all JSON):
  GET    /v1/info                     version, ptxv binary hash, workers, running jobs
  HEAD   /v1/cases/<sha>              200 if the case is stored
  PUT    /v1/cases/<sha>              body: tar.gz of the case directory
  POST   /v1/jobs                     {"kind", "cases": [sha...], "args"} -> {"id", "cached"}
  GET    /v1/jobs/<id>                {"state", "progress", "result" | "error"}
  GET    /v1/jobs/<id>/events         newline-delimited JSON: progress lines, then the end
  DELETE /v1/jobs/<id>                cancel (kills the job's process group)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import _binary
from ._jobs import Job, JobSpec, LocalBackend, result_from_cache, store_cache
from ._util import cache_dir, job_key, tree_hash

VERSION = 1


def pack_case(path: str | os.PathLike) -> bytes:
    """tar.gz of a case directory's regular files (relative paths)."""
    buf = io.BytesIO()
    root = Path(path)
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.relative_to(root).parts[0] != "out":
                tf.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
    return buf.getvalue()


def unpack_case(data: bytes, dst: Path) -> None:
    """Extract only regular files with relative, non-escaping paths."""
    dst.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = Path(m.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"bad path in bundle: {m.name}")
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as f, open(out, "wb") as g:  # type: ignore[union-attr]
                shutil.copyfileobj(f, g)


class Daemon:
    def __init__(
        self,
        root: Path,
        workers: int = 2,
        mem_cap: str | None = "64G",
        threads: int | None = None,
        slurm: Any = None,
        binary: str | None = None,
    ) -> None:
        self.root = root
        (root / "cases").mkdir(parents=True, exist_ok=True)
        (root / "results").mkdir(parents=True, exist_ok=True)
        self.binary = binary or str(_binary.find())
        self.ptxv_hash = _binary.binary_hash(self.binary)
        self.local = LocalBackend(
            workers, mem_cap, threads, binary=self.binary, root=root / "jobs"
        )
        self.slurm = slurm
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def case_path(self, sha: str) -> Path:
        if not all(c in "0123456789abcdef" for c in sha) or len(sha) != 64:
            raise ValueError("bad case hash")
        return self.root / "cases" / sha

    def put_case(self, sha: str, data: bytes) -> None:
        dst = self.case_path(sha)
        if dst.exists():
            return
        tmp = dst.with_name(f"{sha}.{threading.get_ident()}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        unpack_case(data, tmp)
        got = tree_hash(tmp)
        if got != sha:
            shutil.rmtree(tmp)
            raise ValueError(f"case hash mismatch: {got}")
        tmp.rename(dst)

    def submit(self, body: dict) -> tuple[str, bool]:
        kind, shas, args = body["kind"], body["cases"], body.get("args", {})
        cases = [str(self.case_path(s)) for s in shas]
        for c in cases:
            if not Path(c).exists():
                raise FileNotFoundError(f"case {Path(c).name} not uploaded")
        backend = args.pop("backend", "local")
        key = job_key(
            kind,
            {k: v for k, v in args.items() if k != "timeout"},
            shas,
            self.ptxv_hash,
        )
        spec = JobSpec(kind, cases, args)
        job = Job(spec, key)
        res = self.root / "results" / f"{key}.json"
        with self.lock:
            if job.id in self.jobs and not self.jobs[job.id].done():
                return job.id, False  # the same job is running: share it
            r = result_from_cache(res) if res.exists() else None
            if r is not None:
                job.cached = True
                job.set_running_or_notify_cancel()
                job.set_result(r)
                self.jobs[job.id] = job
                return job.id, True
            self.jobs[job.id] = job
        if backend == "slurm":
            if self.slurm is None:
                raise ValueError("this daemon has no Slurm backend (--slurm-host)")
            self.slurm.submit(spec, job)
        else:
            self.local.submit(spec, job)

        def save(f: Job) -> None:
            if (
                not f.cancelled()
                and f.exception() is None
                and not f.result().get("dry_run")
            ):
                store_cache(res, f.result())

        job.add_done_callback(save)
        return job.id, False

    def status(self, jid: str) -> dict:
        job = self.jobs[jid]
        d: dict[str, Any] = {
            "id": jid,
            "kind": job.spec.kind,
            "progress": job.progress,
            "cached": job.cached,
        }
        if job.cancelled():
            d["state"] = "cancelled"
        elif job.done():
            e = job.exception()
            if e is None:
                d["state"], d["result"] = "done", job.result()
            else:
                d["state"], d["error"] = "error", f"{type(e).__name__}: {e}"
        else:
            d["state"] = "running" if job.running() else "queued"
        return d


def make_handler(d: Daemon):  # noqa: C901 (HTTP route table closed over the daemon)
    class H(BaseHTTPRequestHandler):
        server_version = "ptxvd/1"

        def log_message(self, fmt: str, *a: Any) -> None:
            if os.environ.get("PTXVD_VERBOSE"):
                super().log_message(fmt, *a)

        def _json(self, code: int, obj: Any) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(n) if n else b""

        def _err(self, e: BaseException) -> None:
            code = 404 if isinstance(e, (KeyError, FileNotFoundError)) else 400
            self._json(code, {"error": f"{type(e).__name__}: {e}"})

        def do_HEAD(self) -> None:  # noqa: N802
            p = self.path.split("/")
            ok = len(p) == 4 and p[2] == "cases" and d.case_path(p[3]).exists()
            self.send_response(200 if ok else 404)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            p = [x for x in self.path.split("/") if x]
            try:
                if p == ["v1", "info"]:
                    self._json(
                        200,
                        {
                            "version": VERSION,
                            "ptxv_hash": d.ptxv_hash,
                            "workers": d.local.workers,
                            "mem_cap": d.local.mem_cap,
                            "running": d.local.pids(),
                            "slurm": d.slurm is not None,
                        },
                    )
                elif len(p) == 3 and p[1] == "jobs":
                    self._json(200, d.status(p[2]))
                elif len(p) == 4 and p[1] == "jobs" and p[3] == "events":
                    self._events(p[2])
                else:
                    self._json(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._err(e)

        def _events(self, jid: str) -> None:
            job = d.jobs[jid]
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            sent = 0
            while True:
                log = job.log
                while sent < len(log):
                    self.wfile.write(
                        (json.dumps({"progress": log[sent]}) + "\n").encode()
                    )
                    sent += 1
                self.wfile.flush()
                if job.done():
                    st = d.status(jid)
                    st.pop("result", None)
                    self.wfile.write((json.dumps({"end": st}) + "\n").encode())
                    return
                time.sleep(0.2)

        def do_PUT(self) -> None:  # noqa: N802
            p = [x for x in self.path.split("/") if x]
            try:
                if len(p) == 3 and p[1] == "cases":
                    d.put_case(p[2], self._body())
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._err(e)

        def do_POST(self) -> None:  # noqa: N802
            p = [x for x in self.path.split("/") if x]
            try:
                if p == ["v1", "jobs"]:
                    jid, cached = d.submit(json.loads(self._body()))
                    self._json(200, {"id": jid, "cached": cached})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._err(e)

        def do_DELETE(self) -> None:  # noqa: N802
            p = [x for x in self.path.split("/") if x]
            try:
                if len(p) == 3 and p[1] == "jobs":
                    self._json(200, {"cancelled": d.jobs[p[2]].cancel()})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._err(e)

    return H


def serve(
    port: int = 8765,
    bind: str = "127.0.0.1",
    root: str | None = None,
    workers: int = 2,
    mem_cap: str | None = "64G",
    threads: int | None = None,
    slurm: Any = None,
    ready: Any = None,
) -> None:
    d = Daemon(
        Path(root) if root else cache_dir("ptxvd"), workers, mem_cap, threads, slurm
    )
    srv = ThreadingHTTPServer((bind, port), make_handler(d))
    srv.daemon_threads = True
    host, port = srv.server_address[:2]
    msg = json.dumps(
        {"ptxvd": f"ptxvd://{host}:{port}", "pid": os.getpid(), "root": str(d.root)}
    )
    print(msg, flush=True)
    if ready:
        ready(srv)
    try:
        srv.serve_forever()
    finally:
        d.local.shutdown()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m ptxv.daemon", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--port", type=int, default=8765, help="0: any free port (printed)")
    ap.add_argument(
        "--bind", default="127.0.0.1", help="address to bind (default: localhost only)"
    )
    ap.add_argument("--root", help="state directory (default: $PTXV_CACHE/ptxvd)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--mem-cap", default="64G", help='per-job MemoryMax ("none": no cap)'
    )
    ap.add_argument(
        "--threads", type=int, help="PTXV_THREADS per job (default: cores / workers)"
    )
    ap.add_argument(
        "--slurm-host", help="ssh host (login node) for the Slurm backend"
    )
    ap.add_argument(
        "--slurm-root",
        help="shared directory on the cluster (default: $PTXV_SLURM_ROOT); "
        "sbatch options come from $PTXV_SLURM_SBATCH_ARGS",
    )
    ap.add_argument(
        "--slurm-dry-run",
        action="store_true",
        help="print sbatch commands, submit nothing",
    )
    a = ap.parse_args(argv)
    if a.bind not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"ptxvd: warning: binding {a.bind}: jobs run arbitrary cases; no authentication",
            file=sys.stderr,
        )
    slurm = None
    if a.slurm_host or a.slurm_dry_run:
        from .slurm import SlurmBackend

        slurm = SlurmBackend(
            host=a.slurm_host, root=a.slurm_root, dry_run=a.slurm_dry_run
        )
    serve(
        a.port,
        a.bind,
        a.root,
        a.workers,
        None if a.mem_cap == "none" else a.mem_cap,
        a.threads,
        slurm,
    )


# ------------------------------------------------------------------------------- client


class DaemonBackend:
    """Client of a ptxvd daemon: uploads case bundles once, submits jobs, follows progress."""

    name = "ptxvd"

    def __init__(self, url: str, poll: float = 0.5) -> None:
        u = urlparse(url.replace("ptxvd://", "http://"))
        self.base = f"http://{u.hostname}:{u.port or 8765}"
        self.poll = poll
        self._info: dict | None = None

    def _req(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        ctype: str = "application/json",
    ) -> Any:
        r = urllib.request.Request(self.base + path, data=body, method=method)
        if body is not None:
            r.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(r, timeout=600) as f:
                data = f.read()
                return json.loads(data) if data else None
        except urllib.error.HTTPError as e:
            if method == "HEAD":
                return None
            raise RuntimeError(
                f"ptxvd {method} {path}: {e.code} {e.read().decode(errors='replace')}"
            ) from None

    def info(self) -> dict:
        if self._info is None:
            self._info = self._req("GET", "/v1/info")
        return self._info

    def ptxv_hash(self) -> str:
        return self.info()["ptxv_hash"]

    def upload(self, case: str) -> str:
        sha = tree_hash(case)
        if self._head(sha) != 200:
            self._req("PUT", f"/v1/cases/{sha}", pack_case(case), "application/gzip")
        return sha

    def _head(self, sha: str) -> int:
        r = urllib.request.Request(f"{self.base}/v1/cases/{sha}", method="HEAD")
        try:
            with urllib.request.urlopen(r, timeout=60) as f:
                return f.status
        except urllib.error.HTTPError as e:
            return e.code

    def submit(self, spec: JobSpec, job: Job) -> Job:
        shas = [self.upload(c) for c in spec.cases]
        r = self._req(
            "POST",
            "/v1/jobs",
            json.dumps({"kind": spec.kind, "cases": shas, "args": spec.args}).encode(),
        )
        jid = r["id"]
        job.id = jid
        job._kill = lambda: self._req("DELETE", f"/v1/jobs/{jid}")

        def follow() -> None:
            if not job.set_running_or_notify_cancel():
                return
            try:
                req = urllib.request.Request(f"{self.base}/v1/jobs/{jid}/events")
                with urllib.request.urlopen(req, timeout=None) as f:
                    for line in f:
                        ev = json.loads(line)
                        if "progress" in ev:
                            job._line(ev["progress"])
                st = self._req("GET", f"/v1/jobs/{jid}")
                if st["state"] == "done":
                    job.cached = st.get("cached", False)
                    job.set_result(st["result"])
                else:
                    job.set_exception(
                        RuntimeError(
                            f"ptxvd job {jid}: {st['state']}: {st.get('error', '')}"
                        )
                    )
            except Exception as e:  # noqa: BLE001
                job.set_exception(e)
            except BaseException as e:
                # KeyboardInterrupt / SystemExit: still fail the job (waiters never hang), then propagate.
                job.set_exception(e)
                raise

        threading.Thread(target=follow, daemon=True).start()
        return job

    def __repr__(self) -> str:
        return f"DaemonBackend({self.base!r})"


if __name__ == "__main__":
    main()
