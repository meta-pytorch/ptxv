# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import signal
import subprocess
import sys

import ptxv
import pytest
from conftest import FIX, needs_binary
from ptxv.daemon import pack_case, unpack_case


def test_bundle_roundtrip_and_hash(tmp_path):
    from ptxv._util import tree_hash

    data = pack_case(FIX / "c_cat")
    unpack_case(data, tmp_path / "x")
    assert tree_hash(tmp_path / "x") == tree_hash(FIX / "c_cat")


def test_bundle_rejects_escaping_paths(tmp_path):
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("../evil")
        ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        unpack_case(buf.getvalue(), tmp_path / "y")


@pytest.fixture
def daemon(tmp_path):
    env = dict(
        os.environ,
        PYTHONPATH=os.path.dirname(os.path.dirname(ptxv.__file__)),
        PTXV_SLURM_ROOT="/shared/ptxv",
    )
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ptxv.daemon",
            "--port",
            "0",
            "--root",
            str(tmp_path / "d"),
            "--workers",
            "1",
            "--mem-cap",
            "16G",
            "--slurm-dry-run",
        ],
        stdout=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    line = json.loads(p.stdout.readline())
    yield line["ptxvd"]
    os.killpg(p.pid, signal.SIGTERM)  # the daemon we started, by PID
    p.wait(timeout=10)


@needs_binary
def test_daemon_jobs(daemon):
    assert daemon.startswith("ptxvd://127.0.0.1:")
    s = ptxv.Session(daemon, cache=False)
    info = s.backend.info()
    assert info["workers"] == 1 and info["slurm"] and len(info["ptxv_hash"]) == 64
    lines = []
    job = s.submit("check", [FIX / "c_cat"])
    job.add_progress_callback(lines.append)
    v = ptxv.Verdict.from_json(job.result(timeout=300))
    assert v.clean and not job.cached
    j2 = s.submit("check", [FIX / "c_cat"])
    j2.result(timeout=60)
    assert j2.cached  # the daemon's own cache
    eq = s.compare(FIX / "c_cumsum", FIX / "c_triton_cumsum")
    assert eq.level == "L1" and eq.bound is not None


@needs_binary
def test_daemon_slurm_dry_run(daemon):
    s = ptxv.Session(daemon, cache=False)
    r = s.submit("check", [FIX / "c_cat"], backend="slurm").result(timeout=60)
    assert r["dry_run"]
    cmds = r["commands"]
    assert any(c.startswith("rsync -a --mkpath") for c in cmds)
    assert (
        "/shared/ptxv/ptxv/tools/ptxv.sbatch check /shared/ptxv/ptxv-runs/cases/"
        in cmds[-1]
    )
