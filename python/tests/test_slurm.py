# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ptxv
import pytest
from conftest import FIX
from ptxv import dist as D
from ptxv._jobs import JobSpec
from ptxv.slurm import SlurmBackend


def test_plan_check():
    sb = SlurmBackend(
        host="login", root="/shared/ptxv", ntasks=16, extra_sbatch=[], dry_run=True
    )
    cmds, paths = sb.plan(
        JobSpec(
            "check", [str(FIX / "c_cat")], {"max_states": 3_000_000_000, "slices": True}
        ),
        "abc",
    )
    assert cmds[0][:3] == ["rsync", "-a", "--mkpath"] and cmds[0][-1].startswith(
        "login:/shared/ptxv/ptxv-runs/cases/"
    )
    sb_cmd = cmds[-1]
    assert sb_cmd[:2] == ["ssh", "login"]
    s = sb_cmd[2]
    assert "sbatch --parsable --ntasks=16 --cpus-per-task=48" in s
    assert (
        "/shared/ptxv/ptxv/tools/ptxv.sbatch check /shared/ptxv/ptxv-runs/cases/"
        in s
    )
    assert "--json" in s and "--slices" in s and "3000000000" in s
    assert "ptxv-runs/py/abc/out" in s
    assert paths["stdout"].endswith("ptxv-runs/%j.out")


def test_plan_errdist_stages_spec():
    sb = SlurmBackend(host=None, root="/shared/ptxv", dry_run=True)
    args = {"inputs": {"x": D.Empirical([1.0, 2.0]).spec()}}
    cmds, paths = sb.plan(
        JobSpec("errdist", [str(FIX / "c_cumsum"), str(FIX / "c_triton_cumsum")], args),
        "j1",
    )
    flat = [" ".join(c) for c in cmds]
    assert any(
        "rsync" in c and "ptxv-runs/py/j1/" in c for c in flat
    )  # spec.json, equiv.toml, empirical file
    assert (
        "errdist /shared/ptxv/ptxv-runs/py/j1/equiv.toml --inputs"
        in flat[-1]
    )
    assert "--ntasks=1" in flat[-1]


def test_dry_run_job_and_url():
    sb = SlurmBackend.from_url("slurm://login?ntasks=4&dry_run=1&root=/shared/ptxv")
    assert sb.dry_run and sb.ntasks == 4 and sb.host == "login"
    s = ptxv.Session(sb)
    r = s.submit("check", [FIX / "c_cat"]).result(timeout=30)
    assert r["dry_run"] and r["commands"][-1].startswith("ssh login")


def test_config_from_env(monkeypatch):
    """Host, root and site sbatch options from the environment (how a cluster is configured)."""
    monkeypatch.setenv("PTXV_SLURM_HOST", "login")
    monkeypatch.setenv("PTXV_SLURM_ROOT", "/shared/ptxv")
    monkeypatch.setenv(
        "PTXV_SLURM_SBATCH_ARGS", "--partition=cpu --account=acct --qos=high"
    )
    sb = SlurmBackend(dry_run=True)
    assert sb.host == "login" and sb.root == "/shared/ptxv"
    cmds, _ = sb.plan(JobSpec("check", [str(FIX / "c_cat")], {}), "e1")
    s = cmds[-1][2]
    assert (
        "--output=/shared/ptxv/ptxv-runs/%j.out --export=ALL,PTXV_SLURM_ROOT=/shared/ptxv "
        "--partition=cpu --account=acct --qos=high /shared/ptxv/ptxv/tools/ptxv.sbatch check "
    ) in s
    # explicit arguments win over the environment
    sb = SlurmBackend(host=None, root="/other", extra_sbatch=[], dry_run=True)
    assert sb.host is None and sb.root == "/other" and sb.extra == []


def test_root_required(monkeypatch):
    monkeypatch.delenv("PTXV_SLURM_ROOT", raising=False)
    with pytest.raises(ValueError):
        SlurmBackend(host="login")
    with pytest.raises(ValueError):
        SlurmBackend(host="login", root="relative/dir", dry_run=True)
