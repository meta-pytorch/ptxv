# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from pathlib import Path

import ptxv
import pytest
from conftest import FIX, needs_binary, vendor_skip
from ptxv._jobs import argv, JobSpec


@needs_binary
def test_local_check_verdict_and_cache():
    s = ptxv.Session(workers=1, mem_cap="16G")
    v = s.check(FIX / "c_cat")
    assert v.classes == ["clean"] and v.certified and v.complete and v.sim_equals_gpu
    assert "E-WELLSYNC" in v.assumptions and "Verdict(classes=['clean']" in repr(v)
    job = s.submit("check", [FIX / "c_cat"])
    assert job.cached and job.result()["rows"][0]["pass"]
    s2 = ptxv.Session(workers=1, cache=False)
    j2 = s2.submit("check", [FIX / "c_cat"])
    assert not j2.cached and j2.result()["pass"]


@needs_binary
def test_case_object_and_session_context():
    c = ptxv.Case(FIX / "c_cumsum")
    assert c.outputs == ["out"] and c.inputs()["x"].shape == (4, 128)
    assert c.expected()["out"].shape == (4, 128)
    with ptxv.Session(workers=1) as s:
        assert ptxv.session() is s
        v = c.check()
    assert v.clean


@needs_binary
def test_compare_fixtures_returns_bound():
    eq = ptxv.compare(FIX / "c_cumsum", FIX / "c_triton_cumsum")
    assert eq.level == "L1" and eq.proven and not eq.blockers
    assert eq.bound is not None and eq.bound.max_u() == pytest.approx(15.0, abs=1e-3)
    assert "PROVEN L1" in eq.headline and eq.markdown.startswith("# ptxv compare")


@needs_binary
def test_future_progress_and_cancel():
    """A long exploration (cuBLAS GEMM, no certificate) is cancelled: its process group dies."""
    vendor_skip("r2_cublas_gemm")
    case = FIX / "r2_cublas_gemm"
    be = ptxv.LocalBackend(workers=1, mem_cap="16G", threads=4)
    s = ptxv.Session(be, cache=False)
    job = s.submit("check", [case], certify=False, max_states=10**9)
    t0 = time.time()
    while not be.pids() and time.time() - t0 < 30:
        time.sleep(0.1)
    pid = next(iter(be.pids().values()))
    time.sleep(2)
    assert job.running() and job.cancel()
    with pytest.raises(ptxv.JobCancelled):
        job.result(timeout=30)
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    be.shutdown()


def test_argv_shapes(tmp_path):
    a, _ = argv(JobSpec("check", [str(FIX / "c_cat")], {"max_states": 5}), tmp_path)
    assert (
        a[:2] == ["check", str(FIX / "c_cat")]
        and "--json" in a
        and "--certify" in a
        and "5" in a
    )
    a, files = argv(
        JobSpec(
            "compare",
            [str(FIX / "c_cat"), str(FIX / "c_triton_cat")],
            {"function": "gelu"},
        ),
        tmp_path,
    )
    assert (
        a[0] == "compare"
        and "--emit-bound" in a
        and a[a.index("--function") + 1] == "gelu"
    )
    assert set(files) == {"bound", "report_md"}
    with pytest.raises(ValueError):
        JobSpec("nope", [])


def test_nonfinite_argv_and_verdict():
    from ptxv.results import Verdict

    c = str(FIX / "c_nonfinite_exp_f16")
    a, _ = argv(JobSpec("check", [c]), tmp := Path("/tmp"))
    assert "--nonfinite" not in a and "--nonfinite-strict" not in a
    a, _ = argv(JobSpec("check", [c], {"nonfinite": True}), tmp)
    assert "--nonfinite" in a
    a, _ = argv(JobSpec("check", [c], {"nonfinite": "strict"}), tmp)
    assert "--nonfinite-strict" in a
    nf = {"outputs": [{"buffer": "out", "inf": 114, "nan": 0}], "births": []}
    doc = {
        "rows": [
            {
                "got": ["nonfinite-output"],
                "pass": False,
                "verdict": {"findings": [], "complete": True, "states": 1},
                "nonfinite": nf,
            }
        ]
    }
    assert Verdict.from_json(doc).nonfinite == nf


def test_verdict_gb300_fields_from_json():
    from ptxv.results import Verdict

    doc = {
        "rows": [
            {
                "got": ["race"],
                "pass": True,
                "refuted_opt_in": [],
                "gb300": {"observable": 0, "latent": 1, "states": 32, "complete": True},
                "verdict": {
                    "complete": True,
                    "states": 32,
                    "findings": [
                        {
                            "kind": "race",
                            "sig": "race:gmem:11:write-read",
                            "msg": "write-read race",
                            "trace": [],
                            "gb300": {"class": "latent", "reason": "no fill of line"},
                        }
                    ],
                },
            }
        ]
    }
    v = Verdict.from_json(doc)
    assert v.latent and not v.observable and v.gb300["latent"] == 1
    assert v.findings[0].gb300_reason == "no fill of line"
    assert "latent on GB300" in repr(v.findings[0])
    assert not v.latent_closed_world
    doc["rows"][0]["verdict"]["findings"][0]["gb300"] = {
        "class": "latent-closed-world",
        "reason": "latent only under E-L1-CLOSED",
        "witness": "stale-l1-havoc:28",
    }
    v = Verdict.from_json(doc)
    assert v.latent_closed_world and not v.latent and not v.observable
    assert v.findings[0].gb300_witness == "stale-l1-havoc:28"
    spec = JobSpec("check", ["x"], {"assume_refuted": ["E-POLL-ACQ"], "gb300": False})
    cmd, _ = argv(spec, Path("/tmp"))
    assert cmd[-3:] == ["--assume-refuted", "E-POLL-ACQ", "--no-gb300"]


@needs_binary
def test_check_classifies_races_on_gb300():
    s = ptxv.Session(workers=1, mem_cap="16G", cache=False)
    v = s.check(FIX / "l1_litmus", variant="pr0", certify=False)
    assert v.classes == ["race"] and v.gb300 is not None
    # Latent only in a closed world: a fill of the payload line from outside the case (a
    # co-resident reader, as pr0-lineshare does itself) makes the read stale.
    assert [f.gb300 for f in v.findings] == ["latent-closed-world"]
    assert v.latent_closed_world and not v.latent and not v.observable
    assert v.findings[0].gb300_witness.startswith("stale-l1-havoc:")
    assert "E-L1-CLOSED" in v.findings[0].gb300_reason
    v = s.check(FIX / "l1_litmus", variant="pr2", certify=False)
    assert "stale" in v.classes and v.observable and not v.latent
