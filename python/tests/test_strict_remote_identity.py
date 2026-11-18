# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the BSD-style license in the repository root.

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import ptxv
from ptxv._strict_inputs import STRICT_MEMORY_CONTRACT
from ptxv._util import job_key, sha256_file, tree_hash
from ptxv.daemon import Daemon, DaemonBackend, pack_case
from ptxv.slurm import SlurmBackend

OPTIONS = {
    "strict_memory_safety": True,
    "max_states": 100,
    "certify": False,
    "gb300": False,
    "timeout": 45,
}
SOURCE = (
    ".version 9.0\n.target sm_103a\n.address_size 64\n"
    ".global .align 8 .u64 x[2]={0,0};\n"
    ".visible .entry k(){.reg .pred %q;setp.eq.u32 %q,1,1;"
    "@%q st.global.u64 [x],1;ret;}\n"
)
MANIFEST = (
    'name="remote-identity"\ndescription="cache provenance control"\n'
    'ptx="kernel.ptx"\nentry="k"\ngrid=[1,1,1]\nblock=[1,1,1]\n'
    'outputs=[]\nparams=[]\n[[buffers]]\nname="x"\nsymbol="x"\n'
    'dtype="u64"\nshape=[2]\ninit="zeros"\n[[variants]]\n'
    'id="good"\nexpect=["clean"]\n'
)


class RecordingBackend:
    cache_tag = "slurm:transport-control:/shared"

    def __init__(self):
        self.jobs = []

    def submit(self, spec, job):
        self.jobs.append(job)
        return job


class RemoteIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = mock.patch.dict(
            os.environ,
            {
                "PTXV_CACHE": str(self.root / "cache"),
                "PTXV_NO_IMPORT_HOOK": "1",
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.case = self.root / "case"
        self.case.mkdir()
        (self.case / "case.toml").write_text(MANIFEST)
        (self.case / "kernel.ptx").write_text(SOURCE)

    def daemon(self, slurm=None, binary="/usr/bin/true"):
        daemon = Daemon(
            self.root / "daemon",
            workers=1,
            mem_cap=None,
            threads=1,
            slurm=slurm,
            binary=str(binary),
        )
        self.addCleanup(daemon.local.shutdown)
        content = tree_hash(self.case)
        daemon.put_case(content, pack_case(self.case))
        return daemon, content

    def request(self, content, *, strict=True, backend="local"):
        return {
            "kind": "check",
            "cases": [content],
            "args": {**(OPTIONS if strict else {}), "backend": backend},
        }

    def test_strict_slurm_does_not_reuse_results_or_running_jobs(self):
        remote = RecordingBackend()
        daemon, content = self.daemon(remote)
        request = self.request(content, backend="slurm")
        first, cached = daemon.submit(request)
        second, repeated = daemon.submit(request)
        self.assertFalse(cached or repeated)
        self.assertNotEqual(first, second)
        self.assertEqual(len(remote.jobs), 2)
        self.assertTrue(all(job.key is None for job in remote.jobs))
        self.assertTrue(all(job.spec.bound_binary is None for job in remote.jobs))
        for job in remote.jobs:
            job.set_result({"transport_control": True})
        self.assertEqual(list((daemon.root / "results").iterdir()), [])
        self.assertEqual(request["args"]["backend"], "slurm")

    def test_ordinary_local_and_slurm_have_separate_cache_namespaces(self):
        remote = RecordingBackend()
        daemon, content = self.daemon(remote)
        local = RecordingBackend()
        daemon.local.shutdown()
        daemon.local = local
        expected_local = job_key("check", {}, [content], daemon.ptxv_hash)
        local_id, _ = daemon.submit(self.request(content, strict=False))
        self.assertEqual(local.jobs[0].key, expected_local)
        local.jobs[0].set_result({"transport_control": "local"})
        slurm_id, cached = daemon.submit(
            self.request(content, strict=False, backend="slurm")
        )
        self.assertFalse(cached)
        self.assertNotEqual(local_id, slurm_id)
        self.assertEqual(len(remote.jobs), 1)
        remote.jobs[0].set_result({"transport_control": "slurm"})
        repeated, cached = daemon.submit(
            self.request(content, strict=False, backend="slurm")
        )
        self.assertTrue(cached)
        self.assertEqual(repeated, slurm_id)
        self.assertEqual(daemon.jobs[repeated].result()["transport_control"], "slurm")

    def test_unknown_route_fails_before_job_or_cache_lookup(self):
        daemon, content = self.daemon()
        with self.assertRaisesRegex(ValueError, "unknown daemon backend"):
            daemon.submit(self.request(content, backend="slrum"))
        self.assertEqual(daemon.jobs, {})

    def test_caller_cannot_inject_slurm_namespace_into_local_key(self):
        daemon, content = self.daemon()
        daemon.local.shutdown()
        local = RecordingBackend()
        daemon.local = local
        request = self.request(content, strict=False)
        request["args"]["_daemon_backend"] = "slurm:transport-control:/shared"
        daemon.submit(request)
        self.assertEqual(
            local.jobs[0].key, job_key("check", {}, [content], daemon.ptxv_hash)
        )
        local.jobs[0].set_result({"transport_control": True})

    def test_client_strict_requests_bypass_local_cache_but_ordinary_reuses_it(self):
        clients = [
            DaemonBackend("ptxvd://127.0.0.1:1"),
            SlurmBackend(host="unused", root="/shared", dry_run=False),
        ]
        for index, backend in enumerate(clients):
            with self.subTest(backend=backend.name):
                submissions = []

                def transport(spec, job, submissions=submissions):
                    submissions.append(job)
                    job.set_result({"transport_control": True})
                    return job

                with mock.patch.object(backend, "submit", side_effect=transport):
                    with mock.patch.object(
                        ptxv.Session,
                        "ptxv_hash",
                        side_effect=AssertionError("unknown remote hash"),
                    ):
                        session = ptxv.Session(backend)
                        first = session.submit("check", [self.case], **OPTIONS)
                        second = session.submit("check", [self.case], **OPTIONS)
                        self.assertIsNone(first.key)
                        self.assertIsNone(second.key)
                        self.assertFalse(first.cached or second.cached)
                        self.assertEqual(len(submissions), 2)
                    with mock.patch.object(
                        ptxv.Session, "ptxv_hash", return_value=f"ordinary-{index}"
                    ):
                        ordinary = ptxv.Session(backend)
                        ordinary.submit("check", [self.case]).result()
                        repeated = ordinary.submit("check", [self.case])
                        self.assertTrue(repeated.cached)
                        self.assertEqual(len(submissions), 3)

    def test_daemon_contract_two_is_refused(self):
        backend = DaemonBackend("ptxvd://127.0.0.1:1")
        backend._info = {"strict_memory_safety_contract": 2}
        with self.assertRaisesRegex(ValueError, f"contract {STRICT_MEMORY_CONTRACT}"):
            ptxv.Session(backend).submit("check", [self.case], **OPTIONS)

    @unittest.skipUnless(
        os.environ.get("PTXV_BINARY_OLD") and os.environ.get("PTXV_BINARY_NEW"),
        "requires the two qualified native binaries",
    )
    def test_server_binds_real_selected_binary_and_keeps_local_cache_and_sharing(self):
        old, new = (
            Path(os.environ["PTXV_BINARY_OLD"]),
            Path(os.environ["PTXV_BINARY_NEW"]),
        )
        daemon, content = self.daemon(binary=new)
        daemon.local.binary = old
        ready, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def hold():
            ready.set()
            if not release.wait(15):
                raise RuntimeError("queue synchronization timed out")

        blocked = daemon.local._pool.submit(hold)
        self.assertTrue(ready.wait(5))
        request = self.request(content)
        first, cached = daemon.submit(request)
        original = daemon.jobs[first]
        binary = Path(original.spec.bound_binary)
        case = Path(original.spec.cases[0])
        self.assertEqual(sha256_file(binary), sha256_file(old))
        self.assertNotEqual(binary, old)
        shared, duplicate_cached = daemon.submit(request)
        self.assertEqual(first, shared)
        self.assertFalse(cached or duplicate_cached)
        self.assertTrue(binary.is_file() and case.is_dir())
        daemon.local.binary = new
        release.set()
        blocked.result(timeout=5)
        result = original.result(timeout=60)
        daemon.local._pool.submit(lambda: None).result(timeout=5)
        self.assertEqual(result["rows"][0]["strict_memory_safety"]["status"], "safe")
        self.assertFalse(binary.exists())
        self.assertFalse(case.parent.exists())
        self.assertEqual(
            json.loads((Path(result["work_dir"]) / "cmd.json").read_text())[0],
            str(binary),
        )
        second, cached = daemon.submit(request)
        self.assertFalse(cached)
        self.assertNotEqual(first, second)
        result = daemon.jobs[second].result(timeout=60)
        daemon.local._pool.submit(lambda: None).result(timeout=5)
        self.assertEqual(
            result["rows"][0]["strict_memory_safety"]["status"], "inconclusive"
        )
        third, cached = daemon.submit(request)
        self.assertTrue(cached)
        self.assertEqual(third, second)
        self.assertEqual(daemon.jobs[third].spec._input_snapshots, [])
        self.assertIsNone(daemon.jobs[third].spec.bound_binary)


if __name__ == "__main__":
    unittest.main()
