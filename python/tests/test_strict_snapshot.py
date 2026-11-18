# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import ptxv
from ptxv import _strict_inputs
from ptxv._jobs import Job, JobSpec, run_job
from ptxv._util import tree_hash
from ptxv.daemon import Daemon, DaemonBackend, pack_case
from ptxv.session import _mapped
from ptxv.slurm import SlurmBackend

SAFE = (
    ".version 9.0\n.target sm_103a\n.address_size 64\n"
    ".global .align 8 .u64 x[2]={0,0};\n"
    ".visible .entry k(){.reg .b64 %r<2>;st.global.u64 [x],1;ret;}\n"
)
UNSAFE = SAFE.replace("[x],1;", "[x+16],1;")
OPTIONS = {
    "strict_memory_safety": True,
    "max_states": 100,
    "certify": False,
    "gb300": False,
    "timeout": 45,
}


def manifest(ptx="kernel.ptx", actors=1):
    return (
        'name="strict-snapshot"\ndescription="owned input control"\nptx='
        + json.dumps(ptx)
        + f'\nentry="k"\ngrid=[1,1,1]\nblock=[{actors},1,1]\n'
        'outputs=[]\nparams=[]\n[[buffers]]\nname="x"\ndtype="u64"\n'
        'shape=[2]\ninit="zeros"\nsymbol="x"\n[[variants]]\nid="good"\n'
        'summary="control"\nexpect=["clean"]\n'
    )


class HoldingBackend:
    def __init__(self):
        self.submissions = []

    def ptxv_hash(self):
        return "test-binary"

    def prepare_spec(self, spec):
        spec.bind_strict_binary("/usr/bin/true")

    def ptxv_hash_for(self, spec):
        self.prepare_spec(spec)
        return spec.bound_binary_hash or self.ptxv_hash()

    def submit(self, spec, job):
        self.submissions.append((spec, job))
        return job


class StrictSnapshotCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = mock.patch.dict(
            os.environ,
            {"PTXV_CACHE": str(self.root / "cache"), "CUDA_VISIBLE_DEVICES": ""},
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.case = self.root / "case"
        self.case.mkdir()
        (self.case / "case.toml").write_text(manifest())
        (self.case / "kernel.ptx").write_text(SAFE)


class StrictSnapshotTests(StrictSnapshotCase):
    def test_key_and_execution_share_independent_owned_bytes(self):
        backend = HoldingBackend()
        session = ptxv.Session(backend)
        first = session.submit("check", [self.case], **OPTIONS)
        spec, queued = backend.submissions[0]
        snapshot = Path(spec.cases[0])
        self.assertIs(first.spec, spec)
        self.assertEqual(
            spec.args["_strict_memory_safety_contract"],
            _strict_inputs.STRICT_MEMORY_CONTRACT,
        )
        self.assertNotEqual(snapshot, self.case)
        self.assertEqual(snapshot.name, self.case.name)
        self.assertNotEqual(
            (snapshot / "kernel.ptx").stat().st_ino,
            (self.case / "kernel.ptx").stat().st_ino,
        )
        self.assertEqual(session.key(spec), first.key)
        (self.case / "kernel.ptx").write_text(UNSAFE)
        self.assertEqual((snapshot / "kernel.ptx").read_text(), SAFE)
        gc.collect()
        self.assertTrue(snapshot.is_dir())
        self.assertTrue(queued.cancel())
        self.assertFalse(snapshot.exists())
        self.assertFalse(snapshot.parent.exists())

    def test_identical_strict_content_is_cacheable_without_snapshot_path(self):
        backend = HoldingBackend()
        session = ptxv.Session(backend)
        first = session.submit("check", [self.case], **OPTIONS)
        first_snapshot = Path(first.spec.cases[0])
        first.set_result({"rows": []})
        self.assertFalse(first_snapshot.parent.exists())
        self.assertEqual(first.spec._input_snapshots, [])
        second = session.submit("check", [self.case], **OPTIONS)
        self.assertTrue(second.cached)
        self.assertEqual(first.key, second.key)
        self.assertEqual(second.spec._input_snapshots, [])
        self.assertEqual(len(backend.submissions), 1)

    def test_shared_job_and_worker_leases_release_only_after_last_reader(self):
        spec = JobSpec("check", [str(self.case)], OPTIONS)
        first, second = Job(spec), Job(spec)
        snapshot = Path(spec.cases[0])
        self.assertEqual(snapshot.name, self.case.name)
        self.assertEqual(snapshot.parent.stat().st_mode & 0o777, 0o700)
        spec.retain_inputs()
        first.set_result({"rows": []})
        second.cancel()
        self.assertTrue(snapshot.is_dir())
        spec.release_inputs()
        self.assertFalse(snapshot.parent.exists())
        self.assertEqual(spec._input_snapshots, [])
        first.release_inputs()
        second.release_inputs()
        spec.release_inputs()

    def test_active_mapping_uses_parent_ownership_and_cancel(self):
        backend = HoldingBackend()
        parent = ptxv.Session(backend, cache=False).submit(
            "check", [self.case], **OPTIONS
        )
        snapshot = Path(parent.spec.cases[0])
        with mock.patch.object(
            JobSpec,
            "retain_inputs",
            side_effect=AssertionError("mapper acquired input lease"),
        ):
            mapped = _mapped(parent, ptxv.Verdict.from_json)
        self.assertEqual(parent.spec._input_users, 1)
        self.assertTrue(snapshot.is_dir())
        self.assertTrue(mapped.cancel())
        self.assertTrue(parent.cancelled())
        self.assertTrue(mapped.done())
        self.assertFalse(snapshot.parent.exists())

    def test_start_callback_failure_reaps_process_before_snapshot_cleanup(self):
        spec = JobSpec("check", [str(self.case)], OPTIONS)
        spec.own_strict_inputs()
        snapshot = Path(spec.cases[0])
        process = mock.Mock(pid=12345)
        process.poll.return_value = None

        def stop(pid):
            self.assertTrue(snapshot.is_dir())

        with mock.patch("ptxv._jobs.subprocess.Popen", return_value=process):
            with mock.patch("ptxv._jobs.kill_group", side_effect=stop) as kill:
                with self.assertRaisesRegex(RuntimeError, "callback failed"):
                    run_job(
                        spec,
                        self.root / "failed-job",
                        binary="/usr/bin/true",
                        mem_cap=None,
                        on_pid=mock.Mock(side_effect=RuntimeError("callback failed")),
                    )
        kill.assert_called_once_with(12345)
        process.wait.assert_called_once_with()
        self.assertFalse(snapshot.parent.exists())

    def test_shutdown_cancels_queued_job_and_releases_inputs(self):
        backend = ptxv.LocalBackend(
            workers=1, mem_cap=None, binary="/usr/bin/true", root=self.root / "jobs"
        )
        ready, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.addCleanup(backend.shutdown)

        def hold():
            ready.set()
            if not release.wait(10):
                raise RuntimeError("queue control timed out")

        blocked = backend._pool.submit(hold)
        self.assertTrue(ready.wait(2))
        job = ptxv.Session(backend, cache=False).submit("check", [self.case], **OPTIONS)
        snapshot = Path(job.spec.cases[0])
        backend.shutdown()
        self.assertTrue(job.cancelled())
        self.assertFalse(snapshot.parent.exists())
        release.set()
        blocked.result(timeout=5)

    def test_running_cancel_keeps_snapshot_until_worker_stops(self):
        backend = ptxv.LocalBackend(
            workers=1, mem_cap=None, binary="/usr/bin/true", root=self.root / "jobs"
        )
        self.addCleanup(backend.shutdown)
        ready, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def read(spec, work, **kwargs):
            ready.set()
            if not release.wait(10):
                raise RuntimeError("reader control timed out")
            self.assertEqual((Path(spec.cases[0]) / "kernel.ptx").read_text(), SAFE)
            return {"rows": []}

        with mock.patch("ptxv._jobs._run_job", side_effect=read):
            job = ptxv.Session(backend, cache=False).submit(
                "check", [self.case], **OPTIONS
            )
            self.assertTrue(ready.wait(2))
            snapshot = Path(job.spec.cases[0])
            self.assertTrue(job.running())
            self.assertTrue(job.cancel())
            self.assertTrue(snapshot.is_dir())
            release.set()
            job.result(timeout=10)
            backend._pool.submit(lambda: None).result(timeout=5)
        self.assertFalse(snapshot.parent.exists())
        self.assertEqual(job.spec._input_snapshots, [])

    def test_daemon_dispatch_failure_completes_retained_job_and_releases_inputs(self):
        daemon = Daemon(self.root / "daemon", mem_cap=None, binary="/usr/bin/true")
        daemon.local.shutdown()
        daemon.local = HoldingBackend()
        content = tree_hash(self.case)
        daemon.put_case(content, pack_case(self.case))
        for backend in ["slurm", "local"]:
            with self.subTest(backend=backend):
                request = {
                    "kind": "check",
                    "cases": [content],
                    "args": {**OPTIONS, "backend": backend},
                }
                with mock.patch.object(
                    daemon.local, "submit", side_effect=RuntimeError("dispatch failed")
                ):
                    with self.assertRaises((ValueError, RuntimeError)):
                        daemon.submit(request)
                retained = list(daemon.jobs.values())
                self.assertTrue(retained)
                for job in retained:
                    self.assertTrue(job.done())
                    self.assertIsNotNone(job.exception())
                    self.assertEqual(job.spec._input_users, 0)
                    self.assertEqual(job.spec._input_snapshots, [])
                    self.assertEqual(daemon.status(job.id)["state"], "error")

    def test_daemon_duplicate_and_retained_completed_jobs_release_snapshots(self):
        daemon = Daemon(self.root / "daemon", mem_cap=None, binary="/usr/bin/true")
        daemon.local.shutdown()
        backend = HoldingBackend()
        daemon.local = backend
        content = tree_hash(self.case)
        daemon.put_case(content, pack_case(self.case))
        request = {"kind": "check", "cases": [content], "args": OPTIONS}
        identity, _ = daemon.submit(json.loads(json.dumps(request)))
        active = daemon.jobs[identity]
        snapshot = Path(active.spec.cases[0])
        copied = []
        original = _strict_inputs.copy_strict_case

        def capture(case):
            owner = original(case)
            copied.append(Path(owner.name))
            return owner

        with mock.patch("ptxv._jobs.copy_strict_case", side_effect=capture):
            repeated, cached = daemon.submit(json.loads(json.dumps(request)))
        self.assertEqual(identity, repeated)
        self.assertFalse(cached)
        self.assertEqual(len(copied), 1)
        self.assertFalse(copied[0].parent.exists())
        self.assertTrue(snapshot.is_dir())
        active.set_result({"rows": []})
        self.assertFalse(snapshot.parent.exists())
        self.assertEqual(active.spec._input_snapshots, [])
        with mock.patch("ptxv._jobs.copy_strict_case", side_effect=capture):
            restored, cached = daemon.submit(json.loads(json.dumps(request)))
        self.assertEqual(identity, restored)
        self.assertTrue(cached)
        self.assertFalse(copied[-1].parent.exists())
        self.assertEqual(daemon.jobs[restored].spec._input_snapshots, [])

    def test_nonportable_strict_inputs_fail_before_cache_or_backend(self):
        for mode in [
            "out",
            "parent",
            "absolute",
            "file-link",
            "directory-link",
            "manifest-link",
            "dot",
        ]:
            for cache in [False, True]:
                with self.subTest(mode=mode, cache=cache):
                    root = self.root / f"{mode}-{cache}"
                    root.mkdir()
                    outside = self.root / f"external-{mode}-{cache}.ptx"
                    outside.write_text(SAFE)
                    path = "kernel.ptx"
                    if mode == "out":
                        (root / "out").mkdir()
                        (root / "out/kernel.ptx").write_text(SAFE)
                        path = "out/kernel.ptx"
                    elif mode == "parent":
                        path = "../" + outside.name
                    elif mode == "absolute":
                        path = str(outside)
                    elif mode == "file-link":
                        (root / "kernel.ptx").symlink_to(outside)
                    elif mode == "directory-link":
                        directory = self.root / f"outside-{cache}"
                        directory.mkdir()
                        (directory / "kernel.ptx").write_text(SAFE)
                        (root / "linked").symlink_to(
                            directory, target_is_directory=True
                        )
                        path = "linked/kernel.ptx"
                    elif mode == "dot":
                        path = "."
                    else:
                        (root / "kernel.ptx").write_text(SAFE)
                    if mode == "manifest-link":
                        document = self.root / f"outside-{cache}.toml"
                        document.write_text(manifest())
                        (root / "case.toml").symlink_to(document)
                    else:
                        (root / "case.toml").write_text(manifest(path))
                    backend = HoldingBackend()
                    with self.assertRaisesRegex(ValueError, "strict memory"):
                        ptxv.Session(backend, cache=cache).submit(
                            "check", [root], **OPTIONS
                        )
                    self.assertEqual(backend.submissions, [])

    def test_daemon_validates_copied_case_identity_before_cache(self):
        daemon = Daemon(self.root / "daemon", mem_cap=None, binary="/usr/bin/true")
        daemon.local.shutdown()
        backend = HoldingBackend()
        daemon.local = backend
        content = tree_hash(self.case)
        daemon.put_case(content, pack_case(self.case))
        request = {"kind": "check", "cases": [content], "args": OPTIONS}
        identity, _ = daemon.submit(json.loads(json.dumps(request)))
        owned = Path(backend.submissions[0][0].cases[0])
        (daemon.case_path(content) / "kernel.ptx").write_text(UNSAFE)
        self.assertEqual((owned / "kernel.ptx").read_text(), SAFE)
        with self.assertRaisesRegex(ValueError, "content hash"):
            daemon.submit(json.loads(json.dumps(request)))
        daemon.jobs[identity].set_result({"rows": []})
        with self.assertRaisesRegex(ValueError, "content hash"):
            daemon.submit(json.loads(json.dumps(request)))

    def test_strict_old_daemon_fails_before_upload(self):
        backend = DaemonBackend("ptxvd://127.0.0.1:1")
        backend._info = {"ptxv_hash": "test-binary"}
        with mock.patch.object(
            backend, "upload", side_effect=AssertionError("uploaded")
        ):
            with self.assertRaisesRegex(ValueError, "upgraded daemon"):
                ptxv.Session(backend, cache=False).submit(
                    "check", [self.case], **OPTIONS
                )

    def test_slurm_plan_retains_owned_inputs_and_unique_output_directories(self):
        backend = SlurmBackend(host=None, root="/shared/test", dry_run=True)
        spec = JobSpec("check", [str(self.case)], OPTIONS)
        commands, _ = backend.plan(spec, "same-key")
        copied = Path(spec.cases[0])
        (self.case / "kernel.ptx").write_text(UNSAFE)
        self.assertEqual((copied / "kernel.ptx").read_text(), SAFE)
        self.assertIn(str(copied) + "/", commands[0])
        repeated, _ = backend.plan(spec, "same-key")
        self.assertNotEqual(commands[-1], repeated[-1])

    def test_cached_new_daemon_info_cannot_authorize_old_job_response(self):
        for acknowledgement in [None, 1, 2]:
            with self.subTest(acknowledgement=acknowledgement):
                backend = DaemonBackend("ptxvd://127.0.0.1:1")
                backend._info = {
                    "ptxv_hash": "test-binary",
                    "strict_memory_safety_contract": _strict_inputs.STRICT_MEMORY_CONTRACT,
                }
                reply = {"id": "old-safe-result", "cached": True}
                if acknowledgement is not None:
                    reply["strict_memory_safety_contract"] = acknowledgement
                with mock.patch.object(backend, "upload", return_value="0" * 64):
                    with mock.patch.object(backend, "_req", return_value=reply):
                        with mock.patch(
                            "ptxv.daemon.threading.Thread.start",
                            side_effect=AssertionError("followed result"),
                        ):
                            with self.assertRaisesRegex(ValueError, "did not confirm"):
                                ptxv.Session(backend, cache=False).submit(
                                    "check", [self.case], **OPTIONS
                                )


@unittest.skipUnless(os.environ.get("PTXV_BIN"), "set PTXV_BIN for actual CLI controls")
class StrictSnapshotCliTests(StrictSnapshotCase):
    def backend(self):
        backend = ptxv.LocalBackend(
            workers=1,
            threads=1,
            mem_cap=None,
            binary=os.environ["PTXV_BIN"],
            root=self.root / "jobs",
        )
        self.addCleanup(backend.shutdown)
        return backend

    def hold(self, backend):
        ready, release = threading.Event(), threading.Event()

        def block():
            ready.set()
            if not release.wait(20):
                raise RuntimeError("queue control timed out")

        blocker = backend._pool.submit(block)
        self.assertTrue(ready.wait(2))
        self.addCleanup(release.set)
        return release, blocker

    def test_queued_source_is_snapshotted_for_cached_and_uncached_checks(self):
        for cache in [False, True]:
            with self.subTest(cache=cache):
                backend = self.backend()
                session = ptxv.Session(backend, cache=cache)
                (self.case / "kernel.ptx").write_text(UNSAFE)
                release, blocker = self.hold(backend)
                job = session.check(self.case, wait=False, **OPTIONS)
                (self.case / "kernel.ptx").write_text(SAFE)
                release.set()
                blocker.result(timeout=2)
                verdict = job.result(timeout=60)
                self.assertFalse(verdict.memory_safe)
                self.assertEqual(verdict.strict_memory_safety["status"], "unsafe")
                backend._pool.submit(lambda: None).result(timeout=5)
                safe_job = session.check(self.case, wait=False, **OPTIONS)
                self.assertTrue(safe_job.result(timeout=60).memory_safe)
                backend._pool.submit(lambda: None).result(timeout=5)
                repeated = session.check(self.case, wait=False, **OPTIONS)
                self.assertTrue(repeated.result(timeout=60).memory_safe)
                self.assertEqual(repeated.cached, cache)

    def test_queued_manifest_uses_original_actor_inventory(self):
        backend = self.backend()
        session = ptxv.Session(backend)
        release, blocker = self.hold(backend)
        job = session.check(self.case, wait=False, **OPTIONS)
        (self.case / "case.toml").write_text(manifest(actors=2))
        release.set()
        blocker.result(timeout=2)
        self.assertTrue(job.result(timeout=60).memory_safe)
        changed = session.check(self.case, wait=False, **OPTIONS)
        self.assertFalse(changed.cached)
        self.assertFalse(changed.result(timeout=60).memory_safe)

    def test_completed_and_cached_result_mapping_never_rereads_source(self):
        backend = self.backend()
        session = ptxv.Session(backend)
        completed = session.submit("check", [self.case], **OPTIONS)
        self.assertTrue(
            ptxv.Verdict.from_json(completed.result(timeout=60)).memory_safe
        )
        backend._pool.submit(lambda: None).result(timeout=5)
        cached = session.submit("check", [self.case], **OPTIONS)
        self.assertTrue(cached.cached)
        (self.case / "kernel.ptx").unlink()
        for parent in [completed, cached]:
            with self.subTest(cached=parent.cached):
                mapped = _mapped(parent, ptxv.Verdict.from_json)
                self.assertTrue(mapped.result(timeout=2).memory_safe)
                self.assertEqual(mapped.spec._input_users, 0)
                self.assertEqual(mapped.spec._input_snapshots, [])
        (self.case / "case.toml").write_text("not valid TOML !")
        with mock.patch.object(
            JobSpec,
            "retain_inputs",
            side_effect=AssertionError("mapped result read inputs"),
        ):
            mapped = _mapped(completed, ptxv.Verdict.from_json)
            self.assertTrue(mapped.result(timeout=2).memory_safe)

    def test_actual_completed_job_preserves_case_label_and_releases_inputs(self):
        backend = self.backend()
        session = ptxv.Session(backend)
        release, blocker = self.hold(backend)
        job = session.submit("check", [self.case], **OPTIONS)
        snapshot = Path(job.spec.cases[0])
        self.assertEqual(snapshot.name, self.case.name)
        release.set()
        blocker.result(timeout=2)
        document = job.result(timeout=60)
        backend._pool.submit(lambda: None).result(timeout=5)
        self.assertFalse(snapshot.parent.exists())
        self.assertEqual(job.spec._input_snapshots, [])
        self.assertEqual(Path(document["case"]).name, self.case.name)
        repeated = session.submit("check", [self.case], **OPTIONS)
        self.assertTrue(repeated.cached)
        self.assertEqual(repeated.spec._input_snapshots, [])
        self.assertEqual(Path(repeated.result()["case"]).name, self.case.name)

    def test_daemon_queue_executes_owned_bytes_and_restored_cache_is_not_safe(self):
        daemon = Daemon(
            self.root / "daemon",
            workers=1,
            threads=1,
            mem_cap=None,
            binary=os.environ["PTXV_BIN"],
        )
        self.addCleanup(daemon.local.shutdown)
        (self.case / "kernel.ptx").write_text(UNSAFE)
        content = tree_hash(self.case)
        daemon.put_case(content, pack_case(self.case))
        release, blocker = self.hold(daemon.local)
        request = {"kind": "check", "cases": [content], "args": OPTIONS}
        identity, _ = daemon.submit(json.loads(json.dumps(request)))
        (daemon.case_path(content) / "kernel.ptx").write_text(SAFE)
        release.set()
        blocker.result(timeout=2)
        document = daemon.jobs[identity].result(timeout=60)
        self.assertFalse(ptxv.Verdict.from_json(document).memory_safe)
        daemon.local._pool.submit(lambda: None).result(timeout=5)
        with self.assertRaisesRegex(ValueError, "content hash"):
            daemon.submit(json.loads(json.dumps(request)))
        (daemon.case_path(content) / "kernel.ptx").write_text(UNSAFE)
        restored, cached = daemon.submit(json.loads(json.dumps(request)))
        self.assertTrue(cached)
        self.assertFalse(
            ptxv.Verdict.from_json(daemon.jobs[restored].result()).memory_safe
        )


if __name__ == "__main__":
    unittest.main()
