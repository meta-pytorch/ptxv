# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import ptxv
from ptxv import _strict_inputs
from ptxv._jobs import store_cache
from ptxv._util import cache_dir, job_key, tree_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def marker(path, value):
    path.write_text(
        "#!/usr/bin/python3\nimport json\n"
        + f"print(json.dumps({{'marker': {value!r}}}))\n"
    )
    path.chmod(0o500)


class StrictBinaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.a, self.b = self.root / "A", self.root / "B"
        marker(self.a, "A")
        marker(self.b, "B")
        self.case = self.root / "case"
        self.case.mkdir()
        (self.case / "case.toml").write_text(
            'name="binary-control"\nptx="kernel.ptx"\n'
        )
        (self.case / "kernel.ptx").write_text(".version 9.0\n")
        environment = mock.patch.dict(
            os.environ,
            {"PTXV_CACHE": str(self.root / "cache"), "PTXV_BIN": str(self.a)},
        )
        environment.start()
        self.addCleanup(environment.stop)

    def backend(self, binary=None):
        backend = ptxv.LocalBackend(
            workers=1, mem_cap=None, binary=binary, root=self.root / "jobs"
        )
        self.addCleanup(backend.shutdown)
        return backend

    def hold(self, backend):
        ready, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def block():
            ready.set()
            if not release.wait(15):
                raise RuntimeError("binary queue control timed out")

        blocker = backend._pool.submit(block)
        self.assertTrue(ready.wait(2))
        return release, blocker

    def finish(self, backend, job):
        result = job.result(timeout=20)
        backend._pool.submit(lambda: None).result(timeout=5)
        return result

    def test_selected_override_hash_and_lazy_construction(self):
        with mock.patch("ptxv._jobs._binary.find", side_effect=AssertionError("find")):
            backend = self.backend(self.b)
            self.assertEqual(backend.ptxv_hash(), digest(self.b))
            session = ptxv.Session(backend)
            selected = session.submit("check", [self.case], strict_memory_safety=True)
            self.assertEqual(self.finish(backend, selected)["marker"], "B")
            expected = job_key(
                "check", selected.spec.args, [tree_hash(self.case)], digest(self.b)
            )
            self.assertEqual(selected.key, expected)
            backend.binary = self.a
            other = session.submit("check", [self.case], strict_memory_safety=True)
            self.assertEqual(self.finish(backend, other)["marker"], "A")
            self.assertNotEqual(selected.key, other.key)
            backend.binary = self.b
            repeated = session.submit("check", [self.case], strict_memory_safety=True)
            self.assertTrue(repeated.cached)
            self.assertEqual(repeated.result()["marker"], "B")
            ordinary = session.submit("check", [self.case])
            self.assertEqual(self.finish(backend, ordinary)["marker"], "B")
            self.assertEqual(
                ordinary.key,
                job_key("check", {}, [tree_hash(self.case)], digest(self.b)),
            )
            self.assertIsNone(ordinary.spec.bound_binary)

    def test_queued_default_environment_change_uses_request_binary(self):
        backend = self.backend()
        session = ptxv.Session(backend)
        release, blocker = self.hold(backend)
        job = session.submit("check", [self.case], strict_memory_safety=True)
        captured = Path(job.spec.bound_binary)
        self.assertEqual(digest(captured), digest(self.a))
        self.assertEqual(captured.stat().st_mode & 0o777, 0o500)
        self.assertEqual(captured.parent.stat().st_mode & 0o777, 0o700)
        self.assertNotEqual(captured.stat().st_ino, self.a.stat().st_ino)
        os.environ["PTXV_BIN"] = str(self.b)
        release.set()
        blocker.result(timeout=5)
        self.assertEqual(self.finish(backend, job)["marker"], "A")
        self.assertFalse(captured.parent.exists())
        os.environ["PTXV_BIN"] = str(self.a)
        replay = session.submit("check", [self.case], strict_memory_safety=True)
        self.assertTrue(replay.cached)
        self.assertEqual(replay.result()["marker"], "A")
        self.assertIsNone(replay.spec.bound_binary)

    def test_queued_backend_override_change_uses_request_binary(self):
        backend = self.backend(self.a)
        release, blocker = self.hold(backend)
        job = ptxv.Session(backend).submit(
            "check", [self.case], strict_memory_safety=True
        )
        captured = Path(job.spec.bound_binary)
        backend.binary = self.b
        release.set()
        blocker.result(timeout=5)
        self.assertEqual(self.finish(backend, job)["marker"], "A")
        self.assertFalse(captured.parent.exists())

    def test_queued_source_binary_replacement_cannot_change_owned_bytes(self):
        for mode in ["replace", "in-place"]:
            with self.subTest(mode=mode):
                selected = self.root / mode
                shutil.copyfile(self.a, selected)
                selected.chmod(0o500)
                backend = self.backend(selected)
                release, blocker = self.hold(backend)
                job = ptxv.Session(backend, cache=False).submit(
                    "check", [self.case], strict_memory_safety=True
                )
                captured = Path(job.spec.bound_binary)
                if mode == "replace":
                    replacement = self.root / "replacement"
                    shutil.copyfile(self.b, replacement)
                    replacement.chmod(0o500)
                    replacement.replace(selected)
                else:
                    original = selected.stat()
                    selected.chmod(0o700)
                    selected.write_bytes(self.b.read_bytes())
                    os.utime(selected, ns=(original.st_atime_ns, original.st_mtime_ns))
                self.assertEqual(digest(captured), digest(self.a))
                release.set()
                blocker.result(timeout=5)
                self.assertEqual(self.finish(backend, job)["marker"], "A")
                self.assertFalse(captured.parent.exists())

    def test_bare_command_uses_request_path_before_queueing(self):
        first, second = self.root / "first", self.root / "second"
        first.mkdir()
        second.mkdir()
        marker(first / "ptxv-control", "A")
        marker(second / "ptxv-control", "B")
        backend = self.backend("ptxv-control")
        release, blocker = self.hold(backend)
        with mock.patch.dict(os.environ, {"PATH": str(first)}):
            self.assertEqual(backend.ptxv_hash(), digest(first / "ptxv-control"))
            job = ptxv.Session(backend, cache=False).submit(
                "check", [self.case], strict_memory_safety=True
            )
            captured = Path(job.spec.bound_binary)
            self.assertEqual(digest(captured), digest(first / "ptxv-control"))
            os.environ["PATH"] = str(second)
            release.set()
            blocker.result(timeout=5)
            self.assertEqual(self.finish(backend, job)["marker"], "A")
            self.assertFalse(captured.parent.exists())

    def test_old_contract_cache_entry_cannot_authorize_new_binding(self):
        old_args = {
            "strict_memory_safety": True,
            "_strict_memory_safety_contract": 2,
        }
        old_key = job_key("check", old_args, [tree_hash(self.case)], digest(self.a))
        store_cache(cache_dir("results") / f"{old_key}.json", {"marker": "B"})
        backend = self.backend(self.a)
        job = ptxv.Session(backend).submit(
            "check", [self.case], strict_memory_safety=True
        )
        self.assertEqual(self.finish(backend, job)["marker"], "A")
        self.assertFalse(job.cached)
        self.assertNotEqual(job.key, old_key)
        self.assertEqual(job.spec.args["_strict_memory_safety_contract"], 3)

    def test_canceled_and_shutdown_queued_jobs_release_executables(self):
        for shutdown in [False, True]:
            with self.subTest(shutdown=shutdown):
                backend = self.backend(self.a)
                release, blocker = self.hold(backend)
                job = ptxv.Session(backend, cache=False).submit(
                    "check", [self.case], strict_memory_safety=True
                )
                captured = Path(job.spec.bound_binary)
                if shutdown:
                    backend.shutdown()
                else:
                    self.assertTrue(job.cancel())
                self.assertTrue(job.cancelled())
                self.assertFalse(captured.parent.exists())
                self.assertEqual(job.spec._input_users, 0)
                release.set()
                blocker.result(timeout=5)

    def test_popen_failure_releases_executable(self):
        captured = []
        original = _strict_inputs.copy_strict_executable

        def copy(path):
            owner = original(path)
            captured.append(Path(owner.name))
            return owner

        backend = self.backend(self.a)
        with mock.patch("ptxv._jobs.copy_strict_executable", side_effect=copy):
            with mock.patch(
                "ptxv._jobs.subprocess.Popen", side_effect=OSError("spawn")
            ):
                job = ptxv.Session(backend, cache=False).submit(
                    "check", [self.case], strict_memory_safety=True
                )
                with self.assertRaisesRegex(OSError, "spawn"):
                    job.result(timeout=5)
                backend._pool.submit(lambda: None).result(timeout=5)
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0].parent.exists())
        self.assertEqual(job.spec._input_users, 0)


@unittest.skipUnless(os.environ.get("PTXV_BIN"), "set PTXV_BIN for actual CLI controls")
class StrictBinaryCliTests(unittest.TestCase):
    def test_qualified_native_executable_is_copied_and_executed(self):
        binary = Path(os.environ["PTXV_BIN"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "native-binary"
            case.mkdir()
            (case / "kernel.ptx").write_text(
                ".version 9.0\n.target sm_103a\n.address_size 64\n"
                ".global .align 8 .u64 x[2]={0,0};\n"
                ".visible .entry k(){st.global.u64 [x],1;ret;}\n"
            )
            (case / "case.toml").write_text(
                'name="native-binary"\ndescription="owned ELF control"\n'
                'ptx="kernel.ptx"\nentry="k"\ngrid=[1,1,1]\nblock=[1,1,1]\n'
                'outputs=[]\nparams=[]\n[[buffers]]\nname="x"\ndtype="u64"\n'
                'shape=[2]\ninit="zeros"\nsymbol="x"\n[[variants]]\nid="good"\n'
                'summary="control"\nexpect=["clean"]\n'
            )
            backend = ptxv.LocalBackend(
                workers=1, mem_cap=None, threads=1, binary=binary, root=root / "jobs"
            )
            with mock.patch.dict(os.environ, {"PTXV_CACHE": str(root / "cache")}):
                try:
                    session = ptxv.Session(backend, cache=False)
                    document = session.submit(
                        "check",
                        [case],
                        strict_memory_safety=True,
                        max_states=100,
                        certify=False,
                        gb300=False,
                        timeout=60,
                    ).result(timeout=65)
                    backend._pool.submit(lambda: None).result(timeout=5)
                    self.assertTrue(ptxv.Verdict.from_json(document).memory_safe)
                    self.assertEqual(document["ptxv_binary_sha256"], digest(binary))
                    command = json.loads(
                        (Path(document["work_dir"]) / "cmd.json").read_text()
                    )
                    self.assertNotEqual(Path(command[0]), binary)
                    self.assertTrue(
                        Path(command[0]).parent.name.startswith("ptxv-strict-binary-")
                    )
                    self.assertFalse(Path(command[0]).parent.exists())
                finally:
                    backend.shutdown()


if __name__ == "__main__":
    unittest.main()
