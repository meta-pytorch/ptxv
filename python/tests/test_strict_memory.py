# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import ptxv
from ptxv._jobs import argv, JobSpec, store_cache
from ptxv._strict_inputs import STRICT_MEMORY_CONTRACT
from ptxv._util import job_key, tree_hash
from ptxv.daemon import Daemon, make_handler, pack_case

SCOPE = "raw_ptx_fixed_launch_all_payloads_legal_placements"


def document(report=None, *, present=False):
    row = {
        "got": ["clean"],
        "pass": True,
        "verdict": {
            "complete": True,
            "states": 1,
            "findings": [],
            "assumptions": ["E-WELLSYNC"],
        },
    }
    if present:
        row["strict_memory_safety"] = report
    return {"rows": [row]}


class Backend:
    def __init__(self):
        self.specs = []

    def ptxv_hash(self):
        return "fixed-binary"

    def prepare_spec(self, spec):
        spec.bind_strict_binary("/usr/bin/true")

    def ptxv_hash_for(self, spec):
        self.prepare_spec(spec)
        return spec.bound_binary_hash or self.ptxv_hash()

    def submit(self, spec, job):
        self.specs.append(spec)
        job.set_running_or_notify_cancel()
        job.set_result(document({"scope": SCOPE, "status": "safe"}, present=True))
        return job


class StrictMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = mock.patch.dict(os.environ, {"PTXV_CACHE": str(self.root / "cache")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.case = self.root / "case"
        self.case.mkdir()
        (self.case / "case.toml").write_text(
            'name = "cache-control"\nptx="kernel.ptx"\n'
        )
        (self.case / "kernel.ptx").write_text(".version 9.0\n")

    def test_strict_flag_and_front_selection(self):
        for options, expected, enabled in [
            ({}, "sass", False),
            ({"strict_memory_safety": False}, "sass", False),
            ({"strict": False}, "sass", False),
            ({"strict": True}, "ptx", True),
            ({"strict": True, "strict_memory_safety": False}, "ptx", True),
            ({"strict": True, "nonfinite": False}, "ptx", True),
            ({"strict": True, "front": "sass"}, "sass", True),
            ({"strict": True, "front": "both"}, "both", True),
            ({"strict_memory_safety": True}, "ptx", True),
            ({"strict_memory_safety": True, "front": "sass"}, "sass", True),
            ({"strict_memory_safety": True, "front": "both"}, "both", True),
        ]:
            with self.subTest(options=options):
                command, _ = argv(JobSpec("check", [self.case], options), self.root)
                self.assertEqual(command[command.index("--front") + 1], expected)
                self.assertEqual("--strict-memory-safety" in command, enabled)
                self.assertEqual(
                    "--nonfinite-strict" in command, bool(options.get("strict"))
                )
                self.assertNotIn("_strict_memory_safety_contract", command)

    def test_legacy_clean_and_repr_do_not_imply_memory_safety(self):
        verdict = ptxv.Verdict.from_json(document())
        self.assertTrue(verdict.clean)
        self.assertTrue(verdict.certified)
        self.assertFalse(verdict.memory_safe)
        self.assertIsNone(verdict.strict_memory_safety)
        self.assertNotIn("strict_memory_safety", repr(verdict))

    def test_missing_requested_report_fails_closed_for_sync_and_async_checks(self):
        backend = Backend()

        def ordinary(spec, job):
            job.set_running_or_notify_cancel()
            job.set_result(document())
            return job

        backend.submit = ordinary
        session = ptxv.Session(backend, cache=False)
        for options in [{"strict_memory_safety": True}, {"strict": True}]:
            for wait in [True, False]:
                with self.subTest(options=options, wait=wait):
                    result = session.check(self.case, wait=wait, **options)
                    verdict = result if wait else result.result()
                    self.assertFalse(verdict.memory_safe)
                    self.assertFalse(verdict.clean)
                    self.assertIn("strict_memory_safety='missing'", repr(verdict))

    def test_strict_shortcut_uses_expanded_cache_identity_and_owned_inputs(self):
        backend = Backend()
        session = ptxv.Session(backend)
        old_key = job_key(
            "check", {"strict": True}, [tree_hash(self.case)], backend.ptxv_hash()
        )
        store_cache(self.root / "cache/results" / f"{old_key}.json", document())
        first = session.submit("check", [self.case], strict=True)
        self.assertFalse(first.cached)
        self.assertNotEqual(first.key, old_key)
        second = session.submit(
            "check", [self.case], strict_memory_safety=True, nonfinite="strict"
        )
        self.assertTrue(second.cached)
        self.assertEqual(first.key, second.key)
        self.assertEqual(len(backend.specs), 1)
        spec = JobSpec("check", [self.case], {"strict": True})
        spec.retain_inputs()
        self.addCleanup(spec.release_inputs)
        self.assertNotEqual(spec.cases, [str(self.case)])
        self.assertEqual(
            Path(spec.cases[0], "kernel.ptx").read_bytes(),
            (self.case / "kernel.ptx").read_bytes(),
        )

    def test_strict_shortcut_is_rejected_for_other_commands(self):
        for kind in ["compare", "equiv", "errdist"]:
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "only for check"):
                    JobSpec(kind, [self.case], {"strict": True})

    def test_report_is_exposed_and_only_exact_safe_scope_is_memory_safe(self):
        report = {"scope": SCOPE, "status": "safe", "states": 4}
        verdict = ptxv.Verdict.from_json(document(report, present=True))
        self.assertEqual(verdict.strict_memory_safety, report)
        self.assertTrue(verdict.memory_safe)
        self.assertTrue(verdict.clean)
        self.assertIn("strict_memory_safety='safe'", repr(verdict))
        self.assertIn("memory_safe=True", repr(verdict))
        for rejected in [
            None,
            [],
            "safe",
            {},
            {"scope": SCOPE, "status": "unsafe"},
            {"scope": SCOPE, "status": "inconclusive"},
            {"scope": SCOPE, "status": "unknown"},
            {"scope": "typed_ir_model", "status": "safe"},
            {"status": "safe"},
            {"scope": SCOPE},
        ]:
            with self.subTest(report=rejected):
                verdict = ptxv.Verdict.from_json(document(rejected, present=True))
                self.assertFalse(verdict.memory_safe)
                self.assertFalse(verdict.clean)
                self.assertIn("memory_safe=False", repr(verdict))

    def test_strict_request_ignores_old_cached_ordinary_result_then_caches_proof(self):
        backend = Backend()
        session = ptxv.Session(backend)
        options = {"strict_memory_safety": True}
        old_key = job_key("check", options, [tree_hash(self.case)], backend.ptxv_hash())
        store_cache(self.root / "cache/results" / f"{old_key}.json", document())
        first = session.submit("check", [self.case], **options)
        self.assertFalse(first.cached)
        self.assertNotEqual(first.key, old_key)
        self.assertTrue(ptxv.Verdict.from_json(first.result()).memory_safe)
        second = session.submit("check", [self.case], **options)
        self.assertTrue(second.cached)
        self.assertEqual(first.key, second.key)
        self.assertEqual(len(backend.specs), 1)
        overridden = JobSpec(
            "check", [self.case], {**options, "_strict_memory_safety_contract": 0}
        )
        self.assertEqual(session.key(overridden), first.key)

    def test_other_request_keys_are_unchanged(self):
        backend = Backend()
        session = ptxv.Session(backend)
        for kind, options in [
            ("check", {}),
            ("check", {"strict_memory_safety": False}),
            ("compare", {"strict_memory_safety": True}),
            ("check", {"front": "sass", "max_states": 5, "timeout": 3}),
        ]:
            with self.subTest(kind=kind, options=options):
                previous = {
                    key: value for key, value in options.items() if key != "timeout"
                }
                expected = job_key(
                    kind, previous, [tree_hash(self.case)], backend.ptxv_hash()
                )
                self.assertEqual(
                    session.key(JobSpec(kind, [self.case], options)), expected
                )

    def test_daemon_strict_request_also_ignores_old_server_cache(self):
        self._assert_daemon_strict_cache({"strict_memory_safety": True})

    def test_daemon_strict_shortcut_ignores_old_server_cache(self):
        self._assert_daemon_strict_cache({"strict": True})

    def _assert_daemon_strict_cache(self, options):
        daemon = Daemon(self.root / "daemon", mem_cap=None, binary="/usr/bin/true")
        daemon.local.shutdown()
        backend = Backend()
        daemon.local = backend
        content = tree_hash(self.case)
        daemon.put_case(content, pack_case(self.case))
        old_key = job_key("check", options, [content], daemon.ptxv_hash)
        store_cache(daemon.root / "results" / f"{old_key}.json", document())
        body = {
            "kind": "check",
            "cases": [content],
            "args": {**options, "_strict_memory_safety_contract": 0},
        }
        identity, cached = daemon.submit(copy.deepcopy(body))
        self.assertFalse(cached)
        self.assertNotEqual(identity, old_key[:16])
        self.assertTrue(
            ptxv.Verdict.from_json(daemon.jobs[identity].result()).memory_safe
        )
        same, cached = daemon.submit(copy.deepcopy(body))
        self.assertEqual(same, identity)
        self.assertTrue(cached)
        self.assertEqual(len(backend.specs), 1)

    def test_daemon_acknowledges_both_strict_request_forms(self):
        daemon = mock.Mock()
        daemon.submit.return_value = ("job-id", False)
        handler_type = make_handler(daemon)
        for options, acknowledged in [
            ({"strict": True}, True),
            ({"strict_memory_safety": True}, True),
            ({"strict": False}, False),
            ({}, False),
        ]:
            with self.subTest(options=options):
                handler = handler_type.__new__(handler_type)
                handler.path = "/v1/jobs"
                handler._body = mock.Mock(
                    return_value=json.dumps({"kind": "check", "args": options}).encode()
                )
                handler._json = mock.Mock()
                handler.do_POST()
                expected = {"id": "job-id", "cached": False}
                if acknowledged:
                    expected["strict_memory_safety_contract"] = STRICT_MEMORY_CONTRACT
                handler._json.assert_called_once_with(200, expected)


@unittest.skipUnless(
    os.environ.get("PTXV_BIN"), "set PTXV_BIN to run actual CLI controls"
)
class StrictMemoryCliTests(unittest.TestCase):
    def test_actual_cli_reports_safe_limits_unsafe_and_source_refused(self):
        artifacts = os.environ.get("PTXV_TEST_ARTIFACTS")
        if artifacts:
            artifact_root = Path(artifacts) / "strict-memory-cli"
            artifact_root.mkdir(parents=True)
            directory = nullcontext(str(artifact_root))
        else:
            directory = tempfile.TemporaryDirectory()
        with directory as temporary:
            root = Path(temporary)
            backend = ptxv.LocalBackend(
                workers=1, threads=4, mem_cap=None, root=root / "jobs"
            )
            session = ptxv.Session(backend, cache=False)
            self.addCleanup(session.close)
            controls = [
                (
                    "safe",
                    "st.global.u64 [x],1; ld.global.u64 %r0,[x];",
                    100,
                    {},
                    "safe",
                ),
                (
                    "cap",
                    "st.global.u64 [x],1; ld.global.u64 %r0,[x];",
                    1,
                    {},
                    "inconclusive",
                ),
                ("oob", "st.global.u64 [x+16],1;", 100, {}, "unsafe"),
                ("unsupported", "membar.gl;", 100, {}, "inconclusive"),
                (
                    "sass",
                    "st.global.u64 [x],1;",
                    100,
                    {"front": "sass"},
                    "inconclusive",
                ),
            ]
            for name, body, cap, extra, status in controls:
                with self.subTest(control=name):
                    case = root / name
                    case.mkdir()
                    (case / "kernel.ptx").write_text(
                        ".version 9.0\n.target sm_103a\n.address_size 64\n"
                        ".global .align 8 .u64 x[2]={0,0};\n"
                        f".visible .entry k(){{ .reg .b64 %r<2>; {body} ret; }}\n"
                    )
                    (case / "case.toml").write_text(
                        'name="python-strict-memory"\ndescription="actual CLI contract"\n'
                        'ptx="kernel.ptx"\nentry="k"\ngrid=[1,1,1]\nblock=[1,1,1]\n'
                        'outputs=[]\nparams=[]\n[[buffers]]\nname="x"\ndtype="u64"\n'
                        'shape=[2]\ninit="zeros"\nsymbol="x"\n[[variants]]\n'
                        'id="good"\nsummary="control"\nexpect=["clean"]\n'
                    )
                    verdict = session.check(
                        case,
                        strict=True,
                        max_states=cap,
                        certify=False,
                        gb300=False,
                        timeout=60,
                        **extra,
                    )
                    self.assertEqual(verdict.strict_memory_safety["scope"], SCOPE)
                    self.assertEqual(verdict.strict_memory_safety["status"], status)
                    self.assertEqual(verdict.memory_safe, status == "safe")
                    self.assertIn(f"strict_memory_safety='{status}'", repr(verdict))
                    if status != "safe":
                        self.assertFalse(verdict.clean)
                        self.assertFalse(verdict.passed)
                    if name == "unsupported":
                        self.assertEqual(
                            verdict.strict_memory_safety["reason"]["kind"],
                            "source_refused",
                        )
                    command = json.loads(
                        sorted(
                            (root / "jobs").glob("*/cmd.json"),
                            key=lambda path: path.stat().st_mtime_ns,
                        )[-1].read_text()
                    )
                    self.assertNotIn("systemd-run", command)
                    self.assertIn("--strict-memory-safety", command)
                    self.assertEqual(
                        command[command.index("--front") + 1], extra.get("front", "ptx")
                    )


if __name__ == "__main__":
    unittest.main()
