#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the BSD-style license in the repository root.
"""Regression tests for strict-memory proof qualification checks."""

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("checker_under_test", HERE / "check.py")
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)


class CheckerControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ptxv-checker-controls-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_checked_release_profile_and_package_overrides(self):
        for text in [
            "[workspace]\n",
            "[profile.release]\ndebug = 1\n",
            "[profile.release]\noverflow-checks = false\n",
            "[profile.release]\noverflow-checks = 1\n",
            '[profile.release]\noverflow-checks = "true"\n',
        ]:
            with self.subTest(text=text):
                self.put(self.root / "Cargo.toml", text)
                with self.assertRaisesRegex(ValueError, "overflow-checks = true"):
                    checker.check_build_mode(self.root)
        enabled = "[profile.release]\noverflow-checks = true\n"
        for name in ["ptxv-memory-write", "ptxv-sim", "*"]:
            for value in ["false", "0", '"true"']:
                with self.subTest(name=name, value=value):
                    self.put(
                        self.root / "Cargo.toml",
                        enabled
                        + f'[profile.release.package."{name}"]\noverflow-checks = {value}\n',
                    )
                    with self.assertRaisesRegex(ValueError, "package override"):
                        checker.check_build_mode(self.root)
        for suffix in [
            "",
            '[profile.release.package."ptxv-memory-write"]\noverflow-checks = true\n',
            '[profile.release.package."*"]\noverflow-checks = true\n',
            '[profile.release.package."ptxv-sim"]\nopt-level = 2\n',
        ]:
            with self.subTest(suffix=suffix):
                self.put(self.root / "Cargo.toml", enabled + suffix)
                checker.check_build_mode(self.root)

    def test_source_and_formal_freshness_require_checked_release(self):
        self.put(
            self.root / "Cargo.toml",
            "[profile.release]\noverflow-checks = false\n",
        )
        with patch.object(checker, "check_specification_inputs") as later_check:
            with self.assertRaisesRegex(ValueError, "overflow-checks = true"):
                checker.freshness(self.root, {})
            later_check.assert_not_called()
        self.put(self.root / "source-manifest.json", "{}")
        for flags in [["--source-only"], []]:
            with (
                self.subTest(flags=flags),
                patch.object(checker, "PROJECT", self.root),
                patch.object(
                    sys, "argv", ["check", "--ptxv-root", str(self.root), *flags]
                ),
            ):
                with self.assertRaisesRegex(ValueError, "overflow-checks = true"):
                    checker.main()

    def test_sealer_rejects_unchecked_release_before_writing(self):
        project = self.root / "project"
        project.mkdir()
        shutil.copyfile(HERE / "check.py", project / "check.py")
        shutil.copyfile(
            HERE.parent / "generated_lean.py", self.root / "generated_lean.py"
        )
        for text in [
            "[workspace]\n",
            "[profile.release]\noverflow-checks = false\n",
            "[profile.release]\noverflow-checks = true\n"
            '[profile.release.package."ptxv-memory-write"]\noverflow-checks = false\n',
        ]:
            with self.subTest(text=text):
                self.put(self.root / "Cargo.toml", text)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(HERE / "seal.py"),
                        "--project",
                        str(project),
                        "--ptxv-root",
                        str(self.root),
                        "--extraction",
                        str(self.root / "not-yet-extracted"),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("overflow-checks", result.stderr)
                self.assertFalse((project / "source-manifest.json").exists())
                self.assertFalse((project / "qualified-extraction.json").exists())

    def test_kernel_option_lexing(self):
        for spelling in [
            "set_option debug.skipKernelTC\n true",
            "set_option /- nested /- comment -/ -/ debug . skipKernelTC true in theorem x : True := .intro",
            "set_option\u2003debug.skipKernelTC\u00a0true",
            "set_option debug.«skipKernelTC» true",
        ]:
            with self.subTest(spelling=spelling), self.assertRaises(ValueError):
                checker.require_kernel_checking(spelling, "fixture")
        checker.require_kernel_checking(
            """/- set_option debug.skipKernelTC true -/
-- set_option debug.skipKernelTC true
 def prose := "set_option debug.skipKernelTC true"
 def raw := r##"set_option debug.skipKernelTC true"##
 def quote := '"'
 def slash := '/'
 set_option debug.skipKernelTC false in theorem x : True := .intro
""",
            "fixture",
        )
        with self.assertRaises(ValueError):
            checker.require_kernel_checking("/- unclosed", "fixture")

    def backend_fixture(self):
        base = self.root / "backend"
        external = self.root / "floatlib"
        for location in (base, external):
            self.put(location / "lakefile.lean", "import Lake\n")
            self.put(location / "Model.lean", "def model := True\n")
            self.put(location / ".lake/build/lib/lean/Model.olean", "compiled model")
            self.put(
                location / ".lake/build/lib/lean/Model.olean.private",
                "private proof terms",
            )
            self.put(
                location / ".lake/build/lib/lean/Model.olean.server",
                "server environment",
            )
        self.put(
            base / "lake-manifest.json",
            json.dumps(
                {
                    "packages": [
                        {
                            "name": "floatlib",
                            "type": "path",
                            "dir": "../floatlib",
                            "configFile": "lakefile.lean",
                        }
                    ]
                }
            ),
        )
        return base, external

    def test_backend_policy_covers_sources_split_oleans_and_external_paths(self):
        base, external = self.backend_fixture()
        original = checker.backend_policy(checker.backend_records(base))
        policy = self.put(
            self.root / "project/backend-inputs.json", json.dumps(original)
        )
        with patch.object(checker, "PROJECT", policy.parent):
            checker.check_backend(base)
            for location in [
                base / "Model.lean",
                external / "Model.lean",
                external / ".lake/build/lib/lean/Model.olean.private",
                external / ".lake/build/lib/lean/Model.olean.server",
            ]:
                prior = location.read_bytes()
                location.write_bytes(prior + b"changed")
                with self.subTest(path=location), self.assertRaises(ValueError):
                    checker.check_backend(base)
                location.write_bytes(prior)
            added = self.put(external / "Added.lean", "theorem added : True := .intro")
            with self.assertRaises(ValueError):
                checker.check_backend(base)
            added.unlink()
            hidden = external / ".lake/build/lib/lean/Hidden"
            hidden.symlink_to(base / ".lake/build/lib/lean", target_is_directory=True)
            with self.assertRaises(ValueError):
                checker.check_backend(base)
            hidden.unlink()
            checker.check_backend(base)
        relocated = self.root / "relocated"
        shutil.copytree(base, relocated / "backend")
        shutil.copytree(external, relocated / "floatlib")
        self.assertEqual(
            original,
            checker.backend_policy(checker.backend_records(relocated / "backend")),
        )

    def extraction_fixture(self):
        project = self.root / "project"
        out = self.root / "extraction"
        inputs = {"crates/core.rs": "a" * 64}
        manifest = {"extraction_inputs": inputs}
        for name in ["Funs.lean", "Types.lean"]:
            content = f"-- THIS FILE WAS AUTOMATICALLY GENERATED BY AENEAS\n-- {name}\n"
            self.put(project / "MemorySafety/Code" / name, content)
            self.put(out / "fresh/MemorySafety/Code" / name, content)
        generated_export = checker.LEAN_EXPORT.publish(
            out / "fresh/MemorySafety/Code", out / "raw-generated"
        )
        for name in checker.LEAN_EXPORT.NAMES:
            shutil.copyfile(
                out / "fresh/MemorySafety/Code" / name,
                project / "MemorySafety/Code" / name,
            )
        self.put(project / "extraction-spec.json", "{}\n")
        for name in ["source-before.json", "source-after.json"]:
            self.put(out / name, json.dumps(inputs))
        self.put(out / "Program.llbc", "genuine extraction fixture")
        outcomes = []
        for index in range(2):
            log = self.put(out / f"command{index}.log", "command finished")
            outcomes.append(
                {"exit_code": 0, "log": str(log), "log_sha256": checker.digest(log)}
            )
        receipt = {
            "status": "pass",
            "generated_byte_equal": {"Funs.lean": True, "Types.lean": True},
            "generated_export": generated_export,
            "outcomes": outcomes,
            "source_manifest_sha256": checker.digest(out / "source-before.json"),
            "specification_sha256": checker.digest(project / "extraction-spec.json"),
            "llbc_sha256": checker.digest(out / "Program.llbc"),
        }
        self.put(out / "receipt.json", json.dumps(receipt))
        return project, out, manifest

    def test_fresh_extraction_result_checks_exact_code_inputs_and_outputs(self):
        project, out, manifest = self.extraction_fixture()
        with patch.object(checker, "PROJECT", project):
            checker.check_fresh_extraction(out, manifest)
            generated = out / "fresh/MemorySafety/Code/Funs.lean"
            prior = generated.read_bytes()
            generated.write_bytes(prior + b"-- stale")
            with self.assertRaises(ValueError):
                checker.check_fresh_extraction(out, manifest)
            generated.write_bytes(prior)
            extra = self.put(
                out / "fresh/MemorySafety/Code/Extern.lean", "axiom external : False"
            )
            with self.assertRaises(ValueError):
                checker.check_fresh_extraction(out, manifest)
            extra.unlink()
            self.put(out / "source-after.json", "{}")
            with self.assertRaises(ValueError):
                checker.check_fresh_extraction(out, manifest)

    def test_generated_export_rejects_raw_body_and_preamble_changes(self):
        project, out, manifest = self.extraction_fixture()
        raw = out / "raw-generated/Funs.lean"
        original = raw.read_bytes()
        with patch.object(checker, "PROJECT", project):
            raw.write_bytes(original + b"-- changed raw tool output")
            with self.assertRaisesRegex(ValueError, "differs from raw Aeneas"):
                checker.check_fresh_extraction(out, manifest)
            raw.write_bytes(original)
            published = out / "fresh/MemorySafety/Code/Funs.lean"
            original = published.read_bytes()
            published.write_bytes(original.replace(b"BSD-style", b"changed", 1))
            with self.assertRaisesRegex(ValueError, "license preamble"):
                checker.check_fresh_extraction(out, manifest)

    def test_generated_export_preserves_every_raw_byte(self):
        generated = self.root / "generated"
        raw = b"-- THIS FILE WAS AUTOMATICALLY GENERATED BY AENEAS\r\nmodule\r\n"
        for name in checker.LEAN_EXPORT.NAMES:
            path = generated / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
        evidence = self.root / "raw"
        record = checker.LEAN_EXPORT.publish(generated, evidence)
        checker.LEAN_EXPORT.verify(generated, record, evidence)
        for name in checker.LEAN_EXPORT.NAMES:
            self.assertEqual((evidence / name).read_bytes(), raw)
            self.assertEqual(
                (generated / name).read_bytes(), checker.LEAN_EXPORT.PREAMBLE + raw
            )
        with self.assertRaisesRegex(ValueError, "raw Aeneas output"):
            checker.LEAN_EXPORT.publish(generated, self.root / "second-export")

    def test_required_type_policy_rejects_missing_extra_and_duplicate(self):
        expected = {"T": {"sha256": checker.hashlib.sha256(b"type").hexdigest()}}
        log = self.put(
            self.root / "audit.log", 'ENDPOINT_TYPE {"name":"T","canonical":"type"}\n'
        )
        checker.check_endpoint_types(log, expected)
        log.write_text('ENDPOINT_TYPE {"name":"T","canonical":"weaker type"}\n')
        with self.assertRaises(ValueError):
            checker.check_endpoint_types(log, expected)
        log.write_text('ENDPOINT_TYPE {"name":"T","canonical":"type"}\n' * 2)
        with self.assertRaises(ValueError):
            checker.check_endpoint_types(log, expected)
        log.write_text(
            'ENDPOINT_TYPE {"name":"T","canonical":"type"}\n'
            'ENDPOINT_TYPE {"name":"Unrequested","canonical":"type"}\n'
        )
        with self.assertRaises(ValueError):
            checker.check_endpoint_types(log, expected)
        log.write_text("")
        with self.assertRaises(ValueError):
            checker.check_endpoint_types(log, expected)

    def test_formal_main_requires_tools_before_any_proof_build(self):
        project = self.root / "project"
        self.put(project / "source-manifest.json", "{}")
        with (
            patch.object(checker, "PROJECT", project),
            patch.object(checker, "freshness", return_value={}),
            patch.object(sys, "argv", ["check", "--ptxv-root", str(self.root)]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as rejected,
        ):
            checker.main()
        self.assertEqual(rejected.exception.code, 2)

    def test_missing_trust_policy_cannot_be_omitted_from_source_seal(self):
        self.put(
            self.root / "Cargo.toml",
            "[profile.release]\noverflow-checks = true\n",
        )
        with self.assertRaisesRegex(ValueError, "trust-policy"):
            checker.freshness(self.root, {"project_inputs": {}})

    def test_reviewed_specification_survives_ordinary_seal_regeneration(self):
        project = self.root / "project"
        for relative in checker.SPECIFICATION_FILES:
            self.put(project / relative, "def reviewed : Prop := True ∧ True\n")
        records = {
            relative: checker.digest(project / relative)
            for relative in checker.SPECIFICATION_FILES
        }
        policy = {
            "version": 1,
            "selection": "reviewed-memory-safety-specification-v1",
            "files": records,
        }
        path = self.put(project / "specification-inputs.json", json.dumps(policy))
        with patch.object(checker, "PROJECT", project):
            checker.check_specification_inputs()
            predicate = project / "MemorySafetyExecution/SourceMemorySafety.lean"
            previous = predicate.read_bytes()
            predicate.write_text("def reviewed : Prop := True\n")
            self.put(
                project / "source-manifest.json",
                json.dumps(
                    {"project_inputs": {str(predicate): checker.digest(predicate)}}
                ),
            )
            with self.assertRaisesRegex(ValueError, "Stale input"):
                checker.check_specification_inputs()
            predicate.write_bytes(previous)
            checker.check_specification_inputs()
            omitted = dict(policy)
            omitted["files"] = dict(records)
            omitted["files"].pop("MemorySafety/Semantics.lean")
            path.write_text(json.dumps(omitted))
            with self.assertRaisesRegex(ValueError, "cover each required source"):
                checker.check_specification_inputs()
            path.write_text(json.dumps(policy))
            checker.check_specification_inputs()


if __name__ == "__main__":
    unittest.main(verbosity=2)
