# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptxv import LocalBackend, Session
from ptxv.results import Verdict

SCOPE = "raw_ptx_fixed_launch_all_payloads_legal_placements"


def row(status=None):
    value = {
        "got": ["clean"],
        "pass": status in [None, "safe"],
        "verdict": {
            "complete": True,
            "states": 3,
            "findings": [],
            "assumptions": ["E-WELLSYNC"],
        },
    }
    if status is not None:
        value["strict_memory_safety"] = {"scope": SCOPE, "status": status}
    return value


class VerdictRowsTests(unittest.TestCase):
    def test_empty_result_remains_an_error(self):
        for document in [{}, {"rows": []}, {"rows": None}]:
            with self.subTest(document=document):
                with self.assertRaisesRegex(ValueError, "check produced no rows"):
                    Verdict.from_json(document)

    def test_multiple_rows_are_rejected_independent_of_order_or_status(self):
        for statuses in [
            ["safe", "unsafe"],
            ["unsafe", "safe"],
            ["safe", "safe"],
            ["unsafe", "unsafe"],
            ["safe", "inconclusive"],
            [None, None],
        ]:
            with self.subTest(statuses=statuses):
                document = {"rows": [row(status) for status in statuses]}
                before = copy.deepcopy(document)
                with self.assertRaises(ValueError) as error:
                    Verdict.from_json(document)
                message = str(error.exception)
                self.assertIn("2 rows", message)
                self.assertIn("single variant", message)
                self.assertIn("single front", message)
                self.assertIn('Session.submit("check", [case]', message)
                self.assertEqual(document, before)

    def test_single_strict_row_preserves_status_and_fields(self):
        for status in ["safe", "unsafe", "inconclusive"]:
            with self.subTest(status=status):
                document = {"rows": [row(status)]}
                verdict = Verdict.from_json(document)
                self.assertIs(verdict.json, document)
                self.assertEqual(
                    verdict.strict_memory_safety,
                    document["rows"][0]["strict_memory_safety"],
                )
                self.assertEqual(verdict.memory_safe, status == "safe")
                self.assertEqual(verdict.clean, status == "safe")
                self.assertEqual(verdict.passed, status == "safe")
                self.assertTrue(verdict.complete)
                self.assertTrue(verdict.certified)
                self.assertEqual(verdict.states, 3)

    def test_single_legacy_row_preserves_ordinary_verdict(self):
        verdict = Verdict.from_json({"rows": [row()]})
        self.assertEqual(verdict.classes, ["clean"])
        self.assertTrue(verdict.clean)
        self.assertTrue(verdict.passed)
        self.assertTrue(verdict.complete)
        self.assertTrue(verdict.certified)
        self.assertFalse(verdict.memory_safe)
        self.assertIsNone(verdict.strict_memory_safety)

    def test_multiple_fronts_are_rejected_even_for_one_variant(self):
        ptx = row("safe")
        ptx.update({"front": "ptx", "variant": "good"})
        sass = row("inconclusive")
        sass.update({"front": "sass", "variant": "good"})
        with self.assertRaisesRegex(ValueError, "single front"):
            Verdict.from_json({"rows": [ptx, sass]})


@unittest.skipUnless(os.environ.get("PTXV_BIN"), "set PTXV_BIN for actual CLI controls")
class VerdictRowsCliTests(unittest.TestCase):
    def test_mixed_variants_require_raw_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "mixed-variants"
            case.mkdir()
            (case / "kernel.ptx").write_text(
                ".version 9.0\n.target sm_103a\n.address_size 64\n"
                ".global .align 8 .u64 x[2]={0,0};\n.visible .entry k(){\n"
                "st.global.u64 [x],0; // STORE\nret;\n}\n"
            )
            (case / "case.toml").write_text(
                'name="mixed-variants"\ndescription="scalar verdict control"\n'
                'ptx="kernel.ptx"\nentry="k"\n'
                "grid=[1,1,1]\nblock=[1,1,1]\noutputs=[]\nparams=[]\n"
                '[[buffers]]\nname="x"\ndtype="u64"\nshape=[2]\n'
                'init="zeros"\nsymbol="x"\n'
                '[[variants]]\nid="good"\nsummary="safe base"\nexpect=["clean"]\n'
                '[[variants]]\nid="bad"\nsummary="out of bounds"\n'
                'marker="STORE"\nfind="[x]"\nreplace="[x+16]"\nexpect=["clean"]\n'
            )
            options = {
                "front": "ptx",
                "strict_memory_safety": True,
                "max_states": 128,
                "certify": False,
                "gb300": False,
                "timeout": 60,
            }
            backend = LocalBackend(
                workers=1,
                threads=1,
                mem_cap=None,
                binary=os.environ["PTXV_BIN"],
                root=root / "jobs",
            )
            with mock.patch.dict(os.environ, {"PTXV_CACHE": str(root / "cache")}):
                with Session(backend, cache=False) as session:
                    document = session.submit(
                        "check", [case], variant="all", **options
                    ).result(timeout=65)
                    self.assertEqual(
                        [
                            item["strict_memory_safety"]["status"]
                            for item in document["rows"]
                        ],
                        ["safe", "unsafe"],
                    )
                    with self.assertRaisesRegex(ValueError, "single variant"):
                        session.check(case, variant="all", **options)
                    pending = session.check(case, variant="all", wait=False, **options)
                    with self.assertRaisesRegex(ValueError, "single variant"):
                        pending.result(timeout=65)
                    selected = session.check(case, variant="good", **options)
                    self.assertTrue(selected.memory_safe)
                    self.assertTrue(selected.clean)
                    self.assertTrue(selected.passed)


if __name__ == "__main__":
    unittest.main()
