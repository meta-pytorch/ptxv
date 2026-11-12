# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`Verdict` reads ptxv's per-comparison `gpu_compare`, never the report text: an output only
within one ulp of the captured GPU output is no match (its line also says "the rest bitwise
equal"), and `certified` needs a complete verdict. No GPU or binary needed.

    python -m pytest python/tests/test_results.py   (or python -m unittest)
"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("PTXV_NO_IMPORT_HOOK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptxv.results import Verdict  # noqa: E402

ULP = (
    "output 0: out0: within one ulp of the captured GPU output: 2 of 4 elements one unit in "
    "the last place off, the rest bitwise equal (multimem.ld_reduce rounding, "
    "N-MULTIMEM-ROUND16)"
)
BITWISE = "output 0: out1: bitwise equal to the captured GPU output (16 bytes)"
MISMATCH = (
    "output 0: out0: MISMATCH vs the captured GPU output: 4 of 16 bytes differ, first at "
    "byte 0"
)


def verdict(lines, gpu_compare=None, complete=True, assumptions=("E-WELLSYNC",)):
    row = {
        "got": ["clean"],
        "pass": True,
        "expect_report": list(lines),
        "verdict": {
            "assumptions": list(assumptions),
            "complete": complete,
            "states": 45,
            "findings": [],
        },
    }
    if gpu_compare is not None:
        row["gpu_compare"] = [
            {"output": o, "part": o, "result": r} for o, r in gpu_compare
        ]
    return Verdict.from_json({"rows": [row]})


class GpuCompare(unittest.TestCase):
    def test_within_one_ulp_is_no_match(self):
        # The r1h_nvls shape (multicast reduction: -0.0 vs NaN and +inf vs FLT_MAX are one
        # raw-bit step apart): out0 within one ulp, out1 bitwise.
        v = verdict([ULP, BITWISE], [("out0", "within_ulp"), ("out1", "bitwise")])
        self.assertEqual(v.gpu_state, "within_ulp")
        self.assertIs(v.sim_equals_gpu, False)
        self.assertIn("sim!=GPU (within_ulp)", repr(v))
        self.assertNotIn("sim=GPU", repr(v))

    def test_results(self):
        self.assertIs(
            verdict([BITWISE], [("out1", "bitwise")]).sim_equals_gpu, True
        )
        v = verdict([MISMATCH, ULP], [("out0", "mismatch"), ("out1", "within_ulp")])
        self.assertEqual((v.gpu_state, v.sim_equals_gpu), ("mismatch", False))
        for r in ("permuted_equal", "not_compared"):
            v = verdict(["output 0: counter word 0: ..."], [("out0", r)])
            self.assertEqual((v.gpu_state, v.sim_equals_gpu), (r, False))
        # Outputs without captured bytes are not compared.
        self.assertIsNone(verdict([], [("out0", "no_capture")]).sim_equals_gpu)
        self.assertIsNone(verdict([]).sim_equals_gpu)

    def test_outputs_without_gpu_compare(self):
        # Binaries before `gpu_compare`: whole bitwise-equal sentences only.
        self.assertIs(verdict([BITWISE]).sim_equals_gpu, True)
        self.assertEqual(verdict([ULP, BITWISE]).gpu_state, "within_ulp")
        self.assertIs(verdict([ULP, BITWISE]).sim_equals_gpu, False)
        self.assertIs(verdict([BITWISE + " (rerun)"]).sim_equals_gpu, False)
        self.assertIs(verdict([MISMATCH]).sim_equals_gpu, False)


class Certified(unittest.TestCase):
    def test_certified_needs_a_complete_verdict(self):
        self.assertTrue(verdict([BITWISE], [("out1", "bitwise")]).certified)
        v = verdict([BITWISE], [("out1", "bitwise")], complete=False)
        self.assertFalse(v.certified)
        self.assertIn("INCOMPLETE", repr(v))
        self.assertFalse(verdict([], assumptions=()).certified)


if __name__ == "__main__":
    unittest.main()
