/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorDecompression
import Mathlib.Tactic.IntervalCases

namespace GB300Math.TensorDecompression

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000

theorem fp4_padding_ignored (x : BitVec 128) :
    expand 4 x = expand 4 (x &&& 0xffffffffffffffff) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  interval_cases i <;> simp [expand, List.range_succ]

theorem fp6_padding_ignored (x : BitVec 128) :
    expand 6 x = expand 6 (x &&& 0xffffffffffffffffffffffff) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  interval_cases i <;> simp [expand, List.range_succ]

theorem fp4_roundtrip (x : BitVec 128) :
    compact 4 (expand 4 x) = x &&& 0xffffffffffffffff := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  interval_cases i <;> simp [compact, expand, List.range_succ]

theorem fp6_roundtrip (x : BitVec 128) :
    compact 6 (expand 6 x) = x &&& 0xffffffffffffffffffffffff := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  interval_cases i <;> simp [compact, expand, List.range_succ]

#print axioms fp4_padding_ignored
#print axioms fp6_padding_ignored
#print axioms fp4_roundtrip
#print axioms fp6_roundtrip
end GB300Math.TensorDecompression
