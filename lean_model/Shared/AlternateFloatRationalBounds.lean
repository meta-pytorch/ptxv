/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AlternateFloatRoundBounds
import AlternateFloatFmaExact

set_option maxRecDepth 100000
set_option exponentiation.threshold 1024
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem nearestEven_rational_half_quantum (m d : Nat) (hd : 0 < d) :
    |((nearestEven m d * 2^d : Nat) : ℚ) / unit - (m : ℚ) / unit| ≤
      (2^(d-1) : ℚ) / unit := by
  obtain ⟨hlo, hhi⟩ := nearestEven_half_quantum m d hd
  have hlo' : ((nearestEven m d * 2^d : Nat) : ℚ) ≤ (m : ℚ) + (2^(d-1) : ℚ) := by exact_mod_cast hlo
  have hhi' : (m : ℚ) ≤ ((nearestEven m d * 2^d : Nat) : ℚ) + (2^(d-1) : ℚ) := by exact_mod_cast hhi
  have hu : (0 : ℚ) < unit := by norm_num [unit, grid]
  rw [← sub_div, abs_div, abs_of_pos hu]
  apply (div_le_div_iff_of_pos_right hu).2
  rw [abs_le]
  constructor <;> linarith

theorem unit_quantum_exact (m : Nat) : nearestEven m 0 = m := by simp [nearestEven]

theorem half_quantum_attained :
    |((nearestEven 5 1 * 2^1 : Nat) : ℚ) / unit - (5 : ℚ) / unit| = 1 / (unit : ℚ) := by
  norm_num [nearestEven, unit, grid]

#print axioms nearestEven_rational_half_quantum
#print axioms unit_quantum_exact
#print axioms half_quantum_attained
end GB300Math.AlternateFloatFloatLib
