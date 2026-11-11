/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AlternateFloatAddExact

set_option maxRecDepth 100000
set_option exponentiation.threshold 1024
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem finite_fma_exact (sa sb sc : Bool) (a b c : Nat)
    (ha : a % 2^385 = 0) (hb : b % 2^385 = 0) :
    rational (some (add (mul (.finite sa a) (.finite sb b)) (.finite sc c))) =
      some ((((if sa then -1 else 1) * (a : ℚ) / unit) *
        ((if sb then -1 else 1) * (b : ℚ) / unit)) +
        ((if sc then -1 else 1) * (c : ℚ) / unit)) := by
  have hu : (unit : ℚ) ≠ 0 := by norm_num [unit, grid]
  have hp : ((a*b/unit : Nat) : ℚ) = (a : ℚ)*(b : ℚ)/(unit : ℚ) := by
    rw [Nat.cast_div (product_grid_divisible a b ha hb) hu, Nat.cast_mul]
  dsimp only [mul]
  rw [finite_add_exact]
  cases sa <;> cases sb <;> cases sc <;> simp <;> rw [hp] <;> ring

#print axioms finite_fma_exact
end GB300Math.AlternateFloatFloatLib
