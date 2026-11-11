/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Numerics.Enclosure.Elementary.Proof

namespace GB300Math.Texture

noncomputable def quantizedWeight (x : ℝ) : ℝ := (⌊256 * x + 1 / 2⌋ : ℝ) / 256

def interpolate (a b weight : ℝ) : ℝ := a * (1 - weight) + b * weight

theorem weight_error (x : ℝ) : |quantizedWeight x - x| ≤ 1 / 512 := by
  have lo := Int.floor_le (256 * x + 1 / 2)
  have hi := Int.lt_floor_add_one (256 * x + 1 / 2)
  unfold quantizedWeight
  apply abs_le.mpr
  constructor <;> linarith

theorem interpolation_error (a b x : ℝ) :
    |interpolate a b (quantizedWeight x) - interpolate a b x| ≤ |b - a| / 512 := by
  have equality : interpolate a b (quantizedWeight x) - interpolate a b x =
      (b - a) * (quantizedWeight x - x) := by unfold interpolate; ring
  rw [equality, abs_mul]
  have h := mul_le_mul_of_nonneg_left (weight_error x) (abs_nonneg (b - a))
  linarith

theorem weight_zero : quantizedWeight 0 = 0 := by norm_num [quantizedWeight]
theorem weight_one : quantizedWeight 1 = 1 := by norm_num [quantizedWeight]

#print axioms weight_error
#print axioms interpolation_error

end GB300Math.Texture
