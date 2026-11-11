/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Mathlib.Tactic
import FloatLib.Numerics.Quantization.Deterministic.Rational
import FloatLib.Numerics.Quantization.Deterministic.ShiftRight

namespace GB300Math.RcpPtxBounds

theorem quarter_error_bound (actual seed ideal bound : ℚ)
    (hseed : |seed - 4 * ideal| ≤ bound)
    (hround : |actual - seed / 4| ≤ 1 / 2^150) :
    |actual - ideal| ≤ 1 / 2^150 + bound / 4 := by
  have h := abs_add_le (actual - seed / 4) ((seed - 4 * ideal) / 4)
  rw [show actual - seed / 4 + (seed - 4 * ideal) / 4 = actual - ideal by ring,
    abs_div] at h
  norm_num at h
  have hdiv := div_le_div_of_nonneg_right hseed (by norm_num : (0 : ℚ) ≤ 4)
  linarith

theorem quarter_error_exponent253 (actual seed ideal : ℚ)
    (hseed : |seed - 4 * ideal| ≤ 1 / 2^148)
    (hround : |actual - seed / 4| ≤ 1 / 2^150) :
    |actual - ideal| ≤ 1 / 2^149 := by
  have h := quarter_error_bound actual seed ideal (1 / 2^148) hseed hround
  norm_num at h ⊢
  exact h

theorem quarter_error_exponent254 (actual seed ideal : ℚ)
    (hseed : |seed - 4 * ideal| ≤ 1 / 2^149)
    (hround : |actual - seed / 4| ≤ 1 / 2^150) :
    |actual - ideal| ≤ 3 / 2^151 := by
  have h := quarter_error_bound actual seed ideal (1 / 2^149) hseed hround
  norm_num at h ⊢
  exact h

theorem rounded_subnormal_error (mantissa shift : Nat) :
    |(FloatLib.Numerics.roundShiftRightEven mantissa shift : ℚ) / 2^149 -
      (mantissa : ℚ) / (2^shift * 2^149)| ≤ 1 / 2^150 := by
  have h : |(FloatLib.Numerics.roundShiftRightEven mantissa shift : ℚ) -
      (mantissa : ℚ) / 2^shift| ≤ 1 / 2 := by
    rw [FloatLib.Numerics.roundShiftRightEven_eq_roundQuotientEven]
    simpa using FloatLib.Numerics.roundQuotientEven_error_le_half mantissa
      (2^shift) (by positivity)
  rw [show (FloatLib.Numerics.roundShiftRightEven mantissa shift : ℚ) / 2^149 -
      (mantissa : ℚ) / (2^shift * 2^149) =
      ((FloatLib.Numerics.roundShiftRightEven mantissa shift : ℚ) -
      (mantissa : ℚ) / 2^shift) / 2^149 by ring, abs_div]
  norm_num only [abs_of_pos (by positivity : (0 : ℚ) < 2^149)]
  calc
    _ ≤ (1 / 2 : ℚ) / 2^149 := div_le_div_of_nonneg_right h (by positivity)
    _ = _ := by norm_num

#print axioms quarter_error_bound
#print axioms quarter_error_exponent253
#print axioms quarter_error_exponent254
#print axioms rounded_subnormal_error

end GB300Math.RcpPtxBounds
