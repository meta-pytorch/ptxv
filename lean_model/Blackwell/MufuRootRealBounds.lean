/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Mathlib
import MufuRootBounds
import MufuReciprocalRealBounds

namespace GB300Math.MufuRootBounds

open MufuReciprocalBounds

theorem normalized_square_bound {q error : ℝ}
    (hq : 0 ≤ q) (he : 0 ≤ error) (_he1 : error ≤ 1)
    (lower : (1-error)^2 ≤ q^2) (upper : q^2 ≤ (1+error)^2) :
    |q-1| ≤ error := by
  apply abs_le.mpr
  constructor <;> nlinarith

theorem relative_normalized {y ideal : ℝ} (hi : 0 < ideal) :
    |y-ideal| / |ideal| = |y/ideal-1| := by
  rw [← abs_div]
  congr 1
  field_simp

theorem sqrt_relative_identity {x y : ℝ}
    (hx : 0 < x) (hy : 0 ≤ y) (below : y^2 ≤ x) :
    |y-Real.sqrt x| / |Real.sqrt x| = 1-Real.sqrt (y^2/x) := by
  have root_pos : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  have root_sq := Real.sq_sqrt hx.le
  have y_below : y ≤ Real.sqrt x := by nlinarith
  rw [abs_of_nonpos (sub_nonpos.mpr y_below), abs_of_pos root_pos]
  rw [Real.sqrt_div (sq_nonneg y), Real.sqrt_sq_eq_abs, abs_of_nonneg hy]
  field_simp
  ring

theorem sqrt_relative_bound {x y lo : ℝ}
    (hx : 0 < x) (hy : 0 ≤ y) (lower : lo ≤ y^2/x) (upper : y^2/x ≤ 1) :
    |y-Real.sqrt x| / |Real.sqrt x| ≤ 1-Real.sqrt lo := by
  rw [sqrt_relative_identity hx hy ((div_le_one hx).mp upper)]
  exact sub_le_sub_left (Real.sqrt_le_sqrt lower) 1

theorem rsq_relative_bound {x y error : ℝ}
    (hx : 0 < x) (hy : 0 ≤ y) (he : 0 ≤ error) (he1 : error ≤ 1)
    (lower : (1-error)^2 ≤ x*y^2) (upper : x*y^2 ≤ (1+error)^2) :
    |y-1/Real.sqrt x| / |1/Real.sqrt x| ≤ error := by
  have root_pos : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  rw [relative_normalized (one_div_pos.mpr root_pos)]
  simp only [div_eq_mul_inv, one_mul, inv_inv]
  apply normalized_square_bound (mul_nonneg hy root_pos.le) he he1
  · rw [mul_pow, Real.sq_sqrt hx.le]
    nlinarith
  · rw [mul_pow, Real.sq_sqrt hx.le]
    nlinarith

theorem ratio_value (bf16 reciprocal : Bool) (input result : Nat) :
    ((ratio bf16 reciprocal input result).1 : ℝ) /
      (ratio bf16 reciprocal input result).2 =
    if reciprocal then magnitudeValue bf16 input * (magnitudeValue bf16 result)^2
    else (magnitudeValue bf16 result)^2 / magnitudeValue bf16 input := by
  cases reciprocal <;>
    simp [ratio, magnitudeValue, div_eq_mul_inv, mul_pow, mul_inv_rev] <;> ring

theorem checked_ratio (bf16 reciprocal : Bool) (input : Nat)
    (inside : domain bf16 input (output bf16 reciprocal input) = true)
    (checked : valid bf16 reciprocal input = true) :
    ((lower bf16 reciprocal).1 : ℝ) / (lower bf16 reciprocal).2 ≤
      ((ratio bf16 reciprocal input (output bf16 reciprocal input)).1 : ℝ) /
        (ratio bf16 reciprocal input (output bf16 reciprocal input)).2 ∧
    ((ratio bf16 reciprocal input (output bf16 reciprocal input)).1 : ℝ) /
        (ratio bf16 reciprocal input (output bf16 reciprocal input)).2 ≤
      ((upper bf16 reciprocal).1 : ℝ) / (upper bf16 reciprocal).2 := by
  have h := checked
  simp only [valid, inside, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
    Bool.and_eq_true, decide_eq_true_eq] at h
  have hd : (0 : ℝ) < (ratio bf16 reciprocal input (output bf16 reciprocal input)).2 := by
    exact_mod_cast h.1.1
  have hl : (0 : ℝ) < (lower bf16 reciprocal).2 := by
    cases bf16 <;> cases reciprocal <;> norm_num [lower]
  have hu : (0 : ℝ) < (upper bf16 reciprocal).2 := by
    cases bf16 <;> cases reciprocal <;> norm_num [upper]
  constructor
  · apply (div_le_div_iff₀ hl hd).mpr
    exact_mod_cast h.1.2
  · apply (div_le_div_iff₀ hd hu).mpr
    exact_mod_cast h.2

theorem magnitude_positive (bf16 : Bool) (word : Nat)
    (positive : 0 < word) (finite : word < infinity bf16) :
    0 < magnitudeValue bf16 word := by
  have small : word < 32768 := by
    cases bf16 <;> simp [infinity] at finite <;> omega
  have hn := numerator_positive bf16 word
    (by simpa [Nat.mod_eq_of_lt small] using positive)
  exact div_pos (by exact_mod_cast hn)
    (by exact_mod_cast denominator_positive bf16 word)

theorem checked_sqrt_relative (bf16 : Bool) (input : Nat)
    (inside : domain bf16 input (output bf16 false input) = true)
    (checked : valid bf16 false input = true) :
    |magnitudeValue bf16 (output bf16 false input) - Real.sqrt (magnitudeValue bf16 input)| /
      |Real.sqrt (magnitudeValue bf16 input)| ≤
      1-Real.sqrt (((lower bf16 false).1 : ℝ)/(lower bf16 false).2) := by
  have h := inside
  simp only [domain, Bool.and_eq_true, decide_eq_true_eq] at h
  have bounds := checked_ratio bf16 false input inside checked
  simp only [ratio_value, Bool.false_eq_true, ↓reduceIte] at bounds
  apply sqrt_relative_bound (magnitude_positive bf16 input h.1.1.1 h.1.1.2)
    (magnitude_positive bf16 _ h.1.2 h.2).le bounds.1
  simpa [upper] using bounds.2

theorem checked_rsq_relative (bf16 : Bool) (input : Nat)
    (inside : domain bf16 input (output bf16 true input) = true)
    (checked : valid bf16 true input = true) :
    |magnitudeValue bf16 (output bf16 true input) - 1/Real.sqrt (magnitudeValue bf16 input)| /
      |1/Real.sqrt (magnitudeValue bf16 input)| ≤ 1/(if bf16 then (128 : ℝ) else 1024) := by
  have h := inside
  simp only [domain, Bool.and_eq_true, decide_eq_true_eq] at h
  have bounds := checked_ratio bf16 true input inside checked
  simp only [ratio_value, ↓reduceIte] at bounds
  apply rsq_relative_bound (magnitude_positive bf16 input h.1.1.1 h.1.1.2)
    (magnitude_positive bf16 _ h.1.2 h.2).le
  · cases bf16 <;> norm_num
  · cases bf16 <;> norm_num
  · cases bf16 <;> norm_num [lower] at bounds ⊢ <;> exact bounds.1
  · cases bf16 <;> norm_num [upper] at bounds ⊢ <;> exact bounds.2

theorem f16_sqrt_sharp :
    |magnitudeValue false (output false false 0x0402) - Real.sqrt (magnitudeValue false 0x0402)| /
      |Real.sqrt (magnitudeValue false 0x0402)| = 1-Real.sqrt ((512 : ℝ)/513) := by
  rw [f16_sqrt_attained.1]
  rw [sqrt_relative_identity
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])]
  norm_num [magnitudeValue, dyadic, fractionBits, bias]

theorem bf16_sqrt_sharp :
    |magnitudeValue true (output true false 0x0082) - Real.sqrt (magnitudeValue true 0x0082)| /
      |Real.sqrt (magnitudeValue true 0x0082)| = 1-Real.sqrt ((64 : ℝ)/65) := by
  rw [bf16_sqrt_attained.1]
  rw [sqrt_relative_identity
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])]
  norm_num [magnitudeValue, dyadic, fractionBits, bias]

theorem rsq_relative_strict_lower {x y error : ℝ}
    (hx : 0 < x) (hy : 0 ≤ y) (he : error < 1)
    (below : x*y^2 < (1-error)^2) :
    error < |y-1/Real.sqrt x| / |1/Real.sqrt x| := by
  have root_pos : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  have qpos : 0 ≤ y * Real.sqrt x := mul_nonneg hy root_pos.le
  have qsq : (y * Real.sqrt x)^2 < (1-error)^2 := by
    rw [mul_pow, Real.sq_sqrt hx.le]
    nlinarith
  have qlt : y * Real.sqrt x < 1-error := by nlinarith
  rw [relative_normalized (one_div_pos.mpr root_pos)]
  simp only [div_eq_mul_inv, one_mul, inv_inv]
  have habs := le_abs_self (-(y * Real.sqrt x - 1))
  rw [abs_neg] at habs
  linarith

theorem f16_rsq_not_half_unit :
    (1 : ℝ)/2048 <
      |magnitudeValue false (output false true 0x0bcd) - 1/Real.sqrt (magnitudeValue false 0x0bcd)| /
      |1/Real.sqrt (magnitudeValue false 0x0bcd)| := by
  have value : output false true 0x0bcd = 0x540c := by decide +kernel
  rw [value]
  apply rsq_relative_strict_lower
  all_goals norm_num [magnitudeValue, dyadic, fractionBits, bias]

theorem bf16_rsq_not_half_unit :
    (1 : ℝ)/256 <
      |magnitudeValue true (output true true 0x0156) - 1/Real.sqrt (magnitudeValue true 0x0156)| /
      |1/Real.sqrt (magnitudeValue true 0x0156)| := by
  have value : output true true 0x0156 = 0x5e8b := by decide +kernel
  rw [value]
  apply rsq_relative_strict_lower
  all_goals norm_num [magnitudeValue, dyadic, fractionBits, bias]

#print axioms normalized_square_bound
#print axioms relative_normalized
#print axioms sqrt_relative_identity
#print axioms sqrt_relative_bound
#print axioms rsq_relative_bound
#print axioms ratio_value
#print axioms checked_ratio
#print axioms magnitude_positive
#print axioms checked_sqrt_relative
#print axioms checked_rsq_relative
#print axioms f16_sqrt_sharp
#print axioms bf16_sqrt_sharp
#print axioms rsq_relative_strict_lower
#print axioms f16_rsq_not_half_unit
#print axioms bf16_rsq_not_half_unit

end GB300Math.MufuRootBounds
