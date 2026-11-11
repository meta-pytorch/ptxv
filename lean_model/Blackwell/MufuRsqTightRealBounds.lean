/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuRootRealBounds
import MufuRsqTightBounds

namespace GB300Math.MufuRsqTightBounds

open MufuRootBounds MufuReciprocalBounds

theorem root_ratio_interval {x y lo hi : ℝ}
    (hx : 0 < x) (hy : 0 ≤ y) (hl : 0 ≤ lo) (hh : 0 ≤ hi)
    (_hl1 : lo ≤ 1) (lower : lo ≤ x*y^2) (upper : x*y^2 ≤ hi)
    (sum : Real.sqrt lo + Real.sqrt hi ≤ 2) :
    |y-1/Real.sqrt x| / |1/Real.sqrt x| ≤ 1-Real.sqrt lo := by
  have root_pos : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  have qpos : 0 ≤ y * Real.sqrt x := mul_nonneg hy root_pos.le
  have qsq : (y * Real.sqrt x)^2 = x*y^2 := by
    rw [mul_pow, Real.sq_sqrt hx.le]
    ring
  have slo := Real.sqrt_nonneg lo
  have shi := Real.sqrt_nonneg hi
  have slosq := Real.sq_sqrt hl
  have shisq := Real.sq_sqrt hh
  have qlo : Real.sqrt lo ≤ y * Real.sqrt x := by nlinarith
  have qhi : y * Real.sqrt x ≤ Real.sqrt hi := by nlinarith
  rw [relative_normalized (one_div_pos.mpr root_pos)]
  simp only [div_eq_mul_inv, one_mul, inv_inv]
  apply abs_le.mpr
  constructor <;> linarith

theorem root_ratio_identity {x y : ℝ}
    (hx : 0 < x) (hy : 0 ≤ y) (below : x*y^2 ≤ 1) :
    |y-1/Real.sqrt x| / |1/Real.sqrt x| = 1-Real.sqrt (x*y^2) := by
  have root_pos : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  have qpos : 0 ≤ y * Real.sqrt x := mul_nonneg hy root_pos.le
  have qsq : (y * Real.sqrt x)^2 = x*y^2 := by
    rw [mul_pow, Real.sq_sqrt hx.le]
    ring
  have qle : y * Real.sqrt x ≤ 1 := by nlinarith
  rw [relative_normalized (one_div_pos.mpr root_pos)]
  simp only [div_eq_mul_inv, one_mul, inv_inv]
  rw [abs_of_nonpos (sub_nonpos.mpr qle), ← qsq, Real.sqrt_sq_eq_abs,
    abs_of_nonneg qpos]
  ring

theorem root_sum_bound (bf16 : Bool) :
    Real.sqrt (((lower bf16).1 : ℝ)/(lower bf16).2) +
      Real.sqrt (((upper bf16).1 : ℝ)/(upper bf16).2) ≤ 2 := by
  cases bf16
  · change Real.sqrt ((133960757 : ℝ)/134217728) +
      Real.sqrt ((268435475 : ℝ)/268435456) ≤ 2
    have lo_sq := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 133960757/134217728)
    have hi_sq := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 268435475/268435456)
    have hlo : Real.sqrt ((133960757 : ℝ)/134217728) ≤ 1-1/1048576 := by
      nlinarith [Real.sqrt_nonneg ((133960757 : ℝ)/134217728)]
    have hhi : Real.sqrt ((268435475 : ℝ)/268435456) ≤ 1+1/1048576 := by
      nlinarith [Real.sqrt_nonneg ((268435475 : ℝ)/268435456)]
    linarith
  · norm_num [lower, upper]
    have h := Real.sqrt_le_sqrt (by norm_num : (2067347 : ℝ)/2097152 ≤ 1)
    norm_num at h ⊢
    linarith

theorem checked_ratio (bf16 : Bool) (input : Nat)
    (inside : domain bf16 input (output bf16 true input) = true)
    (checked : valid bf16 input = true) :
    ((lower bf16).1 : ℝ)/(lower bf16).2 ≤
      magnitudeValue bf16 input * (magnitudeValue bf16 (output bf16 true input))^2 ∧
    magnitudeValue bf16 input * (magnitudeValue bf16 (output bf16 true input))^2 ≤
      ((upper bf16).1 : ℝ)/(upper bf16).2 := by
  have h := checked
  simp only [valid, inside, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
    Bool.and_eq_true, decide_eq_true_eq] at h
  have hd : (0 : ℝ) < (ratio bf16 true input (output bf16 true input)).2 := by
    exact_mod_cast h.1.1
  have hl : (0 : ℝ) < (lower bf16).2 := by cases bf16 <;> norm_num [lower]
  have hu : (0 : ℝ) < (upper bf16).2 := by cases bf16 <;> norm_num [upper]
  have value := ratio_value bf16 true input (output bf16 true input)
  simp only [↓reduceIte] at value
  rw [← value]
  constructor
  · apply (div_le_div_iff₀ hl hd).mpr
    exact_mod_cast h.1.2
  · apply (div_le_div_iff₀ hd hu).mpr
    exact_mod_cast h.2

theorem checked_rsq_relative (bf16 : Bool) (input : Nat)
    (inside : domain bf16 input (output bf16 true input) = true)
    (checked : valid bf16 input = true) :
    |magnitudeValue bf16 (output bf16 true input) - 1/Real.sqrt (magnitudeValue bf16 input)| /
      |1/Real.sqrt (magnitudeValue bf16 input)| ≤
      1-Real.sqrt (((lower bf16).1 : ℝ)/(lower bf16).2) := by
  have h := inside
  simp only [domain, Bool.and_eq_true, decide_eq_true_eq] at h
  have bounds := checked_ratio bf16 input inside checked
  apply root_ratio_interval (hi := ((upper bf16).1 : ℝ)/(upper bf16).2) (magnitude_positive bf16 input h.1.1.1 h.1.1.2)
    (magnitude_positive bf16 _ h.1.2 h.2).le
  · cases bf16 <;> norm_num [lower]
  · cases bf16 <;> norm_num [upper]
  · cases bf16 <;> norm_num [lower]
  · exact bounds.1
  · exact bounds.2
  · exact root_sum_bound bf16

theorem f16_rsq_sharp :
    |magnitudeValue false (output false true 0x0bcd) - 1/Real.sqrt (magnitudeValue false 0x0bcd)| /
      |1/Real.sqrt (magnitudeValue false 0x0bcd)| =
      1-Real.sqrt ((133960757 : ℝ)/134217728) := by
  rw [f16_attained.1]
  rw [root_ratio_identity
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])]
  norm_num [magnitudeValue, dyadic, fractionBits, bias]

theorem bf16_rsq_sharp :
    |magnitudeValue true (output true true 0x0156) - 1/Real.sqrt (magnitudeValue true 0x0156)| /
      |1/Real.sqrt (magnitudeValue true 0x0156)| =
      1-Real.sqrt ((2067347 : ℝ)/2097152) := by
  rw [bf16_attained.1]
  rw [root_ratio_identity
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])
    (by norm_num [magnitudeValue, dyadic, fractionBits, bias])]
  norm_num [magnitudeValue, dyadic, fractionBits, bias]

#print axioms root_ratio_interval
#print axioms root_ratio_identity
#print axioms root_sum_bound
#print axioms checked_ratio
#print axioms checked_rsq_relative
#print axioms f16_rsq_sharp
#print axioms bf16_rsq_sharp

end GB300Math.MufuRsqTightBounds
