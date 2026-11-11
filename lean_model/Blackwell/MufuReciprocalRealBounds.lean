/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Mathlib
import MufuReciprocalBounds
import MufuReciprocalFloatLib

namespace GB300Math.MufuReciprocalBounds

noncomputable def magnitudeValue (bf16 : Bool) (word : Nat) : ℝ :=
  (dyadic bf16 word).1 / (dyadic bf16 word).2

theorem floatLib_real_value (bf16 : Bool) (word : Nat)
    (finite : word % 32768 < infinity bf16)
    (checked : decodeMatches bf16 word = true) :
    (floatLibValue bf16 word).map (fun q : ℚ => (q : ℝ)) =
      some (if word / 32768 % 2 = 1 then -magnitudeValue bf16 word
            else magnitudeValue bf16 word) := by
  rw [decoded_finite bf16 word finite checked]
  by_cases sign : word / 32768 % 2 = 1 <;>
    simp [rationalValue, magnitudeValue, sign]

theorem denominator_positive (bf16 : Bool) (word : Nat) :
    0 < (dyadic bf16 word).2 := by
  simp only [dyadic]
  positivity

theorem numerator_positive (bf16 : Bool) (word : Nat) (h : 0 < word % 32768) :
    0 < (dyadic bf16 word).1 := by
  dsimp only [dyadic]
  apply Nat.mul_pos
  · have hp : 0 < 2^fractionBits bf16 := by positivity
    split_ifs with he
    · have hm : word % 32768 < 2^fractionBits bf16 := by
        cases bf16 <;> norm_num [fractionBits] at he ⊢ <;> omega
      simpa [Nat.mod_eq_of_lt hm] using h
    · omega
  · positivity

theorem bound_denominator_positive (bf16 : Bool) (word : Nat) :
    0 < (bound bf16 word).2 := by
  cases bf16 <;> simp only [bound] <;> split_ifs <;> decide

theorem reciprocal_relative_from_product {x y error : ℝ}
    (hx : 0 < x) (hproduct : x * y ≤ 1) (herror : 1 - x * y ≤ error) :
    |y - 1/x| / |1/x| ≤ error := by
  have hy : y ≤ 1/x := (le_div_iff₀ hx).mpr (by nlinarith)
  rw [abs_of_nonpos (sub_nonpos.mpr hy), abs_of_pos (one_div_pos.mpr hx)]
  have heq : -(y - 1/x) / (1/x) = 1 - x*y := by field_simp; ring
  rw [heq]
  exact herror

theorem dyadic_relative_bound (xn xd yn yd bn bd : Nat)
    (hxn : 0 < xn) (hxd : 0 < xd) (hyd : 0 < yd) (hbd : 0 < bd)
    (hproduct : xn * yn ≤ xd * yd)
    (hbound : (xd * yd - xn * yn) * bd ≤ bn * xd * yd) :
    |(yn : ℝ)/yd - 1/((xn : ℝ)/xd)| / |1/((xn : ℝ)/xd)| ≤ (bn : ℝ)/bd := by
  have hx : (0 : ℝ) < xn := by exact_mod_cast hxn
  have hd : (0 : ℝ) < xd := by exact_mod_cast hxd
  have he : (0 : ℝ) < yd := by exact_mod_cast hyd
  have hb : (0 : ℝ) < bd := by exact_mod_cast hbd
  have hp : (xn : ℝ) * yn ≤ (xd : ℝ) * yd := by exact_mod_cast hproduct
  have hh : ((xd : ℝ)*yd - (xn : ℝ)*yn)*bd ≤ (bn : ℝ)*xd*yd := by
    have lifted : (((xd*yd-xn*yn)*bd : Nat) : ℝ) ≤ ((bn*xd*yd : Nat) : ℝ) := by
      exact_mod_cast hbound
    simpa only [Nat.cast_mul, Nat.cast_sub hproduct] using lifted
  have product : (xn : ℝ)/xd*((yn : ℝ)/yd) = ((xn : ℝ)*yn)/((xd : ℝ)*yd) := by ring
  apply reciprocal_relative_from_product (div_pos hx hd)
  · rw [product]
    exact (div_le_one (mul_pos hd he)).mpr hp
  · rw [product]
    have identity : 1 - ((xn : ℝ)*yn)/((xd : ℝ)*yd) =
        ((xd : ℝ)*yd-(xn : ℝ)*yn)/((xd : ℝ)*yd) := by field_simp
    rw [identity]
    exact (div_le_div_iff₀ (mul_pos hd he) hb).mpr (by nlinarith [hh])

theorem flush_relative_error {x : ℝ} (hx : x ≠ 0) :
    |(0 : ℝ) - 1/x| / |1/x| = 1 := by
  rw [zero_sub, abs_neg]
  exact div_self (abs_ne_zero.mpr (one_div_ne_zero hx))

theorem checked_relative_bound (bf16 : Bool) (word : Nat)
    (domain : inDomain bf16 word (output bf16 word) = true)
    (checked : valid bf16 word = true) :
    |magnitudeValue bf16 (output bf16 word) - 1/(magnitudeValue bf16 word)| /
      |1/(magnitudeValue bf16 word)| ≤
      ((bound bf16 (output bf16 word)).1 : ℝ) / (bound bf16 (output bf16 word)).2 := by
  have input_positive : 0 < word % 32768 := by
    have hd := domain
    simp only [inDomain, Bool.and_eq_true, decide_eq_true_eq] at hd
    exact hd.1.1.1
  have checks := checked
  simp only [valid, domain, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
    Bool.and_eq_true, decide_eq_true_eq] at checks
  exact dyadic_relative_bound _ _ _ _ _ _
    (numerator_positive bf16 word input_positive)
    (denominator_positive bf16 word)
    (denominator_positive bf16 (output bf16 word))
    (bound_denominator_positive bf16 (output bf16 word)) checks.1.2 checks.2

theorem signed_reciprocal_relative {x y : ℝ} :
    |-y - 1/(-x)| / |1/(-x)| = |y - 1/x| / |1/x| := by
  rw [div_neg, abs_neg]
  congr 1
  rw [show -y - -(1/x) = -(y - 1/x) by ring, abs_neg]

theorem checked_floatLib_relative_bound (bf16 : Bool) (word : Nat)
    (domain : inDomain bf16 word (output bf16 word) = true)
    (checked : valid bf16 word = true)
    (input_decode : decodeMatches bf16 word = true)
    (output_decode : decodeMatches bf16 (output bf16 word) = true) :
    ∃ x y : ℝ,
      (floatLibValue bf16 word).map (fun q : ℚ => (q : ℝ)) = some x ∧
      (floatLibValue bf16 (output bf16 word)).map (fun q : ℚ => (q : ℝ)) = some y ∧
      |y - 1/x| / |1/x| ≤
        ((bound bf16 (output bf16 word)).1 : ℝ) / (bound bf16 (output bf16 word)).2 := by
  have hd := domain
  simp only [inDomain, Bool.and_eq_true, decide_eq_true_eq] at hd
  have checks := checked
  simp only [valid, domain, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
    Bool.and_eq_true, decide_eq_true_eq] at checks
  refine ⟨_, _, floatLib_real_value bf16 word hd.1.1.2 input_decode,
    floatLib_real_value bf16 (output bf16 word) hd.2 output_decode, ?_⟩
  rw [← checks.1.1.2]
  split_ifs
  · rw [signed_reciprocal_relative]
    exact checked_relative_bound bf16 word domain checked
  · exact checked_relative_bound bf16 word domain checked

theorem f16_normal_sharp :
    |magnitudeValue false (output false 0x07d5) - 1/magnitudeValue false 0x07d5| /
      |1/magnitudeValue false 0x07d5| = 1927/2097152 := by
  rw [f16_normal_attained.1]
  norm_num [magnitudeValue, dyadic, fractionBits, bias]

theorem f16_subnormal_sharp :
    |magnitudeValue false (output false 0x7bc2) - 1/magnitudeValue false 0x7bc2| /
      |1/magnitudeValue false 0x7bc2| = 985/262144 := by
  rw [f16_subnormal_attained.1]
  norm_num [magnitudeValue, dyadic, fractionBits, bias]

theorem bf16_normal_sharp :
    |magnitudeValue true (output true 0x00f1) - 1/magnitudeValue true 0x00f1| /
      |1/magnitudeValue true 0x00f1| = 233/32768 := by
  rw [bf16_normal_attained.1]
  norm_num [magnitudeValue, dyadic, fractionBits, bias]

theorem bf16_flushed_sharp :
    |magnitudeValue true (output true 0x7e81) - 1/magnitudeValue true 0x7e81| /
      |1/magnitudeValue true 0x7e81| = 1 := by
  rw [bf16_flush_boundary.2]
  norm_num [magnitudeValue, dyadic, fractionBits, bias]

#print axioms reciprocal_relative_from_product
#print axioms dyadic_relative_bound
#print axioms flush_relative_error
#print axioms denominator_positive
#print axioms numerator_positive
#print axioms bound_denominator_positive
#print axioms checked_relative_bound
#print axioms floatLib_real_value
#print axioms signed_reciprocal_relative
#print axioms checked_floatLib_relative_bound
#print axioms f16_normal_sharp
#print axioms f16_subnormal_sharp
#print axioms bf16_normal_sharp
#print axioms bf16_flushed_sharp

end GB300Math.MufuReciprocalBounds
