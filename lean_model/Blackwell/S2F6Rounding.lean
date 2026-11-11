/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SmallConversionError

namespace GB300Math.S2F6Rounding

open GB300Math.SmallConversions GB300Math.SmallConversionError

def shiftedValue (m : Nat) (shift : Int) : ℚ :=
  if shift ≤ 0 then (m : ℚ)*2^(-shift).toNat else (m : ℚ)/2^shift.toNat

theorem shifted_value_pow (m : Nat) (shift : Int) :
    shiftedValue m shift = (m : ℚ)*(2 : ℚ)^(-shift) := by
  by_cases h : shift ≤ 0
  · have hn : (((-shift).toNat : Nat) : Int) = -shift := Int.toNat_of_nonneg (by omega)
    simp only [shiftedValue, h, ↓reduceIte]
    rw [← hn, zpow_natCast]
    simp only [Int.toNat_natCast]
  · have hn : ((shift.toNat : Nat) : Int) = shift := Int.toNat_of_nonneg (by omega)
    simp only [shiftedValue, h, ↓reduceIte]
    rw [← hn, zpow_neg, zpow_natCast, div_eq_mul_inv]
    simp only [Int.toNat_natCast]

theorem scaled_input_value (m exponent scale : Nat) :
    shiftedValue m (144-(exponent : Int)+((scale : Int)-127))/64 =
      (m : ℚ)*(2 : ℚ)^((exponent : Int)-150)*(2 : ℚ)^(127-(scale : Int)) := by
  rw [shifted_value_pow]
  have hexp : -(144-(exponent : Int)+((scale : Int)-127)) =
      ((exponent : Int)-150+(127-(scale : Int)))+6 := by omega
  rw [hexp, zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),
    zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
  norm_num
  ring

theorem clipped_shift_error (m limit : Nat) (shift : Int) :
    |((min (roundShift m shift) limit : Nat) : ℚ) -
      min (shiftedValue m shift) limit| ≤ 1/2 := by
  by_cases h : shift ≤ 0
  · simp [roundShift, shiftedValue, h, Nat.cast_min]
  · simpa [roundShift, shiftedValue, h, Nat.cast_min] using
      clipped_rational_error m shift.toNat limit

def encoded (negative : Bool) (m : Nat) : Nat :=
  if negative then (256-min m 128)%256 else min m 127

def decoded (code : Nat) : ℚ :=
  if code < 128 then (code : ℚ)/64 else ((code : ℚ)-256)/64

theorem decode_encoded (negative : Bool) (m : Nat) :
    decoded (encoded negative m) =
      (if negative then -(min m 128 : ℚ) else (min m 127 : ℚ))/64 := by
  cases negative
  · have h : min m 127 < 128 := by omega
    simp [encoded, decoded, h, Nat.cast_min]
  · have hmax : min m 128 ≤ 128 := Nat.min_le_right _ _
    by_cases hzero : min m 128 = 0
    · have hzeroq : min (m : ℚ) 128 = 0 := by exact_mod_cast hzero
      simp [encoded, decoded, hzero, hzeroq]
    · have hsmall : 256-min m 128 < 256 := by omega
      have hlarge : ¬256-min m 128 < 128 := by omega
      have hsub : min m 128 ≤ 256 := by omega
      simp [encoded, decoded, Nat.mod_eq_of_lt hsmall, hlarge, Nat.cast_sub hsub,
        Nat.cast_min]

theorem rounded_error (negative : Bool) (m : Nat) (shift : Int) :
    |decoded (encoded negative (roundShift m shift)) -
      (if negative then -min (shiftedValue m shift) 128
        else min (shiftedValue m shift) 127)/64| ≤ 1/128 := by
  rw [decode_encoded]
  cases negative <;> simp only [Bool.false_eq_true, ↓reduceIte]
  · have h := clipped_shift_error m 127 shift
    rw [← sub_div, abs_div]
    norm_num only [abs_of_pos (by norm_num : (0 : ℚ) < 64)]
    push_cast at h
    linarith
  · have h := clipped_shift_error m 128 shift
    rw [← sub_div]
    have heq (a b : ℚ) : -a - -b = -(a-b) := by ring
    rw [heq, abs_div, abs_neg]
    norm_num only [abs_of_pos (by norm_num : (0 : ℚ) < 64)]
    push_cast at h
    linarith

theorem finite_scaled_conversion_error (x scale : Nat)
    (finite : x % 2^31 < 0x7f800000) (validScale : scale < 255) :
    let mag := x % 2^31
    let negative : Bool := x / 2^31 % 2 = 1
    let mantissa := mag % 2^23 + if mag / 2^23 = 0 then 0 else 2^23
    let shift : Int := 144 - ((max (mag / 2^23) 1 : Nat) : Int) + ((scale : Int)-127)
    |decoded (downS2F6 true false true x scale) -
      (if negative then -min (shiftedValue mantissa shift) 128
        else min (shiftedValue mantissa shift) 127)/64| ≤ 1/128 := by
  have hf : ¬x % 2^31 > 0x7f800000 := by omega
  have hi : x % 2^31 ≠ 0x7f800000 := by omega
  have hs : scale ≠ 255 := by omega
  have h := rounded_error (x / 2^31 % 2 = 1)
    (x % 2^31 % 2^23 + if x % 2^31 / 2^23 = 0 then 0 else 2^23)
    (144 - ((max (x % 2^31 / 2^23) 1 : Nat) : Int) + ((scale : Int)-127))
  simpa only [downS2F6, hf, hi, hs, roundShift, encoded, Bool.false_eq_true,
    and_false, false_and, false_or, decide_eq_true_eq, ↓reduceIte] using h

theorem finite_scaled_relu_error (x scale : Nat) (relu : Bool)
    (finite : x % 2^31 < 0x7f800000) (validScale : scale < 255) :
    let mag := x % 2^31
    let mantissa := mag % 2^23 + if mag / 2^23 = 0 then 0 else 2^23
    let shift : Int := 144 - ((max (mag / 2^23) 1 : Nat) : Int) + ((scale : Int)-127)
    let exact := if x / 2^31 % 2 = 1 then -min (shiftedValue mantissa shift) 128 /64
      else min (shiftedValue mantissa shift) 127 /64
    |decoded (downS2F6 true relu true x scale) -
      (if relu = true ∧ x / 2^31 % 2 = 1 then 0 else exact)| ≤ 1/128 := by
  have h := finite_scaled_conversion_error x scale finite validScale
  have hf : ¬x % 2^31 > 0x7f800000 := by omega
  have hs : scale ≠ 255 := by omega
  cases relu
  · simpa [ite_div, decide_eq_true_eq] using h
  · by_cases hn : x / 2^31 % 2 = 1
    · simp only [downS2F6, hf, hs, hn, and_false, false_or, and_true, ↓reduceIte]
      norm_num [decoded]
    · have heq : downS2F6 true true true x scale = downS2F6 true false true x scale := by
        simp only [downS2F6, hn, and_false, ↓reduceIte]
      simpa only [heq, hn, decide_eq_true_eq, and_false, ↓reduceIte] using h

#print axioms clipped_shift_error
#print axioms shifted_value_pow
#print axioms scaled_input_value
#print axioms decode_encoded
#print axioms rounded_error
#print axioms finite_scaled_conversion_error
#print axioms finite_scaled_relu_error

end GB300Math.S2F6Rounding
