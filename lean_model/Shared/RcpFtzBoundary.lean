/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu
import Mathlib.Tactic

namespace GB300Math.RcpFtzBoundary

def inputWord (negative : Bool) (fraction : Nat) : Nat :=
  (if negative then 2^31 else 0) + 254*2^23 + fraction

def finiteValue (word : Nat) : ℚ :=
  (if word < 2^31 then 1 else -1) *
  (if Mufu.exponent word = 0 then (Mufu.fraction word : ℚ)*(2 : ℚ)^(-149 : Int)
   else (2^23+(Mufu.fraction word : ℚ))*(2 : ℚ)^((Mufu.exponent word : Int)-150))

theorem input_fields (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    Mufu.exponent (inputWord negative fraction) = 254 ∧
    Mufu.fraction (inputWord negative fraction) = fraction ∧
    Mufu.magnitude (inputWord negative fraction) < 0x7f800000 := by
  cases negative <;> simp only [inputWord, Bool.false_eq_true, ↓reduceIte] <;>
    dsimp [Mufu.exponent, Mufu.fraction, Mufu.magnitude] <;> omega

theorem small_sign (word : Nat) (h : word < 2^31) : Mufu.sign word = 0 := by
  have hl : word &&& 2^31 ≤ word := Nat.and_le_left
  have hm := Nat.and_mod_two_pow (a := word) (b := 2^31) (n := 31)
  rw [Nat.mod_eq_of_lt (show word &&& 2^31 < 2^31 by omega)] at hm
  norm_num at hm
  exact hm

theorem high_sign (word : Nat) (hw : word < 2^31) : Mufu.sign (2^31+word) = 2^31 := by
  have hl : (2^31+word) &&& 2^31 ≤ 2^31 := Nat.and_le_right
  have hh := Nat.and_div_two_pow (a := 2^31+word) (b := 2^31) (n := 31)
  have hd : (2^31+word)/2^31 = 1 := by omega
  rw [hd] at hh
  norm_num at hh
  dsimp [Mufu.sign]
  norm_num at hl
  omega

theorem flush_exponent254 (word : Nat) (he : Mufu.exponent word = 254)
    (hm : Mufu.magnitude word < 0x7f800000) :
    Mufu.rcp word = Mufu.sign word := by
  by_cases hi : (Mufu.interpolate MufuTables.rcp (Mufu.fraction word)).toNat = 2^24
  · simp [Mufu.rcp,he,not_lt.mpr (le_of_lt hm),hi]
  · norm_num at hi
    simp [Mufu.rcp,he,not_lt.mpr (le_of_lt hm),hi]

theorem signed_zero_bits (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    Mufu.rcp (inputWord negative fraction) = if negative then 2^31 else 0 := by
  have h := input_fields negative fraction hf
  rw [flush_exponent254 _ h.1 h.2.2]
  cases negative
  · simp only [inputWord, Bool.false_eq_true, ↓reduceIte, Nat.zero_add]
    exact small_sign _ (by omega)
  · simp only [inputWord, ↓reduceIte]
    rw [Nat.add_assoc]
    exact high_sign _ (by omega)

theorem input_value (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    finiteValue (inputWord negative fraction) =
      (if negative then -(1 : ℚ) else 1)*(2^23+(fraction : ℚ))*2^104 := by
  have h := input_fields negative fraction hf
  unfold finiteValue
  rw [h.1,h.2.1]
  cases negative
  · have hl : inputWord false fraction < 2^31 := by dsimp [inputWord]; omega
    simp only [Bool.false_eq_true,ite_false,hl,ite_true]
    norm_num
  · have hn : ¬inputWord true fraction < 2^31 := by dsimp [inputWord]; omega
    simp only [↓reduceIte,hn]
    norm_num
    ring

theorem input_nonzero (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    finiteValue (inputWord negative fraction) ≠ 0 := by
  rw [input_value negative fraction hf]
  have hp : (0 : ℚ) < 2^23+(fraction : ℚ) := by positivity
  apply mul_ne_zero
  · apply mul_ne_zero
    · cases negative <;> norm_num
    · exact ne_of_gt hp
  · norm_num

theorem exact_reciprocal_nonzero (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    1/finiteValue (inputWord negative fraction) ≠ 0 :=
  one_div_ne_zero (input_nonzero negative fraction hf)

theorem returned_value_zero (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    finiteValue (Mufu.rcp (inputWord negative fraction)) = 0 := by
  rw [signed_zero_bits negative fraction hf]
  cases negative <;> norm_num [finiteValue,Mufu.exponent,Mufu.fraction]

theorem reciprocal_relative_error_one (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    |finiteValue (Mufu.rcp (inputWord negative fraction)) -
      1/finiteValue (inputWord negative fraction)| /
      |1/finiteValue (inputWord negative fraction)| = 1 := by
  rw [returned_value_zero negative fraction hf,zero_sub,abs_neg]
  exact div_self (abs_ne_zero.mpr (exact_reciprocal_nonzero negative fraction hf))

theorem normalized_relative_error_one (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    |finiteValue (Mufu.rcp (inputWord negative fraction))*
      finiteValue (inputWord negative fraction)-1| = 1 := by
  rw [returned_value_zero negative fraction hf]
  norm_num

#print axioms signed_zero_bits
#print axioms input_value
#print axioms exact_reciprocal_nonzero
#print axioms returned_value_zero
#print axioms reciprocal_relative_error_one
#print axioms normalized_relative_error_one
end GB300Math.RcpFtzBoundary
