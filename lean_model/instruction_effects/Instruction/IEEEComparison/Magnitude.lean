/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Model

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.IEEEComparison

theorem finite_units_exponent_zero (fraction : Nat) : finiteUnits 0 fraction = fraction := by
  simp [finiteUnits]

theorem finite_units_lower (exponent fraction : Nat) (positive : 0 < exponent) :
    fractionBase * 2 ^ (exponent - 1) ≤ finiteUnits exponent fraction := by
  simp only [finiteUnits, Nat.ne_of_gt positive, ite_false]
  exact Nat.mul_le_mul_right _ (by omega)

theorem finite_units_upper (exponent fraction : Nat) (bounded : fraction < fractionBase)
    (positive : 0 < exponent) : finiteUnits exponent fraction < fractionBase * 2 ^ exponent := by
  have mantissa : fractionBase + fraction < fractionBase * 2 := by omega
  have power : 0 < 2 ^ (exponent - 1) := by positivity
  have inequality := Nat.mul_lt_mul_of_pos_right mantissa power
  have successor : exponent - 1 + 1 = exponent := by omega
  simp only [finiteUnits, Nat.ne_of_gt positive, ite_false]
  calc
    (fractionBase + fraction) * 2 ^ (exponent - 1) < fractionBase * 2 * 2 ^ (exponent - 1) := inequality
    _ = fractionBase * (2 ^ (exponent - 1) * 2) := by ring
    _ = fractionBase * 2 ^ exponent := by rw [← pow_succ, successor]

theorem finite_units_exponent_lt (first second firstFraction secondFraction : Nat)
    (firstBound : firstFraction < fractionBase) (ordered : first < second) :
    finiteUnits first firstFraction < finiteUnits second secondFraction := by
  have secondPositive : 0 < second := by omega
  have lower := finite_units_lower second secondFraction secondPositive
  by_cases zero : first = 0
  · subst first
    have powerPositive : 0 < 2 ^ (second - 1) := by positivity
    have baseLower : fractionBase ≤ fractionBase * 2 ^ (second - 1) := by nlinarith
    rw [finite_units_exponent_zero]
    exact lt_of_lt_of_le firstBound (le_trans baseLower lower)
  · have positive : 0 < first := by omega
    have upper := finite_units_upper first firstFraction firstBound positive
    have exponentOrder : first ≤ second - 1 := by omega
    have powers : 2 ^ first ≤ 2 ^ (second - 1) := Nat.pow_le_pow_right (by omega) exponentOrder
    have scales := Nat.mul_le_mul_left fractionBase powers
    omega

theorem finite_units_fraction_lt (exponent first second : Nat) (ordered : first < second) :
    finiteUnits exponent first < finiteUnits exponent second := by
  by_cases zero : exponent = 0
  · simp [finiteUnits, zero, ordered]
  · simp only [finiteUnits, zero, ite_false]
    exact Nat.mul_lt_mul_of_pos_right (by omega) (by positivity)

theorem finite_units_order_iff (firstExponent secondExponent firstFraction secondFraction : Nat)
    (firstBound : firstFraction < fractionBase) (secondBound : secondFraction < fractionBase) :
    finiteUnits firstExponent firstFraction < finiteUnits secondExponent secondFraction ↔
      firstExponent * fractionBase + firstFraction < secondExponent * fractionBase + secondFraction := by
  rcases lt_trichotomy firstExponent secondExponent with ordered | same | reversed
  · have unitOrder := finite_units_exponent_lt firstExponent secondExponent firstFraction secondFraction firstBound ordered
    have bitOrder : firstExponent * fractionBase + firstFraction < secondExponent * fractionBase + secondFraction := by
      dsimp only [fractionBase] at *
      omega
    exact iff_of_true unitOrder bitOrder
  · subst secondExponent
    rcases lt_trichotomy firstFraction secondFraction with ordered | same | reversed
    · exact iff_of_true (finite_units_fraction_lt firstExponent firstFraction secondFraction ordered) (by omega)
    · subst secondFraction
      simp
    · have unitOrder := finite_units_fraction_lt firstExponent secondFraction firstFraction reversed
      exact iff_of_false (by omega) (by omega)
  · have unitOrder := finite_units_exponent_lt secondExponent firstExponent secondFraction firstFraction secondBound reversed
    have bitOrder : secondExponent * fractionBase + secondFraction < firstExponent * fractionBase + firstFraction := by
      dsimp only [fractionBase] at *
      omega
    exact iff_of_false (by omega) (by omega)

end Instruction.IEEEComparison
