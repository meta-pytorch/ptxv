/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Representation

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.IEEEComparison

def bitLess (negativeFirst negativeSecond : Bool) (first second : Nat) : Bool :=
  if negativeFirst = negativeSecond then
    if negativeFirst then decide (second < first) else decide (first < second)
  else negativeFirst && !decide (first = 0 ∧ second = 0)

def bitEqual (negativeFirst negativeSecond : Bool) (first second : Nat) : Bool :=
  if negativeFirst = negativeSecond then decide (first = second) else decide (first = 0 ∧ second = 0)

theorem negative_positive_less (first second : Nat) :
    decide (-(first : Int) < (second : Int)) = !(first == 0 && second == 0) := by
  apply Bool.eq_iff_iff.mpr
  simp <;> omega

theorem positive_negative_less (first second : Nat) :
    decide ((first : Int) < -(second : Int)) = false := by
  simp <;> omega

theorem negative_positive_equal (first second : Nat) :
    decide (-(first : Int) = (second : Int)) = (first == 0 && second == 0) := by
  apply Bool.eq_iff_iff.mpr
  simp <;> omega

theorem positive_negative_equal (first second : Nat) :
    decide ((first : Int) = -(second : Int)) = (first == 0 && second == 0) := by
  apply Bool.eq_iff_iff.mpr
  simp <;> omega

theorem encoded_less_ordered (negativeFirst negativeSecond : Bool) (first second : Nat)
    (firstOrdered : first ≤ infinityMagnitude) (secondOrdered : second ≤ infinityMagnitude) :
    less (encodedValue negativeFirst first) (encodedValue negativeSecond second) =
      bitLess negativeFirst negativeSecond first second := by
  have notFirstNan : ¬ infinityMagnitude < first := by omega
  have notSecondNan : ¬ infinityMagnitude < second := by omega
  cases negativeFirst <;> cases negativeSecond <;>
    by_cases firstInfinity : first = infinityMagnitude <;>
    by_cases secondInfinity : second = infinityMagnitude <;>
    simp [encodedValue, less, bitLess, firstInfinity, secondInfinity, notFirstNan, notSecondNan,
      negative_positive_less, positive_negative_less, magnitude_units_order_iff] <;>
    try omega
  all_goals
    apply Bool.eq_iff_iff.mpr
    simp

theorem encoded_equal_ordered (negativeFirst negativeSecond : Bool) (first second : Nat)
    (firstOrdered : first ≤ infinityMagnitude) (secondOrdered : second ≤ infinityMagnitude) :
    equal (encodedValue negativeFirst first) (encodedValue negativeSecond second) =
      bitEqual negativeFirst negativeSecond first second := by
  have notFirstNan : ¬ infinityMagnitude < first := by omega
  have notSecondNan : ¬ infinityMagnitude < second := by omega
  cases negativeFirst <;> cases negativeSecond <;>
    by_cases firstInfinity : first = infinityMagnitude <;>
    by_cases secondInfinity : second = infinityMagnitude <;>
    simp [encodedValue, equal, bitEqual, firstInfinity, secondInfinity, notFirstNan, notSecondNan,
      negative_positive_equal, positive_negative_equal, magnitude_units_equal_iff] <;>
    try omega

theorem encoded_unordered (negativeFirst negativeSecond : Bool) (first second : Nat) :
    unordered (encodedValue negativeFirst first) (encodedValue negativeSecond second) =
      (decide (infinityMagnitude < first) || decide (infinityMagnitude < second)) := by
  cases negativeFirst <;> cases negativeSecond <;>
    by_cases firstNan : infinityMagnitude < first <;>
    by_cases secondNan : infinityMagnitude < second <;>
    by_cases firstInfinity : first = infinityMagnitude <;>
    by_cases secondInfinity : second = infinityMagnitude <;>
    simp [encodedValue, unordered, firstNan, secondNan, firstInfinity, secondInfinity]

theorem bit_less_reverse (negativeFirst negativeSecond : Bool) (first second : Nat) :
    bitLess negativeSecond negativeFirst second first =
      (!bitEqual negativeFirst negativeSecond first second &&
        !bitLess negativeFirst negativeSecond first second) := by
  cases negativeFirst <;> cases negativeSecond <;>
    apply Bool.eq_iff_iff.mpr <;> simp [bitLess, bitEqual] <;> omega

end Instruction.IEEEComparison
