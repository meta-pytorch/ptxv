/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Magnitude

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxRecDepth 4096
namespace Instruction.IEEEComparison

abbrev infinityMagnitude : Nat := 2139095040

def magnitudeUnits (magnitude : Nat) : Nat :=
  finiteUnits (magnitude / fractionBase) (magnitude % fractionBase)

def encodedValue (negative : Bool) (magnitude : Nat) : Value :=
  if infinityMagnitude < magnitude then .nan
  else if magnitude = infinityMagnitude then
    if negative then .negativeInfinity else .positiveInfinity
  else .finite (if negative then -(magnitudeUnits magnitude : Int) else (magnitudeUnits magnitude : Int))

def normalizedMagnitude (magnitude : Nat) (flush : Bool) : Nat :=
  if flush && magnitude < fractionBase then 0 else magnitude

theorem magnitude_units_order_iff (first second : Nat) :
    magnitudeUnits first < magnitudeUnits second ↔ first < second := by
  have firstBound : first % fractionBase < fractionBase := Nat.mod_lt _ (by decide)
  have secondBound : second % fractionBase < fractionBase := Nat.mod_lt _ (by decide)
  have order := finite_units_order_iff (first / fractionBase) (second / fractionBase)
    (first % fractionBase) (second % fractionBase) firstBound secondBound
  have firstEncoding : first / fractionBase * fractionBase + first % fractionBase = first := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod first fractionBase
  have secondEncoding : second / fractionBase * fractionBase + second % fractionBase = second := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod second fractionBase
  simpa only [magnitudeUnits, firstEncoding, secondEncoding] using order

theorem magnitude_units_equal_iff (first second : Nat) :
    magnitudeUnits first = magnitudeUnits second ↔ first = second := by
  constructor
  · intro same
    rcases lt_trichotomy first second with ordered | same | reversed
    · have := (magnitude_units_order_iff first second).mpr ordered
      omega
    · exact same
    · have := (magnitude_units_order_iff second first).mpr reversed
      omega
  · rintro rfl
    rfl

@[simp] theorem magnitude_units_zero : magnitudeUnits 0 = 0 := by
  simp [magnitudeUnits, finiteUnits]

@[simp] theorem magnitude_units_zero_iff (magnitude : Nat) :
    magnitudeUnits magnitude = 0 ↔ magnitude = 0 := by
  simpa using magnitude_units_equal_iff magnitude 0

theorem decoded_value_encoding (bits : U32) (flush : Bool) :
    decode bits flush = encodedValue (decide (bits.val ≥ signBase))
      (normalizedMagnitude (bits.val % signBase) flush) := by
  have magnitudeBound : bits.val % signBase < signBase := Nat.mod_lt _ (by decide)
  have fractionBound : bits.val % signBase % fractionBase < fractionBase := Nat.mod_lt _ (by decide)
  have exponentBound : bits.val % signBase / fractionBase ≤ 255 := by
    dsimp only [fractionBase, signBase] at *
    omega
  have exponentInfinity : bits.val % signBase / fractionBase = 255 ↔
      infinityMagnitude ≤ bits.val % signBase := by
    dsimp only [fractionBase, signBase, infinityMagnitude] at *
    omega
  have fractionInfinity : bits.val % signBase % fractionBase = 0 ∧
      bits.val % signBase / fractionBase = 255 ↔ bits.val % signBase = infinityMagnitude := by
    dsimp only [fractionBase, signBase, infinityMagnitude] at *
    omega
  unfold decode encodedValue normalizedMagnitude
  by_cases zeroExponent : bits.val % signBase / fractionBase = 0
  · have small : bits.val % signBase < fractionBase := by
      dsimp only [fractionBase] at *
      omega
    have notInfinity : ¬ infinityMagnitude < bits.val % signBase := by
      dsimp only [fractionBase, infinityMagnitude] at *
      omega
    have notEqual : bits.val % signBase ≠ infinityMagnitude := by
      dsimp only [fractionBase, infinityMagnitude] at *
      omega
    cases flush <;> simp [zeroExponent, small, notInfinity, notEqual, magnitudeUnits, finiteUnits]
  · have notSmall : ¬ bits.val % signBase < fractionBase := by
      dsimp only [fractionBase] at *
      omega
    simp only [notSmall, Bool.and_false, Bool.false_eq_true, ite_false]
    by_cases exponent255 : bits.val % signBase / fractionBase = 255
    · by_cases fractionZero : bits.val % signBase % fractionBase = 0
      · have same : bits.val % signBase = infinityMagnitude := fractionInfinity.mp ⟨fractionZero, exponent255⟩
        simp [same, infinityMagnitude, fractionBase]
      · have greater : infinityMagnitude < bits.val % signBase := by
          dsimp only [fractionBase, infinityMagnitude] at *
          omega
        simp [exponent255, fractionZero, greater]
    · have smaller : bits.val % signBase < infinityMagnitude := by
        dsimp only [fractionBase, infinityMagnitude] at *
        omega
      simp [exponent255, smaller, ne_of_lt smaller, not_lt_of_ge (Nat.le_of_lt smaller),
        magnitudeUnits, zeroExponent]

theorem signed_finite_less (negativeFirst negativeSecond : Bool) (first second : Nat) :
    (if negativeFirst then -(first : Int) else (first : Int)) <
      (if negativeSecond then -(second : Int) else (second : Int)) ↔
    (if negativeFirst = negativeSecond then
      if negativeFirst then second < first else first < second
    else negativeFirst = true ∧ (first ≠ 0 ∨ second ≠ 0)) := by
  cases negativeFirst <;> cases negativeSecond <;> simp <;> omega

theorem signed_finite_equal (negativeFirst negativeSecond : Bool) (first second : Nat) :
    (if negativeFirst then -(first : Int) else (first : Int)) =
      (if negativeSecond then -(second : Int) else (second : Int)) ↔
    (if negativeFirst = negativeSecond then first = second else first = 0 ∧ second = 0) := by
  cases negativeFirst <;> cases negativeSecond <;> simp <;> omega

end Instruction.IEEEComparison
