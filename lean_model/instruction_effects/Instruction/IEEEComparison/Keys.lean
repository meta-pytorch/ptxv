/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Comparison

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.IEEEComparison

def signedKey (negative : Bool) (magnitude : Nat) : Int :=
  if negative then -(magnitude : Int) else (magnitude : Int)

def keyTruth (mode : PtxvSim.ptxv_ir.FCmp) (first second : Int) (unordered : Bool) : Bool :=
  match mode with
  | .F => false
  | .T => true
  | .Lt => !unordered && decide (first < second)
  | .Eq => !unordered && decide (first = second)
  | .Le => !unordered && decide (first ≤ second)
  | .Gt => !unordered && decide (second < first)
  | .Ne => !unordered && !decide (first = second)
  | .Ge => !unordered && decide (second ≤ first)
  | .Num => !unordered
  | .Nan => unordered
  | .Ltu => unordered || decide (first < second)
  | .Equ => unordered || decide (first = second)
  | .Leu => unordered || decide (first ≤ second)
  | .Gtu => unordered || decide (second < first)
  | .Neu => unordered || !decide (first = second)
  | .Geu => unordered || decide (second ≤ first)

theorem key_less_exact (negativeFirst negativeSecond : Bool) (first second : Nat) :
    decide (signedKey negativeFirst first < signedKey negativeSecond second) =
      bitLess negativeFirst negativeSecond first second := by
  cases negativeFirst <;> cases negativeSecond <;>
    apply Bool.eq_iff_iff.mpr <;> simp [signedKey, bitLess] <;> omega

theorem key_equal_exact (negativeFirst negativeSecond : Bool) (first second : Nat) :
    decide (signedKey negativeFirst first = signedKey negativeSecond second) =
      bitEqual negativeFirst negativeSecond first second := by
  cases negativeFirst <;> cases negativeSecond <;>
    apply Bool.eq_iff_iff.mpr <;> simp [signedKey, bitEqual] <;> omega

theorem integer_le_split (first second : Int) :
    decide (first ≤ second) = (decide (first < second) || decide (first = second)) := by
  apply Bool.eq_iff_iff.mpr
  simp
  omega

theorem bit_equal_symmetric (negativeFirst negativeSecond : Bool) (first second : Nat) :
    bitEqual negativeSecond negativeFirst second first = bitEqual negativeFirst negativeSecond first second := by
  rw [← key_equal_exact, ← key_equal_exact]
  simp only [eq_comm]

theorem key_truth_facts_exact (mode : PtxvSim.ptxv_ir.FCmp)
    (negativeFirst negativeSecond : Bool) (first second : Nat) :
    keyTruth mode (signedKey negativeFirst first) (signedKey negativeSecond second)
      (decide (infinityMagnitude < first ∨ infinityMagnitude < second)) =
      modeTruth mode (bitFacts negativeFirst negativeSecond first second) := by
  by_cases exceptional : infinityMagnitude < first ∨ infinityMagnitude < second
  · cases mode <;> simp [keyTruth, bitFacts, modeTruth, exceptional]
  · cases mode <;>
      simp only [keyTruth, bitFacts, modeTruth, exceptional, decide_false, ite_false,
        Bool.not_false, Bool.true_and, Bool.false_or, integer_le_split,
        key_less_exact, key_equal_exact,
        bit_equal_symmetric negativeFirst negativeSecond first second,
        bit_less_reverse negativeFirst negativeSecond first second]

theorem comparison_keys_exact (mode : PtxvSim.ptxv_ir.FCmp) (first second : U32) (flush : Bool) :
    keyTruth mode
      (signedKey (decide (first.val ≥ signBase)) (normalizedMagnitude (first.val % signBase) flush))
      (signedKey (decide (second.val ≥ signBase)) (normalizedMagnitude (second.val % signBase) flush))
      (decide (infinityMagnitude < normalizedMagnitude (first.val % signBase) flush ∨
        infinityMagnitude < normalizedMagnitude (second.val % signBase) flush)) =
      compare mode first second flush := by
  rw [key_truth_facts_exact, comparison_bits_exact]

end Instruction.IEEEComparison
