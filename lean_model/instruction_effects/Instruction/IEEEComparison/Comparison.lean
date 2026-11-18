/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Facts

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.IEEEComparison

structure ComparisonFacts where
  unordered : Bool
  equal : Bool
  less : Bool
  greater : Bool
  deriving DecidableEq

def factsOfValues (first second : Value) : ComparisonFacts :=
  ⟨unordered first second, equal first second, less first second, less second first⟩

def bitFacts (negativeFirst negativeSecond : Bool) (first second : Nat) : ComparisonFacts :=
  if infinityMagnitude < first ∨ infinityMagnitude < second then ⟨true, false, false, false⟩
  else
    let equal := bitEqual negativeFirst negativeSecond first second
    let less := bitLess negativeFirst negativeSecond first second
    ⟨false, equal, less, !equal && !less⟩

def modeTruth (mode : PtxvSim.ptxv_ir.FCmp) (facts : ComparisonFacts) : Bool :=
  match mode with
  | .F => false
  | .T => true
  | .Lt => facts.less
  | .Eq => facts.equal
  | .Le => facts.less || facts.equal
  | .Gt => facts.greater
  | .Ne => !facts.unordered && !facts.equal
  | .Ge => facts.greater || facts.equal
  | .Num => !facts.unordered
  | .Nan => facts.unordered
  | .Ltu => facts.unordered || facts.less
  | .Equ => facts.unordered || facts.equal
  | .Leu => facts.unordered || facts.less || facts.equal
  | .Gtu => facts.unordered || facts.greater
  | .Neu => facts.unordered || !facts.equal
  | .Geu => facts.unordered || facts.greater || facts.equal

@[simp] theorem facts_nan_left (value : Value) :
    factsOfValues .nan value = ⟨true, false, false, false⟩ := by cases value <;> rfl

@[simp] theorem facts_nan_right (value : Value) :
    factsOfValues value .nan = ⟨true, false, false, false⟩ := by cases value <;> rfl

theorem encoded_facts_exact (negativeFirst negativeSecond : Bool) (first second : Nat) :
    factsOfValues (encodedValue negativeFirst first) (encodedValue negativeSecond second) =
      bitFacts negativeFirst negativeSecond first second := by
  by_cases firstNan : infinityMagnitude < first
  · simp only [encodedValue, firstNan, ite_true, facts_nan_left, bitFacts, or_true, true_or]
  · by_cases secondNan : infinityMagnitude < second
    · have secondValue : encodedValue negativeSecond second = .nan := by simp [encodedValue, secondNan]
      rw [secondValue, facts_nan_right]
      simp [bitFacts, secondNan]
    · have firstOrdered : first ≤ infinityMagnitude := by omega
      have secondOrdered : second ≤ infinityMagnitude := by omega
      simp only [factsOfValues, encoded_unordered, firstNan, secondNan, decide_false, Bool.false_or,
        bitFacts, firstNan, secondNan, or_self, ite_false,
        encoded_less_ordered negativeFirst negativeSecond first second firstOrdered secondOrdered,
        encoded_less_ordered negativeSecond negativeFirst second first secondOrdered firstOrdered,
        encoded_equal_ordered negativeFirst negativeSecond first second firstOrdered secondOrdered]
      rw [bit_less_reverse negativeFirst negativeSecond first second]

theorem mode_truth_exact (mode : PtxvSim.ptxv_ir.FCmp) (first second : Value) :
    modeTruth mode (factsOfValues first second) = truth mode first second := by
  cases mode <;> rfl

theorem comparison_bits_exact (mode : PtxvSim.ptxv_ir.FCmp) (first second : U32) (flush : Bool) :
    modeTruth mode (bitFacts (decide (first.val ≥ signBase)) (decide (second.val ≥ signBase))
      (normalizedMagnitude (first.val % signBase) flush)
      (normalizedMagnitude (second.val % signBase) flush)) = compare mode first second flush := by
  unfold compare
  rw [← mode_truth_exact, decoded_value_encoding first flush, decoded_value_encoding second flush,
    encoded_facts_exact]

end Instruction.IEEEComparison
