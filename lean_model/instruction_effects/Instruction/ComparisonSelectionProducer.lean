/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ComparisonSelectionRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
namespace Instruction

def comparisonArgument (x y : U32) : Val → U32
  | .Reg register _ =>
      if key register = ⟨.R, 0⟩ then x else if key register = ⟨.R, 1⟩ then y else 0#u32
  | _ => 0#u32

def comparisonInstructionResult (compare : FCmp → U32 → U32 → Bool)
    (operation : Op) (x y : U32) : Option (Reg × Bool) :=
  match operation, plainComparison operation with
  | .Fsetp destination _ a b _ _ _ _ _, some comparison =>
      some (destination, compare comparison (comparisonArgument x y a) (comparisonArgument x y b))
  | _, _ => none

def selectionInstructionResult (operation : Op) (predicate : Reg) (answer : Bool)
    (x y : U32) : Option U32 :=
  match operation with
  | .Sel _ a b condition =>
      let raw := match condition.reg with
        | none => true
        | some register => if key register = key predicate then answer else false
      some (comparisonArgument x y (if raw != condition.neg then a else b))
  | _ => none

def comparisonPairResult (compare : FCmp → U32 → U32 → Bool)
    (pair : Op × Op) (x y : U32) : Option U32 := do
  let (predicate, answer) ← comparisonInstructionResult compare pair.1 x y
  selectionInstructionResult pair.2 predicate answer x y

theorem comparison_pair_all_inputs (compare : FCmp → U32 → U32 → Bool)
    (plan : ComparisonSelection) (x y : U32) :
    comparisonPairResult compare (comparisonPair plan) x y =
      some (comparisonSelectionBits compare plan x y) := by
  cases hswap : plan.swap <;>
    cases hnegate : plan.negate <;>
    cases hcompare : compare plan.comparison x y <;>
    simp [comparisonPairResult, comparisonPair, comparisonInstructionResult, plainComparison,
      comparisonArgument, selectionInstructionResult, key, comparisonSelectionBits,
      hswap, hnegate, hcompare, Array.repeat_val]

@[step] theorem actual_comparison_pair_semantics (compare : FCmp → U32 → U32 → Bool)
    (plan : comparison_select.Plan) (x y : U32) :
    comparison_select.operations plan ⦃ pair =>
      comparisonPairResult compare pair x y =
        some (comparisonSelectionBits compare (observedComparisonPlan plan) x y) ⦄ := by
  step with comparison_pair_spec plan as ⟨pair, hpair⟩
  simpa [hpair] using comparison_pair_all_inputs compare (observedComparisonPlan plan) x y

@[step] theorem actual_comparison_producer_all_inputs {C : Type}
    (comparatorInst : comparison_select.Comparator C) (comparator : C)
    (compare : FCmp → U32 → U32 → Bool) (hcontract : ComparatorContract comparatorInst compare)
    (operation : Op) (negate swap : Bool) (x y : U32) :
    (do
      let plan ← comparison_select.from_operation operation negate swap
      match plan with
      | none => ok none
      | some plan =>
          let pair ← comparison_select.operations plan
          let bits ← comparison_select.evaluate comparatorInst plan x y comparator
          ok (some (pair, bits))) ⦃ result =>
      match comparisonPlan operation negate swap, result with
      | none, none => True
      | some expected, some (pair, bits) =>
          bits = comparisonSelectionBits compare expected x y ∧
          comparisonPairResult compare pair x y = some bits
      | _, _ => False ⦄ := by
  step with actual_comparison_plan_spec operation negate swap as ⟨produced, hproduced⟩
  cases produced with
  | none =>
      simp only [Option.map_none] at hproduced
      simp [← hproduced]
  | some plan =>
      step with comparison_pair_spec plan as ⟨pair, hpair⟩
      step with comparison_evaluate_spec comparatorInst comparator compare hcontract plan x y
        as ⟨bits, hbits⟩
      simp only [Option.map_some] at hproduced
      simp only [← hproduced]
      refine ⟨hbits, ?_⟩
      simpa [hpair, hbits] using comparison_pair_all_inputs compare (observedComparisonPlan plan) x y

end Instruction
