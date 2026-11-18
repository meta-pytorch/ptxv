/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ComparisonSelectionModel
import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

def observedComparisonPlan (plan : comparison_select.Plan) : ComparisonSelection :=
  ⟨plan.comparison, plan.negate, plan.swap⟩

@[step] theorem actual_comparison_plan_spec (operation : Op) (negate swap : Bool) :
    comparison_select.from_operation operation negate swap ⦃ plan =>
      plan.map observedComparisonPlan = comparisonPlan operation negate swap ⦄ := by
  cases operation <;> try (solve | simp [comparison_select.from_operation, comparisonPlan, plainComparison])
  case Fsetp destination comparison x y neg abs condition bop ftz =>
    obtain ⟨n0, n1, hneg⟩ := (List.length_eq_two (l := neg.val)).mp (by simp)
    obtain ⟨a0, a1, habs⟩ := (List.length_eq_two (l := abs.val)).mp (by simp)
    rcases condition with ⟨condition, cn⟩
    cases bop <;> cases ftz <;> cases condition <;> cases cn <;>
      cases n0 <;> cases n1 <;> cases a0 <;> cases a1 <;>
      simp [comparison_select.from_operation, comparisonPlan, plainComparison,
        Array.index_usize, hneg, habs, observedComparisonPlan]

def ComparatorContract {C : Type} (comparatorInst : comparison_select.Comparator C)
    (compare : FCmp → U32 → U32 → Bool) : Prop :=
  ∀ comparator comparison x y, comparatorInst.compare comparator comparison x y ⦃ result =>
    result = compare comparison x y ⦄

@[step] theorem comparison_evaluate_spec {C : Type}
    (comparatorInst : comparison_select.Comparator C) (comparator : C)
    (compare : FCmp → U32 → U32 → Bool) (hcontract : ComparatorContract comparatorInst compare)
    (plan : comparison_select.Plan) (x y : U32) :
    comparison_select.evaluate comparatorInst plan x y comparator ⦃ result =>
      result = comparisonSelectionBits compare (observedComparisonPlan plan) x y ⦄ := by
  unfold comparison_select.evaluate
  step with hcontract comparator plan.comparison x y as ⟨answer, hanswer⟩
  subst answer
  split <;> split <;> simp_all [comparisonSelectionBits, observedComparisonPlan]

def comparisonPair (plan : ComparisonSelection) : Op × Op :=
  let x : Reg := ⟨.R, 0#u16⟩
  let y : Reg := ⟨.R, 1#u16⟩
  let predicate : Reg := ⟨.P, 0#u16⟩
  let destination : Reg := ⟨.R, 2#u16⟩
  (.Fsetp predicate plan.comparison (.Reg x 1#u8) (.Reg y 1#u8)
      (Array.repeat 2#usize false) (Array.repeat 2#usize false) ⟨none, false⟩ .And false,
    .Sel destination (.Reg (if plan.swap then y else x) 1#u8)
      (.Reg (if plan.swap then x else y) 1#u8) ⟨some predicate, plan.negate⟩)

@[step] theorem comparison_pair_spec (plan : comparison_select.Plan) :
    comparison_select.operations plan ⦃ pair => pair = comparisonPair (observedComparisonPlan plan) ⦄ := by
  cases hswap : plan.swap <;>
    simp [comparison_select.operations, comparisonPair, observedComparisonPlan, hswap]

end Instruction
