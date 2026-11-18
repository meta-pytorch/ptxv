/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ComparisonSelectionOperands
import Instruction.ComparisonSelectionProducer

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[step] theorem actual_selection_pair_plan_spec {R : Type}
    (readerInst : comparison_select.NodeReader R) (reader : R)
    (nodeOf : Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (front : Front) (comparison selection : Op) (x y : U32) :
    comparison_select.from_pair readerInst front comparison selection x y reader ⦃ plan =>
      plan.map observedComparisonPlan = comparisonPairPlan front comparison selection x y nodeOf ⦄ := by
  unfold comparison_select.from_pair
  step with actual_comparison_plan_spec comparison false false as ⟨produced, hproduced⟩
  cases produced with
  | none =>
      have hnone : plainComparison comparison = none := by
        simpa [comparisonPlan] using hproduced.symm
      simp [comparisonPairPlan, hnone]
  | some plan =>
      have hmode : plainComparison comparison = some plan.comparison := by
        cases hplain : plainComparison comparison <;> simp_all [comparisonPlan, observedComparisonPlan]
      cases comparison <;> try (solve | simp [plainComparison] at hmode)
      case Fsetp destination mode a b neg abs condition bop ftz =>
        cases selection <;> try (solve | simp [comparisonPairPlan, hmode])
        case Sel result first second predicate =>
          step with RegisterOwner.source_comparison_plan 0#usize front destination true
            as ⟨policy, hkey, hremember⟩
          simp only [Bool.true_and] at hremember
          by_cases hwritable : RegisterOwner.writablePredicate front destination
          · simp only [hremember, hwritable, ite_true]
            rw [RegisterOwner.source_normalize_predicate]
            simp only [bind_ok]
            cases hpredicate : (RegisterOwner.normalizedPredicate front predicate).reg with
            | none => simp [comparisonPairPlan, hmode, hwritable, hpredicate]
            | some register =>
                step with comparison_same_register_spec destination register as ⟨same, hsame⟩
                by_cases halias : key destination = key register
                · simp only [halias, decide_true] at hsame
                  simp only [hsame, ite_true]
                  step with hreader first as ⟨left, hleft⟩
                  cases left with
                  | none => simp [comparisonPairPlan, hmode, hwritable, hpredicate, halias, ← hleft,
                      comparisonSelectionFromOperands]
                  | some left =>
                      step with hreader second as ⟨right, hright⟩
                      cases right with
                      | none => simp [comparisonPairPlan, hmode, hwritable, hpredicate, halias, ← hleft, ← hright,
                          comparisonSelectionFromOperands]
                      | some right =>
                          have hmodel : comparisonPairPlan front
                              (.Fsetp destination mode a b neg abs condition bop ftz)
                              (.Sel result first second predicate) x y nodeOf =
                              comparisonSelectionFromOperands plan.comparison predicate.neg x y (some left) (some right) := by
                            simp [comparisonPairPlan, hmode, hwritable, hpredicate, halias, ← hleft, ← hright]
                          rw [hmodel]
                          clear hleft hright
                          by_cases hxy : x = y <;> by_cases hleftx : left = x <;> by_cases hrighty : right = y <;>
                            by_cases hlefty : left = y <;> by_cases hrightx : right = x <;>
                            simp_all [comparisonPairPlan, comparisonSelectionFromOperands, observedComparisonPlan]
                · simp [hsame, halias, comparisonPairPlan, hmode, hwritable, hpredicate]
          · simp [hremember, hwritable, comparisonPairPlan, hmode]

end Instruction
