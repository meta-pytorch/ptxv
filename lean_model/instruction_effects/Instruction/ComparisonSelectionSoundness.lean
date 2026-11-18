/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ComparisonSelectionPair

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
namespace Instruction

theorem comparison_capture_has_actual_mode (front : Front) (operation : Op)
    (nodeOf : Val → Option U32) (evidence : ComparisonEvidence)
    (hcaptured : comparisonCapture front operation nodeOf = some evidence) :
    plainComparison operation = some evidence.comparison := by
  cases hplain : plainComparison operation with
  | none => simp [comparisonCapture, hplain] at hcaptured
  | some mode =>
      cases operation <;> try (solve | simp [comparisonCapture, hplain] at hcaptured)
      case Fsetp destination cmp a b neg abs condition bop ftz =>
        by_cases hwritable : RegisterOwner.writablePredicate front destination
        · cases hleft : nodeOf a with
          | none => simp [comparisonCapture, hplain, hwritable, hleft] at hcaptured
          | some left =>
              cases hright : nodeOf b with
              | none => simp [comparisonCapture, hplain, hwritable, hleft, hright] at hcaptured
              | some right =>
                  have hevidence : (⟨key destination, mode, left, right⟩ : ComparisonEvidence) = evidence := by
                    simpa [comparisonCapture, hplain, hwritable, hleft, hright] using hcaptured
                  cases hevidence
                  rfl
        · simp [comparisonCapture, hplain, hwritable] at hcaptured

@[step] theorem actual_capture_has_actual_mode {R : Type}
    (readerInst : comparison_select.NodeReader R) (reader : R)
    (nodeOf : Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (front : Front) (operation : Op) :
    comparison_select.capture readerInst front operation reader ⦃ produced =>
      match produced with
      | none => True
      | some evidence => plainComparison operation = some evidence.comparison ⦄ := by
  step with actual_comparison_capture_spec readerInst reader nodeOf hreader front operation
    as ⟨produced, hproduced⟩
  cases produced with
  | none => trivial
  | some evidence =>
      exact comparison_capture_has_actual_mode front operation nodeOf (observedComparisonEvidence evidence)
        (by simpa using hproduced.symm)

def selectedNodeBits (compare : FCmp → U32 → U32 → Bool) (valuation : U32 → U32)
    (mode : FCmp) (predicate : Pred) (x y : U32) (first second : Val)
    (nodeOf : Val → Option U32) : Option U32 :=
  if compare mode (valuation x) (valuation y) != predicate.neg then
    (nodeOf first).map valuation
  else (nodeOf second).map valuation

theorem comparison_pair_plan_preserves_all_node_valuations
    (compare : FCmp → U32 → U32 → Bool) (valuation : U32 → U32)
    (front : Front) (comparison : Op) (mode : FCmp)
    (destination : Reg) (first second : Val) (predicate : Pred) (x y : U32)
    (nodeOf : Val → Option U32) (plan : ComparisonSelection)
    (hmode : plainComparison comparison = some mode)
    (hplan : comparisonPairPlan front comparison (.Sel destination first second predicate) x y nodeOf = some plan) :
    some (comparisonSelectionBits compare plan (valuation x) (valuation y)) =
      selectedNodeBits compare valuation mode predicate x y first second nodeOf := by
  cases comparison <;> try (solve | simp [plainComparison] at hmode)
  case Fsetp register instructionMode a b neg abs condition bop ftz =>
    by_cases hwritable : RegisterOwner.writablePredicate front register
    · cases hpredicate : (RegisterOwner.normalizedPredicate front predicate).reg with
      | none => simp [comparisonPairPlan, hmode, hwritable, hpredicate] at hplan
      | some predicateRegister =>
          by_cases halias : key register = key predicateRegister
          · have hoperands : comparisonSelectionFromOperands mode predicate.neg x y (nodeOf first) (nodeOf second) = some plan := by
              simpa [comparisonPairPlan, hmode, hwritable, hpredicate, halias] using hplan
            cases hfirst : nodeOf first with
            | none => simp [comparisonSelectionFromOperands, hfirst] at hoperands
            | some firstNode =>
                cases hsecond : nodeOf second with
                | none => simp [comparisonSelectionFromOperands, hsecond] at hoperands
                | some secondNode =>
                    have hnode := comparison_selection_refines_node_semantics compare valuation mode predicate.neg
                      x y firstNode secondNode plan (by simpa [hfirst, hsecond] using hoperands)
                    rw [hnode]
                    simp only [selectedNodeBits, hfirst, hsecond, Option.map_some]
                    split <;> rfl
          · simp [comparisonPairPlan, hmode, hwritable, hpredicate, halias] at hplan
    · simp [comparisonPairPlan, hmode, hwritable] at hplan

@[step] theorem actual_selection_preserves_all_node_valuations {R : Type}
    (readerInst : comparison_select.NodeReader R) (reader : R)
    (nodeOf : Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (compare : FCmp → U32 → U32 → Bool)
    (front : Front) (comparison : Op) (mode : FCmp)
    (destination : Reg) (first second : Val) (predicate : Pred) (x y : U32)
    (hmode : plainComparison comparison = some mode) :
    comparison_select.from_pair readerInst front comparison (.Sel destination first second predicate) x y reader
      ⦃ produced => match produced with
        | none => True
        | some plan => ∀ valuation : U32 → U32,
            some (comparisonSelectionBits compare (observedComparisonPlan plan) (valuation x) (valuation y)) =
              selectedNodeBits compare valuation mode predicate x y first second nodeOf ⦄ := by
  step with actual_selection_pair_plan_spec readerInst reader nodeOf hreader front comparison
    (.Sel destination first second predicate) x y as ⟨produced, hproduced⟩
  cases produced with
  | none => simp
  | some plan =>
      intro valuation
      exact comparison_pair_plan_preserves_all_node_valuations compare valuation front comparison mode
        destination first second predicate x y nodeOf (observedComparisonPlan plan) hmode (by simpa using hproduced.symm)

@[step] theorem actual_selection_pair_execution_refines_source_operands {R C : Type}
    (readerInst : comparison_select.NodeReader R) (reader : R)
    (nodeOf : Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (comparatorInst : comparison_select.Comparator C) (comparator : C)
    (compare : FCmp → U32 → U32 → Bool) (hcompare : ComparatorContract comparatorInst compare)
    (valuation : U32 → U32) (front : Front) (comparison : Op) (mode : FCmp)
    (destination : Reg) (first second : Val) (predicate : Pred) (x y : U32)
    (hmode : plainComparison comparison = some mode) :
    (do
      let plan ← comparison_select.from_pair readerInst front comparison (.Sel destination first second predicate) x y reader
      match plan with
      | none => ok none
      | some plan =>
          let pair ← comparison_select.operations plan
          let bits ← comparison_select.evaluate comparatorInst plan (valuation x) (valuation y) comparator
          ok (some (pair, bits))) ⦃ produced =>
      match produced with
      | none => True
      | some (pair, bits) =>
          comparisonPairResult compare pair (valuation x) (valuation y) = some bits ∧
          some bits = selectedNodeBits compare valuation mode predicate x y first second nodeOf ⦄ := by
  step with actual_selection_preserves_all_node_valuations readerInst reader nodeOf hreader compare
    front comparison mode destination first second predicate x y hmode as ⟨produced, hproduced⟩
  cases produced with
  | none => simp
  | some plan =>
      step with actual_comparison_pair_semantics compare plan (valuation x) (valuation y) as ⟨pair, hpair⟩
      step with comparison_evaluate_spec comparatorInst comparator compare hcompare plan (valuation x) (valuation y)
        as ⟨bits, hbits⟩
      exact ⟨by simpa [hbits] using hpair, by simpa [hbits] using hproduced valuation⟩

end Instruction
