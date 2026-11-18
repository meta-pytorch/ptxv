/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ComparisonSelectionRefinement
import Instruction.OwnerComparisonPlan

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
namespace Instruction

structure ComparisonEvidence where
  destination : Key
  comparison : FCmp
  left : U32
  right : U32

def observedComparisonEvidence (evidence : comparison_select.Evidence) : ComparisonEvidence :=
  ⟨key evidence.destination, evidence.comparison, evidence.left, evidence.right⟩

def comparisonCapture (front : Front) (operation : Op) (nodeOf : Val → Option U32) :
    Option ComparisonEvidence := do
  let comparison ← plainComparison operation
  match operation with
  | .Fsetp destination _ a b _ _ _ _ _ =>
      if RegisterOwner.writablePredicate front destination then
        let left ← nodeOf a
        let right ← nodeOf b
        some ⟨key destination, comparison, left, right⟩
      else none
  | _ => none

def comparisonPairPlan (front : Front) (comparison selection : Op) (x y : U32)
    (nodeOf : Val → Option U32) : Option ComparisonSelection := do
  let mode ← plainComparison comparison
  match comparison, selection with
  | .Fsetp destination _ _ _ _ _ _ _ _, .Sel _ a b predicate =>
      if RegisterOwner.writablePredicate front destination then
        let register ← (RegisterOwner.normalizedPredicate front predicate).reg
        if key destination = key register then
          comparisonSelectionFromOperands mode predicate.neg x y (nodeOf a) (nodeOf b)
        else none
      else none
  | _, _ => none

def NodeReaderContract {R : Type} (readerInst : comparison_select.NodeReader R)
    (reader : R) (nodeOf : Val → Option U32) : Prop :=
  ∀ value, readerInst.node_of reader value ⦃ nodes => nodes = nodeOf value ⦄

@[step] theorem comparison_same_register_spec (left right : Reg) :
    comparison_select.same_register left right ⦃ same => same = decide (key left = key right) ⦄ := by
  rcases left with ⟨leftFile, leftIndex⟩
  rcases right with ⟨rightFile, rightIndex⟩
  have hindex : leftIndex = rightIndex ↔ leftIndex.val = rightIndex.val := by
    constructor
    · intro h; subst rightIndex; rfl
    · exact UScalar.eq_of_val_eq
  cases leftFile <;> cases rightFile <;>
    simp [comparison_select.same_register, read_discriminant, File.read_discriminant, key, hindex,
      lift, bind_ok, WP.spec_ok]

@[step] theorem actual_comparison_capture_spec {R : Type}
    (readerInst : comparison_select.NodeReader R) (reader : R)
    (nodeOf : Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (front : Front) (operation : Op) :
    comparison_select.capture readerInst front operation reader ⦃ evidence =>
      evidence.map observedComparisonEvidence = comparisonCapture front operation nodeOf ⦄ := by
  unfold comparison_select.capture
  step with actual_comparison_plan_spec operation false false as ⟨produced, hproduced⟩
  cases produced with
  | none =>
      have hnone : plainComparison operation = none := by
        simpa [comparisonPlan] using hproduced.symm
      simp [comparisonCapture, hnone]
  | some plan =>
      have hmode : plainComparison operation = some plan.comparison := by
        cases hplain : plainComparison operation <;> simp_all [comparisonPlan, observedComparisonPlan]
      cases operation <;> try (solve | simp [plainComparison] at hmode)
      case Fsetp destination mode a b neg abs condition bop ftz =>
        step with RegisterOwner.source_comparison_plan 0#usize front destination true
          as ⟨policy, hkey, hremember⟩
        simp only [Bool.true_and] at hremember
        by_cases hwritable : RegisterOwner.writablePredicate front destination
        · simp only [hremember, hwritable, ite_true]
          step with hreader a as ⟨left, hleft⟩
          cases left with
          | none => simp [comparisonCapture, hmode, hwritable, ← hleft]
          | some left =>
              step with hreader b as ⟨right, hright⟩
              cases right with
              | none => simp [comparisonCapture, hmode, hwritable, ← hleft, ← hright]
              | some right => simp [comparisonCapture, hmode, hwritable, ← hleft, ← hright, observedComparisonEvidence]
        · simp [hremember, hwritable, comparisonCapture, hmode]

end Instruction
