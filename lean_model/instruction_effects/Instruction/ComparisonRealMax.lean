/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ComparisonSelectionRefinement

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

def realMaximumPlan (plan : ComparisonSelection) : Bool :=
  match plan.comparison with
  | .Gt | .Ge | .Gtu | .Geu => plan.negate == plan.swap
  | .Lt | .Le | .Ltu | .Leu => plan.negate != plan.swap
  | _ => false

def orderedComparison {α : Type} [LinearOrder α] (comparison : FCmp) (x y : α) : Bool :=
  match comparison with
  | .Gt | .Gtu => decide (y < x)
  | .Ge | .Geu => decide (y ≤ x)
  | .Lt | .Ltu => decide (x < y)
  | .Le | .Leu => decide (x ≤ y)
  | _ => false

def orderedSelection {α : Type} [LinearOrder α] (plan : ComparisonSelection) (x y : α) : α :=
  if orderedComparison plan.comparison x y != plan.negate then
    if plan.swap then y else x
  else if plan.swap then x else y

theorem classified_selection_is_real_maximum {α : Type} [LinearOrder α]
    (plan : ComparisonSelection) (x y : α) (accepted : realMaximumPlan plan = true) :
    orderedSelection plan x y = max x y := by
  rcases plan with ⟨comparison, negate, swap⟩
  rcases lt_trichotomy x y with less | equal | greater
  · cases comparison <;> cases negate <;> cases swap <;>
      simp_all [realMaximumPlan, orderedSelection, orderedComparison,
        le_of_lt, not_le_of_gt, not_lt_of_ge]
  · subst y
    simp [orderedSelection]
  · cases comparison <;> cases negate <;> cases swap <;>
      simp_all [realMaximumPlan, orderedSelection, orderedComparison,
        le_of_lt, not_le_of_gt, not_lt_of_ge]

@[step] theorem source_real_max (plan : comparison_select.Plan) :
    comparison_select.real_max plan ⦃ result =>
      result = realMaximumPlan (observedComparisonPlan plan) ⦄ := by
  rcases plan with ⟨comparison, negate, swap⟩
  cases comparison <;> cases negate <;> cases swap <;>
    simp [comparison_select.real_max, realMaximumPlan, observedComparisonPlan]

theorem bit_selection_interprets_as_real_maximum {α : Type} [LinearOrder α]
    (interpret : U32 → α) (compare : FCmp → U32 → U32 → Bool)
    (plan : ComparisonSelection) (x y : U32)
    (accepted : realMaximumPlan plan = true)
    (truth : compare plan.comparison x y =
      orderedComparison plan.comparison (interpret x) (interpret y)) :
    interpret (comparisonSelectionBits compare plan x y) = max (interpret x) (interpret y) := by
  have projected : interpret (comparisonSelectionBits compare plan x y) =
      orderedSelection plan (interpret x) (interpret y) := by
    unfold comparisonSelectionBits orderedSelection
    rw [truth]
    split <;> split <;> rfl
  rw [projected]
  exact classified_selection_is_real_maximum plan (interpret x) (interpret y) accepted

end Instruction
