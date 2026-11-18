/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Model

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

structure ComparisonSelection where
  comparison : FCmp
  negate : Bool
  swap : Bool

def plainComparison : Op → Option FCmp
  | .Fsetp _ comparison _ _ neg abs ⟨none, false⟩ .And false =>
      if neg.val = [false, false] ∧ abs.val = [false, false] then some comparison else none
  | _ => none

def comparisonPlan (operation : Op) (negate swap : Bool) : Option ComparisonSelection :=
  (plainComparison operation).map fun comparison => ⟨comparison, negate, swap⟩

-- The IEEE primitive is deliberately a parameter: selection must preserve its answer.
def comparisonSelectionBits (compare : FCmp → U32 → U32 → Bool)
    (plan : ComparisonSelection) (x y : U32) : U32 :=
  if compare plan.comparison x y != plan.negate then
    if plan.swap then y else x
  else if plan.swap then x else y

def comparisonSelectionExpected (compare : FCmp → U32 → U32 → Bool)
    (comparison : FCmp) (negate swap : Bool) (x y : U32) : U32 :=
  let first := if swap then y else x
  let second := if swap then x else y
  if compare comparison x y != negate then first else second

theorem comparison_selection_all_inputs (compare : FCmp → U32 → U32 → Bool)
    (comparison : FCmp) (negate swap : Bool) (x y : U32) :
    comparisonSelectionBits compare ⟨comparison, negate, swap⟩ x y =
      comparisonSelectionExpected compare comparison negate swap x y := by
  rfl

theorem comparison_selection_preserves_tie (compare : FCmp → U32 → U32 → Bool)
    (comparison : FCmp) (negate swap : Bool) (x y : U32)
    (hcompare : compare comparison x y = negate) :
    comparisonSelectionBits compare ⟨comparison, negate, swap⟩ x y =
      (if swap then x else y) := by
  simp [comparisonSelectionBits, hcompare]

theorem comparison_selection_preserves_same_source (compare : FCmp → U32 → U32 → Bool)
    (plan : ComparisonSelection) (x : U32) :
    comparisonSelectionBits compare plan x x = x := by
  simp [comparisonSelectionBits]

def comparisonSelectionFromOperands (comparison : FCmp) (negate : Bool)
    (x y : U32) (first second : Option U32) : Option ComparisonSelection := do
  let first ← first
  let second ← second
  if first = x ∧ second = y then some ⟨comparison, negate, false⟩
  else if first = y ∧ second = x then some ⟨comparison, negate, true⟩
  else none

theorem comparison_selection_sources_complete (comparison : FCmp) (negate : Bool)
    (x y first second : U32) (plan : ComparisonSelection)
    (hplan : comparisonSelectionFromOperands comparison negate x y (some first) (some second) = some plan) :
    plan.comparison = comparison ∧ plan.negate = negate ∧
      (if plan.swap then (first = y ∧ second = x) else (first = x ∧ second = y)) := by
  by_cases hxy : first = x ∧ second = y
  · rcases hxy with ⟨rfl, rfl⟩
    simp [comparisonSelectionFromOperands] at hplan
    subst plan
    simp
  · by_cases hyx : first = y ∧ second = x
    · rcases hyx with ⟨rfl, rfl⟩
      simp [comparisonSelectionFromOperands, hxy] at hplan
      subst plan
      simp
    · simp [comparisonSelectionFromOperands, hxy, hyx] at hplan


theorem comparison_selection_refines_node_semantics
    (compare : FCmp → U32 → U32 → Bool) (valuation : U32 → U32)
    (comparison : FCmp) (negate : Bool) (x y first second : U32)
    (plan : ComparisonSelection)
    (hplan : comparisonSelectionFromOperands comparison negate x y (some first) (some second) = some plan) :
    comparisonSelectionBits compare plan (valuation x) (valuation y) =
      (if compare comparison (valuation x) (valuation y) != negate then valuation first else valuation second) := by
  obtain ⟨hcomparison, hnegate, hsources⟩ :=
    comparison_selection_sources_complete comparison negate x y first second plan hplan
  cases hswap : plan.swap <;> simp only [hswap, Bool.false_eq_true, ite_false, ite_true] at hsources
  · rcases hsources with ⟨rfl, rfl⟩
    simp [comparisonSelectionBits, hcomparison, hnegate, hswap]
  · rcases hsources with ⟨rfl, rfl⟩
    simp [comparisonSelectionBits, hcomparison, hnegate, hswap]

end Instruction
