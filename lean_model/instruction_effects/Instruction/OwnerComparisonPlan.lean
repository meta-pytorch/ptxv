/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerPredicate

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false

namespace RegisterOwner

def writablePredicate (front : Front) (register : Reg) : Bool :=
  match register.file with
  | .P | .UP => !immutablePredicate front register
  | _ => false

@[step]
theorem source_comparison_plan (thread : Usize) (front : Front) (destination : Reg)
    (plain : Bool) :
    register_owner.comparison_plan thread front destination plain ⦃ plan =>
      keyView plan.key = canonicalKey thread.val [] destination ∧
      plan.remember = (plain && writablePredicate front destination) ⦄ := by
  rcases destination with ⟨file, index⟩
  cases front <;> cases file <;> cases plain <;>
    simp [register_owner.comparison_plan, source_normalize_predicate,
      normalizedPredicate, immutablePredicate, writablePredicate, keyView,
      canonicalKey, shared, fileTag, UScalar.cast_val_eq, read_discriminant,
      File.read_discriminant, lift, bind_ok, WP.spec_ok]
  all_goals split <;> simp_all

def applyComparisonPlan {Evidence : Type} (plan : register_owner.ComparisonPlan)
    (before : Key → Option Evidence) (evidence : Evidence) : Key → Option Evidence :=
  fun key => if key = keyView plan.key then
    if plan.remember then some evidence else none
  else before key

def ComparisonSound {Evidence : Type} (valid : Key → Evidence → Prop)
    (cache : Key → Option Evidence) : Prop :=
  ∀ key evidence, cache key = some evidence → valid key evidence

theorem source_ignored_predicate_write_creates_no_record {Evidence : Type}
    (thread : Usize) (front : Front) (destination : Reg) (plain : Bool)
    (ignored : writablePredicate front destination = false)
    (before : Key → Option Evidence) (evidence : Evidence) :
    register_owner.comparison_plan thread front destination plain ⦃ plan =>
      applyComparisonPlan plan before evidence (canonicalKey thread.val [] destination) = none ∧
      (∀ key, key ≠ canonicalKey thread.val [] destination →
        applyComparisonPlan plan before evidence key = before key) ⦄ := by
  step with source_comparison_plan thread front destination plain as ⟨plan, key, remember⟩
  have disabled : plan.remember = false := by simp [ignored] at remember; exact remember
  constructor
  · simp [applyComparisonPlan, key, disabled]
  · intro other different
    simp [applyComparisonPlan, key, different]

theorem source_comparison_plan_preserves_soundness {Evidence : Type}
    (thread : Usize) (front : Front) (destination : Reg) (plain : Bool)
    (valid : Key → Evidence → Prop) (before : Key → Option Evidence) (evidence : Evidence)
    (initial : ComparisonSound valid before)
    (produced : plain = true → writablePredicate front destination = true →
      valid (canonicalKey thread.val [] destination) evidence) :
    register_owner.comparison_plan thread front destination plain ⦃ plan =>
      ComparisonSound valid (applyComparisonPlan plan before evidence) ⦄ := by
  step with source_comparison_plan thread front destination plain as ⟨plan, key, remember⟩
  intro target candidate present
  by_cases same : target = keyView plan.key
  · by_cases kept : plan.remember = true
    · have enabled : plain = true ∧ writablePredicate front destination = true := by
        rw [kept] at remember
        simpa using remember.symm
      have identical : evidence = candidate := by
        simpa [applyComparisonPlan, same, kept] using present
      subst candidate
      simpa [same, key] using produced enabled.1 enabled.2
    · have disabled : plan.remember = false := Bool.eq_false_iff.mpr kept
      simp [applyComparisonPlan, same, disabled] at present
  · apply initial target candidate
    simpa [applyComparisonPlan, same] using present

end RegisterOwner
