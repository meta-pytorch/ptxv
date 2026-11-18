/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerComparison
import MemoryWrite.MetadataPlanRefinement

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false

namespace RegisterOwner.Comparison

/-- List semantics of the native comparison-cache retention predicate. -/
def sourceRetain {Claim : Type} (thread : Usize) (owners : Slice U32)
    (register : Reg) : List (Entry Claim) → Result (List (Entry Claim))
  | [] => .ok []
  | entry :: rest => do
    let aliases ← register_owner.record_aliases thread owners register entry.register
    let retained ← sourceRetain thread owners register rest
    pure (if aliases then retained else entry :: retained)

@[step] theorem source_retain_exact {Claim : Type} (thread : Usize) (owners : Slice U32)
    (register : Reg) (entries : List (Entry Claim)) :
    sourceRetain thread owners register entries ⦃ result =>
      result = invalidate (owners.val.map UScalar.val)
        (canonicalKey thread.val (owners.val.map UScalar.val) register) entries ⦄ := by
  induction entries with
  | nil => simp [sourceRetain, invalidate, WP.spec_ok]
  | cons entry rest ih =>
    simp only [sourceRetain]
    step with rust_record_aliases thread owners register entry.register as ⟨aliases, exact_alias⟩
    step with ih as ⟨retained, exact_retained⟩
    rw [exact_alias, exact_retained]
    simp only [invalidate, List.filter_cons]
    by_cases aliases : recordKey (owners.val.map UScalar.val) entry.register =
        canonicalKey thread.val (owners.val.map UScalar.val) register
    · simp [aliases]
    · simp [aliases, Ne.symm aliases]

/-- Compose the extracted metadata decision with the extracted physical-alias test. -/
def sourceGuardedRetention {Claim : Type} (thread : Usize) (owners : Slice U32)
    (register : Reg) (active dependent hookWrote sourceTainted : Bool)
    (entries : List (Entry Claim)) : Result (List (Entry Claim)) := do
  let plan ← PtxvMemoryWrite.symbolic.destination_plan active dependent hookWrote sourceTainted
  if active || plan.forget_half then sourceRetain thread owners register entries
  else pure entries

theorem source_guarded_retention_preserves_all_input_evidence {Input Value Claim : Type}
    (thread : Usize) (owners : Slice U32) (register : Reg)
    (domain : Input → Prop) (capture : Input) (guard : Input → Bool)
    (dependent hookWrote sourceTainted : Bool)
    (independent : dependent = false → ∀ input, domain input → guard input = guard capture)
    (store : Store Input Value) (meaning : Claim → Input → Value)
    (entries : List (Entry Claim))
    (sound : Sound domain (owners.val.map UScalar.val) store meaning entries)
    (replacement : Input → Value) :
    sourceGuardedRetention thread owners register (guard capture) dependent hookWrote sourceTainted entries
      ⦃ result => Sound domain (owners.val.map UScalar.val)
        (guardedWrite store (canonicalKey thread.val (owners.val.map UScalar.val) register)
          guard replacement) meaning result ⦄ := by
  simp only [sourceGuardedRetention]
  step with MemoryWrite.Symbolic.destination_plan_spec
    (guard capture) dependent hookWrote sourceTainted as ⟨plan, exact_plan⟩
  have forget_exact : plan.forget_half = dependent := by
    have field := congrArg MemoryWrite.Symbolic.DestinationPlan.forgetHalf exact_plan
    simpa [MemoryWrite.Symbolic.destinationPlanView, MemoryWrite.Symbolic.destinationPlan] using field
  rw [forget_exact]
  split
  · step with source_retain_exact thread owners register entries as ⟨result, exact_result⟩
    rw [exact_result]
    exact invalidation_preserves_guarded_evidence domain (owners.val.map UScalar.val)
      store meaning entries sound _ guard replacement
  · rename_i inactive
    have guard_false : guard capture = false := by simp_all
    have dependent_false : dependent = false := by simp_all
    simp only [WP.spec_ok]
    intro entry member input valid_input
    have no_write := independent dependent_false input valid_input
    simp only [guard_false] at no_write
    by_cases same : recordKey (owners.val.map UScalar.val) entry.register =
        canonicalKey thread.val (owners.val.map UScalar.val) register
    · simpa [guardedWrite, write, no_write, ← same] using sound entry member input valid_input
    · simpa [guardedWrite, write, no_write, same] using sound entry member input valid_input

end RegisterOwner.Comparison
