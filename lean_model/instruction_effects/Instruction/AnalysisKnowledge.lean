/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerTemporal

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.OwnerTemporal

open RegisterOwner PredicateRead

def valuePrivate : Val → Bool
  | .Reg register _ => !shared register.file
  | _ => true

theorem source_value_thread_private (value : Val) :
    analysis_knowledge.value_is_thread_private value = .ok (valuePrivate value) := by
  cases value <;> simp [analysis_knowledge.value_is_thread_private, valuePrivate, source_thread_private]

def normalizedShared (front : Front) (value : Pred) : Bool :=
  match (normalizedPredicate front value).reg with
  | none => false
  | some register => shared register.file

theorem source_predicate_dispatch {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (value : Pred) :
    analysis_knowledge.predicate reader state front value =
      if normalizedShared front value then .ok (none, state)
      else predicate_read.evaluate reader state front (normalizedPredicate front value) := by
  unfold analysis_knowledge.predicate
  rw [source_normalize_predicate]
  simp only [bind_ok]
  cases present : (normalizedPredicate front value).reg with
  | none => simp [normalizedShared, present]
  | some register =>
      simp only [source_thread_private, bind_ok]
      cases isShared : shared register.file <;> simp [normalizedShared, present, isShared]

theorem source_shared_predicate_never_reads {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (value : Pred) (register : Reg)
    (present : (normalizedPredicate front value).reg = some register)
    (isShared : shared register.file = true) :
    analysis_knowledge.predicate reader state front value = .ok (none, state) := by
  simp [source_predicate_dispatch, normalizedShared, present, isShared]

theorem source_constant_predicate_never_reads {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (value : Pred)
    (constant : (normalizedPredicate front value).reg = none) :
    analysis_knowledge.predicate reader state front value = .ok (some (!value.neg), state) := by
  rw [source_predicate_dispatch]
  simp only [normalizedShared, constant, Bool.false_eq_true, ite_false]
  rw [source_constant_no_read reader state front (normalizedPredicate front value)
    (by simpa only [normalized_predicate_idempotent] using constant)]
  rw [normalized_neg]

@[step]
theorem rust_predicate {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (value : Pred) (known : Reg → Option Bool)
    (frame : R → Prop) (unchanged : frame state)
    (reads : ∀ register, (normalizedPredicate front value).reg = some register →
      shared register.file = false → reader.read state register ⦃ result =>
        result.1 = known register ∧ frame result.2 ⦄) :
    analysis_knowledge.predicate reader state front value ⦃ result =>
      result.1 = meaning front value (privateKnowledge known) ∧ frame result.2 ⦄ := by
  cases present : (normalizedPredicate front value).reg with
  | none =>
      rw [source_constant_predicate_never_reads reader state front value present, WP.spec_ok]
      refine ⟨?_, unchanged⟩
      rw [←meaning_normalized front value (privateKnowledge known)]
      unfold meaning
      rw [present, normalized_neg]
      cases value.neg <;> rfl
  | some register =>
      cases isShared : shared register.file with
      | true =>
          rw [source_shared_predicate_never_reads reader state front value register present isShared,
            WP.spec_ok]
          refine ⟨?_, unchanged⟩
          rw [←meaning_normalized front value (privateKnowledge known)]
          unfold meaning
          rw [present]
          have mutable := normalized_register_is_mutable front value register present
          simp [mutable, privateKnowledge, isShared]
      | false =>
          rw [source_predicate_dispatch]
          simp only [normalizedShared, present, isShared, Bool.false_eq_true, ite_false]
          apply WP.spec_mono (rust_evaluate reader state front (normalizedPredicate front value)
            (privateKnowledge known) frame unchanged ?_)
          · intro found post
            simpa only [meaning_normalized] using post
          · intro other found
            have original : (normalizedPredicate front value).reg = some other := by
              simpa only [normalized_predicate_idempotent] using found
            have same : other = register := by simpa [present] using original.symm
            subst other
            simpa [privateKnowledge, isShared] using reads register present isShared

@[step]
theorem rust_predicate_after_shared_interference {R : Type}
    (reader : predicate_read.KnownPredicate R) (state : R) (front : Front) (value : Pred)
    (known : Reg → Option Bool) (before after : Reg → Bool)
    (sound : KnowledgeSound known before) (peers : PrivateAgree before after)
    (frame : R → Prop) (unchanged : frame state)
    (reads : ∀ register, (normalizedPredicate front value).reg = some register →
      shared register.file = false → reader.read state register ⦃ result =>
        result.1 = known register ∧ frame result.2 ⦄) :
    analysis_knowledge.predicate reader state front value ⦃ result =>
      (∀ truth, result.1 = some truth → truth = predicateRead front after value) ∧ frame result.2 ⦄ := by
  apply WP.spec_mono (rust_predicate reader state front value known frame unchanged reads)
  rintro result ⟨meaningEq, framed⟩
  refine ⟨?_, framed⟩
  intro truth present
  exact meaning_conservative front value (privateKnowledge known) after
    (private_knowledge_survives_shared_interference known before after sound peers) truth
    (meaningEq.symm.trans present)

end Instruction.OwnerTemporal
