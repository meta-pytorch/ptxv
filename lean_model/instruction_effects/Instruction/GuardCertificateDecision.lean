/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateLookup

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial

set_option autoImplicit false
set_option maxHeartbeats 5000000

namespace Instruction.GuardCertificate

open RegisterOwner Instruction.PredicateRead Instruction.OwnerTemporal

def knowledge (facts : List Fact) (register : Reg) : Option Bool :=
  (lookup facts register).map fun value => value != 0#u64

def predicateEligible (front : Front) (predicate : Pred) : Bool :=
  match (normalizedPredicate front predicate).reg with
  | none => true
  | some register => match register.file with | .P | .UP => true | _ => false

def decision (front : Front) (facts : List Fact) (predicate : Pred) : Option Bool :=
  if predicateEligible front predicate then
    meaning front predicate (privateKnowledge (knowledge facts))
  else none

@[step]
theorem rust_predicates_read (state : guard_certificate.Predicates) (register : Reg) :
    guard_certificate.Predicates.Insts.Ptxv_irPredicate_readKnownPredicate.read state register
      ⦃ result => result.1 = knowledge state.facts.val register ∧ result.2 = state ⦄ := by
  unfold guard_certificate.Predicates.Insts.Ptxv_irPredicate_readKnownPredicate.read
  step with rust_lookup state.facts register as ⟨found, foundEq⟩
  cases found <;>
    simp [core.option.Option.map,
      guard_certificate.KnownPredicatePredicates.read.closure.Insts.CoreOpsFunctionFnOnceTupleU64Bool,
      guard_certificate.KnownPredicatePredicates.read.closure.Insts.CoreOpsFunctionFnOnceTupleU64Bool.call_once,
      knowledge, ←foundEq]

@[step]
theorem rust_decision (front : Front) (facts : Slice Fact) (predicate : Pred) :
    guard_certificate.decision front facts predicate
      ⦃ result => result = decision front facts.val predicate ⦄ := by
  have evaluated := Instruction.OwnerTemporal.rust_predicate
    guard_certificate.Predicates.Insts.Ptxv_irPredicate_readKnownPredicate
    ({ facts } : guard_certificate.Predicates) front (normalizedPredicate front predicate)
    (knowledge facts.val) (fun _ => True) trivial
    (by
      intro register _ _
      apply WP.spec_mono (rust_predicates_read { facts } register)
      intro result property
      exact ⟨property.1, trivial⟩)
  unfold guard_certificate.decision
  rw [source_normalize_predicate]
  simp only [bind_ok]
  cases present : (normalizedPredicate front predicate).reg with
  | none =>
      step with evaluated as ⟨result, state, resultEq, _⟩
      simpa only [decision, predicateEligible, present, ite_true, meaning_normalized] using resultEq
  | some register =>
      cases file : register.file
      case V => simp [decision, predicateEligible, present, file]
      case R => simp [decision, predicateEligible, present, file]
      case UR => simp [decision, predicateEligible, present, file]
      case P | UP =>
        simp only [file, bind_ok, ite_true]
        step with evaluated as ⟨result, state, resultEq, _⟩
        simpa only [decision, predicateEligible, present, file, ite_true, meaning_normalized] using resultEq

theorem lookup_meaning_sound (facts : List Fact) (registers : Reg → U64)
    (sound : FactsSound facts registers) (register : Reg) (value : U64)
    (found : lookup facts register = some value) : registers register = value := by
  induction facts with
  | nil => simp at found
  | cons first rest ih =>
      rw [lookup_cons] at found
      split at found
      · rename_i same
        have valueEq := Option.some.inj found
        simpa only [same, valueEq] using sound first (by simp)
      · exact ih (fun fact member => sound fact (by simp [member])) found

theorem knowledge_sound (facts : List Fact) (registers : Reg → U64)
    (sound : FactsSound facts registers) :
    KnowledgeSound (knowledge facts) (fun register => registers register != 0#u64) := by
  intro register truth present
  unfold knowledge at present
  cases found : lookup facts register with
  | none => simp [found] at present
  | some value =>
      have valueEq := lookup_meaning_sound facts registers sound register value found
      simpa [found, valueEq] using present.symm

theorem decision_sound (front : Front) (facts : List Fact) (registers : Reg → U64)
    (sound : FactsSound facts registers) (predicate : Pred) (truth : Bool)
    (observed : decision front facts predicate = some truth) :
    truth = predicateRead front (fun register => registers register != 0#u64) predicate := by
  unfold decision at observed
  split at observed
  · exact meaning_conservative front predicate (privateKnowledge (knowledge facts)) _
      (private_knowledge_survives_shared_interference (knowledge facts) _ _
        (knowledge_sound facts registers sound) (fun register _ => rfl)) truth observed
  · cases observed

theorem source_decision_sound (front : Front) (facts : Slice Fact) (registers : Reg → U64)
    (sound : FactsSound facts.val registers) (predicate : Pred) (truth : Bool)
    (returned : guard_certificate.decision front facts predicate = .ok (some truth)) :
    truth = predicateRead front (fun register => registers register != 0#u64) predicate := by
  have exactValue := rust_decision front facts predicate
  simp only [returned, WP.spec_ok] at exactValue
  exact decision_sound front facts.val registers sound predicate truth exactValue.symm

end Instruction.GuardCertificate
