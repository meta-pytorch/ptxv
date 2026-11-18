/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateFrame
import Instruction.GuardCertificateAssignment
import Instruction.GuardCertificateWritten

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.GuardCertificate

structure OpPost (front : Front) (operation : Op) (bank : U32 → U8 → Option U64)
    (before after : Reg → U64) : Prop where
  frame : ∀ writes, frameWrites operation = some writes →
    ∀ register, RegisterOwner.shared register.file = false → key register ∉ writes →
      after register = before register
  assignment : ∀ fact, AssignmentTruth front operation before bank fact →
    after fact.register = fact.value

def StepPost (front : Front) (instruction : Inst) (bank : U32 → U8 → Option U64)
    (before after : Reg → U64) : Prop :=
  if RegisterOwner.predicateRead front (fun register => before register != 0#u64) instruction.guard then
    OpPost front instruction.op bank before after
  else ∀ register, RegisterOwner.shared register.file = false → after register = before register

def postAfter {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C) (register : Reg) (old : Option U64)
    (certain : Bool) : Result (Option U64) := do
  let writes ← guard_certificate.frame context.instruction.op
  match writes with
  | none => .ok none
  | some writes =>
    let written ← guard_certificate.post_value_loop register writes 0#usize false
    if written then
      let assigned ← guard_certificate.assignment reader context.front context.instruction context.facts context.constants
      let value ← match assigned with
        | none => .ok none
        | some fact => do
          let same ← guard_certificate.same_register fact.register register
          if same then .ok (some fact.value) else .ok none
      if certain then .ok value else .ok none
    else .ok old

theorem assignment_result_sound {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C) (before after : Reg → U64)
    (bank : U32 → U8 → Option U64) (sound : FactsSound context.facts.val before)
    (constantSound : ConstantsSound reader context.constants bank)
    (operation : OpPost context.front context.instruction.op bank before after)
    (fact : Fact)
    (returned : guard_certificate.assignment reader context.front context.instruction context.facts context.constants = .ok (some fact)) :
    after fact.register = fact.value := by
  exact operation.assignment fact (source_assignment_truth reader context.front context.instruction
    context.facts context.constants before bank sound constantSound fact returned)


@[simp] theorem key_eq (left right : Reg) : key left = key right ↔ left = right := by
  cases left; cases right
  simp only [key, Key.mk.injEq, Reg.mk.injEq, UScalar.eq_equiv]

def finish (certain : Bool) (assigned old : Option U64) : Result (Option U64) :=
  if certain then .ok assigned else .ok none

theorem successful_finish (certain : Bool) (assigned old : Option U64) (value : U64)
    (returned : finish certain assigned old = .ok (some value)) :
    assigned = some value ∧ (certain = true ∨ old = some value) := by
  cases certain with
  | true => simpa [finish] using returned
  | false => simp [finish] at returned


theorem postAfter_sound {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C) (register : Reg) (old : Option U64)
    (certain : Bool) (before after : Reg → U64) (bank : U32 → U8 → Option U64)
    (fits : FrameFits context.instruction.op) (isPrivate : RegisterOwner.shared register.file = false)
    (sound : FactsSound context.facts.val before)
    (constantSound : ConstantsSound reader context.constants bank)
    (oldSound : ∀ value, old = some value → before register = value)
    (stepPost : StepPost context.front context.instruction bank before after)
    (certainSound : certain = true → RegisterOwner.predicateRead context.front
      (fun register => before register != 0#u64) context.instruction.guard = true)
    (value : U64) (returned : postAfter reader context register old certain = .ok (some value)) :
    after register = value := by
  unfold postAfter at returned
  obtain ⟨frame, frameAt, tail⟩ := bind_success _ _ _ returned
  have frameMeaning := source_frame context.instruction.op fits
  simp only [frameAt, WP.spec_ok] at frameMeaning
  cases frame with
  | none => simp at tail
  | some writes =>
    obtain ⟨written, writtenAt, tail⟩ := bind_success _ _ _ tail
    have writtenMeaning := rust_written register writes
    simp only [writtenAt, WP.spec_ok] at writtenMeaning
    cases written with
    | false =>
      simp only [Bool.false_eq_true, ite_false] at tail
      have oldAt := Result.ok_injective tail
      have oldTruth := oldSound value oldAt
      have notWritten : register ∉ writes.val := by simpa using writtenMeaning.symm
      have notKey : key register ∉ writes.val.map key := by simpa using notWritten
      have beforeEq : after register = before register := by
        cases truth : RegisterOwner.predicateRead context.front
          (fun register => before register != 0#u64) context.instruction.guard with
        | false =>
          simp only [StepPost, truth, Bool.false_eq_true, ite_false] at stepPost
          exact stepPost register isPrivate
        | true =>
          simp only [StepPost, truth, ite_true] at stepPost
          have expected : frameWrites context.instruction.op = some (writes.val.map key) := by
            simpa [registerObservation] using frameMeaning.symm
          exact stepPost.frame _ expected register isPrivate notKey
      exact beforeEq.trans oldTruth
    | true =>
      simp only [ite_true] at tail
      obtain ⟨assigned, assignedAt, tail⟩ := bind_success _ _ _ tail
      obtain ⟨selected, selectedAt, finished⟩ := bind_success _ _ _ tail
      have final := successful_finish certain selected old value finished
      cases assigned with
      | none =>
        have noneEq := Result.ok_injective selectedAt
        simp only [←noneEq] at final
        cases final.1
      | some fact =>
        dsimp only at selectedAt
        rw [source_same_register] at selectedAt
        simp only [bind_ok] at selectedAt
        split at selectedAt
        · rename_i same
          have sameRegister : fact.register = register := by simpa using same
          have selectedEq := Result.ok_injective selectedAt
          have valueEq : fact.value = value := by simpa [←selectedEq] using final.1
          cases truth : RegisterOwner.predicateRead context.front
            (fun register => before register != 0#u64) context.instruction.guard with
          | true =>
            simp only [StepPost, truth, ite_true] at stepPost
            simpa only [sameRegister, valueEq] using assignment_result_sound reader context before after bank
              sound constantSound stepPost fact assignedAt
          | false =>
            have oldAt : old = some value := final.2.resolve_left (by
              intro present
              have impossible := certainSound present
              simp [truth] at impossible)
            simp only [StepPost, truth, Bool.false_eq_true, ite_false] at stepPost
            exact (stepPost register isPrivate).trans (oldSound value oldAt)
        · have noneEq := Result.ok_injective selectedAt
          simp only [←noneEq] at final
          cases final.1


theorem source_post_value_sound {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C) (register : Reg)
    (before after : Reg → U64) (bank : U32 → U8 → Option U64)
    (fits : FrameFits context.instruction.op) (isPrivate : RegisterOwner.shared register.file = false)
    (sound : FactsSound context.facts.val before)
    (constantSound : ConstantsSound reader context.constants bank)
    (stepPost : StepPost context.front context.instruction bank before after)
    (value : U64)
    (returned : guard_certificate.post_value reader context register = .ok (some value)) :
    after register = value := by
  unfold guard_certificate.post_value at returned
  obtain ⟨old, oldAt, tail⟩ := bind_success _ _ _ returned
  obtain ⟨known, knownAt, tail⟩ := bind_success _ _ _ tail
  have oldSound (found : U64) (present : old = some found) : before register = found := by
    exact lookup_sound context.facts before sound register found (by simpa only [present] using oldAt)
  cases known with
  | none =>
    apply postAfter_sound reader context register old false before after bank fits isPrivate sound
      constantSound oldSound stepPost (by simp) value
    simp only [bind_ok, Bool.false_eq_true, ite_false] at tail
    unfold postAfter
    convert tail using 1
    congr 1
    funext frame
    cases frame <;>
      simp [core.option.Option.Insts.CoreOpsTry_traitTry.branch,
        core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionNever.from_residual]
    all_goals
      congr 1
  | some known =>
    cases known with
    | false =>
      simp only [bind_ok, Bool.false_eq_true, ite_false, ite_true] at tail
      have oldEq := Result.ok_injective tail
      have guardEq := source_decision_sound context.front context.facts before sound
        context.instruction.guard false knownAt
      simp only [StepPost, ←guardEq, Bool.false_eq_true, ite_false] at stepPost
      exact (stepPost register isPrivate).trans (oldSound value oldEq)
    | true =>
      refine postAfter_sound reader context register old true before after bank fits isPrivate sound
        constantSound oldSound stepPost (fun _ =>
          (source_decision_sound context.front context.facts before sound
            context.instruction.guard true knownAt).symm) value ?_
      simp only [bind_ok, Bool.false_eq_true, ite_false, ite_true] at tail
      unfold postAfter
      convert tail using 1
      congr 1
      funext frame
      cases frame <;>
        simp [core.option.Option.Insts.CoreOpsTry_traitTry.branch,
          core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionNever.from_residual]
      all_goals
        congr 1

end Instruction.GuardCertificate
