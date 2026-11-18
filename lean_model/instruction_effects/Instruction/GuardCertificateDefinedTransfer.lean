/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateDefinedAssignment
import Instruction.GuardCertificateTransfer

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.GuardCertificate

structure OpPostDefined (front : Front) (operation : Op) (bank : U32 → U8 → Option U64)
    (before after : PartialRegisters) : Prop where
  frame : ∀ writes, frameWrites operation = some writes →
    ∀ register, RegisterOwner.shared register.file = false → key register ∉ writes →
      after register = before register
  assignment : ∀ fact, AssignmentDefined front operation before bank fact →
    after fact.register = some fact.value

def StepPostDefined (front : Front) (instruction : Inst) (bank : U32 → U8 → Option U64)
    (before after : PartialRegisters) : Prop :=
  match partialPredicate front before instruction.guard with
  | some true => OpPostDefined front instruction.op bank before after
  | some false => ∀ register, RegisterOwner.shared register.file = false → after register = before register
  | none => ∀ writes, frameWrites instruction.op = some writes →
      ∀ register, RegisterOwner.shared register.file = false → key register ∉ writes →
        after register = before register

theorem step_defined_preserves_unwritten (front : Front) (instruction : Inst)
    (bank : U32 → U8 → Option U64) (before after : PartialRegisters)
    (post : StepPostDefined front instruction bank before after)
    (writes : List Key) (frameAt : frameWrites instruction.op = some writes)
    (register : Reg) (isPrivate : RegisterOwner.shared register.file = false)
    (unwritten : key register ∉ writes) : after register = before register := by
  unfold StepPostDefined at post
  cases guardAt : partialPredicate front before instruction.guard with
  | none =>
    simp only [guardAt] at post
    exact post writes frameAt register isPrivate unwritten
  | some truth =>
    cases truth with
    | false =>
      simp only [guardAt] at post
      exact post register isPrivate
    | true =>
      simp only [guardAt] at post
      exact post.frame writes frameAt register isPrivate unwritten

theorem postAfter_defined {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C) (register : Reg) (old : Option U64)
    (certain : Bool) (before after : PartialRegisters) (bank : U32 → U8 → Option U64)
    (fits : FrameFits context.instruction.op) (isPrivate : RegisterOwner.shared register.file = false)
    (sound : FactsSoundDefined context.facts.val before)
    (constantSound : ConstantsSound reader context.constants bank)
    (oldSound : ∀ value, old = some value → before register = some value)
    (stepPost : StepPostDefined context.front context.instruction bank before after)
    (certainSound : certain = true → partialPredicate context.front before context.instruction.guard = some true)
    (value : U64) (returned : postAfter reader context register old certain = .ok (some value)) :
    after register = some value := by
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
      have expected : frameWrites context.instruction.op = some (writes.val.map key) := by
        simpa [registerObservation] using frameMeaning.symm
      exact (step_defined_preserves_unwritten context.front context.instruction bank before after
        stepPost _ expected register isPrivate notKey).trans oldTruth
    | true =>
      simp only [ite_true] at tail
      obtain ⟨assigned, assignedAt, tail⟩ := bind_success _ _ _ tail
      obtain ⟨selected, selectedAt, finished⟩ := bind_success _ _ _ tail
      have certainAt : certain = true := by
        cases certain <;> simp_all
      simp only [certainAt, ite_true] at finished
      have selectedEq : selected = some value := Result.ok_injective finished
      have truth := certainSound certainAt
      simp only [StepPostDefined, truth] at stepPost
      cases assigned with
      | none => simp at selectedAt; simp_all
      | some fact =>
        dsimp only at selectedAt
        rw [source_same_register] at selectedAt
        simp only [bind_ok] at selectedAt
        split at selectedAt
        · rename_i same
          have sameRegister : fact.register = register := by simpa using same
          have sourceSelected := Result.ok_injective selectedAt
          have valueEq : fact.value = value := by simpa [←sourceSelected] using selectedEq
          have assignment := source_assignment_defined reader context.front context.instruction context.facts
            context.constants before bank sound constantSound fact assignedAt
          simpa only [sameRegister, valueEq] using stepPost.assignment fact assignment
        · have noneEq := Result.ok_injective selectedAt
          simp_all

theorem source_post_value_defined {C : Type} (reader : guard_certificate.Constants C)
    (context : guard_certificate.Evaluation C) (register : Reg)
    (before after : PartialRegisters) (bank : U32 → U8 → Option U64)
    (fits : FrameFits context.instruction.op) (isPrivate : RegisterOwner.shared register.file = false)
    (sound : FactsSoundDefined context.facts.val before)
    (constantSound : ConstantsSound reader context.constants bank)
    (stepPost : StepPostDefined context.front context.instruction bank before after)
    (value : U64)
    (returned : guard_certificate.post_value reader context register = .ok (some value)) :
    after register = some value := by
  unfold guard_certificate.post_value at returned
  obtain ⟨old, oldAt, tail⟩ := bind_success _ _ _ returned
  obtain ⟨known, knownAt, tail⟩ := bind_success _ _ _ tail
  have oldSound (found : U64) (present : old = some found) : before register = some found := by
    exact source_lookup_defined context.facts before sound register found (by simpa only [present] using oldAt)
  cases known with
  | none =>
    apply postAfter_defined reader context register old false before after bank fits isPrivate sound
      constantSound oldSound stepPost (by simp) value
    simp only [bind_ok, Bool.false_eq_true, ite_false] at tail
    unfold postAfter
    convert tail using 1
    congr 1
    funext frame
    cases frame <;>
      simp [core.option.Option.Insts.CoreOpsTry_traitTry.branch,
        core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionNever.from_residual]
    all_goals congr 1
  | some known =>
    cases known with
    | false =>
      simp only [bind_ok, Bool.false_eq_true, ite_false, ite_true] at tail
      have oldEq := Result.ok_injective tail
      have guardEq := source_decision_defined context.front context.facts before sound
        context.instruction.guard false knownAt
      simp only [StepPostDefined, guardEq] at stepPost
      exact (stepPost register isPrivate).trans (oldSound value oldEq)
    | true =>
      refine postAfter_defined reader context register old true before after bank fits isPrivate sound
        constantSound oldSound stepPost (fun _ => source_decision_defined context.front context.facts before sound
          context.instruction.guard true knownAt) value ?_
      simp only [bind_ok, Bool.false_eq_true, ite_false, ite_true] at tail
      unfold postAfter
      convert tail using 1
      congr 1
      funext frame
      cases frame <;>
        simp [core.option.Option.Insts.CoreOpsTry_traitTry.branch,
          core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionNever.from_residual]
      all_goals congr 1

end Instruction.GuardCertificate
