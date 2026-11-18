/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateWords
import Instruction.GuardCertificateOperands

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial
set_option autoImplicit false
set_option maxHeartbeats 5000000
namespace Instruction.GuardCertificate

def combined (or comparison predicate : Bool) : Bool :=
  if or then comparison || predicate else comparison && predicate

theorem source_comparison_closure {C : Type} (reader : guard_certificate.Constants C)
    (destination : Reg) (or predicate comparison : Bool) :
    guard_certificate.comparison_assignment.closure.Insts.CoreOpsFunctionFnOnceTupleBoolFact.call_once
      reader (destination, or, predicate) comparison =
      .ok ({register := destination, value := if combined or comparison predicate then 1#u64 else 0#u64} : Fact) := by
  cases or <;> cases predicate <;> cases comparison <;>
    simp [guard_certificate.comparison_assignment.closure.Insts.CoreOpsFunctionFnOnceTupleBoolFact.call_once,
      combined, lift, UScalar.cast_fromBool] <;> rfl

def ComparisonTruth (plan : guard_certificate.Comparison) (operands : Val × Val)
    (registers : Reg → U64) (bank : U32 → U8 → Option U64) (fact : Fact) : Prop :=
  ∃ x y c,
    operandValue registers bank operands.1 = some x ∧
    operandValue registers bank operands.2 = some y ∧
    c = RegisterOwner.predicateRead plan.front (fun register => registers register != 0#u64) plan.and ∧
    0 < plan.bits.val ∧ plan.bits.val ≤ 64 ∧
    fact.register = plan.destination ∧
    fact.value = if combined plan.or (compareWords plan.comparison plan.bits.val x y) c
      then 1#u64 else 0#u64

def comparisonBody {C : Type} (reader : guard_certificate.Constants C)
    (plan : guard_certificate.Comparison) (operands : Val × Val)
    (facts : Slice Fact) (constants : C) : Result (Option Fact) := do
  let xo ← guard_certificate.value reader facts constants operands.1
  let yo ← guard_certificate.value reader facts constants operands.2
  let co ← guard_certificate.decision plan.front facts plan.and
  match xo with
  | none => .ok none
  | some x => match yo with
    | none => .ok none
    | some y => match co with
      | none => .ok none
      | some c =>
        let compared ← guard_certificate.compare plan.comparison plan.bits x y
        core.option.Option.map
          (guard_certificate.comparison_assignment.closure.Insts.CoreOpsFunctionFnOnceTupleBoolFact reader)
          compared (plan.destination, plan.or, c)

theorem source_comparison_body_sound {C : Type} (reader : guard_certificate.Constants C)
    (plan : guard_certificate.Comparison) (operands : Val × Val)
    (facts : Slice Fact) (constants : C) (registers : Reg → U64)
    (bank : U32 → U8 → Option U64) (sound : FactsSound facts.val registers)
    (constantSound : ConstantsSound reader constants bank) (fact : Fact)
    (returned : comparisonBody reader plan operands facts constants = .ok (some fact)) :
    ComparisonTruth plan operands registers bank fact := by
  unfold comparisonBody at returned
  obtain ⟨xo, xAt, tail⟩ := bind_success _ _ _ returned
  obtain ⟨yo, yAt, tail⟩ := bind_success _ _ _ tail
  obtain ⟨co, cAt, tail⟩ := bind_success _ _ _ tail
  cases xo with
  | none => simp at tail
  | some x =>
    cases yo with
    | none => simp at tail
    | some y =>
      cases co with
      | none => simp at tail
      | some c =>
        obtain ⟨compared, compareAt, mapped⟩ := bind_success _ _ _ tail
        cases compared with
        | none => simp [core.option.Option.map] at mapped
        | some comparison =>
          simp only [core.option.Option.map] at mapped
          rw [source_comparison_closure] at mapped
          obtain ⟨produced, producedAt, wrapped⟩ := bind_success _ _ _ mapped
          have producedEq := Result.ok_injective producedAt
          have factAt := Option.some.inj (Result.ok_injective wrapped)
          rw [←producedEq] at factAt
          have comparedSound := successful_compare plan.comparison plan.bits x y comparison compareAt
          refine ⟨x, y, c,
            source_value_sound reader facts constants registers bank sound constantSound operands.1 x xAt,
            source_value_sound reader facts constants registers bank sound constantSound operands.2 y yAt,
            source_decision_sound plan.front facts registers sound plan.and c cAt,
            comparedSound.1, comparedSound.2.1, ?_, ?_⟩
          · simpa only [←factAt]
          · simp only [←factAt, comparedSound.2.2]

theorem source_comparison_assignment_sound {C : Type} (reader : guard_certificate.Constants C)
    (plan : guard_certificate.Comparison) (operands : Val × Val)
    (facts : Slice Fact) (constants : C) (registers : Reg → U64)
    (bank : U32 → U8 → Option U64) (sound : FactsSound facts.val registers)
    (constantSound : ConstantsSound reader constants bank) (fact : Fact)
    (returned : guard_certificate.comparison_assignment reader plan operands facts constants = .ok (some fact)) :
    plan.destination.file = .P ∧
    ¬(plan.front = .Sass ∧ plan.destination.idx = 7#u16) ∧
    ComparisonTruth plan operands registers bank fact := by
  cases operands with
  | mk left right =>
    cases file : plan.destination.file <;>
      simp [guard_certificate.comparison_assignment, file,
        core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default, File.Insts.CoreCmpPartialEqFile.eq,
        File.read_discriminant] at returned
    cases front : plan.front <;>
      simp [front, Front.Insts.CoreCmpPartialEqFront.eq, Front.read_discriminant] at returned
    case Ptx =>
      exact ⟨rfl, by simp, source_comparison_body_sound reader plan (left, right)
        facts constants registers bank sound constantSound fact (by
          unfold comparisonBody
          rw [front]
          convert returned using 1 <;> rfl)⟩
    case Sass =>
      split at returned
      · simp at returned
      · rename_i notConstant
        exact ⟨rfl, by simpa using notConstant, source_comparison_body_sound reader plan (left, right)
          facts constants registers bank sound constantSound fact (by
          unfold comparisonBody
          rw [front]
          convert returned using 1 <;> rfl)⟩

end Instruction.GuardCertificate
