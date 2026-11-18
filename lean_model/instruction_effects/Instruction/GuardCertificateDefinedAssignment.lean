/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateDefinedFacts

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.GuardCertificate

theorem source_move_assignment_defined {C : Type} (reader : guard_certificate.Constants C)
    (destination : Reg) (operand : Val) (facts : Slice Fact) (constants : C)
    (registers : PartialRegisters) (bank : U32 → U8 → Option U64)
    (sound : FactsSoundDefined facts.val registers) (constantSound : ConstantsSound reader constants bank)
    (fact : Fact)
    (returned : guard_certificate.move_assignment reader destination operand facts constants =
      .ok (some fact)) :
    (destination.file = .V ∨ destination.file = .R ∧ destination.idx.val < 255) ∧
    scalarMove destination operand = true ∧
    ∃ value, partialOperandValue registers bank operand = some value ∧
      fact.register = destination ∧ fact.value.val = moveWord destination value := by
  have reduced (file : File) (fileAt : destination.file = file)
      (scalar : guard_certificate.scalar_move destination operand = .ok true)
      (tail : (do
        let observed ← guard_certificate.value reader facts constants operand
        core.option.Option.map
          (guard_certificate.move_assignment.closure.Insts.CoreOpsFunctionFnOnceTupleU64Fact reader)
          observed destination) = .ok (some fact)) :
      scalarMove destination operand = true ∧
      ∃ value, partialOperandValue registers bank operand = some value ∧
        fact.register = destination ∧ fact.value.val = moveWord destination value := by
    have scalarAt := source_scalar_move destination operand
    rw [scalar] at scalarAt
    have eligible := Result.ok_injective scalarAt
    obtain ⟨observed, observedAt, mapped⟩ := bind_success _ _ _ tail
    cases observed with
    | none => simp [core.option.Option.map] at mapped
    | some value =>
        simp only [core.option.Option.map] at mapped
        rw [source_move_closure] at mapped
        obtain ⟨produced, producedAt, wrapped⟩ := bind_success _ _ _ mapped
        have producedEq := Result.ok_injective producedAt
        have factAt := Option.some.inj (Result.ok_injective wrapped)
        rw [←producedEq] at factAt
        refine ⟨eligible.symm, value,
          source_value_defined reader facts constants registers bank sound constantSound
            operand value observedAt, ?_, ?_⟩
        · simpa only [←factAt]
        · rw [←factAt]
          simp only [moveWord]
          split
          · rfl
          · simp only [UScalar.val_and]
            exact Nat.and_two_pow_sub_one_eq_mod value.val 32
  cases file : destination.file <;>
    simp [guard_certificate.move_assignment, file,
      File.Insts.CoreCmpPartialEqFile.eq, File.read_discriminant] at returned
  case V =>
    have same : ({ file := .V, idx := destination.idx } : Reg) = destination := by
      cases destination; simp_all
    rw [same] at returned
    obtain ⟨scalar, scalarAt, tail⟩ := bind_success _ _ _ returned
    cases scalar with
    | false => simp at tail
    | true =>
      simp only [ite_true] at tail
      have result := reduced .V file scalarAt tail
      exact ⟨Or.inl rfl, result⟩
  case R =>
    split at returned
    · simp at returned
    · rename_i writable
      have same : ({ file := .R, idx := destination.idx } : Reg) = destination := by
        cases destination; simp_all
      rw [same] at returned
      obtain ⟨scalar, scalarAt, tail⟩ := bind_success _ _ _ returned
      cases scalar with
      | false => simp at tail
      | true =>
        simp only [ite_true] at tail
        have result := reduced .R file scalarAt tail
        exact ⟨Or.inr ⟨rfl, by scalar_tac⟩, result⟩

def ComparisonDefined (plan : guard_certificate.Comparison) (operands : Val × Val)
    (registers : PartialRegisters) (bank : U32 → U8 → Option U64) (fact : Fact) : Prop :=
  ∃ x y c,
    partialOperandValue registers bank operands.1 = some x ∧
    partialOperandValue registers bank operands.2 = some y ∧
    partialPredicate plan.front registers plan.and = some c ∧
    0 < plan.bits.val ∧ plan.bits.val ≤ 64 ∧
    fact.register = plan.destination ∧
    fact.value = if combined plan.or (compareWords plan.comparison plan.bits.val x y) c
      then 1#u64 else 0#u64

theorem source_comparison_body_defined {C : Type} (reader : guard_certificate.Constants C)
    (plan : guard_certificate.Comparison) (operands : Val × Val)
    (facts : Slice Fact) (constants : C) (registers : PartialRegisters)
    (bank : U32 → U8 → Option U64) (sound : FactsSoundDefined facts.val registers)
    (constantSound : ConstantsSound reader constants bank) (fact : Fact)
    (returned : comparisonBody reader plan operands facts constants = .ok (some fact)) :
    ComparisonDefined plan operands registers bank fact := by
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
            source_value_defined reader facts constants registers bank sound constantSound operands.1 x xAt,
            source_value_defined reader facts constants registers bank sound constantSound operands.2 y yAt,
            source_decision_defined plan.front facts registers sound plan.and c cAt,
            comparedSound.1, comparedSound.2.1, ?_, ?_⟩
          · simpa only [←factAt]
          · simp only [←factAt, comparedSound.2.2]

theorem source_comparison_assignment_defined {C : Type} (reader : guard_certificate.Constants C)
    (plan : guard_certificate.Comparison) (operands : Val × Val)
    (facts : Slice Fact) (constants : C) (registers : PartialRegisters)
    (bank : U32 → U8 → Option U64) (sound : FactsSoundDefined facts.val registers)
    (constantSound : ConstantsSound reader constants bank) (fact : Fact)
    (returned : guard_certificate.comparison_assignment reader plan operands facts constants = .ok (some fact)) :
    plan.destination.file = .P ∧
    ¬(plan.front = .Sass ∧ plan.destination.idx = 7#u16) ∧
    ComparisonDefined plan operands registers bank fact := by
  cases operands with
  | mk left right =>
    cases file : plan.destination.file <;>
      simp [guard_certificate.comparison_assignment, file,
        core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default, File.Insts.CoreCmpPartialEqFile.eq,
        File.read_discriminant] at returned
    cases front : plan.front <;>
      simp [front, Front.Insts.CoreCmpPartialEqFront.eq, Front.read_discriminant] at returned
    case Ptx =>
      exact ⟨rfl, by simp, source_comparison_body_defined reader plan (left, right)
        facts constants registers bank sound constantSound fact (by
          unfold comparisonBody
          rw [front]
          convert returned using 1 <;> rfl)⟩
    case Sass =>
      split at returned
      · simp at returned
      · rename_i notConstant
        exact ⟨rfl, by simpa using notConstant, source_comparison_body_defined reader plan (left, right)
          facts constants registers bank sound constantSound fact (by
          unfold comparisonBody
          rw [front]
          convert returned using 1 <;> rfl)⟩

def AssignmentDefined (front : Front) (operation : Op) (registers : PartialRegisters)
    (bank : U32 → U8 → Option U64) (fact : Fact) : Prop :=
  match operation with
  | .Mov destination operand =>
    (destination.file = .V ∨ destination.file = .R ∧ destination.idx.val < 255) ∧
    scalarMove destination operand = true ∧
    ∃ value, partialOperandValue registers bank operand = some value ∧
      fact.register = destination ∧ fact.value.val = moveWord destination value
  | .Setp destination comparison bits left right and or false none =>
    destination.file = .P ∧ ¬(front = .Sass ∧ destination.idx = 7#u16) ∧
    ComparisonDefined {front, destination, comparison, bits, and, or} (left, right) registers bank fact
  | _ => False

theorem source_assignment_defined {C : Type} (reader : guard_certificate.Constants C)
    (front : Front) (instruction : Inst) (facts : Slice Fact) (constants : C)
    (registers : PartialRegisters) (bank : U32 → U8 → Option U64)
    (sound : FactsSoundDefined facts.val registers) (constantSound : ConstantsSound reader constants bank)
    (fact : Fact)
    (returned : guard_certificate.assignment reader front instruction facts constants = .ok (some fact)) :
    AssignmentDefined front instruction.op registers bank fact := by
  cases instruction with
  | mk guard operation uniform loc text =>
    cases operation <;> try (solve | simp [guard_certificate.assignment] at returned)
    case Mov destination operand =>
      exact source_move_assignment_defined reader destination operand facts constants registers bank
        sound constantSound fact returned
    case Setp destination comparison bits left right and or xor extra =>
      cases extra with
      | some _ => simp [guard_certificate.assignment] at returned
      | none =>
        cases xor with
        | true => simp [guard_certificate.assignment] at returned
        | false =>
          exact source_comparison_assignment_defined reader
            {front, destination, comparison, bits, and, or} (left, right)
            facts constants registers bank sound constantSound fact returned


end Instruction.GuardCertificate
