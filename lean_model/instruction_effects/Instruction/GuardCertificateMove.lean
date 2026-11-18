/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateOperands

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial

set_option autoImplicit false
set_option maxHeartbeats 5000000

namespace Instruction.GuardCertificate

def moveWord (destination : Reg) (value : U64) : Nat :=
  if destination.file = .V then value.val else value.val % 2 ^ 32

theorem source_move_closure {C : Type} (reader : guard_certificate.Constants C)
    (destination : Reg) (value : U64) :
    guard_certificate.move_assignment.closure.Insts.CoreOpsFunctionFnOnceTupleU64Fact.call_once
      reader destination value =
      .ok ({ register := destination, value := (if destination.file = .V then value else value &&& 4294967295#u64) } : Fact) := by
  cases file : destination.file <;>
    simp [guard_certificate.move_assignment.closure.Insts.CoreOpsFunctionFnOnceTupleU64Fact.call_once,
      File.Insts.CoreCmpPartialEqFile.eq, file, File.read_discriminant, lift]

theorem source_move_assignment_sound {C : Type} (reader : guard_certificate.Constants C)
    (destination : Reg) (operand : Val) (facts : Slice Fact) (constants : C)
    (registers : Reg → U64) (bank : U32 → U8 → Option U64)
    (sound : FactsSound facts.val registers) (constantSound : ConstantsSound reader constants bank)
    (fact : Fact)
    (returned : guard_certificate.move_assignment reader destination operand facts constants =
      .ok (some fact)) :
    (destination.file = .V ∨ destination.file = .R ∧ destination.idx.val < 255) ∧
    scalarMove destination operand = true ∧
    ∃ value, operandValue registers bank operand = some value ∧
      fact.register = destination ∧ fact.value.val = moveWord destination value := by
  have reduced (file : File) (fileAt : destination.file = file)
      (scalar : guard_certificate.scalar_move destination operand = .ok true)
      (tail : (do
        let observed ← guard_certificate.value reader facts constants operand
        core.option.Option.map
          (guard_certificate.move_assignment.closure.Insts.CoreOpsFunctionFnOnceTupleU64Fact reader)
          observed destination) = .ok (some fact)) :
      scalarMove destination operand = true ∧
      ∃ value, operandValue registers bank operand = some value ∧
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
          source_value_sound reader facts constants registers bank sound constantSound
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

end Instruction.GuardCertificate
