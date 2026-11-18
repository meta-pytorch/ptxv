/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateUnknownGuard

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial
set_option autoImplicit false
set_option maxHeartbeats 5000000
namespace Instruction.GuardCertificate

abbrev PartialRegisters := Reg → Option U64

def FactsSoundDefined (facts : List Fact) (registers : PartialRegisters) : Prop :=
  ∀ fact ∈ facts, registers fact.register = some fact.value

def partialPredicate (front : Front) (registers : PartialRegisters) (predicate : Pred) : Option Bool :=
  Instruction.PredicateRead.meaning front predicate
    (fun register => (registers register).map fun value => value != 0#u64)

theorem lookup_defined (facts : List Fact) (registers : PartialRegisters)
    (sound : FactsSoundDefined facts registers) (register : Reg) (value : U64)
    (found : lookup facts register = some value) : registers register = some value := by
  induction facts with
  | nil => simp at found
  | cons first rest ih =>
    rw [lookup_cons] at found
    split at found
    · rename_i same
      have valueEq := Option.some.inj found
      simpa only [same, valueEq] using sound first (by simp)
    · exact ih (fun fact member => sound fact (by simp [member])) found

theorem source_lookup_defined (facts : Slice Fact) (registers : PartialRegisters)
    (sound : FactsSoundDefined facts.val registers) (register : Reg) (value : U64)
    (found : guard_certificate.lookup facts register = .ok (some value)) :
    registers register = some value := by
  obtain ⟨fact, member, same, valueEq⟩ := successful_lookup facts register value found
  simpa only [same, valueEq] using sound fact member

theorem decision_defined (front : Front) (facts : List Fact) (registers : PartialRegisters)
    (sound : FactsSoundDefined facts registers) (predicate : Pred) (truth : Bool)
    (observed : decision front facts predicate = some truth) :
    partialPredicate front registers predicate = some truth := by
  unfold decision at observed
  split at observed
  · cases predicate with
    | mk register neg =>
      cases register with
      | none => simpa [partialPredicate, Instruction.PredicateRead.meaning] using observed
      | some register =>
        by_cases immutable : RegisterOwner.immutablePredicate front register
        · simpa [partialPredicate, Instruction.PredicateRead.meaning, immutable] using observed
        · by_cases shared : RegisterOwner.shared register.file
          · simp [Instruction.PredicateRead.meaning, immutable,
              Instruction.OwnerTemporal.privateKnowledge, shared] at observed
          · unfold knowledge at observed
            cases found : lookup facts register with
            | none => simp [Instruction.PredicateRead.meaning, immutable,
                Instruction.OwnerTemporal.privateKnowledge, shared, found] at observed
            | some value =>
              have defined := lookup_defined facts registers sound register value found
              simpa [partialPredicate, Instruction.PredicateRead.meaning, immutable,
                Instruction.OwnerTemporal.privateKnowledge, shared, found, defined] using observed
  · cases observed

theorem source_decision_defined (front : Front) (facts : Slice Fact) (registers : PartialRegisters)
    (sound : FactsSoundDefined facts.val registers) (predicate : Pred) (truth : Bool)
    (returned : guard_certificate.decision front facts predicate = .ok (some truth)) :
    partialPredicate front registers predicate = some truth := by
  have exactValue := rust_decision front facts predicate
  simp only [returned, WP.spec_ok] at exactValue
  exact decision_defined front facts.val registers sound predicate truth exactValue.symm

def partialOperandValue (registers : PartialRegisters) (bank : U32 → U8 → Option U64) : Val → Option U64
  | .Imm value => some value
  | .Const offset bytes => bank offset bytes
  | .Reg register slots =>
      match register.file with
      | .V => registers register
      | .R => if slots = 1#u8 then
          if register.idx.val ≥ 255 then some 0#u64 else registers register
        else none
      | _ => none
  | .SReg _ => none

theorem source_value_defined {C : Type} (reader : guard_certificate.Constants C)
    (facts : Slice Fact) (constants : C) (registers : PartialRegisters)
    (bank : U32 → U8 → Option U64) (sound : FactsSoundDefined facts.val registers)
    (constantSound : ConstantsSound reader constants bank) (operand : Val) (value : U64)
    (returned : guard_certificate.value reader facts constants operand = .ok (some value)) :
    partialOperandValue registers bank operand = some value := by
  cases operand with
  | Imm immediate => simpa [guard_certificate.value, partialOperandValue] using returned
  | Const offset bytes => exact constantSound offset bytes value returned
  | SReg special => simp [guard_certificate.value] at returned
  | Reg register slots =>
      cases file : register.file <;>
        simp [guard_certificate.value, file, File.Insts.CoreCmpPartialEqFile.eq,
          read_discriminant, File.read_discriminant, bind_ok, ite_true, ite_false,
          Bool.false_eq_true] at returned
      case V =>
        have same := source_lookup_defined facts registers sound register value returned
        simp [partialOperandValue, file, same]
      case R =>
        split at returned
        · rename_i scalar
          split at returned
          · rename_i zero
            have result := Option.some.inj (Result.ok_injective returned)
            have zero' : register.idx.val ≥ 255 := by scalar_tac
            simp [partialOperandValue, file, scalar, zero', ←result]
          · rename_i nonzero
            have same := source_lookup_defined facts registers sound register value returned
            have nonzero' : ¬register.idx.val ≥ 255 := by scalar_tac
            simp [partialOperandValue, file, scalar, nonzero', same]
        · simp at returned
      all_goals simp at returned

end Instruction.GuardCertificate
