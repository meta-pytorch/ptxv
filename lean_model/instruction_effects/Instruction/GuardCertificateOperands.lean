/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateDecision

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial

set_option autoImplicit false
set_option maxHeartbeats 5000000

namespace Instruction.GuardCertificate

def operandValue (registers : Reg → U64) (bank : U32 → U8 → Option U64) : Val → Option U64
  | .Imm value => some value
  | .Const offset bytes => bank offset bytes
  | .Reg register slots =>
      match register.file with
      | .V => some (registers register)
      | .R => if slots = 1#u8 then
          some (if register.idx.val ≥ 255 then 0#u64 else registers register)
        else none
      | _ => none
  | .SReg _ => none

def ConstantsSound {C : Type} (reader : guard_certificate.Constants C) (constants : C)
    (bank : U32 → U8 → Option U64) : Prop :=
  ∀ offset bytes value, reader.read constants offset bytes = .ok (some value) →
    bank offset bytes = some value

theorem source_value_sound {C : Type} (reader : guard_certificate.Constants C)
    (facts : Slice Fact) (constants : C) (registers : Reg → U64)
    (bank : U32 → U8 → Option U64) (sound : FactsSound facts.val registers)
    (constantSound : ConstantsSound reader constants bank) (operand : Val) (value : U64)
    (returned : guard_certificate.value reader facts constants operand = .ok (some value)) :
    operandValue registers bank operand = some value := by
  cases operand with
  | Imm immediate => simpa [guard_certificate.value, operandValue] using returned
  | Const offset bytes => exact constantSound offset bytes value returned
  | SReg special => simp [guard_certificate.value] at returned
  | Reg register slots =>
      cases file : register.file <;>
        simp [guard_certificate.value, file, File.Insts.CoreCmpPartialEqFile.eq,
          read_discriminant, File.read_discriminant, bind_ok, ite_true, ite_false,
          Bool.false_eq_true] at returned
      case V =>
        have same := lookup_sound facts registers sound register value returned
        simp [operandValue, file, same]
      case R =>
        split at returned
        · rename_i scalar
          split at returned
          · rename_i zero
            have result := Option.some.inj (Result.ok_injective returned)
            have zero' : register.idx.val ≥ 255 := by scalar_tac
            simp [operandValue, file, scalar, zero', ←result]
          · rename_i nonzero
            have same := lookup_sound facts registers sound register value returned
            have nonzero' : ¬register.idx.val ≥ 255 := by scalar_tac
            simp [operandValue, file, scalar, nonzero', same]
        · simp at returned
      all_goals simp at returned

def scalarMove (destination : Reg) (operand : Val) : Bool :=
  if (match operand with | .Const _ bytes => decide (bytes.val = 16) | _ => false) then false
  else if destination.file = .V then true
  else match operand with
    | .Const _ bytes => decide (bytes.val ≤ 4)
    | .Reg _ slots => decide (slots = 1#u8)
    | _ => true

theorem source_scalar_move (destination : Reg) (operand : Val) :
    guard_certificate.scalar_move destination operand = .ok (scalarMove destination operand) := by
  have literal : (16#8#uscalar : U8).val = 16 := rfl
  cases destination with
  | mk file index =>
      cases file <;> cases operand <;>
        simp [guard_certificate.scalar_move, scalarMove,
          File.Insts.CoreCmpPartialEqFile.eq, read_discriminant, File.read_discriminant,
          UScalar.le_equiv]
      all_goals
        split <;> simp_all [UScalar.eq_equiv, literal]

end Instruction.GuardCertificate
