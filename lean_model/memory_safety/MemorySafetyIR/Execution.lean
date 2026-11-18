/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.ExecutionBase
import MemorySafetyIR.Instruction

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

def checkedInstructionMemoryModel : CheckedIRInstruction → MemorySafety.Instruction
  | .Access request => .access (requestMemoryModel request)
  | .Free allocation => .free allocation.val
  | .Choice targets => .choice (targets.val.map UScalar.val)
  | .Skip => .skip
  | .Halt => .halt
  | .Unsupported => .unsupported

theorem scalar_width_meaning (bytes : U8) (supported : scalarWidth bytes = true) :
    scalarMemoryWidth bytes := by
  unfold scalarWidth at supported
  split at supported
  · exact .inl (UScalar.eq_of_val_eq (by rfl))
  · exact .inr (.inl (UScalar.eq_of_val_eq (by rfl)))
  · exact .inr (.inr (.inl (UScalar.eq_of_val_eq (by rfl))))
  · exact .inr (.inr (.inr (UScalar.eq_of_val_eq (by rfl))))
  · cases supported

theorem interpret_supported_access (instruction : ptxv_ir.Inst)
    (supported : SupportedScalarAccess instruction.op) (valuation : Valuation) :
    interpretInstruction valuation instruction =
      match instructionAccess valuation instruction with
      | some request => .access (requestMemoryModel request)
      | none => .skip := by
  cases operation : instruction.op with
  | Ld destination count space bytes operand sync sext nc =>
    cases space <;> cases sync <;> cases nc <;>
      simp only [operation, SupportedScalarAccess] at supported
    obtain ⟨countEq, widthEq⟩ := supported
    have widthSupported := scalar_width_meaning bytes widthEq
    cases guardEq : predicate valuation instruction.guard <;>
      simp [interpretInstruction, instructionAccess, globalAccess, operation, countEq, widthSupported, guardEq]
  | St space bytes operand source sync =>
    cases source <;> cases space <;> cases sync <;>
      simp only [operation, SupportedScalarAccess] at supported
    have widthSupported := scalar_width_meaning bytes supported
    cases guardEq : predicate valuation instruction.guard <;>
      simp [interpretInstruction, instructionAccess, globalAccess, operation, valueDefined,
        widthSupported, guardEq]
  | _ => simp only [operation, SupportedScalarAccess] at supported

theorem exact_instruction_interpret (instruction : ptxv_ir.Inst) (target : CheckedIRInstruction)
    (exact : ExactInstruction instruction target) (valuation : Valuation) :
    interpretInstruction valuation instruction = checkedInstructionMemoryModel target := by
  cases exact with
  | access request supported footprint =>
    rw [interpret_supported_access instruction supported valuation, footprint valuation]
    rfl
  | skip operation guard => simp [interpretInstruction, operation, checkedInstructionMemoryModel]
  | halt operation guard => simp [interpretInstruction, operation, guard valuation, checkedInstructionMemoryModel]

theorem exact_code_interpret (source : List ptxv_ir.Inst) (target : List CheckedIRInstruction)
    (exact : List.Forall₂ ExactInstruction source target) :
    List.Forall₂ (fun instruction result => ∀ valuation,
      interpretInstruction valuation instruction = result) source (target.map checkedInstructionMemoryModel) := by
  induction exact with
  | nil => exact .nil
  | @cons instruction result source target first rest ih =>
    exact .cons (exact_instruction_interpret instruction result first) ih

theorem exact_code_with_globals (source : List ptxv_ir.Inst) (target : List CheckedIRInstruction)
    (globals : List GlobalBinding) (exact : List.Forall₂ ExactInstruction source target) :
    List.Forall₂ (fun instruction result => ∀ valuation,
      interpretInstruction (bindGlobals globals valuation) instruction = result)
      source (target.map checkedInstructionMemoryModel) := by
  induction exact with
  | nil => exact .nil
  | @cons instruction result source target first rest ih =>
    exact .cons (fun valuation => exact_instruction_interpret instruction result first
      (bindGlobals globals valuation)) ih

end MemorySafetyIR
