/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.Execution

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

def globalOperand (bindings : List GlobalBinding) : Option ptxv_ir.Val → Option U64
  | none => some 0#u64
  | some (.Imm bits) => some bits
  | some (.SReg (.GlobalAddr index)) => globalAddress bindings index
  | _ => none

def globalResolvedAddress (bindings : List GlobalBinding) (operand : ptxv_ir.Addr) : Option U64 := do
  let first ← globalOperand bindings operand.base
  let second ← globalOperand bindings operand.base2
  pure ⟨operand.off.bv + first.bv + second.bv⟩

def literalAddress (bits : U64) : ptxv_ir.Addr := ⟨some (.Imm bits), none, 0#i64⟩

theorem global_operand_exact (bindings : List GlobalBinding)
    (operand : Option ptxv_ir.Val) (bits : U64)
    (resolved : globalOperand bindings operand = some bits) (valuation : Valuation) :
    optionalValue (bindGlobals bindings valuation) operand = bits := by
  cases operand with
  | none => simpa [globalOperand, optionalValue] using resolved
  | some operand =>
    cases operand with
    | Imm immediate => simpa [globalOperand, optionalValue, value] using resolved
    | Reg register slots => simp [globalOperand] at resolved
    | Const offset bytes => simp [globalOperand] at resolved
    | SReg register =>
      cases register <;>
        simp [globalOperand] at resolved
      simp [optionalValue, value, bindGlobals, resolved]

theorem global_address_exact (bindings : List GlobalBinding)
    (operand : ptxv_ir.Addr) (bits : U64)
    (resolved : globalResolvedAddress bindings operand = some bits) (valuation : Valuation) :
    address (bindGlobals bindings valuation) operand = bits := by
  unfold globalResolvedAddress at resolved
  cases first : globalOperand bindings operand.base with
  | none => simp [first] at resolved
  | some firstValue =>
    cases second : globalOperand bindings operand.base2 with
    | none => simp [first, second] at resolved
    | some secondValue =>
      simp [first, second] at resolved
      cases resolved
      simp only [address, global_operand_exact bindings operand.base firstValue first valuation,
        global_operand_exact bindings operand.base2 secondValue second valuation]

theorem literal_address_exact (bits : U64) (valuation : Valuation) :
    address valuation (literalAddress bits) = bits := by
  simp [address, literalAddress, optionalValue, value]

theorem global_address_specialization_preserves_all_valuations
    (bindings : List GlobalBinding) (operand : ptxv_ir.Addr) (bits : U64)
    (resolved : globalResolvedAddress bindings operand = some bits) (valuation : Valuation) :
    address (bindGlobals bindings valuation) operand =
      address valuation (literalAddress bits) := by
  rw [global_address_exact bindings operand bits resolved valuation, literal_address_exact]

inductive SpecializesOpcode (bindings : List GlobalBinding) : ptxv_ir.Op → ptxv_ir.Op → Prop
  | load (destination : ptxv_ir.Reg) (bytes : U8) (operand : ptxv_ir.Addr)
      (sync : Option ptxv_ir.MemSync) (sext nc : Bool) (bits : U64)
      (resolved : globalResolvedAddress bindings operand = some bits) :
      SpecializesOpcode bindings (.Ld destination 1#u8 .Global bytes operand sync sext nc)
        (.Ld destination 1#u8 .Global bytes (literalAddress bits) sync sext nc)
  | store (bytes : U8) (operand : ptxv_ir.Addr) (value : U64)
      (sync : Option ptxv_ir.MemSync) (bits : U64)
      (resolved : globalResolvedAddress bindings operand = some bits) :
      SpecializesOpcode bindings (.St .Global bytes operand (.Imm value) sync)
        (.St .Global bytes (literalAddress bits) (.Imm value) sync)
  | nop : SpecializesOpcode bindings .Nop .Nop
  | exit : SpecializesOpcode bindings .Exit .Exit

def SpecializesInstruction (bindings : List GlobalBinding)
    (source target : ptxv_ir.Inst) : Prop :=
  source.guard = target.guard ∧ SpecializesOpcode bindings source.op target.op

theorem bound_predicate_exact (bindings : List GlobalBinding) (valuation : Valuation)
    (guard : ptxv_ir.Pred) :
    predicate (bindGlobals bindings valuation) guard = predicate valuation guard := by
  simp [predicate, bindGlobals]

theorem global_specialization_interpretation_exact
    (bindings : List GlobalBinding) (source target : ptxv_ir.Inst)
    (specialized : SpecializesInstruction bindings source target) (valuation : Valuation) :
    interpretInstruction (bindGlobals bindings valuation) source = interpretInstruction valuation target := by
  rcases source with ⟨sourceGuard, sourceOp, sourceUniform, sourceLoc, sourceText⟩
  rcases target with ⟨targetGuard, targetOp, targetUniform, targetLoc, targetText⟩
  rcases specialized with ⟨guardEq, opcode⟩
  dsimp only at guardEq opcode
  subst targetGuard
  cases opcode with
  | load destination bytes operand sync sext nc bits resolved =>
      cases sync <;> cases nc <;>
        simp [interpretInstruction, bound_predicate_exact,
          global_address_exact bindings operand bits resolved valuation, literal_address_exact]
  | store bytes operand value sync bits resolved =>
      cases sync <;>
        simp [interpretInstruction, bound_predicate_exact,
          global_address_exact bindings operand bits resolved valuation, literal_address_exact]
  | nop => rfl
  | exit => simp [interpretInstruction, bound_predicate_exact]

theorem specialized_code_interpretation_exact
    (bindings : List GlobalBinding) (source specialized : List ptxv_ir.Inst)
    (target : List CheckedIRInstruction)
    (specialization : List.Forall₂ (SpecializesInstruction bindings) source specialized)
    (projection : List.Forall₂ ExactInstruction specialized target) :
    List.Forall₂ (fun instruction result => ∀ valuation,
      interpretInstruction (bindGlobals bindings valuation) instruction = result)
      source (target.map checkedInstructionMemoryModel) := by
  induction specialization generalizing target with
  | nil => cases projection; exact .nil
  | @cons instruction literal source specialized first rest ih =>
    cases projection with
    | @cons _ checked _ checkedTail literalExact remaining =>
      refine .cons (fun valuation => ?_) (ih checkedTail remaining)
      rw [global_specialization_interpretation_exact bindings instruction literal first valuation]
      exact exact_instruction_interpret literal checked literalExact valuation

end MemorySafetyIR
