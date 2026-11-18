/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Funs
import MemorySafetyIR.Semantics

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemorySafetyIR

def constantOperand : Option ptxv_ir.Val → Option U64
  | none => some 0#u64
  | some (.Imm bits) => some bits
  | _ => none

def constantAddress (operand : ptxv_ir.Addr) : Option U64 := do
  let first ← constantOperand operand.base
  let second ← constantOperand operand.base2
  pure ⟨operand.off.bv + first.bv + second.bv⟩

def scalarWidth (bytes : U8) : Bool :=
  match bytes with
  | 1#uscalar => true
  | 2#uscalar => true
  | 4#uscalar => true
  | 8#uscalar => true
  | _ => false

theorem source_scalar_width (bytes : U8) :
    strict_memory_admission.width_supported bytes = .ok (scalarWidth bytes) := by
  unfold strict_memory_admission.width_supported
  split <;> simp_all [scalarWidth]

def virtualFile : ptxv_ir.File → Bool
  | .V => true
  | _ => false

theorem source_nonvirtual_file (file : ptxv_ir.File) :
    core.cmp.PartialEq.ne.trait_default ptxv_ir.File.Insts.CoreCmpPartialEqFile file .V =
      .ok (!virtualFile file) := by
  cases file <;>
    simp [core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default,
      ptxv_ir.File.Insts.CoreCmpPartialEqFile.eq, ptxv_ir.File.read_discriminant, virtualFile]

theorem source_constant_address (operand : ptxv_ir.Addr) :
    strict_memory_admission.constant_address operand = .ok (constantAddress operand) := by
  rcases operand with ⟨first, second, offset⟩
  cases first with
  | none =>
      cases second with
      | none => simp [strict_memory_admission.constant_address, constantAddress,
          constantOperand, IScalar.hcast, core.num.U64.wrapping_add, UScalar.wrapping_add, lift]
      | some other =>
          cases other <;> simp [strict_memory_admission.constant_address, constantAddress,
            constantOperand, IScalar.hcast, core.num.U64.wrapping_add, UScalar.wrapping_add, lift]
  | some first =>
      cases first with
      | Reg register slots =>
          simp [strict_memory_admission.constant_address, constantAddress, constantOperand]
      | Const offset bytes =>
          simp [strict_memory_admission.constant_address, constantAddress, constantOperand]
      | SReg register =>
          simp [strict_memory_admission.constant_address, constantAddress, constantOperand]
      | Imm bits =>
          cases second with
          | none => simp [strict_memory_admission.constant_address, constantAddress,
              constantOperand, IScalar.hcast, core.num.U64.wrapping_add, UScalar.wrapping_add, lift]
          | some other =>
              cases other <;> simp [strict_memory_admission.constant_address, constantAddress,
                constantOperand, IScalar.hcast, core.num.U64.wrapping_add, UScalar.wrapping_add, lift]

theorem source_constant_address_preserves_every_valuation
    (operand : ptxv_ir.Addr) (bits : U64)
    (accepted : strict_memory_admission.constant_address operand = .ok (some bits))
    (valuation : Valuation) : address valuation operand = bits := by
  rw [source_constant_address] at accepted
  rcases operand with ⟨first, second, offset⟩
  cases first with
  | none =>
      cases second with
      | none => simp_all [constantAddress, constantOperand, address, optionalValue]
      | some other =>
          cases other <;> simp_all [constantAddress, constantOperand, address, optionalValue, value]
  | some first =>
      cases first with
      | Reg register slots => simp_all [constantAddress, constantOperand]
      | Const offset bytes => simp_all [constantAddress, constantOperand]
      | SReg register => simp_all [constantAddress, constantOperand]
      | Imm bits =>
          cases second with
          | none => simp_all [constantAddress, constantOperand, address, optionalValue, value]
          | some other =>
              cases other <;> simp_all [constantAddress, constantOperand, address, optionalValue, value]

theorem source_admission_requires_true_nonuniform_guard
    (instruction : ptxv_ir.Inst) (nregs : U16) (pc : Usize) (last : Bool)
    (projected : Option ptxv_memory_write.memory_safety.Instruction)
    (accepted : strict_memory_admission.project_instruction instruction nregs pc last =
      .ok (.Ok projected)) :
    instruction.guard.reg = none ∧ instruction.guard.neg = false ∧ instruction.uniform = false := by
  cases guardRegister : instruction.guard.reg with
  | some register =>
      simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
        guardRegister] at accepted
  | none =>
      cases negated : instruction.guard.neg with
      | true =>
          simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
            guardRegister, negated] at accepted
      | false =>
          cases uniform : instruction.uniform with
          | true =>
              simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
                guardRegister, negated, uniform] at accepted
          | false => exact ⟨rfl, rfl, rfl⟩

theorem source_admitted_guard_is_true_in_every_valuation
    (instruction : ptxv_ir.Inst) (nregs : U16) (pc : Usize) (last : Bool)
    (projected : Option ptxv_memory_write.memory_safety.Instruction)
    (accepted : strict_memory_admission.project_instruction instruction nregs pc last =
      .ok (.Ok projected)) (valuation : Valuation) :
    predicate valuation instruction.guard = true := by
  obtain ⟨register, negated, _⟩ :=
    source_admission_requires_true_nonuniform_guard instruction nregs pc last projected accepted
  simp [predicate, register, negated]

inductive MemoryOnly : ptxv_ir.Op → Prop
  | load (d : ptxv_ir.Reg) (n : U8) (space : ptxv_ir.Space) (bytes : U8)
      (operand : ptxv_ir.Addr) (sync : Option ptxv_ir.MemSync) (sext nc : Bool) :
      MemoryOnly (.Ld d n space bytes operand sync sext nc)
  | store (space : ptxv_ir.Space) (bytes : U8) (operand : ptxv_ir.Addr)
      (source : ptxv_ir.Val) (sync : Option ptxv_ir.MemSync) :
      MemoryOnly (.St space bytes operand source sync)
  | nop : MemoryOnly .Nop
  | exit : MemoryOnly .Exit

theorem source_admission_refuses_every_other_opcode
    (instruction : ptxv_ir.Inst) (nregs : U16) (pc : Usize) (last : Bool)
    (projected : Option ptxv_memory_write.memory_safety.Instruction)
    (accepted : strict_memory_admission.project_instruction instruction nregs pc last =
      .ok (.Ok projected)) : MemoryOnly instruction.op := by
  obtain ⟨register, negated, uniform⟩ :=
    source_admission_requires_true_nonuniform_guard instruction nregs pc last projected accepted
  cases operation : instruction.op <;> first
    | constructor
    | simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
        register, negated, uniform, operation] at accepted

end MemorySafetyIR
