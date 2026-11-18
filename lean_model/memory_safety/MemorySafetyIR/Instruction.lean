/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.Footprint

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 8000000

namespace MemorySafetyIR

abbrev CheckedIRInstruction := ptxv_memory_write.memory_safety.Instruction

def SupportedScalarAccess : ptxv_ir.Op → Prop
  | .Ld _ count .Global bytes _ none _ false => count = 1#u8 ∧ scalarWidth bytes = true
  | .St .Global bytes _ (.Imm _) none => scalarWidth bytes = true
  | _ => False

inductive ExactInstruction (instruction : ptxv_ir.Inst) : CheckedIRInstruction → Prop
  | access (request : Request) (supported : SupportedScalarAccess instruction.op)
      (exact : ExactAccess instruction request) :
      ExactInstruction instruction (.Access request)
  | skip (operation : instruction.op = .Nop)
      (guard : ∀ valuation, predicate valuation instruction.guard = true) :
      ExactInstruction instruction .Skip
  | halt (operation : instruction.op = .Exit)
      (guard : ∀ valuation, predicate valuation instruction.guard = true) :
      ExactInstruction instruction .Halt

theorem source_projected_instruction_shape
    (instruction : ptxv_ir.Inst) (nregs : U16) (pc : Usize) (last : Bool)
    (target : CheckedIRInstruction)
    (accepted : strict_memory_admission.project_instruction instruction nregs pc last =
      .ok (.Ok (some target))) :
    (∃ request, target = .Access request) ∨
      (target = .Skip ∧ instruction.op = .Nop) ∨
      (target = .Halt ∧ instruction.op = .Exit) := by
  obtain ⟨guardReg, guardNeg, uniform⟩ :=
    source_admission_requires_true_nonuniform_guard instruction nregs pc last _ accepted
  have classified := source_admission_refuses_every_other_opcode instruction nregs pc last _ accepted
  rcases instruction with ⟨⟨greg, gneg⟩, operation, uni, loc, text⟩
  dsimp only at guardReg guardNeg uniform classified
  subst greg
  subst gneg
  subst uni
  cases classified with
  | load destination count space bytes operand sync sext nc =>
    cases space <;> cases sync <;> cases nc <;>
      simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
        source_scalar_width, source_nonvirtual_file, source_constant_address, lift] at accepted
    all_goals
      split_ifs at accepted <;> try simp at accepted
      cases resolved : constantAddress operand with
      | none => simp [resolved] at accepted
      | some resolved =>
        simp [resolved] at accepted
        cases accepted
        exact .inl ⟨_, rfl⟩
  | store space bytes operand source sync =>
    cases source <;> cases space <;> cases sync <;>
      simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
        source_scalar_width, source_constant_address, lift] at accepted
    all_goals
      split_ifs at accepted <;> try simp at accepted
      cases resolved : constantAddress operand with
      | none => simp [resolved] at accepted
      | some resolved =>
        simp [resolved] at accepted
        cases accepted
        exact .inl ⟨_, rfl⟩
  | nop =>
    simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always] at accepted
    cases accepted
    exact .inr (.inl ⟨rfl, rfl⟩)
  | exit =>
    cases last <;>
      simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always] at accepted
    cases accepted
    exact .inr (.inr ⟨rfl, rfl⟩)

theorem source_projected_access_supported
    (instruction : ptxv_ir.Inst) (nregs : U16) (pc : Usize) (last : Bool)
    (request : Request)
    (accepted : strict_memory_admission.project_instruction instruction nregs pc last =
      .ok (.Ok (some (.Access request)))) : SupportedScalarAccess instruction.op := by
  obtain ⟨guardReg, guardNeg, uniform⟩ :=
    source_admission_requires_true_nonuniform_guard instruction nregs pc last _ accepted
  have classified := source_admission_refuses_every_other_opcode instruction nregs pc last _ accepted
  rcases instruction with ⟨⟨greg, gneg⟩, operation, uni, loc, text⟩
  dsimp only at guardReg guardNeg uniform classified
  subst greg
  subst gneg
  subst uni
  cases classified with
  | load destination count space bytes operand sync sext nc =>
    cases space <;> cases sync <;> cases nc <;>
      simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
        source_scalar_width, source_nonvirtual_file, source_constant_address, lift] at accepted
    all_goals
      split_ifs at accepted <;> try simp at accepted
      simp_all [SupportedScalarAccess]; scalar_tac
  | store space bytes operand source sync =>
    cases source <;> cases space <;> cases sync <;>
      simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
        source_scalar_width, source_constant_address, lift] at accepted
    all_goals
      split_ifs at accepted <;> try simp at accepted
      simp_all [SupportedScalarAccess]
  | nop => simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always] at accepted
  | exit => cases last <;>
      simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always] at accepted

theorem source_projected_instruction_is_exact
    (instruction : ptxv_ir.Inst) (nregs : U16) (pc : Usize) (last : Bool)
    (target : CheckedIRInstruction)
    (accepted : strict_memory_admission.project_instruction instruction nregs pc last =
      .ok (.Ok (some target))) : ExactInstruction instruction target := by
  rcases source_projected_instruction_shape instruction nregs pc last target accepted with
    ⟨request, rfl⟩ | ⟨rfl, operation⟩ | ⟨rfl, operation⟩
  · exact .access request (source_projected_access_supported instruction nregs pc last request accepted)
      (source_projected_access_is_exact instruction nregs pc last request accepted)
  · exact .skip operation
      (source_admitted_guard_is_true_in_every_valuation instruction nregs pc last _ accepted)
  · exact .halt operation
      (source_admitted_guard_is_true_in_every_valuation instruction nregs pc last _ accepted)

end MemorySafetyIR
