/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.Admission

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 8000000

namespace MemorySafetyIR

theorem source_projected_access_is_exact
    (instruction : ptxv_ir.Inst) (nregs : U16) (pc : Usize) (last : Bool)
    (request : Request)
    (accepted : strict_memory_admission.project_instruction instruction nregs pc last =
      .ok (.Ok (some (.Access request)))) : ExactAccess instruction request := by
  obtain ⟨guardReg, guardNeg, uniform⟩ :=
    source_admission_requires_true_nonuniform_guard instruction nregs pc last _ accepted
  have classified := source_admission_refuses_every_other_opcode instruction nregs pc last _ accepted
  rcases instruction with ⟨⟨greg, gneg⟩, operation, uni, loc, text⟩
  dsimp only at guardReg guardNeg uniform classified
  subst greg
  subst gneg
  subst uni
  intro valuation
  cases classified with
  | load destination count space bytes operand sync sext nc =>
      cases space <;> cases sync <;> cases nc <;>
        simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
          source_scalar_width, source_nonvirtual_file, source_constant_address, lift] at accepted
      all_goals
        split_ifs at accepted <;> try simp at accepted
        cases resolvedEquation : constantAddress operand with
        | none => simp [resolvedEquation] at accepted
        | some resolved =>
            have exactAddress : address valuation operand = resolved :=
              source_constant_address_preserves_every_valuation operand resolved
                (by rw [source_constant_address, resolvedEquation]) valuation
            simp [resolvedEquation] at accepted
            cases accepted
            simp [instructionAccess, predicate, globalAccess, width, UScalar.cast, exactAddress]
  | store space bytes operand source sync =>
      cases source <;> cases space <;> cases sync <;>
        simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
          source_scalar_width, source_constant_address, lift] at accepted
      all_goals
        split_ifs at accepted <;> try simp at accepted
        cases resolvedEquation : constantAddress operand with
        | none => simp [resolvedEquation] at accepted
        | some resolved =>
            have exactAddress : address valuation operand = resolved :=
              source_constant_address_preserves_every_valuation operand resolved
                (by rw [source_constant_address, resolvedEquation]) valuation
            simp [resolvedEquation] at accepted
            cases accepted
            simp [instructionAccess, predicate, globalAccess, valueDefined, width,
              UScalar.cast, exactAddress]
  | nop =>
      simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always] at accepted
  | exit =>
      cases last <;>
        simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always] at accepted

end MemorySafetyIR
