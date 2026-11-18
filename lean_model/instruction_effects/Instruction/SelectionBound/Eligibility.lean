/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBound.Model

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegBound

@[simp] theorem source_direct_value (value : PtxvSim.ptxv_ir.Val) :
    PtxvSim.seg.bound.direct_value value = .ok (directValue value) := by
  cases value <;> rfl

@[simp] theorem source_eligible (operation : PtxvSim.ptxv_ir.Op) :
    PtxvSim.seg.bound.eligible operation = .ok (eligibleOperation operation) := by
  cases operation <;>
    simp only [PtxvSim.seg.bound.eligible, eligibleOperation, source_direct_value, bind_ok] <;>
    split <;> simp_all

@[step] theorem actual_constructor_eligibility
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right comparisonPC selectionPC : U32) :
    PtxvSim.seg.comparison_program_parts ⟨plan, left, right, comparisonPC, selectionPC⟩ ⦃ parts =>
      ∀ instruction ∈ parts.2.1.val, eligibleInstruction instruction = true ⦄ := by
  step with SegProgram.actual_program_shape plan left right comparisonPC selectionPC as ⟨parts, shape⟩
  rename_i instruction member
  have hops := congrArg (fun p : SegProgram.Parts => p.2.1) shape
  dsimp only [SegProgram.observe, SegProgram.expected] at hops
  rw [hops] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with member | member
  · rw [member]
    simp [SegProgram.opcodePair, eligibleInstruction, eligibleOperation, directValue]
  · rw [member]
    cases hswap : plan.swap <;>
      simp [SegProgram.opcodePair, eligibleInstruction, eligibleOperation, directValue, hswap]

@[simp] theorem source_try_unsupported {C : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (copyInst : core.marker.Copy C)
    (instruction : PtxvSim.seg.SOp) (inputs : Inputs) (primitive : C)
    (unsupported : eligibleInstruction instruction = false) :
    PtxvSim.seg.bound.try_evaluate primitiveInst copyInst instruction inputs primitive = .ok none := by
  cases hkind : instruction.kind with
  | Guard predicate negative new old => simp [PtxvSim.seg.bound.try_evaluate, hkind]
  | DivRn a b => simp [PtxvSim.seg.bound.try_evaluate, hkind]
  | Inst operation destination bindings constants =>
    have ineligible : eligibleOperation operation = false := by
      simpa [eligibleInstruction, hkind] using unsupported
    simp [PtxvSim.seg.bound.try_evaluate, hkind, source_eligible, ineligible]

end Instruction.SegBound
