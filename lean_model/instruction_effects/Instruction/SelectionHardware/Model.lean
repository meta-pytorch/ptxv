/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Composition

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SelectionHardware

open IEEEComparison SegProgram

def hardwareBits (plan : PtxvSim.ptxv_ir.comparison_select.Plan)
    (first second : U32) : U32 :=
  comparisonSelectionBits (fun mode a b => oracle mode a b false)
    (planView plan) first second

theorem hardware_chooses_captured_operand
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (first second : U32) :
    hardwareBits plan first second = first ∨ hardwareBits plan first second = second := by
  unfold hardwareBits comparisonSelectionBits
  split <;> split <;> simp_all

theorem greater_signed_zero_retains_negative_zero :
    hardwareBits ⟨.Gt, false, false⟩ 0#u32 0x80000000#u32 = 0x80000000#u32 := by
  rfl

@[step] theorem actual_program_hardware_agreement {F : Type}
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (fallback : F)
    (valuation : U32 → U32) (context : PtxvSim.seg.ComparisonContext)
    (values : alloc.vec.Vec U64) :
    (do
      let parts ← PtxvSim.seg.comparison_program_parts context
      SegBound.executeBound primitive primitiveCopy fallbackInst (concreteEvaluator fallback)
        valuation parts values) ⦃ output =>
      output.1 = .Ok (hardwareBits context.plan (valuation context.left) (valuation context.right)) ∧
      output.2.2 = concreteEvaluator fallback ⦄ := by
  step with SegBound.actual_constructor_bound_driver_execution primitive primitiveCopy
    fallbackInst (concreteEvaluator fallback) oracle concrete_primitive_contract valuation context.plan
    context.left context.right context.comparison_pc context.selection_pc values as ⟨parts, _, hrun⟩
  exact hrun

end Instruction.SelectionHardware
