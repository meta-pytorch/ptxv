/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionHardware.Refinement

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SelectionHardware

open IEEEComparison SegProgram

@[step] theorem source_hardware_matches_actual_program {F : Type}
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (fallback : F)
    (valuation : U32 → U32) (context : PtxvSim.seg.ComparisonContext)
    (values : alloc.vec.Vec U64) :
    (do
      let hardware ← PtxvSim.seg.hardware.evaluate context.plan
        (valuation context.left) (valuation context.right)
      let parts ← PtxvSim.seg.comparison_program_parts context
      let (answer, scratch, evaluator) ← SegBound.executeBound primitive primitiveCopy
        fallbackInst (concreteEvaluator fallback) valuation parts values
      .ok (hardware, answer, scratch, evaluator)) ⦃ output =>
        output.2.1 = .Ok output.1 ∧ output.2.2.2 = concreteEvaluator fallback ⦄ := by
  step with source_hardware_value_spec context.plan
    (valuation context.left) (valuation context.right) as ⟨hardware, hhardware⟩
  step with SegBound.actual_constructor_bound_driver_execution primitive primitiveCopy
    fallbackInst (concreteEvaluator fallback) oracle concrete_primitive_contract valuation context.plan
    context.left context.right context.comparison_pc context.selection_pc values as ⟨parts, _, hrun⟩
  step with hrun as ⟨answer, scratch, evaluator, hanswer, hevaluator⟩
  exact ⟨by simpa [hhardware, hardwareBits] using hanswer, hevaluator⟩

@[step] theorem actual_op_hardware_and_program_source_selection {R F : Type}
    (readerInst : PtxvIr.comparison_select.NodeReader R) (reader : R)
    (nodeOf : PtxvIr.Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (fallback : F)
    (valuation : U32 → U32)
    (front : PtxvIr.Front) (comparison : PtxvIr.Op) (mode : PtxvIr.FCmp)
    (destination : PtxvIr.Reg) (first second : PtxvIr.Val) (predicate : PtxvIr.Pred)
    (left right comparisonPC selectionPC : U32) (values : alloc.vec.Vec U64)
    (hmode : plainComparison comparison = some mode) :
    (do
      let plan ← PtxvIr.comparison_select.from_pair readerInst front comparison
        (.Sel destination first second predicate) left right reader
      match plan with
      | none => .ok none
      | some plan =>
        let context := PtxvSim.seg.ComparisonContext.mk
          (importPlan plan) left right comparisonPC selectionPC
        let hardware ← PtxvSim.seg.hardware.evaluate context.plan
          (valuation context.left) (valuation context.right)
        let parts ← PtxvSim.seg.comparison_program_parts context
        let (answer, scratch, evaluator) ← SegBound.executeBound primitive primitiveCopy
          fallbackInst (concreteEvaluator fallback) valuation parts values
        .ok (some (hardware, answer, scratch, evaluator))) ⦃ produced => match produced with
          | none => True
          | some (hardware, answer, _, evaluator) =>
            answer = .Ok hardware ∧ evaluator = concreteEvaluator fallback ∧
            some hardware = selectedNodeBits (fun mode a b => oracle mode a b false)
              valuation mode predicate left right first second nodeOf ⦄ := by
  step with actual_selection_preserves_all_node_valuations readerInst reader nodeOf hreader
    (fun mode a b => oracle mode a b false) front comparison mode destination first second predicate
    left right hmode as ⟨produced, hproduced⟩
  cases produced with
  | none => simp
  | some plan =>
    step with source_hardware_value_spec (importPlan plan) (valuation left) (valuation right) as
      ⟨hardware, hhardware⟩
    step with SegBound.actual_constructor_bound_driver_execution primitive primitiveCopy
      fallbackInst (concreteEvaluator fallback) oracle concrete_primitive_contract valuation
      (importPlan plan) left right comparisonPC selectionPC values as ⟨parts, _, hrun⟩
    step with hrun as ⟨answer, scratch, evaluator, hanswer, hevaluator⟩
    have hselection := hproduced valuation
    rw [← imported_plan_view plan] at hselection
    exact ⟨by simpa [hhardware, hardwareBits] using hanswer, hevaluator, by simpa [hhardware, hardwareBits] using hselection⟩

end Instruction.SelectionHardware
