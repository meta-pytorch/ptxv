/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionDriver.Observations

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegDriver

open SegProgram

@[step] theorem actual_op_source_driver_execution {R : Type}
    (readerInst : PtxvIr.comparison_select.NodeReader R) (reader : R)
    (nodeOf : PtxvIr.Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (compare : Comparison) (valuation : U32 → U32)
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
        let parts ← PtxvSim.seg.comparison_program_parts
          ⟨importPlan plan, left, right, comparisonPC, selectionPC⟩
        let (answer, scratch, _) ← executeSelected compare valuation parts values
        .ok (some (answer, scratch))) ⦃ produced => match produced with
          | none => True
          | some (answer, _) => ∃ bits, answer = .Ok bits ∧
            some bits = selectedNodeBits (fun mode a b => compare mode a b false)
              valuation mode predicate left right first second nodeOf ⦄ := by
  step with actual_selection_preserves_all_node_valuations readerInst reader nodeOf hreader
    (fun mode a b => compare mode a b false) front comparison mode destination first second predicate
    left right hmode as ⟨produced, hproduced⟩
  cases produced with
  | none => simp
  | some plan =>
    step with actual_constructor_driver_execution compare valuation (importPlan plan)
      left right comparisonPC selectionPC values as ⟨parts, _, hdriver⟩
    step with hdriver as ⟨answer, scratch, houtput⟩
    rw [imported_plan_view] at houtput
    exact ⟨_, houtput, hproduced valuation⟩

end Instruction.SegDriver
