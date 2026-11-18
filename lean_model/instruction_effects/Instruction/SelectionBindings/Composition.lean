/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBindings.Bounds

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegBinding

open SegProgram

@[step] theorem actual_constructor_bound_execution (compare : Comparison) (valuation : U32 → U32)
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right comparisonPC selectionPC : U32) :
    PtxvSim.seg.comparison_program_parts ⟨plan, left, right, comparisonPC, selectionPC⟩
      ⦃ parts => programReady (observe parts) ∧ executeProducedPair compare valuation parts = .ok
        (comparisonSelectionBits (fun mode a b => compare mode a b false)
          (planView plan) (valuation left) (valuation right)) ⦄ :=
  WP.spec_and (actual_constructor_no_defaults_or_bad_references plan left right comparisonPC selectionPC)
    (actual_constructor_source_execution compare valuation plan left right comparisonPC selectionPC)

@[step] theorem actual_op_bound_program_execution {R : Type}
    (readerInst : PtxvIr.comparison_select.NodeReader R) (reader : R)
    (nodeOf : PtxvIr.Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (compare : Comparison) (valuation : U32 → U32)
    (front : PtxvIr.Front) (comparison : PtxvIr.Op) (mode : PtxvIr.FCmp)
    (destination : PtxvIr.Reg) (first second : PtxvIr.Val) (predicate : PtxvIr.Pred)
    (left right comparisonPC selectionPC : U32) (hmode : plainComparison comparison = some mode) :
    (do
      let plan ← PtxvIr.comparison_select.from_pair readerInst front comparison
        (.Sel destination first second predicate) left right reader
      match plan with
      | none => .ok none
      | some plan =>
        let parts ← PtxvSim.seg.comparison_program_parts
          ⟨importPlan plan, left, right, comparisonPC, selectionPC⟩
        .ok (some parts)) ⦃ produced => match produced with
          | none => True
          | some parts => ∃ bits,
            executeProducedPair compare valuation parts = .ok bits ∧
            some bits = selectedNodeBits (fun mode a b => compare mode a b false)
              valuation mode predicate left right first second nodeOf ∧
            programReady (observe parts) ⦄ := by
  step with actual_selection_preserves_all_node_valuations readerInst reader nodeOf hreader
    (fun mode a b => compare mode a b false) front comparison mode destination first second predicate
    left right hmode as ⟨produced, hproduced⟩
  cases produced with
  | none => simp
  | some plan =>
    step with actual_constructor_bound_execution compare valuation (importPlan plan)
      left right comparisonPC selectionPC as ⟨parts, hready, hexec⟩
    rw [imported_plan_view] at hexec
    exact ⟨_, hexec, hproduced valuation, hready⟩

end Instruction.SegBinding
