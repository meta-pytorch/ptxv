/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionProgram.Semantics
import Instruction.SelectionProgram.MachineControl

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
namespace Instruction.SegProgram

def importComparator : PtxvIr.FCmp → PtxvSim.ptxv_ir.FCmp
  | .Eq => .Eq | .Ne => .Ne | .Lt => .Lt | .Le => .Le | .Gt => .Gt | .Ge => .Ge
  | .Num => .Num | .Nan => .Nan | .Equ => .Equ | .Neu => .Neu | .Ltu => .Ltu
  | .Leu => .Leu | .Gtu => .Gtu | .Geu => .Geu | .F => .F | .T => .T

def importPlan (plan : comparison_select.Plan) : PtxvSim.ptxv_ir.comparison_select.Plan :=
  ⟨importComparator plan.comparison, plan.negate, plan.swap⟩

theorem comparator_view_import (comparison : PtxvIr.FCmp) :
    comparatorView (importComparator comparison) = comparison := by
  cases comparison <;> rfl

theorem comparator_import_view (comparison : PtxvSim.ptxv_ir.FCmp) :
    importComparator (comparatorView comparison) = comparison := by
  cases comparison <;> rfl

theorem imported_plan_view (plan : comparison_select.Plan) :
    planView (importPlan plan) = observedComparisonPlan plan := by
  simp [planView, importPlan, observedComparisonPlan, comparator_view_import]

@[step] theorem actual_selection_program_refines_source_operands {R : Type}
    (readerInst : comparison_select.NodeReader R) (reader : R)
    (nodeOf : Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (compare : FCmp → U32 → U32 → Bool) (valuation : U32 → U32)
    (front : Front) (comparison : Op) (mode : FCmp)
    (destination : Reg) (first second : Val) (predicate : Pred) (x y : U32)
    (comparisonPC selectionPC : U32) (hmode : plainComparison comparison = some mode) :
    (do
      let plan ← comparison_select.from_pair readerInst front comparison
        (.Sel destination first second predicate) x y reader
      match plan with
      | none => ok none
      | some plan =>
          let parts ← PtxvSim.seg.comparison_program_parts ⟨importPlan plan, x, y, comparisonPC, selectionPC⟩
          ok (some parts)) ⦃ produced => match produced with
            | none => True
            | some parts => result compare valuation (observe parts) =
                selectedNodeBits compare valuation mode predicate x y first second nodeOf ⦄ := by
  step with actual_selection_preserves_all_node_valuations readerInst reader nodeOf hreader compare
    front comparison mode destination first second predicate x y hmode as ⟨produced, hproduced⟩
  cases produced with
  | none => simp
  | some plan =>
      step with actual_program_preserves_all_node_valuations compare valuation (importPlan plan)
        x y comparisonPC selectionPC as ⟨parts, hparts⟩
      rw [imported_plan_view] at hparts
      exact hparts.trans (hproduced valuation)

end Instruction.SegProgram
