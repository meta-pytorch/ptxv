/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionProgram.Shape

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegProgram

theorem expected_preserves_all_node_valuations
    (compare : PtxvIr.FCmp → U32 → U32 → Bool)
    (valuation : U32 → U32) (plan : PtxvSim.ptxv_ir.comparison_select.Plan)
    (x y comparisonPC selectionPC : U32) :
    result compare valuation (expected plan x y comparisonPC selectionPC) =
      some (comparisonSelectionBits compare (planView plan) (valuation x) (valuation y)) := by
  cases hswap : plan.swap <;>
    cases hnegate : plan.negate <;>
    cases hcompare : compare (comparatorView plan.comparison) (valuation x) (valuation y) <;>
    simp [result, expected, runOps, instructionResult, opcodePair, operandBinding,
      registerBinding, registerKey, sourceBinding, xreg, yreg, preg, dreg,
      planView, comparisonSelectionBits, hswap, hnegate, hcompare,
      pairVec, tripleVec]

@[step] theorem actual_program_preserves_all_node_valuations
    (compare : PtxvIr.FCmp → U32 → U32 → Bool)
    (valuation : U32 → U32) (plan : PtxvSim.ptxv_ir.comparison_select.Plan)
    (x y comparisonPC selectionPC : U32) :
    PtxvSim.seg.comparison_program_parts ⟨plan, x, y, comparisonPC, selectionPC⟩ ⦃ parts =>
      result compare valuation (observe parts) =
        some (comparisonSelectionBits compare (planView plan) (valuation x) (valuation y)) ⦄ := by
  step with actual_program_shape plan x y comparisonPC selectionPC as ⟨parts, hparts⟩
  simpa [hparts] using
    expected_preserves_all_node_valuations compare valuation plan x y comparisonPC selectionPC

end Instruction.SegProgram
