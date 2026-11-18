/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionHardware.Model

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SelectionHardware

open IEEEComparison SegProgram

@[simp] theorem source_hardware_comparator
    (mode : PtxvSim.ptxv_ir.FCmp) (first second : U32) :
    PtxvSim.seg.hardware.IeeeComparator.Insts.Ptxv_irComparison_selectComparator.compare
      () mode first second = .ok (IEEEComparison.compare mode first second false) := by
  exact source_f32_cmp first second mode false

@[simp] theorem source_hardware_value
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (first second : U32) :
    PtxvSim.seg.hardware.evaluate plan first second = .ok (hardwareBits plan first second) := by
  simp only [PtxvSim.seg.hardware.evaluate, PtxvSim.ptxv_ir.comparison_select.evaluate,
    source_hardware_comparator, bind_ok]
  simp only [hardwareBits, comparisonSelectionBits, planView, oracle, comparator_import_view]
  split <;> split <;> simp_all

@[step] theorem source_hardware_value_spec
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (first second : U32) :
    PtxvSim.seg.hardware.evaluate plan first second ⦃ output =>
      output = hardwareBits plan first second ⦄ := by
  rw [source_hardware_value]
  simp

end Instruction.SelectionHardware
