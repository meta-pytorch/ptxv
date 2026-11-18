/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Refinement
import Instruction.SelectionBound.Composition

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.IEEEComparison

abbrev primitive := PtxvSim.seg.bound.IeeeComparison.Insts.Ptxv_simSegBoundComparisonPrimitive
abbrev primitiveCopy := PtxvSim.seg.bound.IeeeComparison.Insts.CoreMarkerCopy

def oracle : SegBound.Comparison :=
  fun mode first second flush => compare (SegProgram.importComparator mode) first second flush

@[simp] theorem source_ieee_primitive (request : PtxvSim.seg.ComparisonRequest) :
    PtxvSim.seg.bound.IeeeComparison.Insts.Ptxv_simSegBoundComparisonPrimitive.compare () request =
      .ok (compare request.comparison request.left request.right request.ftz) := by
  exact source_f32_cmp request.left request.right request.comparison request.ftz

theorem concrete_primitive_contract : SegBound.PrimitiveContract primitive () oracle := by
  intro request
  simp only [primitive,
    PtxvSim.seg.bound.IeeeComparison.Insts.Ptxv_simSegBoundComparisonPrimitive,
    source_ieee_primitive, oracle, SegProgram.comparator_import_view]

@[simp] theorem source_signed_zeros_compare_equal (flush : Bool) :
    PtxvSim.fp.f32_cmp 0#u32 0x80000000#u32 .Eq flush = .ok true := by
  rw [source_f32_cmp]
  cases flush <;> rfl

@[simp] theorem source_signed_zero_tie_not_greater (flush : Bool) :
    PtxvSim.fp.f32_cmp 0#u32 0x80000000#u32 .Gt flush = .ok false := by
  rw [source_f32_cmp]
  cases flush <;> rfl

end Instruction.IEEEComparison
