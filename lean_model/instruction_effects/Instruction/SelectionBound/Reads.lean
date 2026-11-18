/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBound.Eligibility

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegBound

@[simp] theorem source_direct_read {C : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (self : Sources C)
    (value : PtxvSim.ptxv_ir.Val) (direct : directValue value = true) :
    PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator.read primitiveInst self value =
      (do let bits ← valueBits self.inputs self.bindings value; .ok (bits, self)) := by
  cases value <;>
    simp_all [directValue, PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator.read,
      valueBits, SegBinding.source_register_bits_exact, lift, Std.bind_assoc]

@[simp] theorem source_predicate_read {C : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (self : Sources C)
    (predicate : PtxvSim.ptxv_ir.Pred) :
    PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator.predicate primitiveInst self predicate =
      (do let bits ← predicateBits self.inputs self.bindings predicate; .ok (bits, self)) := by
  simp [PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator.predicate,
    predicateBits, SegBinding.source_read_predicate_exact]

@[simp] theorem source_primitive_read {C : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (self : Sources C)
    (compare : Comparison) (primitive : PrimitiveContract primitiveInst self.comparison compare)
    (request : PtxvSim.seg.ComparisonRequest) :
    PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator.compare primitiveInst self request =
      .ok (compare (SegProgram.comparatorView request.comparison) request.left request.right request.ftz, self) := by
  simp [PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator.compare, primitive request]

@[simp] theorem previous_direct_read (compare : Comparison) (inputs : Inputs)
    (bindings : Slice SegBinding.Binding) (value : PtxvSim.ptxv_ir.Val) :
    (SegBinding.callbacks compare inputs.args inputs.prior bindings).read () value =
      (do let bits ← valueBits inputs bindings value; .ok (bits, ())) := by
  cases value <;> simp [SegBinding.callbacks, valueBits, SegBinding.source_register_bits_exact,
    Std.bind_assoc]

@[simp] theorem previous_predicate_read (compare : Comparison) (inputs : Inputs)
    (bindings : Slice SegBinding.Binding) (predicate : PtxvSim.ptxv_ir.Pred) :
    (SegBinding.callbacks compare inputs.args inputs.prior bindings).predicate () predicate =
      (do let bits ← predicateBits inputs bindings predicate; .ok (bits, ())) := by
  simp [SegBinding.callbacks, predicateBits, SegBinding.source_read_predicate_exact]

@[simp] theorem previous_primitive_read (compare : Comparison) (inputs : Inputs)
    (bindings : Slice SegBinding.Binding) (request : PtxvSim.seg.ComparisonRequest) :
    (SegBinding.callbacks compare inputs.args inputs.prior bindings).compare () request =
      .ok (compare (SegProgram.comparatorView request.comparison) request.left request.right request.ftz, ()) := rfl

end Instruction.SegBound
