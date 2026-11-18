/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBound.Semantics

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction.SegBound

open SegProgram

def executeBound {C F : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (copyInst : core.marker.Copy C)
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (self : Evaluator C F)
    (valuation : U32 → U32) (parts : SegBinding.NativeParts) (values : alloc.vec.Vec U64) :
    Result (SegDriver.Output (Evaluator C F)) :=
  PtxvSim.seg.program.execute
    (PtxvSim.seg.bound.Evaluator.Insts.Ptxv_simSegProgramProgramEvaluator primitiveInst copyInst fallbackInst)
    (SegDriver.selectedProgram valuation parts) values self

theorem source_bound_pair_state {C F : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (copyInst : core.marker.Copy C)
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (self : Evaluator C F)
    (compare : Comparison) (contract : PrimitiveContract primitiveInst self.comparison compare)
    (valuation : U32 → U32) (parts : SegBinding.NativeParts) (values : alloc.vec.Vec U64)
    (two : parts.2.1.val.length = 2)
    (eligible : ∀ instruction ∈ parts.2.1.val, eligibleInstruction instruction = true) :
    executeBound primitiveInst copyInst fallbackInst self valuation parts values =
      (do let (answer, scratch, _) ← SegDriver.executeSelected compare valuation parts values
          .ok (answer, scratch, self)) := by
  obtain ⟨first, second, hop⟩ := List.length_eq_two.mp two
  have hf : eligibleInstruction first = true := eligible first (by simp [hop])
  have hs : eligibleInstruction second = true := eligible second (by simp [hop])
  unfold executeBound SegDriver.executeSelected
  rw [SegDriver.source_execute_exact, SegDriver.source_execute_exact]
  simp only [SegDriver.expected, SegDriver.selectedProgram, SegBinding.binding_slice_val, hop,
    SegDriver.interpret,
    PtxvSim.seg.bound.Evaluator.Insts.Ptxv_simSegProgramProgramEvaluator,
    source_direct_dispatch primitiveInst copyInst fallbackInst self compare contract first _ _ hf,
    SegDriver.selectedCallbacks, Std.bind_assoc, bind_ok, uncurry_apply_pair]
  apply SegDriver.bind_continuation_congr
  intro firstBits
  simp only [SegDriver.source_push_empty, bind_ok, SegDriver.one_value_slice,
    source_direct_dispatch primitiveInst copyInst fallbackInst self compare contract second _ _ hs,
    Std.bind_assoc, bind_ok, uncurry_apply_pair]
  apply SegDriver.bind_continuation_congr
  intro secondBits
  simp only [SegDriver.source_push_one, bind_ok, SegDriver.finish,
    Std.bind_assoc, bind_ok, uncurry_apply_pair]

@[step] theorem actual_constructor_bound_driver_execution {C F : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (copyInst : core.marker.Copy C)
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (self : Evaluator C F)
    (compare : Comparison) (contract : PrimitiveContract primitiveInst self.comparison compare)
    (valuation : U32 → U32) (plan : PtxvSim.ptxv_ir.comparison_select.Plan)
    (left right comparisonPC selectionPC : U32) (values : alloc.vec.Vec U64) :
    PtxvSim.seg.comparison_program_parts ⟨plan, left, right, comparisonPC, selectionPC⟩
      ⦃ parts => SegBinding.programReady (observe parts) ∧
        executeBound primitiveInst copyInst fallbackInst self valuation parts values ⦃ output =>
          output.1 = .Ok (comparisonSelectionBits (fun mode a b => compare mode a b false)
            (planView plan) (valuation left) (valuation right)) ∧ output.2.2 = self ⦄ ⦄ := by
  apply WP.spec_mono (WP.spec_and
    (WP.spec_and
      (actual_program_shape plan left right comparisonPC selectionPC)
      (actual_constructor_eligibility plan left right comparisonPC selectionPC))
    (SegDriver.actual_constructor_driver_execution compare valuation plan left right comparisonPC selectionPC values))
  intro parts specification
  rcases specification with ⟨⟨shape, eligible⟩, ready, execution⟩
  refine ⟨ready, ?_⟩
  have two : parts.2.1.val.length = 2 := by
    have hlength := congrArg (fun p : Parts => p.2.1.length) shape
    simpa [observe, SegProgram.expected, opcodePair] using hlength
  rw [source_bound_pair_state primitiveInst copyInst fallbackInst self compare contract valuation parts values two eligible]
  step with execution as ⟨answer, scratch, houtput⟩
  exact houtput

end Instruction.SegBound
