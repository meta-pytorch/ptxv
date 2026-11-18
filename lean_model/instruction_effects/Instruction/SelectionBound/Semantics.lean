/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBound.Reads

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction.SegBound

theorem source_comparison_direct {C : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (self : Sources C)
    (compare : Comparison) (primitive : PrimitiveContract primitiveInst self.comparison compare)
    (operation : PtxvSim.ptxv_ir.Op) (eligible : eligibleOperation operation = true) :
    PtxvSim.seg.comparison_instruction
      (PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator primitiveInst) operation self =
      (do
        let (answer, _) ← PtxvSim.seg.comparison_instruction
          (SegBinding.callbacks compare self.inputs.args self.inputs.prior self.bindings) operation ()
        .ok (answer, self)) := by
  cases operation <;> try (solve | simp [eligibleOperation] at eligible)
  case Fsetp destination comparison a b negative absolute condition boolean ftz =>
    have both : directValue a = true ∧ directValue b = true := by
      simpa [eligibleOperation] using eligible
    rcases both with ⟨ha, hb⟩
    simp only [SegEvaluator.actual_evaluator_exact_callbacks, SegEvaluator.expected,
      PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator,
      source_direct_read primitiveInst self a ha, source_direct_read primitiveInst self b hb,
      source_predicate_read primitiveInst self, source_primitive_read primitiveInst self compare primitive,
      previous_direct_read compare self.inputs self.bindings,
      previous_predicate_read compare self.inputs self.bindings,
      previous_primitive_read compare self.inputs self.bindings,
      Std.bind_assoc, bind_ok, uncurry_apply_pair]
  case Sel destination a b predicate =>
    have both : directValue a = true ∧ directValue b = true := by
      simpa [eligibleOperation] using eligible
    rcases both with ⟨ha, hb⟩
    simp only [SegEvaluator.actual_evaluator_exact_callbacks, SegEvaluator.expected,
      PtxvSim.seg.bound.Sources.Insts.Ptxv_simSegComparisonEvaluator,
      source_predicate_read primitiveInst self,
      previous_predicate_read compare self.inputs self.bindings,
      Std.bind_assoc, bind_ok, uncurry_apply_pair]
    apply SegDriver.bind_continuation_congr
    intro selected
    cases selected <;>
      simp only [Bool.false_eq_true, ite_false, ite_true,
        source_direct_read primitiveInst self a ha, source_direct_read primitiveInst self b hb,
        previous_direct_read compare self.inputs self.bindings,
        Std.bind_assoc, bind_ok, uncurry_apply_pair]

theorem source_try_direct {C : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (copyInst : core.marker.Copy C)
    (primitive : C) (compare : Comparison) (contract : PrimitiveContract primitiveInst primitive compare)
    (instruction : PtxvSim.seg.SOp) (inputs : Inputs)
    (eligible : eligibleInstruction instruction = true) :
    PtxvSim.seg.bound.try_evaluate primitiveInst copyInst instruction inputs primitive =
      (do let bits ← SegBinding.executeInstruction compare inputs.args inputs.prior instruction
          .ok (some bits)) := by
  cases hkind : instruction.kind with
  | Guard predicate negative new old => simp [eligibleInstruction, hkind] at eligible
  | DivRn a b => simp [eligibleInstruction, hkind] at eligible
  | Inst operation destination bindings constants =>
    have hop : eligibleOperation operation = true := by
      simpa [eligibleInstruction, hkind] using eligible
    simp only [PtxvSim.seg.bound.try_evaluate, hkind, source_eligible, hop,
      bind_ok, ite_true, SegDriver.deref_eq_slice]
    rw [source_comparison_direct primitiveInst ⟨bindings.slice, inputs, primitive⟩ compare contract operation hop]
    unfold SegBinding.executeInstruction
    rw [hkind]
    simp only [Std.bind_assoc]
    cases operation <;> try (solve | simp [eligibleOperation] at hop)
    all_goals simp only [SegEvaluator.actual_evaluator_exact_callbacks, SegEvaluator.expected,
      previous_direct_read, previous_predicate_read, previous_primitive_read,
      Std.bind_assoc, bind_ok, uncurry_apply_pair]
    all_goals
      apply SegDriver.bind_continuation_congr
      intro firstBits
    case Inst.Fsetp =>
      apply SegDriver.bind_continuation_congr
      intro secondBits
      simp [lift]
    case Inst.Sel =>
      apply SegDriver.bind_continuation_congr
      intro chosen
      simp [lift]


theorem source_direct_dispatch {C F : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (copyInst : core.marker.Copy C)
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (self : Evaluator C F)
    (compare : Comparison) (contract : PrimitiveContract primitiveInst self.comparison compare)
    (instruction : PtxvSim.seg.SOp) (args : Slice U32) (prior : Slice U64)
    (eligible : eligibleInstruction instruction = true) :
    PtxvSim.seg.bound.Evaluator.Insts.Ptxv_simSegProgramProgramEvaluator.evaluate
      primitiveInst copyInst fallbackInst self instruction args prior =
      (do let bits ← SegBinding.executeInstruction compare args prior instruction
          .ok (.Ok bits, self)) := by
  simp only [PtxvSim.seg.bound.Evaluator.Insts.Ptxv_simSegProgramProgramEvaluator.evaluate,
    source_try_direct primitiveInst copyInst self.comparison compare contract instruction ⟨args, prior⟩ eligible,
    Std.bind_assoc, bind_ok]

theorem source_fallback_dispatch {C F : Type}
    (primitiveInst : PtxvSim.seg.bound.ComparisonPrimitive C) (copyInst : core.marker.Copy C)
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (self : Evaluator C F)
    (instruction : PtxvSim.seg.SOp) (args : Slice U32) (prior : Slice U64)
    (unsupported : eligibleInstruction instruction = false) :
    PtxvSim.seg.bound.Evaluator.Insts.Ptxv_simSegProgramProgramEvaluator.evaluate
      primitiveInst copyInst fallbackInst self instruction args prior =
      (do let (answer, next) ← fallbackInst.evaluate self.fallback instruction ⟨args, prior⟩
          .ok (answer, {self with fallback := next})) := by
  simp only [PtxvSim.seg.bound.Evaluator.Insts.Ptxv_simSegProgramProgramEvaluator.evaluate,
    source_try_unsupported primitiveInst copyInst instruction ⟨args, prior⟩ self.comparison unsupported,
    bind_ok]

end Instruction.SegBound
