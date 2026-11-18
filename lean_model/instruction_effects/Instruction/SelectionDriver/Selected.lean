/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionDriver.Refinement
import Instruction.SelectionBindings.Composition

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction.SegDriver

open SegProgram

abbrev Comparison := SegBinding.Comparison

def selectedCallbacks (compare : Comparison) : PtxvSim.seg.program.ProgramEvaluator Unit where
  evaluate state instruction args prior := do
    let bits ← SegBinding.executeInstruction compare args prior instruction
    .ok (.Ok bits, state)

def selectedProgram (valuation : U32 → U32) (parts : SegBinding.NativeParts) :
    PtxvSim.seg.program.Program :=
  ⟨SegBinding.arguments valuation parts.1, parts.2.1.slice, parts.2.2⟩

def executeSelected (compare : Comparison) (valuation : U32 → U32)
    (parts : SegBinding.NativeParts) (values : alloc.vec.Vec U64) : Result (Output Unit) :=
  PtxvSim.seg.program.execute (selectedCallbacks compare) (selectedProgram valuation parts) values ()

def projected {E : Type} (computation : Result (Output E)) : Result U32 := do
  let (answer, _, _) ← computation
  match answer with
  | .Ok bits => .ok bits
  | .Err _ => .fail .undef

def oneValue (bits : U64) : alloc.vec.Vec U64 :=
  alloc.vec.Vec.from [bits] (by simp; scalar_tac)

@[simp] theorem source_push_empty (bits : U64) :
    alloc.vec.Vec.push emptyValues bits = .ok (oneValue bits) := by
  have bound : (1 : Nat) ≤ U32.max := by scalar_tac
  simp [alloc.vec.Vec.push, emptyValues, oneValue, bound]

@[simp] theorem source_push_one (first second : U64) :
    alloc.vec.Vec.push (oneValue first) second = .ok (pairVec first second) := by
  have bound : (2 : Nat) ≤ U32.max := by scalar_tac
  simp [alloc.vec.Vec.push, oneValue, pairVec, bound]

@[simp] theorem empty_slice : emptyValues.slice = SegBinding.emptyPrior := rfl
@[simp] theorem one_value_slice (bits : U64) : (oneValue bits).slice = SegBinding.onePrior bits := rfl
theorem source_selected_pair_projection (compare : Comparison) (valuation : U32 → U32)
    (parts : SegBinding.NativeParts) (values : alloc.vec.Vec U64)
    (two : parts.2.1.val.length = 2) :
    projected (executeSelected compare valuation parts values) =
      SegBinding.executeProducedPair compare valuation parts := by
  obtain ⟨first, second, hop⟩ := List.length_eq_two.mp two
  unfold executeSelected
  rw [source_execute_exact]
  simp only [expected, selectedProgram, SegBinding.binding_slice_val, hop, interpret,
    selectedCallbacks, Std.bind_assoc, bind_ok, source_push_empty, source_push_one,
    empty_slice, one_value_slice, finish, projected, uncurry_apply_pair]
  unfold SegBinding.executeProducedPair
  rw [hop]
  apply bind_continuation_congr
  intro firstBits
  apply bind_continuation_congr
  intro secondBits
  rw [SegBinding.source_read_sval_exact]
  simp only [SegBinding.twoPrior, Slice.from_val, pairVec, alloc.vec.Vec.from_val]

theorem projected_success_spec {E : Type} (computation : Result (Output E)) (bits : U32)
    (success : projected computation = .ok bits) :
    computation ⦃ output => output.1 = .Ok bits ⦄ := by
  cases computation using Result.cases with
  | ret output =>
    rcases output with ⟨answer, values, evaluator⟩
    cases answer with
    | Err error => simp [projected] at success
    | Ok answer =>
      simp only [projected, bind_ok, uncurry_apply_pair] at success
      have same : answer = bits := Result.ok_injective success
      subst answer
      simp
  | vis effect continuation =>
    simp only [projected, bind_vis] at success
    exact False.elim (vis_not_ok success)
  | div => simp [projected] at success

@[step] theorem actual_constructor_driver_execution (compare : Comparison) (valuation : U32 → U32)
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right comparisonPC selectionPC : U32)
    (values : alloc.vec.Vec U64) :
    PtxvSim.seg.comparison_program_parts ⟨plan, left, right, comparisonPC, selectionPC⟩
      ⦃ parts => SegBinding.programReady (observe parts) ∧
        executeSelected compare valuation parts values ⦃ output => output.1 = .Ok
          (comparisonSelectionBits (fun mode a b => compare mode a b false)
            (planView plan) (valuation left) (valuation right)) ⦄ ⦄ := by
  apply WP.spec_mono (WP.spec_and
    (actual_program_shape plan left right comparisonPC selectionPC)
    (SegBinding.actual_constructor_bound_execution compare valuation plan left right comparisonPC selectionPC))
  intro parts specification
  rcases specification with ⟨shape, ready, pairResult⟩
  refine ⟨ready, projected_success_spec _ _ ?_⟩
  have two : parts.2.1.val.length = 2 := by
    have hlength := congrArg (fun p : Parts => p.2.1.length) shape
    simpa [observe, SegProgram.expected, opcodePair] using hlength
  rw [source_selected_pair_projection compare valuation parts values two, pairResult]

end Instruction.SegDriver
