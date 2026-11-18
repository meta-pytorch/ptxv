/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowLookup

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.ControlFlow

@[step]
theorem rust_graph_successors (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool))
    (returns : Slice (alloc.vec.Vec Usize))
    (space : ∀ pc < code.val.length, (graphRows code.val cut fold.val returns.val)[pc]!.length ≤ Usize.max) :
    control_flow.graph_successors code cut fold returns ⦃ result =>
      result.val.map (fun row => row.val.map UScalar.val) = graphRows code.val cut fold.val returns.val ⦄ := by
  unfold control_flow.graph_successors control_flow.graph_successors_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (alloc.vec.Vec Usize) × Usize => code.val.length - state.2.val)
    (fun state => state.2.val ≤ code.val.length ∧
      state.1.val.map (fun row => row.val.map UScalar.val) =
        (graphRows code.val cut fold.val returns.val).take state.2.val)
  · rintro ⟨current, pc⟩ ⟨pcBound, currentEq⟩
    have pcLe : pc.val ≤ code.val.length := pcBound
    have currentValue : current.val.map (fun row => row.val.map UScalar.val) =
        (graphRows code.val cut fold.val returns.val).take pc.val := currentEq
    have currentLength : current.val.length = pc.val := by
      have lengthEq := congrArg List.length currentValue
      simp only [List.length_map, List.length_take, graph_rows_length, Nat.min_eq_left pcLe] at lengthEq
      exact lengthEq
    unfold control_flow.graph_successors_loop.body
    by_cases inside : pc.val < code.val.length
    · have test : pc < Slice.len code := by scalar_tac
      simp only [test, ite_true]
      step with rust_fold_at fold pc as ⟨decision, decisionValue⟩
      step with rust_returns_at returns pc as ⟨destinations, destinationsValue⟩
      step with Slice.index_usize_spec code pc inside as ⟨instruction, instructionValue⟩
      have contextEqual : contextView
          { pc, count := Slice.len code, cut, fold := decision, returns := destinations } =
          rowContext code.val.length cut fold.val returns.val pc.val := by
        simp only [contextView, rowContext, decisionValue, destinationsValue, Slice.len_val,
          Option.map_map, Function.comp_def, alloc.vec.Vec.deref, Slice.from_val, Slice.length]
      have rowSpace : (successors instruction.op instruction.guard
          (contextView { pc, count := Slice.len code, cut, fold := decision, returns := destinations })).length ≤ Usize.max := by
        rw [contextEqual, instructionValue]
        simpa only [graph_row_at code.val cut fold.val returns.val pc.val inside] using space pc.val inside
      step with rust_successors instruction.op instruction.guard
        { pc, count := Slice.len code, cut, fold := decision, returns := destinations } rowSpace
        as ⟨row, rowPost⟩
      have rowValue : row.val.map UScalar.val = (graphRows code.val cut fold.val returns.val)[pc.val]! := by
        rw [graph_row_at code.val cut fold.val returns.val pc.val inside]
        simpa only [RowPost, instructionValue, contextEqual] using rowPost
      step with alloc.vec.Vec.push_spec current row (by have := code.property; omega)
        as ⟨next, nextValue⟩
      step with Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac) as ⟨nextPc, nextPcValue⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      have outputInside : pc.val < (graphRows code.val cut fold.val returns.val).length := by
        simpa only [graph_rows_length] using inside
      rw [nextValue, List.map_append, List.map_cons, List.map_nil, currentValue, rowValue, nextPcValue,
        List.take_succ_eq_append_getElem outputInside,
        List.Inhabited_getElem_eq_getElem! _ _ outputInside]
    · have test : ¬pc < Slice.len code := by scalar_tac
      have completed : pc.val = code.val.length := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [completed, ←graph_rows_length code.val cut fold.val returns.val, List.take_length] using currentValue
  · exact ⟨by scalar_tac, by simp⟩

end Instruction.ControlFlow
