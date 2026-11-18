/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowModel

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

def contextView (context : control_flow.Context) : Context :=
  ⟨context.pc.val, context.count.val, context.cut, context.fold,
    context.returns.map (fun values => values.val.map UScalar.val)⟩

def RowPost (expected : List Nat) (result : alloc.vec.Vec Usize) : Prop := result.val.map UScalar.val = expected

theorem RowPost.length {expected : List Nat} {result : alloc.vec.Vec Usize}
    (post : RowPost expected result) : result.val.length = expected.length := by
  simpa only [List.length_map] using congrArg List.length post

theorem rust_pred_always (guard : Pred) : Pred.is_always guard = .ok (always guard) := by
  rcases guard with ⟨register, negative⟩
  cases register <;> cases negative <;> rfl

@[step]
theorem rust_append (initial : alloc.vec.Vec Usize) (values : Slice Usize)
    (space : initial.val.length + values.val.length ≤ Usize.max) :
    control_flow.append initial values ⦃ result => result.val = initial.val ++ values.val ⦄ := by
  unfold control_flow.append control_flow.append_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec Usize × Usize => values.val.length - state.2.val)
    (fun state => state.2.val ≤ values.val.length ∧ state.1.val = initial.val ++ values.val.take state.2.val)
  · rintro ⟨current, position⟩ ⟨positionBound, currentEq⟩
    have positionLe : position.val ≤ values.val.length := positionBound
    have currentValue : current.val = initial.val ++ values.val.take position.val := currentEq
    unfold control_flow.append_loop.body
    by_cases inside : position.val < values.val.length
    · have test : position < Slice.len values := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec values position inside as ⟨value, valueEq⟩
      have currentLength : current.val.length = initial.val.length + position.val := by
        rw [currentValue, List.length_append, List.length_take, Nat.min_eq_left positionLe]
      step with alloc.vec.Vec.push_spec current value (by omega) as ⟨next, nextEq⟩
      step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac) as ⟨nextPosition, nextValue⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [nextEq, currentValue, List.append_assoc, nextValue, List.take_succ_eq_append_getElem inside, valueEq]
    · have test : ¬position < Slice.len values := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      have completed : position.val = values.val.length := by omega
      simp only [currentValue, completed, List.take_length]
  · exact ⟨by scalar_tac, by simp⟩

@[step]
theorem rust_append_next (initial : alloc.vec.Vec Usize) (context : control_flow.Context)
    (space : initial.val.length + (next (contextView context)).length ≤ Usize.max) :
    control_flow.append_next initial context ⦃ result =>
      RowPost (initial.val.map UScalar.val ++ next (contextView context)) result ⦄ := by
  unfold control_flow.append_next
  by_cases pcInside : context.pc.val < context.count.val
  · have test : context.pc < context.count := by scalar_tac
    simp only [test, ite_true]
    step with Usize.add_spec (x := context.pc) (y := 1#usize) (by scalar_tac) as ⟨destination, destinationEq⟩
    by_cases nextInside : destination.val < context.count.val
    · have nextTest : destination < context.count := by scalar_tac
      have logicalInside : context.pc.val + 1 < context.count.val := by scalar_tac
      have capacity : initial.val.length < Usize.max := by
        simp only [next, contextView, logicalInside, ite_true, List.length_singleton] at space
        omega
      simp only [nextTest, ite_true]
      step with alloc.vec.Vec.push_spec initial destination capacity as ⟨result, resultEq⟩
      simp only [RowPost, resultEq, List.map_append, List.map_cons, List.map_nil, next, contextView,
        logicalInside, ite_true, destinationEq]
    · have nextTest : ¬destination < context.count := by scalar_tac
      have logicalOutside : ¬context.pc.val + 1 < context.count.val := by scalar_tac
      simp only [nextTest, ite_false, WP.spec_ok, RowPost, next, contextView,
        logicalOutside, ite_false, List.append_nil]
  · have test : ¬context.pc < context.count := by scalar_tac
    have logicalOutside : ¬context.pc.val + 1 < context.count.val := by omega
    simp only [test, ite_false, WP.spec_ok, RowPost, next, contextView,
      logicalOutside, ite_false, List.append_nil]

end Instruction.ControlFlow
