/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowPrimitives

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

@[step]
theorem rust_range (initial : alloc.vec.Vec Usize) (count : Usize)
    (space : initial.val.length + count.val ≤ Usize.max) :
    control_flow.successors_loop count initial 0#usize ⦃ result =>
      RowPost (initial.val.map UScalar.val ++ List.range count.val) result ⦄ := by
  unfold control_flow.successors_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec Usize × Usize => count.val - state.2.val)
    (fun state => state.2.val ≤ count.val ∧
      RowPost (initial.val.map UScalar.val ++ List.range state.2.val) state.1)
  · rintro ⟨current, position⟩ ⟨positionBound, currentPost⟩
    have positionLe : position.val ≤ count.val := positionBound
    have post : RowPost (initial.val.map UScalar.val ++ List.range position.val) current := currentPost
    have currentLength : current.val.length = initial.val.length + position.val := by
      simpa only [List.length_append, List.length_map, List.length_range] using post.length
    change current.val.map UScalar.val = initial.val.map UScalar.val ++ List.range position.val at post
    unfold control_flow.successors_loop.body
    by_cases inside : position.val < count.val
    · have test : position < count := by scalar_tac
      simp only [test, ite_true]
      step with alloc.vec.Vec.push_spec current position (by omega) as ⟨next, nextEq⟩
      step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac) as ⟨nextPosition, nextValue⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      simp only [RowPost, nextEq, List.map_append, List.map_cons, List.map_nil, post,
        nextValue, List.range_succ, List.append_assoc]
    · have test : ¬position < count := by scalar_tac
      have completed : position.val = count.val := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [RowPost, completed] using post
  · exact ⟨by scalar_tac, by simp [RowPost]⟩

end Instruction.ControlFlow
