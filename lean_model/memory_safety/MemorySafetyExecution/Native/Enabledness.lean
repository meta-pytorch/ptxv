/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core
import MemorySafety.SourceVectors
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem terminal_loop_spec (state : CoreState) (terminal : Bool) (actor : Usize) :
    ptxv_memory_write.execution_core.terminal_loop state terminal actor
      ⦃ output => output = (terminal &&
        (state.threads.val.drop actor.val).all (fun thread => !thread.running)) ⦄ := by
  unfold ptxv_memory_write.execution_core.terminal_loop
  apply loop.spec_decr_nat
    (fun current : Bool × Usize => state.threads.val.length - current.2.val)
    (fun current : Bool × Usize =>
      (current.1 && (state.threads.val.drop current.2.val).all (fun thread => !thread.running)) =
        (terminal && (state.threads.val.drop actor.val).all (fun thread => !thread.running)))
  · rintro ⟨current, index⟩ invariant
    unfold ptxv_memory_write.execution_core.terminal_loop.body
    by_cases inside : index.val < state.threads.val.length
    · have test : index < alloc.vec.Vec.len state.threads := by scalar_tac
      simp only [test, ite_true]
      cases current with
      | false => simpa only [Bool.false_eq_true, ite_false, WP.spec_ok, Bool.false_and] using invariant
      | true =>
        simp only [ite_true]
        step as ⟨entry, entryEq⟩
        step as ⟨next, nextVal⟩
        refine ⟨?_, by scalar_tac⟩
        rw [nextVal]
        rw [List.drop_eq_getElem_cons inside] at invariant
        have boolNot (value : Bool) : decide (¬value = true) = !value := by
          cases value <;> rfl
        simpa only [List.all_cons, Bool.true_and, entryEq, boolNot] using invariant
    · have test : ¬index < alloc.vec.Vec.len state.threads := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      simpa only [List.drop_eq_nil_of_le (show state.threads.val.length ≤ index.val by omega),
        List.all_nil, Bool.and_true] using invariant
  · rfl

theorem source_terminal (state : CoreState) :
    ptxv_memory_write.execution_core.terminal state =
      .ok (state.threads.val.all (fun thread => !thread.running)) := by
  have specification := terminal_loop_spec state true 0#usize
  obtain ⟨result, computed, same⟩ := MemoryWrite.spec_returns _ _ specification
  simpa [ptxv_memory_write.execution_core.terminal, same] using computed

end MemorySafetyExecution.Native
