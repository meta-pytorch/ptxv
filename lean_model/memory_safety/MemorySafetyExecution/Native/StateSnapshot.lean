/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core
import ValueFlow.SourceSnapshot
import MemorySafety.SourceVectors
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem snapshot_threads (state : CoreState)
    (threads : alloc.vec.Vec ptxv_memory_write.execution_core.ThreadState) (index : Usize)
    (bounded : index.val ≤ state.threads.val.length)
    (copied : threads.val = state.threads.val.take index.val) :
    ptxv_memory_write.execution_core.snapshot_state_loop0 state threads index
      ⦃ output => output = (state.allocations, state.threads) ⦄ := by
  unfold ptxv_memory_write.execution_core.snapshot_state_loop0
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec ptxv_memory_write.execution_core.ThreadState × Usize =>
      state.threads.val.length - current.2.val)
    (fun current : alloc.vec.Vec ptxv_memory_write.execution_core.ThreadState × Usize =>
      current.2.val ≤ state.threads.val.length ∧
      current.1.val = state.threads.val.take current.2.val)
  · rintro ⟨threads, index⟩ ⟨bounded, copied⟩
    change index.val ≤ state.threads.val.length at bounded
    change threads.val = state.threads.val.take index.val at copied
    unfold ptxv_memory_write.execution_core.snapshot_state_loop0.body
    by_cases inside : index.val < state.threads.val.length
    · have test : index < alloc.vec.Vec.len state.threads := by scalar_tac
      simp only [test, ite_true]
      step as ⟨entry, entryEq⟩
      rw [ValueFlow.Source.source_snapshot_registers]
      simp only [bind_ok]
      have copiedLength : threads.val.length = index.val := by
        simp [copied, List.length_take, Nat.min_eq_left bounded]
      step with alloc.vec.Vec.push_spec threads _
        (by have := alloc.vec.Vec.len_ineq state.threads; omega) as ⟨updated, appended⟩
      step as ⟨next, nextVal⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [appended, nextVal, List.take_succ_eq_append_getElem inside, copied]
      exact congrArg (fun value => state.threads.val.take index.val ++ [value]) entryEq
    · have test : ¬index < alloc.vec.Vec.len state.threads := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      have same : threads = state.threads := by
        apply alloc.vec.Vec.ext
        rw [copied]
        exact List.take_of_length_le (by omega)
      simp only [same]
  · exact ⟨bounded, copied⟩

private theorem snapshot_allocations
    (values allocations : alloc.vec.Vec ptxv_memory_write.execution_core.AllocationBytes)
    (index : Usize) (bounded : index.val ≤ values.val.length)
    (copied : allocations.val = values.val.take index.val) :
    ptxv_memory_write.execution_core.snapshot_state_loop1 values index allocations
      ⦃ output => output = values ⦄ := by
  unfold ptxv_memory_write.execution_core.snapshot_state_loop1
  apply loop.spec_decr_nat
    (fun current : Usize × alloc.vec.Vec ptxv_memory_write.execution_core.AllocationBytes =>
      values.val.length - current.1.val)
    (fun current : Usize × alloc.vec.Vec ptxv_memory_write.execution_core.AllocationBytes =>
      current.1.val ≤ values.val.length ∧ current.2.val = values.val.take current.1.val)
  · rintro ⟨index, allocations⟩ ⟨bounded, copied⟩
    change index.val ≤ values.val.length at bounded
    change allocations.val = values.val.take index.val at copied
    unfold ptxv_memory_write.execution_core.snapshot_state_loop1.body
    by_cases inside : index.val < values.val.length
    · have test : index < alloc.vec.Vec.len values := by scalar_tac
      simp only [test, ite_true]
      step as ⟨entry, entryEq⟩
      rw [MemorySafety.Source.vec_clone_exact core.clone.CloneU8 (by intro value; rfl)]
      simp only [bind_ok]
      rw [MemorySafety.Source.vec_clone_exact core.clone.CloneBool (by intro value; rfl)]
      simp only [bind_ok]
      have copiedLength : allocations.val.length = index.val := by
        simp [copied, List.length_take, Nat.min_eq_left bounded]
      step with alloc.vec.Vec.push_spec allocations _
        (by have := alloc.vec.Vec.len_ineq values; omega) as ⟨updated, appended⟩
      step as ⟨next, nextVal⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [appended, nextVal, List.take_succ_eq_append_getElem inside, copied]
      exact congrArg (fun value => values.val.take index.val ++ [value]) entryEq
    · have test : ¬index < alloc.vec.Vec.len values := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      apply alloc.vec.Vec.ext
      rw [copied]
      exact List.take_of_length_le (by omega)
  · exact ⟨bounded, copied⟩

theorem source_snapshot_state (state : CoreState) :
    ptxv_memory_write.execution_core.snapshot_state state = .ok state := by
  have threadSpec := snapshot_threads state (alloc.vec.Vec.new _) 0#usize
    (by scalar_tac) (by simp)
  obtain ⟨threadPair, threadRun, sameThreads⟩ := MemoryWrite.spec_returns _ _ threadSpec
  have allocationSpec := snapshot_allocations state.allocations (alloc.vec.Vec.new _) 0#usize
    (by scalar_tac) (by simp)
  obtain ⟨allocationCopy, allocationRun, sameAllocations⟩ := MemoryWrite.spec_returns _ _ allocationSpec
  unfold ptxv_memory_write.execution_core.snapshot_state
  rw [threadRun]
  simp only [sameThreads, bind_ok]
  change Aeneas.Std.bind
    (ptxv_memory_write.execution_core.snapshot_state_loop1 state.allocations
      0#usize (alloc.vec.Vec.new ptxv_memory_write.execution_core.AllocationBytes))
    (fun allocations => Result.ok ({ threads := state.threads, allocations } : CoreState)) = .ok state
  rw [allocationRun]
  simp only [bind_ok, sameAllocations]

end MemorySafetyExecution.Native
