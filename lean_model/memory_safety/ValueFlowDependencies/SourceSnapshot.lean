/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlowDependencies.SourceTypes
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlowDependencies.Source

private theorem snapshot_metadata_loop (values : Slice Metadata)
    (result : alloc.vec.Vec Metadata) (index : Usize)
    (bounded : index.val ≤ values.val.length)
    (copied : result.val = values.val.take index.val) :
    value_flow_dependencies.snapshot_metadata_loop values result index
      ⦃ output => output.val = values.val ⦄ := by
  unfold value_flow_dependencies.snapshot_metadata_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec Metadata × Usize => values.val.length - current.2.val)
    (fun current : alloc.vec.Vec Metadata × Usize =>
      current.2.val ≤ values.val.length ∧ current.1.val = values.val.take current.2.val)
  · rintro ⟨result, index⟩ ⟨bounded, copied⟩
    change index.val ≤ values.val.length at bounded
    change result.val = values.val.take index.val at copied
    unfold value_flow_dependencies.snapshot_metadata_loop.body
    by_cases inside : index.val < values.val.length
    · have test : index < Slice.len values := by scalar_tac
      simp only [test, ite_true]
      step as ⟨entry, entryEq⟩
      have copiedLength : result.val.length = index.val := by
        simp [copied, List.length_take, Nat.min_eq_left bounded]
      step with alloc.vec.Vec.push_spec result _
        (by have := Slice.length_ineq values; omega) as ⟨updated, appended⟩
      step as ⟨next, nextVal⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [appended, nextVal, List.take_succ_eq_append_getElem inside, copied]
      exact congrArg (fun value => values.val.take index.val ++ [value]) entryEq
    · have test : ¬index < Slice.len values := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      rw [copied]
      exact List.take_of_length_le (by omega)
  · exact ⟨bounded, copied⟩

theorem source_snapshot_metadata (values : alloc.vec.Vec Metadata) :
    value_flow_dependencies.snapshot_metadata (alloc.vec.Vec.deref values) = .ok values := by
  have spec := snapshot_metadata_loop (alloc.vec.Vec.deref values) (alloc.vec.Vec.new _) 0#usize
    (by scalar_tac) (by simp)
  obtain ⟨copied, run, valuesEq⟩ := MemoryWrite.spec_returns _ _ spec
  have same : copied = values := by
    apply alloc.vec.Vec.ext
    simpa only [alloc.vec.Vec.deref, Slice.from_val] using valuesEq
  simpa only [value_flow_dependencies.snapshot_metadata, same] using run

theorem source_snapshot_registers (registers : Registers) :
    value_flow_dependencies.snapshot_registers registers = .ok registers := by
  simp only [value_flow_dependencies.snapshot_registers, source_snapshot_metadata, bind_ok]

private theorem snapshot_threads (state : State) (threads : alloc.vec.Vec Registers) (index : Usize)
    (bounded : index.val ≤ state.threads.val.length)
    (copied : threads.val = state.threads.val.take index.val) :
    value_flow_dependencies.snapshot_state_loop0 state threads index
      ⦃ output => output = (state.memory, state.threads) ⦄ := by
  unfold value_flow_dependencies.snapshot_state_loop0
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec Registers × Usize => state.threads.val.length - current.2.val)
    (fun current : alloc.vec.Vec Registers × Usize =>
      current.2.val ≤ state.threads.val.length ∧ current.1.val = state.threads.val.take current.2.val)
  · rintro ⟨threads, index⟩ ⟨bounded, copied⟩
    change index.val ≤ state.threads.val.length at bounded
    change threads.val = state.threads.val.take index.val at copied
    unfold value_flow_dependencies.snapshot_state_loop0.body
    by_cases inside : index.val < state.threads.val.length
    · have test : index < alloc.vec.Vec.len state.threads := by scalar_tac
      simp only [test, ite_true]
      step as ⟨entry, entryEq⟩
      rw [source_snapshot_registers]
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

private theorem snapshot_memory
    (values memory : alloc.vec.Vec (alloc.vec.Vec Metadata)) (index : Usize)
    (bounded : index.val ≤ values.val.length)
    (copied : memory.val = values.val.take index.val) :
    value_flow_dependencies.snapshot_state_loop1 values memory index
      ⦃ output => output = values ⦄ := by
  unfold value_flow_dependencies.snapshot_state_loop1
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (alloc.vec.Vec Metadata) × Usize => values.val.length - current.2.val)
    (fun current : alloc.vec.Vec (alloc.vec.Vec Metadata) × Usize =>
      current.2.val ≤ values.val.length ∧ current.1.val = values.val.take current.2.val)
  · rintro ⟨memory, index⟩ ⟨bounded, copied⟩
    change index.val ≤ values.val.length at bounded
    change memory.val = values.val.take index.val at copied
    unfold value_flow_dependencies.snapshot_state_loop1.body
    by_cases inside : index.val < values.val.length
    · have test : index < alloc.vec.Vec.len values := by scalar_tac
      simp only [test, ite_true]
      step as ⟨entry, entryEq⟩
      rw [source_snapshot_metadata]
      simp only [bind_ok]
      have copiedLength : memory.val.length = index.val := by
        simp [copied, List.length_take, Nat.min_eq_left bounded]
      step with alloc.vec.Vec.push_spec memory _
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

theorem source_snapshot_state (state : State) :
    value_flow_dependencies.snapshot_state state = .ok state := by
  have threadSpec := snapshot_threads state (alloc.vec.Vec.new _) 0#usize (by scalar_tac) (by simp)
  obtain ⟨threadPair, threadRun, sameThreads⟩ := MemoryWrite.spec_returns _ _ threadSpec
  have memorySpec := snapshot_memory state.memory (alloc.vec.Vec.new _) 0#usize (by scalar_tac) (by simp)
  obtain ⟨memory, memoryRun, sameMemory⟩ := MemoryWrite.spec_returns _ _ memorySpec
  unfold value_flow_dependencies.snapshot_state
  rw [threadRun]
  simp only [sameThreads, bind_ok]
  change Aeneas.Std.bind
    (value_flow_dependencies.snapshot_state_loop1 state.memory (alloc.vec.Vec.new (alloc.vec.Vec Metadata)) 0#usize)
    (fun memory => Result.ok ({ threads := state.threads, memory } : State)) = .ok state
  rw [memoryRun]
  simp only [bind_ok, sameMemory]

end ValueFlowDependencies.Source
