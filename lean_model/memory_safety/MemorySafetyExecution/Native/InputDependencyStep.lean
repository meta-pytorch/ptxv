/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputPreparation

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

structure DependencyStepContext (program : CoreProgram) (state : CoreState)
    (shadow after : value_flow_dependencies.State) (actor : Usize) where
  commit : execution_core.Commit
  thread : execution_core.ThreadState
  instruction : V.GuardedInstruction
  metadata : DependencyRegisters
  loaded : alloc.vec.Vec Metadata
  transferred : DependencyTransfer
  prepared : execution_core.prepare program state actor = .ok (.Ok commit)
  threadAt : state.threads.val[actor.val]? = some thread
  instructionAt : program.code.val[thread.pc.val]? = some instruction
  metadataAt : shadow.threads.val[actor.val]? = some metadata
  loadedComputed : input_independent_execution.action_metadata program state shadow commit.action = .ok (some loaded)
  transferComputed : value_flow_dependencies.transfer instruction commit.action metadata
    (alloc.vec.Vec.deref loaded) = .ok (.Ok transferred)
  applied : input_independent_execution.apply_transfer program state shadow actor commit.action transferred = .ok (some after)

theorem source_dependency_step_context (program : CoreProgram) (state : CoreState)
    (shadow after : value_flow_dependencies.State) (actor : Usize)
    (accepted : input_independent_execution.advance_dependencies program state shadow actor = .ok (some after)) :
    Nonempty (DependencyStepContext program state shadow after actor) := by
  unfold input_independent_execution.advance_dependencies at accepted
  obtain ⟨preparation, prepared, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  cases preparation with
  | Err fault => simp at accepted
  | Ok commit =>
    by_cases outside : actor ≥ alloc.vec.Vec.len shadow.threads
    · simp [outside] at accepted
    · simp only [outside, ite_false] at accepted
      obtain ⟨thread, threadRead, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      obtain ⟨index, indexCast, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      obtain ⟨instruction, instructionRead, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      obtain ⟨loadedResult, loadedComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      cases loadedResult with
      | none => simp at accepted
      | some loaded =>
        obtain ⟨metadata, metadataRead, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        obtain ⟨transferredResult, transferComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        cases transferredResult with
        | Err fault => simp at accepted
        | Ok transferred =>
          have threadAt := MemorySafety.Source.vec_index_success state.threads actor thread
            (by simpa only [alloc.vec.Vec.index_slice_index] using threadRead)
          have metadataAt := MemorySafety.Source.vec_index_success shadow.threads actor metadata
            (by simpa only [alloc.vec.Vec.index_slice_index] using metadataRead)
          have indexVal : index.val = thread.pc.val := by
            have spec := UScalar.cast_inBounds_spec .Usize thread.pc (by
              rcases Usize.bounds_eq with narrow | wide <;> scalar_tac)
            simpa only [indexCast, WP.spec_ok] using spec
          have instructionAt : program.code.val[thread.pc.val]? = some instruction := by
            have atIndex := MemorySafety.Source.vec_index_success program.code index instruction
              (by simpa only [alloc.vec.Vec.index_slice_index] using instructionRead)
            simpa only [indexVal] using atIndex
          exact ⟨⟨commit, thread, instruction, metadata, loaded, transferred, prepared, threadAt,
            instructionAt, metadataAt, loadedComputed, transferComputed, accepted⟩⟩

theorem source_dependency_step_prepares_inputs (program : CoreProgram)
    (left right : CoreState) (shadow after : value_flow_dependencies.State) (actor : Usize)
    (related : StateInputsRelated shadow left right)
    (accepted : input_independent_execution.advance_dependencies program left shadow actor = .ok (some after)) :
    ∃ (context : DependencyStepContext program left shadow after actor) (other : execution_core.Commit),
      execution_core.prepare program right actor = .ok (.Ok other) ∧
      other.actor = context.commit.actor ∧ other.next_pc = context.commit.next_pc ∧
      ActionsRelated (transferTaint context.transferred)
        (actionModel context.commit.action) (actionModel other.action) := by
  obtain ⟨context⟩ := source_dependency_step_context program left shadow after actor accepted
  obtain ⟨other, prepared, actorSame, nextSame, actions⟩ := source_prepare_inputs program shadow left right related
    actor context.commit context.thread context.instruction context.metadata
    (alloc.vec.Vec.deref context.loaded) context.transferred context.threadAt context.instructionAt
    context.metadataAt context.transferComputed context.prepared
  exact ⟨context, other, prepared, actorSame, nextSame, actions⟩

end MemorySafetyExecution.Native
