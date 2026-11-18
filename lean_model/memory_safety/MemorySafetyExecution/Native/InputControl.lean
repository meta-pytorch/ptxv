/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputFrames
import MemorySafetyExecution.Native.CommitRules

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow.Source

private theorem vector_set_found {α : Type} (values : alloc.vec.Vec α) (index : Usize) (value : α)
    (found : values.val[index.val]? = some value) : values.set index value = values := by
  obtain ⟨inside,equal⟩ := List.getElem?_eq_some_iff.mp found
  apply alloc.vec.Vec.ext
  simp only [alloc.vec.Vec.set_val_eq,←equal,List.set_getElem_self inside]

theorem input_state_control (shadow : value_flow_dependencies.State) (left right : CoreState)
    (related : StateInputsRelated shadow left right) (actor : Usize) (metadata : DependencyRegisters)
    (leftThread rightThread : execution_core.ThreadState)
    (inside : actor.val < left.threads.val.length)
    (metadataAt : shadow.threads.val[actor.val]? = some metadata)
    (threads : ThreadInputsRelated metadata leftThread rightThread) :
    StateInputsRelated shadow (update_thread left actor leftThread) (update_thread right actor rightThread) := by
  have lifted := input_state_set_thread shadow left right related actor metadata leftThread rightThread inside threads
  have unchanged := vector_set_found shadow.threads actor metadata metadataAt
  simpa only [unchanged,update_thread] using lifted

theorem input_thread_advance (metadata : DependencyRegisters) (left right : execution_core.ThreadState)
    (related : ThreadInputsRelated metadata left right) (pc : U32) :
    ThreadInputsRelated metadata {left with pc} {right with pc} := by
  exact ⟨rfl,related.running,related.ids,related.registers⟩

theorem input_thread_halt (metadata : DependencyRegisters) (left right : execution_core.ThreadState)
    (related : ThreadInputsRelated metadata left right) :
    ThreadInputsRelated metadata {left with running:=false} {right with running:=false} := by
  exact ⟨related.pc,rfl,related.ids,related.registers⟩

end MemorySafetyExecution.Native
