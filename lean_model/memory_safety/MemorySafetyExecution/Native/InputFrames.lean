/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputState

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow.Source

theorem input_state_set_thread (shadow : value_flow_dependencies.State) (left right : CoreState)
    (related : StateInputsRelated shadow left right) (actor : Usize)
    (metadata : DependencyRegisters) (leftThread rightThread : execution_core.ThreadState)
    (inside : actor.val < left.threads.val.length)
    (threads : ThreadInputsRelated metadata leftThread rightThread) :
    StateInputsRelated { shadow with threads := shadow.threads.set actor metadata }
      { left with threads := left.threads.set actor leftThread }
      { right with threads := right.threads.set actor rightThread } := by
  have rightInside : actor.val < right.threads.val.length := by have := related.threadsLength; omega
  have shadowInside : actor.val < shadow.threads.val.length := by have := related.threadMetadataLength; omega
  constructor
  · simpa only [alloc.vec.Vec.set_val_eq, List.length_set] using related.threadsLength
  · exact related.allocationsLength
  · simpa only [alloc.vec.Vec.set_val_eq, List.length_set] using related.threadMetadataLength
  · exact related.allocationMetadataLength
  · intro index l r metadata foundLeft foundRight foundShadow
    by_cases selected : actor.val = index
    · subst index
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_self inside, Option.some.injEq] at foundLeft
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_self rightInside, Option.some.injEq] at foundRight
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_self shadowInside, Option.some.injEq] at foundShadow
      subst l; subst r; subst metadata
      exact threads
    · simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_ne selected] at foundLeft foundRight foundShadow
      exact related.threads index l r metadata foundLeft foundRight foundShadow
  · exact related.allocations

theorem input_state_set_memory (shadow : value_flow_dependencies.State) (left right : CoreState)
    (related : StateInputsRelated shadow left right) (allocation : Usize)
    (metadata : alloc.vec.Vec Metadata) (leftMemory rightMemory : execution_core.AllocationBytes)
    (inside : allocation.val < left.allocations.val.length)
    (memory : MemoryInputsRelated metadata leftMemory rightMemory) :
    StateInputsRelated { shadow with memory := shadow.memory.set allocation metadata }
      { left with allocations := left.allocations.set allocation leftMemory }
      { right with allocations := right.allocations.set allocation rightMemory } := by
  have rightInside : allocation.val < right.allocations.val.length := by have := related.allocationsLength; omega
  have shadowInside : allocation.val < shadow.memory.val.length := by have := related.allocationMetadataLength; omega
  constructor
  · exact related.threadsLength
  · simpa only [alloc.vec.Vec.set_val_eq, List.length_set] using related.allocationsLength
  · exact related.threadMetadataLength
  · simpa only [alloc.vec.Vec.set_val_eq, List.length_set] using related.allocationMetadataLength
  · exact related.threads
  · intro index l r metadata foundLeft foundRight foundShadow
    by_cases selected : allocation.val = index
    · subst index
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_self inside, Option.some.injEq] at foundLeft
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_self rightInside, Option.some.injEq] at foundRight
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_self shadowInside, Option.some.injEq] at foundShadow
      subst l; subst r; subst metadata
      exact memory
    · simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_ne selected] at foundLeft foundRight foundShadow
      exact related.allocations index l r metadata foundLeft foundRight foundShadow

end MemorySafetyExecution.Native
