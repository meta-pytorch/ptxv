/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentStore
import MemorySafetyExecution.Native.IndependentResolution

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent

theorem source_store_model (program : CoreProgram) (before after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (event : memory_safety.Event)
    (bytes : U8) (pointer : Pointer) (value : U64)
    (canonical : Canonical before) (selected : before.threads.val[actor.val]?=some thread)
    (nextPc : pc.val=thread.pc.val+1)
    (computed : execution_core.commit_action program before
      {actor,next_pc:=pc,action:=.Store bytes pointer value}=.ok (.Ok event,after)) :
    MachineModel.applyAction (programModel program) (stateModel before) actor.val (threadModel thread)
      (.store bytes.val (ValueFlow.Source.pointerModel pointer) value.val)=
      some (MemorySafety.Source.eventModel event,stateModel after) ∧ Canonical after := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  obtain ⟨access,written,resolved,stored,eventEq,afterEq⟩ := source_store_commit_components program before after actor pc
    thread bytes pointer value event inside selected computed
  obtain ⟨descriptor,layout,memory,layoutAt,memoryAt,widthKnown,offsetVal,addressVal,
    live,originBound,alignment,aligned,offsetAligned,baseAligned,physicalBound,
    storageBound,definedShape,initialized⟩ := source_resolve_range program before pointer bytes true access resolved
  obtain ⟨other,otherAt,resolution⟩ := source_resolution_model program before pointer bytes true access resolved
  have same : other=memory := Option.some.inj (otherAt.symm.trans memoryAt)
  subst other
  have small : access.width.val  ≤  8 := by rcases widthKnown with h|h|h|h <;> simp_all only [descriptor.2.1] <;> omega
  have bounded : access.offset.val+access.width.val  ≤  memory.bytes.val.length := by simpa only [offsetVal,descriptor.2.1] using storageBound
  have definedBound : access.offset.val+access.width.val  ≤  memory.defined.val.length := by simpa only [definedShape] using bounded
  obtain ⟨writtenModel,threadsFrame⟩ := source_store_state_model before written access memory value
    (by simpa only [descriptor.1] using memoryAt) small bounded definedBound stored
  constructor
  · rw [afterEq,model_thread_set]
    simp only [MachineModel.applyAction,resolution,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
    rw [eventEq]
    simp only [MemorySafety.Source.eventModel,writtenModel,threadModel,nextPc,descriptor.1,
      descriptor.2.1,offsetVal,ValueFlow.Source.pointerModel]
  · rw [afterEq]
    apply canonical_thread_set written actor _
    · simpa only [Canonical,threadsFrame] using canonical
    · exact canonical thread (List.mem_of_getElem? selected)

end MemorySafetyExecution.Native.Independent
