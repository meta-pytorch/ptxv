/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementCommit
import MemorySafetyExecution.Native.PlacementPreparation
import MemorySafetyExecution.Native.InputDependencyStep

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

theorem source_advance_native_placements_valid (program rightProgram : CoreProgram) (left right leftAfter : CoreState)
    (shadow shadowAfter : value_flow_dependencies.State) (actor : Usize) (event : memory_safety.Event)
    (sameCode : program.code=rightProgram.code)
    (placements : ProgramPlacements program rightProgram)
    (rightLayout : execution_core.layout_valid rightProgram right=.ok true)
    (related : StateInputsRelated shadow left right)
    (advanced : input_independent_execution.advance_dependencies program left shadow actor=.ok (some shadowAfter))
    (captured : execution_core.execute program left actor=.ok (.Ok event,leftAfter)) :
    ∃ rightEvent rightAfter, execution_core.execute rightProgram right actor=.ok (.Ok rightEvent,rightAfter) ∧
      EventPlacement program.layouts rightProgram.layouts event rightEvent ∧
      StateInputsRelated shadowAfter leftAfter rightAfter := by
  obtain ⟨context⟩ := source_dependency_step_context program left shadow shadowAfter actor advanced
  obtain ⟨other,otherPrepared,otherActor,otherPc,actions⟩ := source_prepare_placements program rightProgram shadow left right
    related sameCode rightLayout actor context.commit context.thread context.instruction context.metadata
    (alloc.vec.Vec.deref context.loaded) context.transferred context.threadAt context.instructionAt context.metadataAt
    context.transferComputed context.prepared
  have preparedMeaning := source_prepare_evaluation program left actor context.commit context.prepared
  obtain ⟨thread,instruction,threadAt,running,instructionAt,commitActor,nextPc,evaluated⟩ := preparedMeaning
  have capturedCommit : execution_core.commit_action program left context.commit=.ok (.Ok event,leftAfter) := by
    simpa only [execution_core.execute,context.prepared,bind_ok] using captured
  have leftInside := (List.getElem?_eq_some_iff.mp context.threadAt).1
  have rightInside : actor.val<right.threads.val.length := by
    have lengths:=related.threadsLength
    omega
  let rightThread:=right.threads.val[actor.val]
  have rightAt : right.threads.val[actor.val]?=some rightThread := List.getElem?_eq_getElem rightInside
  have commitEq : ({actor,next_pc:=context.commit.next_pc,action:=context.commit.action}:execution_core.Commit)=context.commit := by
    cases e:context.commit
    simp only [e] at commitActor ⊢
    simp only [commitActor]
  have otherEq : other=({actor,next_pc:=context.commit.next_pc,action:=other.action}:execution_core.Commit) := by
    have otherActorEq : other.actor=actor := otherActor.trans commitActor
    cases e:other
    simp only [e] at otherActorEq otherPc ⊢
    simp only [otherActorEq,otherPc]
  have capturedNormal : execution_core.commit_action program left
      {actor,next_pc:=context.commit.next_pc,action:=context.commit.action}=.ok (.Ok event,leftAfter) := by
    simpa only [commitEq] using capturedCommit
  obtain ⟨rightEvent,rightAfter,otherCommit,eventPlacement,afterRelated⟩ := source_commit_placements program rightProgram sameCode left right leftAfter shadow shadowAfter
    actor context.commit.next_pc context.thread rightThread context.metadata context.instruction context.commit.action other.action
    context.loaded context.transferred event placements related context.threadAt rightAt context.metadataAt actions context.loadedComputed
    context.transferComputed context.applied capturedNormal
  refine ⟨rightEvent,rightAfter,?_,eventPlacement,afterRelated⟩
  simp only [execution_core.execute,otherPrepared,bind_ok]
  rw [otherEq]
  exact otherCommit

end MemorySafetyExecution.Native
