/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputControl
import MemorySafetyExecution.Native.CommitRuns

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow.Source

theorem source_next_commit_placements (program rightProgram : CoreProgram)
    (sameCode : program.code=rightProgram.code) (left right leftAfter : CoreState)
    (shadow : value_flow_dependencies.State) (actor : Usize) (pc : U32)
    (leftThread rightThread : execution_core.ThreadState) (dependencies : DependencyRegisters) (event : memory_safety.Event)
    (related : StateInputsRelated shadow left right)
    (leftAt : left.threads.val[actor.val]?=some leftThread)
    (rightAt : right.threads.val[actor.val]?=some rightThread)
    (metadataAt : shadow.threads.val[actor.val]?=some dependencies)
    (captured : execution_core.commit_action program left {actor,next_pc:=pc,action:=.Next}=.ok (.Ok event,leftAfter)) :
    ∃ rightAfter, execution_core.commit_action rightProgram right {actor,next_pc:=pc,action:=.Next}=.ok (.Ok event,rightAfter) ∧ event=.Control ∧
      StateInputsRelated shadow leftAfter rightAfter := by
  have leftInside := (List.getElem?_eq_some_iff.mp leftAt).1
  have rightInside := (List.getElem?_eq_some_iff.mp rightAt).1
  have threads := related.threads actor.val leftThread rightThread dependencies leftAt rightAt metadataAt
  rw [source_commit_next program left actor pc leftThread leftInside leftAt] at captured
  have meaning : event=.Control ∧ leftAfter=update_thread left actor {leftThread with pc} := by
    simpa only [ok.injEq,Prod.mk.injEq,core.result.Result.Ok.injEq] using captured.symm
  obtain ⟨eventEq,afterEq⟩ := meaning
  refine ⟨update_thread right actor {rightThread with pc},?_,eventEq,?_⟩
  · rw [eventEq]
    exact source_commit_next rightProgram right actor pc rightThread rightInside rightAt
  · rw [afterEq]
    exact input_state_control shadow left right related actor dependencies _ _ leftInside metadataAt
      (input_thread_advance dependencies leftThread rightThread threads pc)

theorem source_halt_commit_placements (program rightProgram : CoreProgram)
    (sameCode : program.code=rightProgram.code) (left right leftAfter : CoreState)
    (shadow : value_flow_dependencies.State) (actor : Usize) (pc : U32)
    (leftThread rightThread : execution_core.ThreadState) (dependencies : DependencyRegisters) (event : memory_safety.Event)
    (related : StateInputsRelated shadow left right)
    (leftAt : left.threads.val[actor.val]?=some leftThread)
    (rightAt : right.threads.val[actor.val]?=some rightThread)
    (metadataAt : shadow.threads.val[actor.val]?=some dependencies)
    (captured : execution_core.commit_action program left {actor,next_pc:=pc,action:=.Halt}=.ok (.Ok event,leftAfter)) :
    ∃ rightAfter, execution_core.commit_action rightProgram right {actor,next_pc:=pc,action:=.Halt}=.ok (.Ok event,rightAfter) ∧ event=.Control ∧
      StateInputsRelated shadow leftAfter rightAfter := by
  have leftInside := (List.getElem?_eq_some_iff.mp leftAt).1
  have rightInside := (List.getElem?_eq_some_iff.mp rightAt).1
  have threads := related.threads actor.val leftThread rightThread dependencies leftAt rightAt metadataAt
  rw [source_commit_halt program left actor pc leftThread leftInside leftAt] at captured
  have meaning : event=.Control ∧ leftAfter=update_thread left actor {leftThread with running:=false} := by
    simpa only [ok.injEq,Prod.mk.injEq,core.result.Result.Ok.injEq] using captured.symm
  obtain ⟨eventEq,afterEq⟩ := meaning
  refine ⟨update_thread right actor {rightThread with running:=false},?_,eventEq,?_⟩
  · rw [eventEq]
    exact source_commit_halt rightProgram right actor pc rightThread rightInside rightAt
  · rw [afterEq]
    exact input_state_control shadow left right related actor dependencies _ _ leftInside metadataAt
      (input_thread_halt dependencies leftThread rightThread threads)

theorem source_jump_commit_placements (program rightProgram : CoreProgram)
    (sameCode : program.code=rightProgram.code) (left right leftAfter : CoreState)
    (shadow : value_flow_dependencies.State) (actor : Usize) (pc target : U32)
    (leftThread rightThread : execution_core.ThreadState) (dependencies : DependencyRegisters) (event : memory_safety.Event)
    (related : StateInputsRelated shadow left right)
    (leftAt : left.threads.val[actor.val]?=some leftThread)
    (rightAt : right.threads.val[actor.val]?=some rightThread)
    (metadataAt : shadow.threads.val[actor.val]?=some dependencies)
    (captured : execution_core.commit_action program left {actor,next_pc:=pc,action:=.Jump target}=.ok (.Ok event,leftAfter)) :
    ∃ rightAfter, execution_core.commit_action rightProgram right {actor,next_pc:=pc,action:=.Jump target}=.ok (.Ok event,rightAfter) ∧ event=.Control ∧
      StateInputsRelated shadow leftAfter rightAfter := by
  have leftInside := (List.getElem?_eq_some_iff.mp leftAt).1
  have rightInside := (List.getElem?_eq_some_iff.mp rightAt).1
  have threads := related.threads actor.val leftThread rightThread dependencies leftAt rightAt metadataAt
  obtain ⟨index,cast,available,eventEq,afterEq⟩ := source_jump_commit_components program left leftAfter actor pc target leftThread event
    leftInside leftAt captured
  refine ⟨update_thread right actor {rightThread with pc:=target},?_,eventEq,?_⟩
  · rw [eventEq]
    have rightAvailable : ¬index ≥ alloc.vec.Vec.len rightProgram.code := by simpa only [←sameCode] using available
    exact source_commit_jump rightProgram right actor pc target index rightThread rightInside rightAt cast rightAvailable
  · rw [afterEq]
    exact input_state_control shadow left right related actor dependencies _ _ leftInside metadataAt
      (input_thread_advance dependencies leftThread rightThread threads target)

end MemorySafetyExecution.Native
