/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputRegisters
import MemorySafetyExecution.Native.PlacementRegisters
import MemorySafetyExecution.Native.InputFrames
import MemorySafetyExecution.Native.CommitRuns
import ValueFlow.SourceDependencyApply

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

theorem source_virtual_commit_placements (program rightProgram : CoreProgram)
    (layoutLengths : program.layouts.val.length=rightProgram.layouts.val.length) (left right leftAfter : CoreState)
    (shadow shadowAfter : DependencyState) (actor : Usize) (pc : U32)
    (leftThread rightThread : execution_core.ThreadState) (dependencies : DependencyRegisters)
    (destination : U16) (leftValue rightValue : V.Value) (metadata : Metadata) (event : memory_safety.Event)
    (related : StateInputsRelated shadow left right)
    (leftAt : left.threads.val[actor.val]? = some leftThread)
    (rightAt : right.threads.val[actor.val]? = some rightThread)
    (metadataAt : shadow.threads.val[actor.val]? = some dependencies)
    (defined : metadata.defined = true)
    (values : ValuesRelated metadata.tainted (valueModel leftValue) (valueModel rightValue))
    (captured : execution_core.commit_action program left {actor,next_pc:=pc,action:=.WriteVirtual destination leftValue} =
      .ok (.Ok event,leftAfter))
    (applied : input_independent_execution.apply_transfer program left shadow actor
      (.WriteVirtual destination leftValue) (.Virtual destination metadata) = .ok (some shadowAfter)) :
    ∃ rightAfter, execution_core.commit_action rightProgram right {actor,next_pc:=pc,action:=.WriteVirtual destination rightValue} =
      .ok (.Ok event,rightAfter) ∧ event=.Control ∧ StateInputsRelated shadowAfter leftAfter rightAfter := by
  have leftInside := (List.getElem?_eq_some_iff.mp leftAt).1
  have rightInside := (List.getElem?_eq_some_iff.mp rightAt).1
  have shadowInside := (List.getElem?_eq_some_iff.mp metadataAt).1
  have threads := related.threads actor.val leftThread rightThread dependencies leftAt rightAt metadataAt
  have shadowSpec := source_apply_transfer_virtual program left shadow actor
    (.WriteVirtual destination leftValue) destination metadata shadowInside
  rw [applied] at shadowSpec
  have shadowMeaning : ∀ output, some shadowAfter=some output → VirtualShadowEffect shadow output actor destination metadata := by
    simpa only [WP.spec_ok] using shadowSpec
  obtain ⟨original,shadowIndex,indexEq,originalAt,shadowBound,shadowEq⟩ := shadowMeaning shadowAfter rfl
  have originalEq : original=dependencies := by simpa only [metadataAt,Option.some.injEq] using originalAt.symm
  subst original
  obtain ⟨leftRegisters,leftWritten,eventEq,afterEq⟩ := source_virtual_commit_components program left leftAfter actor pc
    leftThread destination leftValue event leftInside leftAt captured
  obtain ⟨rightRegisters,rightWritten,registers⟩ := source_virtual_write_placements dependencies leftThread.registers
    rightThread.registers leftRegisters destination shadowIndex metadata leftValue rightValue
    (alloc.vec.Vec.deref program.layouts) (alloc.vec.Vec.deref rightProgram.layouts)
    (by simpa [alloc.vec.Vec.deref] using layoutLengths) threads.registers indexEq defined values leftWritten
  let rightAfter := update_thread right actor {rightThread with registers:=rightRegisters,pc}
  refine ⟨rightAfter,?_,eventEq,?_⟩
  · rw [eventEq]
    exact source_commit_virtual rightProgram right actor pc rightThread destination rightValue rightRegisters rightInside rightAt rightWritten
  · rw [afterEq,shadowEq]
    exact input_state_set_thread shadow left right related actor
      {dependencies with virtuals:=dependencies.virtuals.set shadowIndex metadata}
      {leftThread with registers:=leftRegisters,pc} {rightThread with registers:=rightRegisters,pc}
      leftInside ⟨rfl,threads.running,threads.ids,registers⟩

theorem source_predicate_commit_placements (program rightProgram : CoreProgram) (left right leftAfter : CoreState)
    (shadow shadowAfter : DependencyState) (actor : Usize) (pc : U32)
    (leftThread rightThread : execution_core.ThreadState) (dependencies : DependencyRegisters)
    (destination : U16) (leftValue rightValue : Bool) (metadata : Metadata) (event : memory_safety.Event)
    (related : StateInputsRelated shadow left right)
    (leftAt : left.threads.val[actor.val]? = some leftThread)
    (rightAt : right.threads.val[actor.val]? = some rightThread)
    (metadataAt : shadow.threads.val[actor.val]? = some dependencies)
    (defined : metadata.defined = true)
    (values : metadata.tainted = false → leftValue = rightValue)
    (captured : execution_core.commit_action program left {actor,next_pc:=pc,action:=.WritePredicate destination leftValue} =
      .ok (.Ok event,leftAfter))
    (applied : input_independent_execution.apply_transfer program left shadow actor
      (.WritePredicate destination leftValue) (.Predicate destination metadata) = .ok (some shadowAfter)) :
    ∃ rightAfter, execution_core.commit_action rightProgram right {actor,next_pc:=pc,action:=.WritePredicate destination rightValue} =
      .ok (.Ok event,rightAfter) ∧ event=.Control ∧ StateInputsRelated shadowAfter leftAfter rightAfter := by
  have leftInside := (List.getElem?_eq_some_iff.mp leftAt).1
  have rightInside := (List.getElem?_eq_some_iff.mp rightAt).1
  have shadowInside := (List.getElem?_eq_some_iff.mp metadataAt).1
  have threads := related.threads actor.val leftThread rightThread dependencies leftAt rightAt metadataAt
  have shadowSpec := source_apply_transfer_predicate program left shadow actor
    (.WritePredicate destination leftValue) destination metadata shadowInside
  rw [applied] at shadowSpec
  have shadowMeaning : ∀ output, some shadowAfter=some output → PredicateShadowEffect shadow output actor destination metadata := by
    simpa only [WP.spec_ok] using shadowSpec
  obtain ⟨original,shadowIndex,indexEq,originalAt,shadowBound,shadowEq⟩ := shadowMeaning shadowAfter rfl
  have originalEq : original=dependencies := by simpa only [metadataAt,Option.some.injEq] using originalAt.symm
  subst original
  obtain ⟨leftRegisters,leftWritten,eventEq,afterEq⟩ := source_predicate_commit_components program left leftAfter actor pc
    leftThread destination leftValue event leftInside leftAt captured
  obtain ⟨rightRegisters,rightWritten,registers⟩ := source_predicate_write_inputs dependencies leftThread.registers
    rightThread.registers leftRegisters destination shadowIndex metadata leftValue rightValue
    threads.registers indexEq defined values leftWritten
  let rightAfter := update_thread right actor {rightThread with registers:=rightRegisters,pc}
  refine ⟨rightAfter,?_,eventEq,?_⟩
  · rw [eventEq]
    exact source_commit_predicate rightProgram right actor pc rightThread destination rightValue rightRegisters rightInside rightAt rightWritten
  · rw [afterEq,shadowEq]
    exact input_state_set_thread shadow left right related actor
      {dependencies with predicates:=dependencies.predicates.set shadowIndex metadata}
      {leftThread with registers:=leftRegisters,pc} {rightThread with registers:=rightRegisters,pc}
      leftInside ⟨rfl,threads.running,threads.ids,registers⟩


end MemorySafetyExecution.Native
