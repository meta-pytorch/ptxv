/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementInputResolution
import MemorySafetyExecution.Native.PlacementResolvedEvent
import MemorySafetyExecution.Native.PlacementRegisters
import MemorySafetyExecution.Native.AccessFields
import MemorySafetyExecution.Native.InputControl
import MemorySafetyExecution.Native.InputMemory
import MemorySafetyExecution.Native.InputResolution
import MemorySafetyExecution.Native.ResolvedStorage
import MemorySafetyExecution.Native.StoreSemantics
import MemorySafetyExecution.Native.CommitRuns
import ValueFlow.SourceDependencyApply

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow.Source

theorem source_store_commit_placements (program rightProgram : CoreProgram) (left right leftAfter : CoreState)
    (shadow shadowAfter : DependencyState) (actor : Usize) (pc : U32)
    (leftThread rightThread : execution_core.ThreadState) (dependencies : DependencyRegisters)
    (bytes : U8) (pointer : value_flow.Pointer) (leftValue rightValue : U64)
    (metadata : Metadata) (event : memory_safety.Event)
    (placements : ProgramPlacements program rightProgram)
    (related : StateInputsRelated shadow left right)
    (leftAt : left.threads.val[actor.val]? = some leftThread)
    (rightAt : right.threads.val[actor.val]? = some rightThread)
    (metadataAt : shadow.threads.val[actor.val]? = some dependencies)
    (defined : metadata.defined = true)
    (values : metadata.tainted = false → leftValue = rightValue)
    (captured : execution_core.commit_action program left {actor,next_pc:=pc,action:=.Store bytes pointer leftValue} =
      .ok (.Ok event,leftAfter))
    (applied : input_independent_execution.apply_transfer program left shadow actor
      (.Store bytes pointer leftValue) (.Store metadata) = .ok (some shadowAfter)) :
    ∃ rightEvent rightAfter, execution_core.commit_action rightProgram right {actor,next_pc:=pc,action:=.Store bytes pointer rightValue} =
      .ok (.Ok rightEvent,rightAfter) ∧
      EventPlacement program.layouts rightProgram.layouts event rightEvent ∧
      StateInputsRelated shadowAfter leftAfter rightAfter := by
  have leftInside := (List.getElem?_eq_some_iff.mp leftAt).1
  have rightInside := (List.getElem?_eq_some_iff.mp rightAt).1
  have threads := related.threads actor.val leftThread rightThread dependencies leftAt rightAt metadataAt
  obtain ⟨access,leftWritten,resolved,stored,eventEq,afterEq⟩ := source_store_commit_components program left leftAfter actor pc
    leftThread bytes pointer leftValue event leftInside leftAt captured
  obtain ⟨shadowAccess,oldShadow,newShadow,shadowResolved,shadowAt,shadowThreads,shadowMemory,shadowFrame⟩ :=
    source_apply_transfer_store_success program left shadow shadowAfter actor bytes pointer leftValue metadata applied
  have sameAccess : shadowAccess=access := by
    simpa only [resolved,ok.injEq,core.result.Result.Ok.injEq] using shadowResolved.symm
  subst shadowAccess
  obtain ⟨rightAccess,rightLayout,rightResolved,rightLayoutAt,allocationEq,offsetEq,widthEq,_,_,_,_⟩ :=
    source_resolve_placements program rightProgram left right pointer bytes true access placements
      (input_storage_shapes shadow left right related) resolved
  have requestPlacement := source_resolved_request_placement program rightProgram left right pointer bytes true
    access rightAccess (fun old fresh oldAt freshAt => by
      obtain ⟨sameBytes,sameAlignment,_⟩ := placements.2.1 pointer.allocation.val old fresh oldAt freshAt
      exact ⟨sameBytes.symm,sameAlignment.symm⟩) resolved rightResolved
  obtain ⟨requestPlacement⟩ := requestPlacement
  obtain ⟨oldLeft,leftMemoryAt,nonzero,small,leftFits,leftShape⟩ :=
    source_resolved_storage program left pointer bytes true access resolved
  obtain ⟨oldRight,rightMemoryAt,_,_,rightFits,rightShape⟩ :=
    source_resolved_storage rightProgram right pointer bytes true rightAccess rightResolved
  rw [allocationEq] at rightMemoryAt
  rw [offsetEq,widthEq] at rightFits
  have memories := related.allocations access.allocation.val oldLeft oldRight oldShadow leftMemoryAt rightMemoryAt shadowAt
  obtain ⟨leftWrittenAgain,newLeft,leftStored,leftWrittenEq,leftFrame,leftPayload⟩ :=
    source_store_semantics left access leftValue oldLeft leftMemoryAt small leftFits (by omega)
  have leftWrittenSame : leftWrittenAgain=leftWritten := by simpa only [stored,ok.injEq] using leftStored.symm
  rw [leftWrittenSame] at leftStored leftWrittenEq
  obtain ⟨rightWritten,newRight,rightStored,rightWrittenEq,rightFrame,rightPayload⟩ :=
    source_store_semantics right access rightValue oldRight rightMemoryAt small rightFits (by omega)
  have rightStoredRebased : execution_core.store_value right rightAccess rightValue=.ok rightWritten := by
    rw [store_value_access_fields right rightAccess access rightValue allocationEq offsetEq widthEq]
    exact rightStored
  have payload : metadata.tainted=false → ∀ index, access.offset.val ≤ index → index < access.offset.val + access.width.val →
      newLeft.bytes.val[index]?=newRight.bytes.val[index]? := by
    intro clean index lower upper
    have valueEq := values clean
    rw [←valueEq] at rightPayload
    have span := stored_payload_same newLeft newRight access.offset.val access.width.val leftValue leftPayload rightPayload
      (index-access.offset.val) (by omega)
    have relative : access.offset.val+(index-access.offset.val)=index := by omega
    simpa only [relative] using span
  have memoryRelation := input_memory_store oldShadow newShadow oldLeft oldRight newLeft newRight
    access.offset.val access.width.val metadata memories leftFrame rightFrame shadowFrame defined payload
  have shadowEq : shadowAfter={shadow with memory:=shadow.memory.set access.allocation newShadow} := by
    have sameMemory : shadowAfter.memory=shadow.memory.set access.allocation newShadow := by
      apply alloc.vec.Vec.ext
      simpa only [alloc.vec.Vec.set_val_eq] using shadowMemory
    cases shadowAfter
    cases shadow
    simp_all
  have allocationInside := (List.getElem?_eq_some_iff.mp leftMemoryAt).1
  have storedRelation := input_state_set_memory shadow left right related access.allocation newShadow newLeft newRight
    allocationInside memoryRelation
  rw [←shadowEq,←leftWrittenEq,←rightWrittenEq] at storedRelation
  have rightThreadFrame : rightWritten.threads=right.threads := by rw [rightWrittenEq]
  have leftThreadFrame : leftWritten.threads=left.threads := by rw [leftWrittenEq]
  let rightAfter := update_thread rightWritten actor {rightThread with pc}
  refine ⟨.Access actor rightAccess.request,rightAfter,?_,?_,?_⟩
  · exact source_commit_store rightProgram right rightWritten actor pc rightThread bytes pointer rightAccess rightValue rightInside rightAt
      rightResolved rightStoredRebased rightThreadFrame
  · rw [eventEq]
    exact .access actor access.request rightAccess.request requestPlacement
  · rw [afterEq]
    apply input_state_control shadowAfter leftWritten rightWritten storedRelation actor dependencies
      {leftThread with pc} {rightThread with pc}
    · simpa only [leftThreadFrame] using leftInside
    · simpa only [shadowThreads] using metadataAt
    · exact input_thread_advance dependencies leftThread rightThread threads pc

end MemorySafetyExecution.Native
