/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementInputResolution
import MemorySafetyExecution.Native.PlacementResolvedEvent
import MemorySafetyExecution.Native.PlacementRegisters
import MemorySafetyExecution.Native.AccessFields
import MemorySafetyExecution.Native.InputRegisters
import MemorySafetyExecution.Native.InputFrames
import MemorySafetyExecution.Native.InputMemory
import MemorySafetyExecution.Native.InputResolution
import MemorySafetyExecution.Native.ResolvedStorage
import MemorySafetyExecution.Native.LoadTotal
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

theorem source_load_commit_placements (program rightProgram : CoreProgram) (left right leftAfter : CoreState)
    (shadow shadowAfter : DependencyState) (actor : Usize) (pc : U32)
    (leftThread rightThread : execution_core.ThreadState) (dependencies : DependencyRegisters)
    (destination : U16) (bits bytes : U8) (signed : Bool) (pointer : V.Pointer)
    (metadata : Metadata) (loadedMetadata : MetadataVector) (event : memory_safety.Event)
    (placements : ProgramPlacements program rightProgram)
    (related : StateInputsRelated shadow left right)
    (leftAt : left.threads.val[actor.val]? = some leftThread)
    (rightAt : right.threads.val[actor.val]? = some rightThread)
    (metadataAt : shadow.threads.val[actor.val]? = some dependencies)
    (defined : metadata.defined = true)
    (loadedComputed : input_independent_execution.action_metadata program left shadow
      (.Load destination bits bytes signed pointer)=.ok (some loadedMetadata))
    (joined : value_flow_dependencies.loaded_metadata loadedMetadata.slice bytes=.ok (.Ok metadata))
    (captured : execution_core.commit_action program left {actor,next_pc:=pc,action:=.Load destination bits bytes signed pointer} =
      .ok (.Ok event,leftAfter))
    (applied : input_independent_execution.apply_transfer program left shadow actor
      (.Load destination bits bytes signed pointer) (.Virtual destination metadata) = .ok (some shadowAfter)) :
    ∃ rightEvent rightAfter, execution_core.commit_action rightProgram right {actor,next_pc:=pc,action:=.Load destination bits bytes signed pointer} =
      .ok (.Ok rightEvent,rightAfter) ∧
      EventPlacement program.layouts rightProgram.layouts event rightEvent ∧
      StateInputsRelated shadowAfter leftAfter rightAfter := by
  have leftInside := (List.getElem?_eq_some_iff.mp leftAt).1
  have rightInside := (List.getElem?_eq_some_iff.mp rightAt).1
  have shadowInside := (List.getElem?_eq_some_iff.mp metadataAt).1
  have threads := related.threads actor.val leftThread rightThread dependencies leftAt rightAt metadataAt
  obtain ⟨access,leftValue,mask,leftRegisters,resolved,leftLoaded,maskRun,leftWritten,eventEq,afterEq⟩ :=
    source_load_commit_components program left leftAfter actor pc leftThread destination bits bytes signed pointer event
      leftInside leftAt captured
  obtain ⟨rightAccess,rightLayout,rightResolved,rightLayoutAt,allocationEq,offsetEq,widthEq,_,_,_,_⟩ :=
    source_resolve_placements program rightProgram left right pointer bytes false access placements
      (input_storage_shapes shadow left right related) resolved
  have requestPlacement := source_resolved_request_placement program rightProgram left right pointer bytes false
    access rightAccess (fun old fresh oldAt freshAt => by
      obtain ⟨sameBytes,sameAlignment,_⟩ := placements.2.1 pointer.allocation.val old fresh oldAt freshAt
      exact ⟨sameBytes.symm,sameAlignment.symm⟩) resolved rightResolved
  obtain ⟨requestPlacement⟩ := requestPlacement
  obtain ⟨oldLeft,leftMemoryAt,nonzero,small,leftFits,leftShape⟩ :=
    source_resolved_storage program left pointer bytes false access resolved
  obtain ⟨oldRight,rightMemoryAt,_,_,rightFits,rightShape⟩ :=
    source_resolved_storage rightProgram right pointer bytes false rightAccess rightResolved
  rw [allocationEq] at rightMemoryAt
  rw [offsetEq,widthEq] at rightFits
  obtain ⟨rightValue,rightLoaded,_⟩ := MemoryWrite.spec_returns _ _
    (source_load_value_total right access oldRight signed rightMemoryAt rightFits nonzero small)
  have rightLoadedRebased : execution_core.load_value right rightAccess signed=.ok rightValue := by
    rw [load_value_access_fields right rightAccess access signed allocationEq offsetEq widthEq]
    exact rightLoaded
  have read : input_independent_execution.loaded_metadata (alloc.vec.Vec.deref shadow.memory) access=.ok (some loadedMetadata) := by
    simpa only [input_independent_execution.action_metadata,resolved,bind_ok] using loadedComputed
  have valuesEqual : metadata.tainted=false → leftValue=rightValue := by
    intro clean
    obtain ⟨oldShadow,shadowAt,shadowFits,allClean⟩ := source_loaded_memory_metadata
      (alloc.vec.Vec.deref shadow.memory) access loadedMetadata bytes metadata read joined clean
    have memories := related.allocations access.allocation.val oldLeft oldRight oldShadow leftMemoryAt rightMemoryAt
      (by simpa [alloc.vec.Vec.deref] using shadowAt)
    have sameBytes := input_memory_loaded_equal oldShadow oldLeft oldRight memories access.offset.val access.width.val allClean shadowFits
    have offsetBound : access.offset.val+access.width.val ≤ UScalar.max .Usize := by
      have memoryBound := alloc.vec.Vec.len_ineq oldLeft.bytes
      scalar_tac
    have calls := source_load_value_same_span left right access access signed oldLeft oldRight leftMemoryAt rightMemoryAt
      rfl rfl offsetBound sameBytes
    simpa only [leftLoaded,rightLoaded,ok.injEq] using calls
  have values : ValuesRelated metadata.tainted
      (valueModel (.Scalar (leftValue &&& mask))) (valueModel (.Scalar (rightValue &&& mask))) := by
    cases clean : metadata.tainted with
    | false => simp only [ValuesRelated,Bool.false_eq_true,ite_false,valueModel,valuesEqual clean]
    | true => exact ⟨_,_,rfl,rfl⟩
  have shadowSpec := source_apply_transfer_virtual program left shadow actor
    (.Load destination bits bytes signed pointer) destination metadata shadowInside
  rw [applied] at shadowSpec
  have shadowMeaning : ∀ output, some shadowAfter=some output → VirtualShadowEffect shadow output actor destination metadata := by
    simpa only [WP.spec_ok] using shadowSpec
  obtain ⟨original,shadowIndex,indexEq,originalAt,shadowBound,shadowEq⟩ := shadowMeaning shadowAfter rfl
  have originalEq : original=dependencies := by simpa only [metadataAt,Option.some.injEq] using originalAt.symm
  subst original
  obtain ⟨rightRegisters,rightWritten,registers⟩ := source_virtual_write_placements dependencies leftThread.registers
    rightThread.registers leftRegisters destination shadowIndex metadata
    (.Scalar (leftValue &&& mask)) (.Scalar (rightValue &&& mask))
    (alloc.vec.Vec.deref program.layouts) (alloc.vec.Vec.deref rightProgram.layouts)
    (by simpa [alloc.vec.Vec.deref] using placements.1) threads.registers indexEq defined values leftWritten
  let rightAfter := update_thread right actor {rightThread with registers:=rightRegisters,pc}
  refine ⟨.Access actor rightAccess.request,rightAfter,?_,?_,?_⟩
  · exact source_commit_load rightProgram right actor pc rightThread destination bits bytes signed pointer rightAccess rightValue mask
      rightRegisters rightInside rightAt rightResolved rightLoadedRebased maskRun rightWritten
  · rw [eventEq]
    exact .access actor access.request rightAccess.request requestPlacement
  · rw [afterEq,shadowEq]
    exact input_state_set_thread shadow left right related actor
      {dependencies with virtuals:=dependencies.virtuals.set shadowIndex metadata}
      {leftThread with registers:=leftRegisters,pc} {rightThread with registers:=rightRegisters,pc}
      leftInside ⟨rfl,threads.running,threads.ids,registers⟩

end MemorySafetyExecution.Native
