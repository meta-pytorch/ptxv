/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputActions
import MemorySafetyExecution.Native.InputCommitControl
import MemorySafetyExecution.Native.InputCommitWrites
import MemorySafetyExecution.Native.InputCommitStore
import MemorySafetyExecution.Native.InputCommitLoad

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

theorem source_commit_inputs (program : CoreProgram) (left right leftAfter : CoreState)
    (shadow shadowAfter : DependencyState) (actor : Usize) (pc : U32)
    (leftThread rightThread : execution_core.ThreadState) (dependencies : DependencyRegisters)
    (instruction : V.GuardedInstruction) (leftAction rightAction : V.Action)
    (loaded : MetadataVector) (transferred : DependencyTransfer) (event : memory_safety.Event)
    (related : StateInputsRelated shadow left right)
    (leftAt : left.threads.val[actor.val]?=some leftThread)
    (rightAt : right.threads.val[actor.val]?=some rightThread)
    (metadataAt : shadow.threads.val[actor.val]?=some dependencies)
    (actions : ActionsRelated (transferTaint transferred) (actionModel leftAction) (actionModel rightAction))
    (loadedComputed : input_independent_execution.action_metadata program left shadow leftAction=.ok (some loaded))
    (transferComputed : value_flow_dependencies.transfer instruction leftAction dependencies (alloc.vec.Vec.deref loaded)=.ok (.Ok transferred))
    (applied : input_independent_execution.apply_transfer program left shadow actor leftAction transferred=.ok (some shadowAfter))
    (captured : execution_core.commit_action program left {actor,next_pc:=pc,action:=leftAction}=.ok (.Ok event,leftAfter)) :
    ∃ rightAfter, execution_core.commit_action program right {actor,next_pc:=pc,action:=rightAction}=.ok (.Ok event,rightAfter) ∧
      StateInputsRelated shadowAfter leftAfter rightAfter := by
  have nativeRelated := source_actions_inputs leftAction rightAction (transferTaint transferred) actions
  obtain ⟨condition,transferDefined,transferShape⟩ := source_transfer_success instruction leftAction dependencies
    (alloc.vec.Vec.deref loaded) transferred transferComputed
  cases leftAction with
  | Next =>
    cases rightAction <;> simp only [NativeActionsRelated,reduceCtorEq] at nativeRelated
    cases transferred <;> simp only [TransferMatches] at transferShape
    have shadowEq : shadowAfter=shadow := by
      simpa only [source_apply_transfer_none,ok.injEq,Option.some.injEq] using applied.symm
    subst shadowAfter
    exact source_next_commit_inputs program left right leftAfter shadow actor pc leftThread rightThread dependencies event
      related leftAt rightAt metadataAt captured
  | Halt =>
    cases rightAction <;> simp only [NativeActionsRelated,reduceCtorEq] at nativeRelated
    cases transferred <;> simp only [TransferMatches] at transferShape
    have shadowEq : shadowAfter=shadow := by
      simpa only [source_apply_transfer_none,ok.injEq,Option.some.injEq] using applied.symm
    subst shadowAfter
    exact source_halt_commit_inputs program left right leftAfter shadow actor pc leftThread rightThread dependencies event
      related leftAt rightAt metadataAt captured
  | Jump target =>
    cases rightAction <;> simp only [NativeActionsRelated,reduceCtorEq,value_flow.Action.Jump.injEq] at nativeRelated
    cases nativeRelated
    cases transferred <;> simp only [TransferMatches] at transferShape
    have shadowEq : shadowAfter=shadow := by
      simpa only [source_apply_transfer_none,ok.injEq,Option.some.injEq] using applied.symm
    subst shadowAfter
    exact source_jump_commit_inputs program left right leftAfter shadow actor pc target leftThread rightThread dependencies event
      related leftAt rightAt metadataAt captured
  | WriteVirtual destination value =>
    cases rightAction <;> simp only [NativeActionsRelated,reduceCtorEq] at nativeRelated
    obtain ⟨rfl,values⟩ := nativeRelated
    cases transferred <;> simp only [TransferMatches] at transferShape
    cases transferShape
    exact source_virtual_commit_inputs program left right leftAfter shadow shadowAfter actor pc leftThread rightThread dependencies
      _ value _ _ event related leftAt rightAt metadataAt transferDefined values captured applied
  | WritePredicate destination value =>
    cases rightAction <;> simp only [NativeActionsRelated,reduceCtorEq] at nativeRelated
    obtain ⟨rfl,values⟩ := nativeRelated
    cases transferred <;> simp only [TransferMatches] at transferShape
    cases transferShape
    exact source_predicate_commit_inputs program left right leftAfter shadow shadowAfter actor pc leftThread rightThread dependencies
      _ value _ _ event related leftAt rightAt metadataAt transferDefined values captured applied
  | Load destination bits bytes signed pointer =>
    cases rightAction <;> simp only [NativeActionsRelated,reduceCtorEq,value_flow.Action.Load.injEq] at nativeRelated
    obtain ⟨rfl,rfl,rfl,rfl,rfl⟩ := nativeRelated
    obtain ⟨metadata,joined,transferEq⟩ := source_transfer_load_computation instruction destination bits bytes signed pointer
      dependencies (alloc.vec.Vec.deref loaded) transferred transferComputed
    rw [transferEq] at applied transferDefined
    have joinedSlice : value_flow_dependencies.loaded_metadata loaded.slice bytes=.ok (.Ok metadata) := by
      simpa [alloc.vec.Vec.deref,alloc.vec.Vec.val] using joined
    exact source_load_commit_inputs program left right leftAfter shadow shadowAfter actor pc leftThread rightThread dependencies
      destination bits bytes signed pointer metadata loaded event related leftAt rightAt metadataAt transferDefined
      loadedComputed joinedSlice captured applied
  | Store bytes pointer value =>
    cases rightAction <;> simp only [NativeActionsRelated,reduceCtorEq] at nativeRelated
    obtain ⟨rfl,rfl,values⟩ := nativeRelated
    cases transferred <;> simp only [TransferMatches] at transferShape
    exact source_store_commit_inputs program left right leftAfter shadow shadowAfter actor pc leftThread rightThread dependencies
      _ _ value _ _ event related leftAt rightAt metadataAt transferDefined values captured applied

end MemorySafetyExecution.Native
