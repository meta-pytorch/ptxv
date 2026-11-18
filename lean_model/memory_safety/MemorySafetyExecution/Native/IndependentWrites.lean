/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentControl

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent

theorem source_virtual_model (program : CoreProgram) (before after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (event : memory_safety.Event)
    (destination : U16) (value : value_flow.Value)
    (canonical : Canonical before) (selected : before.threads.val[actor.val]?=some thread)
    (nextPc : pc.val=thread.pc.val+1)
    (computed : execution_core.commit_action program before
      {actor,next_pc:=pc,action:=.WriteVirtual destination value}=.ok (.Ok event,after)) :
    MachineModel.applyAction (programModel program) (stateModel before) actor.val (threadModel thread)
      (.writeVirtual destination.val (ValueFlow.Source.valueModel value))=
      some (MemorySafety.Source.eventModel event,stateModel after) ∧ Canonical after := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  obtain ⟨registers,written,rfl,rfl⟩ := source_virtual_commit_components program before after actor pc
    thread destination value event inside selected computed
  have original := canonical thread (List.mem_of_getElem? selected)
  obtain ⟨word,rendered,bank,proper⟩ := source_virtual_bank_model program thread.registers registers
    destination value original written
  constructor
  · simp only [MachineModel.applyAction,rendered,Option.bind_some,Option.pure_def,
      MemorySafety.Source.eventModel,model_thread_set,threadModel,nextPc,bank]
    rfl
  · exact canonical_thread_set before actor _ canonical proper

theorem source_predicate_model (program : CoreProgram) (before after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (event : memory_safety.Event)
    (destination : U16) (value : Bool)
    (canonical : Canonical before) (selected : before.threads.val[actor.val]?=some thread)
    (nextPc : pc.val=thread.pc.val+1)
    (computed : execution_core.commit_action program before
      {actor,next_pc:=pc,action:=.WritePredicate destination value}=.ok (.Ok event,after)) :
    MachineModel.applyAction (programModel program) (stateModel before) actor.val (threadModel thread)
      (.writePredicate destination.val value)=
      some (MemorySafety.Source.eventModel event,stateModel after) ∧ Canonical after := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  obtain ⟨registers,written,rfl,rfl⟩ := source_predicate_commit_components program before after actor pc
    thread destination value event inside selected computed
  have bank := source_predicate_bank_model thread.registers registers destination value written
  constructor
  · simp only [MachineModel.applyAction,MemorySafety.Source.eventModel,model_thread_set,threadModel,nextPc,bank]
  · obtain ⟨bounds,raw,rawRun,afterEq⟩ :=
      ValueFlow.Source.source_write_predicate_components thread.registers registers destination value written
    have rawMeaning := ValueFlow.Storage.MachineRegisters.source_write_predicate thread.registers.raw
      destination value true bounds.1 bounds.2
    rw [rawRun] at rawMeaning
    simp only [WP.spec_ok] at rawMeaning
    apply canonical_thread_set before actor _ canonical
    change ValueFlow.Storage.MachineRegistersVirtualMap.Canonical registers.raw.vund.val
    rw [afterEq,rawMeaning.2.2.1]
    exact canonical thread (List.mem_of_getElem? selected)

end MemorySafetyExecution.Native.Independent
