/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentSemantics
import MemorySafetyExecution.Native.InstructionAccess
import ValueFlow.SourceWriterExecution
import MemorySafety.SourceModel

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native
namespace Independent

abbrev M := MachineModel.State

def layoutModel (layout : value_flow.AllocationLayout) : MachineModel.Layout :=
  ⟨layout.base.val,layout.bytes.val,layout.alignment.val⟩

def memoryModel (memory : execution_core.AllocationBytes) : MachineModel.Memory :=
  ⟨memory.bytes.val.map UScalar.val,memory.defined.val,memory.live⟩

def threadModel (thread : execution_core.ThreadState) : MachineModel.Thread :=
  ⟨thread.pc.val,thread.running,ValueFlow.Source.idsModel thread.ids,
    ValueFlow.Source.bankModel thread.registers⟩

def stateModel (state : CoreState) : MachineModel.State :=
  ⟨state.threads.val.map threadModel,state.allocations.val.map memoryModel⟩

def programModel (program : CoreProgram) : MachineModel.Program :=
  ⟨program.code.val.map ValueFlow.Source.guardedModel,
    program.layouts.val.map layoutModel,program.outputs.val.map UScalar.val⟩

def Canonical (state : CoreState) : Prop :=
  ∀thread∈state.threads.val,
    ValueFlow.Storage.MachineRegistersVirtualMap.Canonical thread.registers.raw.vund.val

theorem model_thread_set (state : CoreState) (actor : Usize)
    (thread : execution_core.ThreadState) :
    stateModel (update_thread state actor thread)=
      MachineModel.setThread (stateModel state) actor.val (threadModel thread) := by
  simp only [stateModel,update_thread,alloc.vec.Vec.set_val_eq,MachineModel.setThread,
    List.map_set]

theorem model_allocations_set (state : CoreState) (index : Usize)
    (memory : execution_core.AllocationBytes) :
    stateModel {state with allocations:=state.allocations.set index memory}=
      {stateModel state with allocations:=(stateModel state).allocations.set index.val (memoryModel memory)} := by
  simp only [stateModel,alloc.vec.Vec.set_val_eq,List.map_set]

theorem canonical_thread_set (state : CoreState) (actor : Usize)
    (thread : execution_core.ThreadState) (before : Canonical state)
    (selected : ValueFlow.Storage.MachineRegistersVirtualMap.Canonical thread.registers.raw.vund.val) :
    Canonical (update_thread state actor thread) := by
  intro other member
  simp only [update_thread,alloc.vec.Vec.set_val_eq] at member
  have m : other∈state.threads.val ∨ other=thread :=
    List.mem_or_eq_of_mem_set member
  rcases m with m|rfl
  · exact before other m
  · exact selected

end Independent
end MemorySafetyExecution.Native
