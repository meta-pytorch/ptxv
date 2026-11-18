/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentModel
import MemorySafetyExecution.Native.StoreValues

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent

theorem store_memory_model (before after : AllocationMemory) (offset count : Nat) (value : U64)
    (frame : ByteFrame before after offset count)
    (payload : StoredPayload after offset count value) :
    memoryModel after=MachineModel.store (memoryModel before) offset count value.val := by
  rcases frame with ⟨live,byteLength,definedLength,outside,defined⟩
  change MachineModel.Memory.mk _ _ after.live=MachineModel.Memory.mk _ _ before.live
  rw [live]
  apply congrArg₂ (fun bytes defined => MachineModel.Memory.mk bytes defined before.live)
  · apply List.ext_getElem
    · simpa only [memoryModel,MachineModel.store,List.length_map,List.length_zipIdx] using byteLength
    · intro index leftBound rightBound
      have oldBound : index<before.bytes.val.length := by simpa only [List.length_map,memoryModel,byteLength] using leftBound
      have afterBound : index<after.bytes.val.length := by simpa only [memoryModel,List.length_map] using leftBound
      simp only [memoryModel,MachineModel.store,List.getElem_map,List.getElem_zipIdx,Nat.zero_add]
      by_cases selected : offset ≤ index ∧ index<offset+count
      · have indexEq : offset+(index-offset)=index := by omega
        have part := payload (index-offset) (by omega)
        simp only [indexEq,List.getElem?_eq_getElem afterBound,Option.map_some,Option.some.injEq] at part
        simpa only [selected,and_self,ite_true] using part
      · have framed := (outside index (by omega)).1
        have words : after.bytes.val[index]=before.bytes.val[index] := by
          simpa only [List.getElem?_eq_getElem afterBound,List.getElem?_eq_getElem oldBound,Option.some.injEq] using framed
        simp only [selected,ite_false,words]
  · apply List.ext_getElem
    · simpa only [memoryModel,MachineModel.store,List.length_map,List.length_zipIdx] using definedLength
    · intro index leftBound rightBound
      have oldBound : index<before.defined.val.length := by simpa only [List.length_map,List.length_zipIdx,memoryModel,MachineModel.store] using rightBound
      have afterBound : index<after.defined.val.length := leftBound
      simp only [memoryModel,MachineModel.store,List.getElem_map,List.getElem_zipIdx,Nat.zero_add]
      by_cases selected : offset ≤ index ∧ index<offset+count
      · have written := defined index selected.1 selected.2
        simpa only [selected,and_self,ite_true,List.getElem?_eq_getElem afterBound,Option.some.injEq] using written
      · have framed := (outside index (by omega)).2
        have words : after.defined.val[index]=before.defined.val[index] := by
          simpa only [List.getElem?_eq_getElem afterBound,List.getElem?_eq_getElem oldBound,Option.some.injEq] using framed
        simp only [selected,ite_false,words]


theorem source_store_state_model (state after : CoreState) (access : execution_core.ResolvedAccess)
    (memory : AllocationMemory) (value : U64)
    (selected : state.allocations.val[access.allocation.val]?=some memory)
    (small : access.width.val ≤ 8)
    (bounded : access.offset.val+access.width.val ≤ memory.bytes.val.length)
    (definedBound : access.offset.val+access.width.val ≤ memory.defined.val.length)
    (stored : execution_core.store_value state access value=.ok after) :
    stateModel after={stateModel state with allocations:=(stateModel state).allocations.set access.allocation.val (MachineModel.store (memoryModel memory) access.offset.val access.width.val value.val)} ∧
      after.threads=state.threads := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  have memoryEq := (List.getElem?_eq_some_iff.mp selected).2
  have frameSpec := source_store_value_frame state access value inside small
    (by simpa only [memoryEq] using bounded) (by simpa only [memoryEq] using definedBound)
  rw [stored] at frameSpec
  have framed : ∃written,
      after={state with allocations:=state.allocations.set access.allocation written} ∧
      ByteFrame memory written access.offset.val access.width.val := by
    simpa only [WP.spec_ok,memoryEq] using frameSpec
  obtain ⟨written,writtenEq,frame⟩ := framed
  have payloadSpec := source_store_value_payload state access value inside small
    (by simpa only [memoryEq] using bounded) (by simpa only [memoryEq] using definedBound)
  rw [stored] at payloadSpec
  have payloads : ∃other,
      after={state with allocations:=state.allocations.set access.allocation other} ∧
      StoredPayload other access.offset.val access.width.val value := by
    simpa only [WP.spec_ok] using payloadSpec
  obtain ⟨other,otherEq,payload⟩ := payloads
  have selectedWritten : after.allocations.val[access.allocation.val]?=some written := by
    simp only [writtenEq,alloc.vec.Vec.set_val_eq,List.getElem?_set_self inside]
  have selectedOther : after.allocations.val[access.allocation.val]?=some other := by
    simp only [otherEq,alloc.vec.Vec.set_val_eq,List.getElem?_set_self inside]
  have same : other=written := Option.some.inj (selectedOther.symm.trans selectedWritten)
  subst other
  constructor
  · rw [writtenEq,model_allocations_set,store_memory_model memory written access.offset.val access.width.val value frame payload]
  · simp only [writtenEq]

end MemorySafetyExecution.Native.Independent
