/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputOutputCopy

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem successful_false (condition : Prop) [Decidable condition] (fault : Fault)
    (before after : alloc.vec.Vec U8)
    (next : Result ((core.result.Result Unit Fault) × alloc.vec.Vec U8))
    (computed : (if condition then .ok (.Err fault,before) else next)=.ok (.Ok (),after)) :
    ¬condition ∧ next=.ok (.Ok (),after) := by
  by_cases h:condition <;> simp_all

private theorem successful_true (condition : Prop) [Decidable condition] (fault : Fault)
    (before after : alloc.vec.Vec U8)
    (next : Result ((core.result.Result Unit Fault) × alloc.vec.Vec U8))
    (computed : (if condition then next else .ok (.Err fault,before))=.ok (.Ok (),after)) :
    condition ∧ next=.ok (.Ok (),after) := by
  by_cases h:condition <;> simp_all

theorem source_append_output_inputs (shadow : value_flow_dependencies.State) (left right : CoreState)
    (index : Usize) (leftOutput rightOutput leftAfter : alloc.vec.Vec U8)
    (related : StateInputsRelated shadow left right)
    (outputLength : leftOutput.val.length=rightOutput.val.length)
    (computed : execution_core.append_output left index leftOutput=.ok (.Ok (),leftAfter)) :
    ∃ rightAfter, execution_core.append_output right index rightOutput=.ok (.Ok (),rightAfter) ∧
      leftAfter.val.length=rightAfter.val.length := by
  unfold execution_core.append_output at computed ⊢
  obtain ⟨available,computed⟩ := successful_false _ _ _ _ _ computed
  obtain ⟨leftMemory,leftRead,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
  obtain ⟨live,computed⟩ := successful_true _ _ _ _ _ computed
  obtain ⟨shape,computed⟩ := successful_false _ _ _ _ _ computed
  obtain ⟨defined,definedRun,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
  cases defined with
  | false => simp at computed
  | true =>
    obtain ⟨copied,copiedRun,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
    have copiedEq : copied=leftAfter := by simpa only [ok.injEq,Prod.mk.injEq,true_and] using computed
    subst copied
    have leftInside : index.val<left.allocations.val.length := by scalar_tac
    have rightInside : index.val<right.allocations.val.length := by have := related.allocationsLength; omega
    have shadowInside : index.val<shadow.memory.val.length := by have := related.allocationMetadataLength; omega
    have leftAt := MemorySafety.Source.vec_index_success left.allocations index leftMemory
      (by simpa only [alloc.vec.Vec.index_slice_index] using leftRead)
    let rightMemory:=right.allocations.val[index.val]
    let metadata:=shadow.memory.val[index.val]
    have rightAt : right.allocations.val[index.val]?=some rightMemory := List.getElem?_eq_getElem rightInside
    have shadowAt : shadow.memory.val[index.val]?=some metadata := List.getElem?_eq_getElem shadowInside
    have memory:=related.allocations index.val leftMemory rightMemory metadata leftAt rightAt shadowAt
    obtain ⟨rightAfter,rightCopied,afterLength⟩ := source_output_copy_inputs leftMemory.bytes rightMemory.bytes
      leftOutput rightOutput leftAfter 0#usize memory.bytesLength outputLength copiedRun
    have rightRead : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice execution_core.AllocationBytes)
        right.allocations index=.ok rightMemory := by
      simp only [alloc.vec.Vec.index_slice_index,alloc.vec.Vec.index_usize,
        alloc.vec.Vec.getElem?_Nat_eq,rightAt]
    have rightAvailable : ¬index≥alloc.vec.Vec.len right.allocations := by scalar_tac
    have byteLengths : alloc.vec.Vec.len leftMemory.bytes=alloc.vec.Vec.len rightMemory.bytes := by
      have same:=memory.bytesLength
      scalar_tac
    refine ⟨rightAfter,?_,afterLength⟩
    simp only [rightAvailable,ite_false,rightRead,bind_ok,←memory.live,live,ite_true,
      ←memory.definitions,←byteLengths,shape,definedRun,rightCopied,Bool.false_eq_true]

end MemorySafetyExecution.Native
