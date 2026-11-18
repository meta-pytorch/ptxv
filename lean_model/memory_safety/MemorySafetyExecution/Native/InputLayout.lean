/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputState

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem layout_body_same_shape (program : CoreProgram) (dependencies : value_flow_dependencies.State)
    (left right : CoreState) (related : StateInputsRelated dependencies left right) (valid : Bool) (index : Usize) :
    execution_core.layout_valid_loop.body program.code program.layouts program.initial program.outputs
      left.allocations valid index =
    execution_core.layout_valid_loop.body program.code program.layouts program.initial program.outputs
      right.allocations valid index := by
  have allocationLengths := related.allocationsLength
  by_cases inside : index.val < left.allocations.val.length
  · have rightInside : index.val < right.allocations.val.length := by omega
    have metadataLength := related.allocationMetadataLength
    have metadataInside : index.val < dependencies.memory.val.length := by omega
    have memory := related.allocations index.val left.allocations.val[index.val]
      right.allocations.val[index.val] dependencies.memory.val[index.val]
      (List.getElem?_eq_getElem inside) (List.getElem?_eq_getElem rightInside)
      (List.getElem?_eq_getElem metadataInside)
    have bytesEq : alloc.vec.Vec.len left.allocations.val[index.val].bytes =
        alloc.vec.Vec.len right.allocations.val[index.val].bytes := by
      apply UScalar.eq_of_val_eq
      simpa using memory.bytesLength
    have definedEq : alloc.vec.Vec.len left.allocations.val[index.val].defined =
        alloc.vec.Vec.len right.allocations.val[index.val].defined := by rw [memory.definitions]
    simp only [execution_core.layout_valid_loop.body, alloc.vec.Vec.index_slice_index,
      alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
      List.getElem?_eq_getElem inside, List.getElem?_eq_getElem rightInside, bind_ok,
      bytesEq, definedEq]
  · have rightOutside : ¬index.val < right.allocations.val.length := by omega
    have leftMissing : left.allocations.val[index.val]? = none := List.getElem?_eq_none (by omega)
    have rightMissing : right.allocations.val[index.val]? = none := List.getElem?_eq_none (by omega)
    simp only [execution_core.layout_valid_loop.body, alloc.vec.Vec.index_slice_index,
      alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
      leftMissing, rightMissing]

theorem source_layout_valid_same_shape (program : CoreProgram) (dependencies : value_flow_dependencies.State)
    (left right : CoreState) (related : StateInputsRelated dependencies left right) :
    execution_core.layout_valid program left = execution_core.layout_valid program right := by
  unfold execution_core.layout_valid
  have lengths : alloc.vec.Vec.len left.allocations = alloc.vec.Vec.len right.allocations := by
    apply UScalar.eq_of_val_eq
    simpa using related.allocationsLength
  rw [lengths]
  unfold execution_core.layout_valid_loop
  dsimp only
  congr 1
  funext current
  exact layout_body_same_shape program dependencies left right related current.1 current.2

end MemorySafetyExecution.Native
