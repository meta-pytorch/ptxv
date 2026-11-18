/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputLayout

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_resolve_inputs (program : CoreProgram) (dependencies : value_flow_dependencies.State)
    (left right : CoreState) (related : StateInputsRelated dependencies left right)
    (pointer : value_flow.Pointer) (width : U8) (write : Bool) :
    execution_core.resolve_access program left pointer width write =
      execution_core.resolve_access program right pointer width write := by
  have allocationLengths := related.allocationsLength
  have lengths : alloc.vec.Vec.len left.allocations = alloc.vec.Vec.len right.allocations := by scalar_tac
  by_cases inside : pointer.allocation.val < left.allocations.val.length
  · have rightInside : pointer.allocation.val < right.allocations.val.length := by omega
    have metadataLength := related.allocationMetadataLength
    have metadataInside : pointer.allocation.val < dependencies.memory.val.length := by omega
    have memory := related.allocations pointer.allocation.val left.allocations.val[pointer.allocation.val]
      right.allocations.val[pointer.allocation.val] dependencies.memory.val[pointer.allocation.val]
      (List.getElem?_eq_getElem inside) (List.getElem?_eq_getElem rightInside)
      (List.getElem?_eq_getElem metadataInside)
    have bytesEq : alloc.vec.Vec.len left.allocations.val[pointer.allocation.val].bytes =
        alloc.vec.Vec.len right.allocations.val[pointer.allocation.val].bytes := by
      apply UScalar.eq_of_val_eq
      simpa using memory.bytesLength
    simp only [execution_core.resolve_access, lengths, alloc.vec.Vec.index_slice_index,
      alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
      List.getElem?_eq_getElem inside, List.getElem?_eq_getElem rightInside, bind_ok,
      bytesEq, memory.definitions, memory.live]
  · have outside : pointer.allocation ≥ alloc.vec.Vec.len right.allocations := by scalar_tac
    simp only [execution_core.resolve_access, lengths, outside, ite_true]

end MemorySafetyExecution.Native
