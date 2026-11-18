/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Preparation

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem load_body_same_span (left right : ptxv_memory_write.execution_core.AllocationBytes)
    (offset count : Usize) (bounded : offset.val + count.val ≤ UScalar.max .Usize)
    (same : ∀ index, index < count.val →
      left.bytes.val[offset.val + index]? = right.bytes.val[offset.val + index]?)
    (value : U64) (index : Usize) :
    ptxv_memory_write.execution_core.load_value_loop.body offset count left value index =
      ptxv_memory_write.execution_core.load_value_loop.body offset count right value index := by
  unfold ptxv_memory_write.execution_core.load_value_loop.body
  by_cases inside : index < count
  · simp only [inside, ite_true]
    have addSpec := UScalar.add_spec (x := offset) (y := index) (by scalar_tac : offset.val + index.val ≤ UScalar.max .Usize)
    obtain ⟨position, added, positionVal⟩ := MemoryWrite.spec_returns _ _ addSpec
    simp only [added, bind_ok]
    have readSame : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U8) left.bytes position =
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U8) right.bytes position := by
      simp only [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
        alloc.vec.Vec.getElem?_Nat_eq, positionVal]
      rw [same index.val (by scalar_tac)]
    rw [readSame]
  · simp only [inside, ite_false]

theorem source_load_loop_same_span (left right : ptxv_memory_write.execution_core.AllocationBytes)
    (offset count : Usize) (bounded : offset.val + count.val ≤ UScalar.max .Usize)
    (same : ∀ index, index < count.val →
      left.bytes.val[offset.val + index]? = right.bytes.val[offset.val + index]?)
    (value : U64) (index : Usize) :
    ptxv_memory_write.execution_core.load_value_loop offset count left value index =
      ptxv_memory_write.execution_core.load_value_loop offset count right value index := by
  unfold ptxv_memory_write.execution_core.load_value_loop
  congr 1
  funext current
  exact load_body_same_span left right offset count bounded same current.1 current.2

theorem source_load_value_same_span (left right : CoreState)
    (leftAccess rightAccess : ptxv_memory_write.execution_core.ResolvedAccess) (signed : Bool)
    (leftMemory rightMemory : ptxv_memory_write.execution_core.AllocationBytes)
    (leftSelected : left.allocations.val[leftAccess.allocation.val]? = some leftMemory)
    (rightSelected : right.allocations.val[rightAccess.allocation.val]? = some rightMemory)
    (sameOffset : leftAccess.offset = rightAccess.offset)
    (sameWidth : leftAccess.width = rightAccess.width)
    (bounded : leftAccess.offset.val + leftAccess.width.val ≤ UScalar.max .Usize)
    (same : ∀ index, index < leftAccess.width.val →
      leftMemory.bytes.val[leftAccess.offset.val + index]? =
        rightMemory.bytes.val[rightAccess.offset.val + index]?) :
    ptxv_memory_write.execution_core.load_value left leftAccess signed =
      ptxv_memory_write.execution_core.load_value right rightAccess signed := by
  unfold ptxv_memory_write.execution_core.load_value
  simp only [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    alloc.vec.Vec.getElem?_Nat_eq, leftSelected, rightSelected, bind_ok]
  rw [source_load_loop_same_span leftMemory rightMemory leftAccess.offset leftAccess.width bounded
    (by simpa only [sameOffset] using same)]
  simp only [sameOffset, sameWidth]

end MemorySafetyExecution.Native
