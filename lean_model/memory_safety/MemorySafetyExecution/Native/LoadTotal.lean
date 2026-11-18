/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.LoadValues

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_load_loop_total (memory : execution_core.AllocationBytes) (offset count : Usize)
    (bounded : offset.val + count.val ≤ memory.bytes.val.length) (small : count.val ≤ 8)
    (value : U64) (index : Usize) (within : index.val ≤ count.val) :
    execution_core.load_value_loop offset count memory value index ⦃ _ => True ⦄ := by
  unfold execution_core.load_value_loop
  apply loop.spec_decr_nat (fun current : U64 × Usize => count.val - current.2.val)
    (fun current : U64 × Usize => current.2.val ≤ count.val)
  · rintro ⟨value, index⟩ within
    change index.val ≤ count.val at within
    unfold execution_core.load_value_loop.body
    by_cases inside : index < count
    · simp only [inside, ite_true]
      step as ⟨position, positionVal⟩
      step as ⟨byte, byteEq⟩
      simp only [lift, bind_ok]
      step as ⟨shift, shiftVal⟩
      step as ⟨part, partVal, partBits⟩
      step as ⟨next, nextVal⟩
      exact ⟨by scalar_tac, by scalar_tac⟩
    · simp only [inside, ite_false, WP.spec_ok]
  · exact within

theorem source_load_value_total (state : CoreState) (access : execution_core.ResolvedAccess)
    (memory : execution_core.AllocationBytes) (signed : Bool)
    (selected : state.allocations.val[access.allocation.val]? = some memory)
    (bounded : access.offset.val + access.width.val ≤ memory.bytes.val.length)
    (nonzero : 0 < access.width.val) (small : access.width.val ≤ 8) :
    execution_core.load_value state access signed ⦃ _ => True ⦄ := by
  unfold execution_core.load_value
  simp only [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    alloc.vec.Vec.getElem?_Nat_eq, selected, bind_ok]
  step with source_load_loop_total memory access.offset access.width bounded small
    0#u64 0#usize (by scalar_tac) as ⟨value⟩
  split
  · split
    · step as ⟨bits, bitsVal⟩
      step as ⟨shift, shiftVal⟩
      step as ⟨shifted, shiftedVal, shiftedBits⟩
      simp only [lift, bind_ok]
      step as ⟨result, resultVal, resultBits⟩
    · simp only [WP.spec_ok]
  · simp only [WP.spec_ok]

end MemorySafetyExecution.Native
