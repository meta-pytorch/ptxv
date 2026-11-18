/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

abbrev AllocationMemory := ptxv_memory_write.execution_core.AllocationBytes

/-- Native stores preserve every byte outside their exact contiguous span. -/
def ByteFrame (before after : AllocationMemory) (offset count : Nat) : Prop :=
  after.live = before.live ∧
  after.bytes.val.length = before.bytes.val.length ∧
  after.defined.val.length = before.defined.val.length ∧
  (∀ index, index < offset ∨ offset + count ≤ index →
    after.bytes.val[index]? = before.bytes.val[index]? ∧
    after.defined.val[index]? = before.defined.val[index]?) ∧
  (∀ index, offset ≤ index → index < offset + count →
    after.defined.val[index]? = some true)

private theorem byte_frame_zero (before : AllocationMemory) (offset : Nat) :
    ByteFrame before before offset 0 := by
  simp [ByteFrame]
  omega

private theorem byte_frame_set (before current : AllocationMemory)
    (offset count : Nat) (index : Usize) (byte : U8)
    (indexEq : index.val = offset + count)
    (inside : index.val < current.defined.val.length)
    (framed : ByteFrame before current offset count) :
    ByteFrame before
      { current with bytes := (current.bytes.set index byte), defined := (current.defined.set index true) } offset (count + 1) := by
  rcases framed with ⟨live, byteLength, definedLength, outside, written⟩
  refine ⟨live, by simpa using byteLength, by simpa using definedLength, ?_, ?_⟩
  · intro cursor beyond
    have different : index.val ≠ cursor := by omega
    simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_ne different]
    exact outside cursor (by omega)
  · intro cursor lower upper
    by_cases same : index.val = cursor
    · subst cursor
      simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_self inside]
    · simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_ne same]
      exact written cursor lower (by omega)

private theorem source_store_loop_frame
    (offset count : Usize) (value : U64) (before memory : AllocationMemory) (cursor : Usize)
    (small : count.val ≤ 8) (progress : cursor.val ≤ count.val)
    (fitsBytes : offset.val + count.val ≤ before.bytes.val.length)
    (fitsDefined : offset.val + count.val ≤ before.defined.val.length)
    (framed : ByteFrame before memory offset.val cursor.val) :
    ptxv_memory_write.execution_core.store_value_loop offset count value memory cursor
      ⦃ output => ByteFrame before output offset.val count.val ⦄ := by
  unfold ptxv_memory_write.execution_core.store_value_loop
  apply loop.spec_decr_nat
    (fun current : AllocationMemory × Usize => count.val - current.2.val)
    (fun current : AllocationMemory × Usize =>
      current.2.val ≤ count.val ∧ ByteFrame before current.1 offset.val current.2.val)
  · rintro ⟨memory, cursor⟩ ⟨progress, framed⟩
    change cursor.val ≤ count.val at progress
    change ByteFrame before memory offset.val cursor.val at framed
    unfold ptxv_memory_write.execution_core.store_value_loop.body
    by_cases inside : cursor.val < count.val
    · have test : cursor < count := by scalar_tac
      simp only [test, ite_true]
      step as ⟨shift, shiftEq⟩
      step as ⟨shifted, shiftedEq⟩
      step as ⟨index, indexEq⟩
      have byteBound : index.val < memory.bytes.val.length := by
        have := framed.2.1
        scalar_tac
      have definedBound : index.val < memory.defined.val.length := by
        have := framed.2.2.1
        scalar_tac
      step as ⟨_, bytesBack, bytesRead, bytesBackEq⟩
      step as ⟨byte, byteEq⟩
      step as ⟨_, definedBack, definedRead, definedBackEq⟩
      step as ⟨next, nextEq⟩
      simp only [bytesBackEq, definedBackEq]
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      have frame := byte_frame_set before memory offset.val cursor.val index byte indexEq definedBound framed
      simpa only [nextEq] using frame
    · have test : ¬cursor < count := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      have same : cursor.val = count.val := by omega
      simpa only [same] using framed
  · exact ⟨progress, framed⟩

theorem source_store_value_frame
    (state : CoreState) (access : ptxv_memory_write.execution_core.ResolvedAccess) (value : U64)
    (inside : access.allocation.val < state.allocations.val.length)
    (small : access.width.val ≤ 8)
    (fitsBytes : access.offset.val + access.width.val ≤
      state.allocations.val[access.allocation.val].bytes.val.length)
    (fitsDefined : access.offset.val + access.width.val ≤
      state.allocations.val[access.allocation.val].defined.val.length) :
    ptxv_memory_write.execution_core.store_value state access value
      ⦃ output => ∃ memory,
        output = { state with allocations := state.allocations.set access.allocation memory } ∧
        ByteFrame state.allocations.val[access.allocation.val] memory
          access.offset.val access.width.val ⦄ := by
  unfold ptxv_memory_write.execution_core.store_value
  step as ⟨memory, back, memoryEq, backEq⟩
  have byteBound : access.offset.val + access.width.val ≤ memory.bytes.val.length := by
    simpa only [memoryEq] using fitsBytes
  have definedBound : access.offset.val + access.width.val ≤ memory.defined.val.length := by
    simpa only [memoryEq] using fitsDefined
  step with source_store_loop_frame access.offset access.width value memory memory 0#usize small
    (by scalar_tac) byteBound definedBound (byte_frame_zero memory access.offset.val)
    as ⟨written, writtenFrame⟩
  exact ⟨written, by simp only [backEq], by simpa only [memoryEq] using writtenFrame⟩

end MemorySafetyExecution.Native
