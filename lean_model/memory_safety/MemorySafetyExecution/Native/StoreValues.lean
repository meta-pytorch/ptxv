/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.ByteEffects

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

def StoredPayload (memory : AllocationMemory) (offset count : Nat) (value : U64) : Prop :=
  ∀ index, index < count → (memory.bytes.val[offset + index]?).map UScalar.val =
    some ((value.val / 2 ^ (8 * index)) % 256)

private theorem payload_set (memory : AllocationMemory) (offset count : Nat)
    (value : U64) (position : Usize) (byte : U8)
    (positionVal : position.val = offset + count)
    (inside : position.val < memory.bytes.val.length)
    (byteVal : byte.val = (value.val / 2 ^ (8 * count)) % 256)
    (payload : StoredPayload memory offset count value) :
    StoredPayload
      { memory with bytes := (memory.bytes.set position byte), defined := (memory.defined.set position true) }
      offset (count + 1) value := by
  intro index bounded
  by_cases same : index = count
  · subst index
    simp only [alloc.vec.Vec.set_val_eq, ←positionVal,
      List.getElem?_set_self inside, Option.map_some, byteVal]
  · have different : position.val ≠ offset + index := by omega
    simp only [alloc.vec.Vec.set_val_eq, List.getElem?_set_ne different]
    exact payload index (by omega)

private theorem source_store_loop_payload (offset count : Usize) (value : U64)
    (before memory : AllocationMemory) (cursor : Usize)
    (small : count.val ≤ 8) (progress : cursor.val ≤ count.val)
    (fitsBytes : offset.val + count.val ≤ before.bytes.val.length)
    (fitsDefined : offset.val + count.val ≤ before.defined.val.length)
    (bytesLength : memory.bytes.val.length = before.bytes.val.length)
    (definedLength : memory.defined.val.length = before.defined.val.length)
    (payload : StoredPayload memory offset.val cursor.val value) :
    execution_core.store_value_loop offset count value memory cursor
      ⦃ output => StoredPayload output offset.val count.val value ⦄ := by
  unfold execution_core.store_value_loop
  apply loop.spec_decr_nat
    (fun current : AllocationMemory × Usize => count.val - current.2.val)
    (fun current : AllocationMemory × Usize => current.2.val ≤ count.val ∧
      current.1.bytes.val.length = before.bytes.val.length ∧
      current.1.defined.val.length = before.defined.val.length ∧
      StoredPayload current.1 offset.val current.2.val value)
  · rintro ⟨memory, cursor⟩ ⟨progress, bytesLength, definedLength, payload⟩
    change cursor.val ≤ count.val at progress
    unfold execution_core.store_value_loop.body
    by_cases inside : cursor.val < count.val
    · have test : cursor < count := by scalar_tac
      simp only [test, ite_true]
      step as ⟨shift, shiftVal⟩
      step as ⟨shifted, shiftedVal⟩
      step as ⟨position, positionVal⟩
      have byteBound : position.val < memory.bytes.val.length := by scalar_tac
      have definedBound : position.val < memory.defined.val.length := by scalar_tac
      step as ⟨_, bytesBack, bytesRead, bytesBackEq⟩
      step as ⟨byte, byteEq⟩
      step as ⟨_, definedBack, definedRead, definedBackEq⟩
      step as ⟨next, nextVal⟩
      simp only [bytesBackEq, definedBackEq]
      have byteVal : byte.val = (value.val / 2 ^ (8 * cursor.val)) % 256 := by
        rw [byteEq]
        simp only [UScalar.cast_val_eq, UScalarTy.numBits, shiftedVal, shiftVal,
          Nat.shiftRight_eq_div_pow, Nat.mul_comm]
        rfl
      refine ⟨by scalar_tac, by simpa using bytesLength, by simpa using definedLength, ?_, by scalar_tac⟩
      have changed := payload_set memory offset.val cursor.val value position byte positionVal byteBound byteVal payload
      simpa only [nextVal] using changed
    · have test : ¬cursor < count := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      have same : cursor.val = count.val := by omega
      simpa only [same] using payload
  · exact ⟨progress, bytesLength, definedLength, payload⟩

theorem source_store_value_payload (state : CoreState) (access : execution_core.ResolvedAccess) (value : U64)
    (inside : access.allocation.val < state.allocations.val.length)
    (small : access.width.val ≤ 8)
    (fitsBytes : access.offset.val + access.width.val ≤
      state.allocations.val[access.allocation.val].bytes.val.length)
    (fitsDefined : access.offset.val + access.width.val ≤
      state.allocations.val[access.allocation.val].defined.val.length) :
    execution_core.store_value state access value ⦃ output => ∃ memory,
      output = { state with allocations := state.allocations.set access.allocation memory } ∧
      StoredPayload memory access.offset.val access.width.val value ⦄ := by
  unfold execution_core.store_value
  step as ⟨memory, back, memoryEq, backEq⟩
  have byteBound : access.offset.val + access.width.val ≤ memory.bytes.val.length := by simpa only [memoryEq] using fitsBytes
  have definedBound : access.offset.val + access.width.val ≤ memory.defined.val.length := by simpa only [memoryEq] using fitsDefined
  step with source_store_loop_payload access.offset access.width value memory memory 0#usize small
    (by scalar_tac) byteBound definedBound rfl rfl (by simp [StoredPayload]) as ⟨written, writtenPayload⟩
  exact ⟨written, by simp only [backEq], writtenPayload⟩

theorem stored_payload_same (left right : AllocationMemory) (offset count : Nat) (value : U64)
    (leftWritten : StoredPayload left offset count value) (rightWritten : StoredPayload right offset count value)
    (index : Nat) (inside : index < count) :
    left.bytes.val[offset + index]? = right.bytes.val[offset + index]? := by
  have leftValue := leftWritten index inside
  have rightValue := rightWritten index inside
  cases l : left.bytes.val[offset + index]? with
  | none => simp [l] at leftValue
  | some a =>
    cases r : right.bytes.val[offset + index]? with
    | none => simp [r] at rightValue
    | some b =>
      have same : a.val = b.val := by
        have la : a.val = (value.val / 2 ^ (8 * index)) % 256 := by simpa only [l, Option.map_some, Option.some.injEq] using leftValue
        have rb : b.val = (value.val / 2 ^ (8 * index)) % 256 := by simpa only [r, Option.map_some, Option.some.injEq] using rightValue
        exact la.trans rb.symm
      rw [UScalar.eq_of_val_eq same]

end MemorySafetyExecution.Native
