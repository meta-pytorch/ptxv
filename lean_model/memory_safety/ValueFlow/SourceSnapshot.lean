/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceTypes
import MemorySafety.Code.Funs
import MemorySafety.SourceVectors
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result
open PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

theorem source_snapshot_loop (values pointers : alloc.vec.Vec (Option V.Pointer)) (index : Usize)
    (bounded : index.val ≤ values.val.length)
    (copied : pointers.val = values.val.take index.val) :
    ptxv_memory_write.value_flow.snapshot_registers_loop values pointers index
      ⦃ output => output = values ⦄ := by
  unfold ptxv_memory_write.value_flow.snapshot_registers_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (Option V.Pointer) × Usize => values.val.length - current.2.val)
    (fun current : alloc.vec.Vec (Option V.Pointer) × Usize =>
      current.2.val ≤ values.val.length ∧ current.1.val = values.val.take current.2.val)
  · rintro ⟨pointers, index⟩ ⟨bounded, copied⟩
    change index.val ≤ values.val.length at bounded
    change pointers.val = values.val.take index.val at copied
    unfold ptxv_memory_write.value_flow.snapshot_registers_loop.body
    by_cases inside : index.val < values.val.length
    · have test : index < alloc.vec.Vec.len values := by scalar_tac
      simp only [test, ite_true]
      step as ⟨entry, entryEq⟩
      cases entry <;> simp only [bind_ok]
      all_goals
        have copiedLength : pointers.val.length = index.val := by
          simp [copied, List.length_take, Nat.min_eq_left bounded]
        step with alloc.vec.Vec.push_spec pointers _
          (by have := alloc.vec.Vec.len_ineq values; omega) as ⟨updated, appended⟩
        step as ⟨next, nextVal⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        rw [appended, nextVal, List.take_succ_eq_append_getElem inside, copied]
        exact congrArg (fun value => values.val.take index.val ++ [value]) entryEq
    · have test : ¬index < alloc.vec.Vec.len values := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      apply alloc.vec.Vec.ext
      rw [copied]
      exact List.take_of_length_le (by omega)
  · exact ⟨bounded, copied⟩

theorem source_snapshot_tags (values : alloc.vec.Vec (Option V.Pointer)) :
    ptxv_memory_write.value_flow.snapshot_registers_loop values
      (alloc.vec.Vec.new (Option V.Pointer)) 0#usize = .ok values := by
  have computed := source_snapshot_loop values (alloc.vec.Vec.new _) 0#usize (by scalar_tac) (by simp)
  obtain ⟨after, accepted, same⟩ := MemoryWrite.spec_returns _ _ computed
  simpa only [same] using accepted

theorem source_snapshot_registers (registers : V.Registers) :
    ptxv_memory_write.value_flow.snapshot_registers registers = .ok registers := by
  unfold ptxv_memory_write.value_flow.snapshot_registers
  rw [MemorySafety.Source.vec_clone_exact core.clone.CloneU64 (by intro value; rfl)]
  simp only [bind_ok]
  rw [MemorySafety.Source.vec_clone_exact core.clone.CloneU64 (by intro value; rfl)]
  simp only [bind_ok]
  rw [MemorySafety.Source.vec_clone_exact core.clone.CloneU64 (by intro value; rfl)]
  simp only [bind_ok]
  rw [MemorySafety.Source.vec_clone_exact core.clone.CloneU64 (by intro value; rfl)]
  simp only [bind_ok]
  rw [MemorySafety.Source.vec_clone_exact (BuiltinClone (U16 × U64)) (by intro value; rfl)]
  simp only [bind_ok]
  rw [source_snapshot_tags]
  simp only [bind_ok]

end ValueFlow.Source
