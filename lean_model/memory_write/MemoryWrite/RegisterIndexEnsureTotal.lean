/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexIntern
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

@[step]
theorem rust_ensure_page_total (before : alloc.vec.Vec register_index.RegisterPage)
    (key : U64) (valid : Directory before.val) (space : before.val.length < Usize.max) :
    register_index.ensure_page before key ⦃ result =>
      PageReady before result.2 key result.1 ∧ result.2.val.length ≤ before.val.length + 1 ⦄ := by
  unfold register_index.ensure_page
  step with rust_page_position (alloc.vec.Vec.deref before) key
    (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using directory_ordered before.val valid)
    as ⟨position, bound⟩
  simp only [alloc.vec.Vec.deref, Slice.from_val] at bound
  by_cases atEnd : position.val = before.val.length
  · have test : position = alloc.vec.Vec.len before := by scalar_tac
    simp only [test, ite_true]
    obtain ⟨after, inserted, page, pageKey, pageSlots, afterValue⟩ := MemoryWrite.spec_returns _ _
      (rust_insert_page before position key bound.1 space)
    rw [←test, inserted]
    simp only [bind_ok, WP.spec_ok]
    refine ⟨inserted_page_ready before after key position valid bound
      (lower_bound_missing_key (directory_ordered before.val valid) bound (Or.inl atEnd)) inserted, ?_⟩
    rw [afterValue, List.length_insertIdx_of_le_length bound.1]
  · have test : position ≠ alloc.vec.Vec.len before := by scalar_tac
    have inside : position.val < before.val.length := by have := bound.1; omega
    simp only [test, ite_false]
    step with alloc.vec.Vec.index_usize_spec before position inside as ⟨page, pageAt⟩
    have rowAt : page = before.val[position.val]! := by
      rw [pageAt, List.Inhabited_getElem_eq_getElem! _ _ inside]
    by_cases matched : page.key = key
    · simp only [matched, bne_self_eq_false, Bool.false_eq_true, ite_false, WP.spec_ok]
      exact ⟨⟨valid, inside, by rw [←rowAt]; exact matched, fun _ _ => Iff.rfl⟩, by omega⟩
    · have different : (page.key != key) = true := by simpa only [bne_iff_ne] using matched
      simp only [different, ite_true]
      obtain ⟨after, inserted, added, addedKey, addedSlots, afterValue⟩ := MemoryWrite.spec_returns _ _
        (rust_insert_page before position key bound.1 space)
      rw [inserted]
      simp only [bind_ok, WP.spec_ok]
      have missing : (before.val[position.val]!).key.val ≠ key.val := by
        intro equal
        apply matched
        exact UScalar.eq_of_val_eq (by simpa only [rowAt] using equal)
      refine ⟨inserted_page_ready before after key position valid bound
        (lower_bound_missing_key (directory_ordered before.val valid) bound (Or.inr missing)) inserted, ?_⟩
      rw [afterValue, List.length_insertIdx_of_le_length bound.1]

end MemoryWrite.RegisterIndex
