/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexPageInsertion

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

def PageReady (before after : alloc.vec.Vec register_index.RegisterPage)
    (key : U64) (position : Usize) : Prop :=
  Directory after.val ∧ position.val < after.val.length ∧
  (after.val[position.val]!).key = key ∧
  ∀ register id, StoredRows after.val register id ↔ StoredRows before.val register id

theorem inserted_page_ready (before after : alloc.vec.Vec register_index.RegisterPage)
    (key : U64) (position : Usize) (valid : Directory before.val)
    (bound : LowerBound before.val.length key.val (fun index => (before.val[index]!).key.val) position)
    (missing : ∀ index < before.val.length, (before.val[index]!).key.val ≠ key.val)
    (success : register_index.insert_page before position key = .ok after) :
    PageReady before after key position := by
  obtain ⟨page, pageKey, pageSlots, afterValue⟩ := successful_insert_page before after position key bound.1 success
  have pageBound : LowerBound before.val.length page.key.val
      (fun index => (before.val[index]!).key.val) position := by simpa only [pageKey] using bound
  have noPage : ∀ index < before.val.length, (before.val[index]!).key.val ≠ page.key.val := by
    simpa only [pageKey] using missing
  have positionBound := bound.1
  have insertedLength := List.length_insertIdx_of_le_length bound.1 page
  have readyBound : position.val < after.val.length := by rw [afterValue, insertedLength]; omega
  refine ⟨?_, readyBound, ?_, ?_⟩
  · rw [afterValue]
    exact directory_insert before.val valid position page (by rw [pageSlots, List.length_replicate]) pageBound noPage
  · rw [afterValue, ←List.Inhabited_getElem_eq_getElem! _ _ (by rw [insertedLength]; omega),
      List.getElem_insertIdx_self]
    exact pageKey
  · intro register id
    rw [afterValue]
    exact empty_insert_preserves_rows before.val position page bound.1 pageSlots register id

theorem successful_ensure_page (before after : alloc.vec.Vec register_index.RegisterPage)
    (key : U64) (position : Usize) (valid : Directory before.val)
    (success : register_index.ensure_page before key = .ok (position, after)) :
    PageReady before after key position := by
  unfold register_index.ensure_page at success
  obtain ⟨found, searchOk, tailOk⟩ := MemoryWrite.bind_success _ _ _ success
  have foundSpec := rust_page_position (alloc.vec.Vec.deref before) key
    (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using directory_ordered before.val valid)
  simp only [searchOk, WP.spec_ok] at foundSpec
  simp only [alloc.vec.Vec.deref, Slice.from_val] at foundSpec
  by_cases atEnd : found.val = before.val.length
  · have test : found = alloc.vec.Vec.len before := by scalar_tac
    simp only [test, ite_true] at tailOk
    obtain ⟨result, inserted, completed⟩ := MemoryWrite.bind_success _ _ _ tailOk
    have equal := Result.ok_injective completed
    have resultEqual : result = after := congrArg Prod.snd equal
    have positionEqual : found = position := by simpa only [test] using congrArg Prod.fst equal
    have inserted' : register_index.insert_page before found key = .ok result := by
      simpa only [test] using inserted
    have ready := inserted_page_ready before result key found valid foundSpec
      (lower_bound_missing_key (directory_ordered before.val valid) foundSpec (Or.inl atEnd)) inserted'
    simpa only [resultEqual, positionEqual] using ready
  · have test : found ≠ alloc.vec.Vec.len before := by scalar_tac
    simp only [test, ite_false] at tailOk
    have inside : found.val < before.val.length := by have := foundSpec.1; omega
    obtain ⟨row, indexed, rowOk⟩ := MemoryWrite.bind_success _ _ _ tailOk
    have rowSpec := rust_vec_index before found inside
    simp only [indexed, WP.spec_ok] at rowSpec
    have rowAt : row = before.val[found.val]! := by
      rw [rowSpec, List.Inhabited_getElem_eq_getElem! _ _ inside]
    by_cases matched : row.key = key
    · simp only [matched, bne_self_eq_false, Bool.false_eq_true, ite_false] at rowOk
      have equal := Result.ok_injective rowOk
      have resultEqual : before = after := congrArg Prod.snd equal
      have positionEqual : found = position := congrArg Prod.fst equal
      have ready : PageReady before before key found :=
        ⟨valid, inside, by rw [←rowAt]; exact matched, fun _ _ => Iff.rfl⟩
      simpa only [←resultEqual, positionEqual] using ready
    · have testDifferent : (row.key != key) = true := by simp only [bne_iff_ne]; exact matched
      simp only [testDifferent, ite_true] at rowOk
      obtain ⟨result, inserted, completed⟩ := MemoryWrite.bind_success _ _ _ rowOk
      have equal := Result.ok_injective completed
      have resultEqual : result = after := congrArg Prod.snd equal
      have positionEqual : found = position := congrArg Prod.fst equal
      have missing : (before.val[found.val]!).key.val ≠ key.val := by
        intro same; apply matched; exact UScalar.eq_of_val_eq (by simpa only [rowAt] using same)
      have ready := inserted_page_ready before result key found valid foundSpec
        (lower_bound_missing_key (directory_ordered before.val valid) foundSpec (Or.inr missing)) inserted
      simpa only [resultEqual, positionEqual] using ready

end MemoryWrite.RegisterIndex
