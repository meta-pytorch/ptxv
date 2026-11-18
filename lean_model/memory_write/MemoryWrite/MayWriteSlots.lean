/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWritePageAddition

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

def cell (sets : cache_graph.WorkSets) (storage offset : Usize) :
    Option (Option (alloc.vec.Vec U64)) :=
  sets.pages.val[storage.val]?.bind fun slots => slots.val[offset.val]?

def setSlot (sets : cache_graph.WorkSets) (storage offset : Usize)
    (value : Option (alloc.vec.Vec U64)) : cache_graph.WorkSets :=
  {sets with pages := (sets.pages.set storage ((sets.pages.val[storage.val]!).set offset value))}

theorem list_some_at {α : Type} [Inhabited α] (values : List α) (index : Nat)
    (bound : index < values.length) : values[index]? = some values[index]! := by
  rw [List.getElem?_eq_getElem bound, List.Inhabited_getElem_eq_getElem! values index bound]

theorem set_slot_cell (sets : cache_graph.WorkSets) (storage offset : Usize)
    (value : Option (alloc.vec.Vec U64))
    (storageBound : storage.val < sets.pages.val.length)
    (offsetBound : offset.val < (sets.pages.val[storage.val]!).val.length)
    (queryStorage queryOffset : Usize) :
    cell (setSlot sets storage offset value) queryStorage queryOffset =
      if queryStorage = storage ∧ queryOffset = offset then some value
      else cell sets queryStorage queryOffset := by
  by_cases storageEqual : queryStorage = storage
  · subst queryStorage
    simp only [cell, setSlot, alloc.vec.Vec.set_val_eq,
      List.getElem?_set_self storageBound, Option.bind_some]
    by_cases offsetEqual : queryOffset = offset
    · subst queryOffset
      simp only [List.getElem?_set_self offsetBound, and_self, ite_true]
    · have indexDifferent : offset.val ≠ queryOffset.val := by
        intro same
        exact offsetEqual (UScalar.eq_of_val_eq same).symm
      simp only [List.getElem?_set_ne indexDifferent, offsetEqual, and_false, ite_false]
      rw [list_some_at sets.pages.val storage.val storageBound]
      rfl
  · have indexDifferent : storage.val ≠ queryStorage.val := by
      intro same
      exact storageEqual (UScalar.eq_of_val_eq same).symm
    simp only [cell, setSlot, alloc.vec.Vec.set_val_eq,
      List.getElem?_set_ne indexDifferent, storageEqual, false_and, ite_false]

theorem set_slot_preserves_invariant {sets : cache_graph.WorkSets}
    (invariant : DirectoryInvariant sets) (storage offset : Usize)
    (value : Option (alloc.vec.Vec U64))
    (storageBound : storage.val < sets.pages.val.length) :
    DirectoryInvariant (setSlot sets storage offset value) := by
  refine ⟨invariant.strict, ?_, ?_, invariant.separated⟩
  · intro entry member
    simpa only [setSlot, alloc.vec.Vec.set_val_eq, List.length_set] using invariant.safe entry member
  · intro slots member
    simp only [setSlot, alloc.vec.Vec.set_val_eq] at member
    rcases List.mem_or_eq_of_mem_set member with old | equal
    · exact invariant.sizes slots old
    · subst slots
      simp only [alloc.vec.Vec.set_val_eq, List.length_set]
      apply invariant.sizes
      rw [←List.Inhabited_getElem_eq_getElem! _ _ storageBound]
      exact List.getElem_mem storageBound

theorem page_at_set_slot (sets : cache_graph.WorkSets) (storage offset : Usize)
    (value : Option (alloc.vec.Vec U64)) (page : Nat) (address : Usize) :
    PageAt (setSlot sets storage offset value) page address ↔ PageAt sets page address := Iff.rfl

theorem words_at_cell (sets : cache_graph.WorkSets) (pc : Nat) (offset : Usize)
    (offsetValue : offset.val = pc % 64) (words : alloc.vec.Vec U64) :
    WordsAt sets pc words ↔ ∃ storage, PageAt sets (pc / 64) storage ∧
      cell sets storage offset = some (some words) := by
  constructor
  · rintro ⟨storage, pageAt, slots, slotsLookup, wordsLookup⟩
    refine ⟨storage, pageAt, ?_⟩
    simp only [cell, slotsLookup, Option.bind_some, offsetValue, wordsLookup]
  · rintro ⟨storage, pageAt, lookup⟩
    unfold cell at lookup
    cases present : sets.pages.val[storage.val]? with
    | none => simp only [present, Option.bind_none, reduceCtorEq] at lookup
    | some slots =>
        exact ⟨storage, pageAt, slots, present, by simpa only [present, Option.bind_some, offsetValue] using lookup⟩

theorem pc_eq_of_page_offset {left right : Nat} (pages : left / 64 = right / 64)
    (offsets : left % 64 = right % 64) : left = right := by
  have leftIdentity := Nat.mod_add_div left 64
  have rightIdentity := Nat.mod_add_div right 64
  omega


theorem page_at_bounds {sets : cache_graph.WorkSets} (invariant : DirectoryInvariant sets)
    {page : Nat} {storage : Usize} (atPage : PageAt sets page storage) :
    storage.val < sets.pages.val.length ∧ (sets.pages.val[storage.val]!).val.length = 64 := by
  obtain ⟨entry, member, _, storageEq⟩ := atPage
  have bound : storage.val < sets.pages.val.length := by simpa only [storageEq] using invariant.safe entry member
  refine ⟨bound, invariant.sizes _ ?_⟩
  rw [←List.Inhabited_getElem_eq_getElem! _ _ bound]
  exact List.getElem_mem bound

theorem page_number_unique {sets : cache_graph.WorkSets} (invariant : DirectoryInvariant sets)
    {left right : Nat} {storage : Usize} (leftAt : PageAt sets left storage)
    (rightAt : PageAt sets right storage) : left = right := by
  obtain ⟨leftEntry, leftMember, leftPage, leftStorage⟩ := leftAt
  obtain ⟨rightEntry, rightMember, rightPage, rightStorage⟩ := rightAt
  have equal := invariant.separated leftEntry leftMember rightEntry rightMember
    (leftStorage.trans rightStorage.symm)
  exact leftPage.symm.trans ((congrArg UScalar.val equal).trans rightPage)

theorem words_at_known_page {sets : cache_graph.WorkSets} (invariant : DirectoryInvariant sets)
    {pc : Nat} {storage offset : Usize} (atPage : PageAt sets (pc / 64) storage)
    (offsetValue : offset.val = pc % 64) (words : alloc.vec.Vec U64) :
    WordsAt sets pc words ↔ cell sets storage offset = some (some words) := by
  rw [words_at_cell sets pc offset offsetValue words]
  constructor
  · rintro ⟨address, pageAt, lookup⟩
    have equal := page_storage_unique invariant atPage pageAt
    simpa only [equal] using lookup
  · intro lookup
    exact ⟨storage, atPage, lookup⟩

theorem set_slot_words_at_target {sets : cache_graph.WorkSets}
    (invariant : DirectoryInvariant sets) {pc : Nat} {storage offset : Usize}
    (atPage : PageAt sets (pc / 64) storage) (offsetValue : offset.val = pc % 64)
    (value : Option (alloc.vec.Vec U64)) (words : alloc.vec.Vec U64) :
    WordsAt (setSlot sets storage offset value) pc words ↔ value = some words := by
  have bounds := page_at_bounds invariant atPage
  have resultInvariant := set_slot_preserves_invariant invariant storage offset value bounds.1
  rw [words_at_known_page resultInvariant atPage offsetValue words,
    set_slot_cell sets storage offset value bounds.1 (by rw [bounds.2, offsetValue]; omega)]
  simp

theorem set_slot_words_at_other {sets : cache_graph.WorkSets}
    (invariant : DirectoryInvariant sets) {pc : Nat} {storage offset : Usize}
    (atPage : PageAt sets (pc / 64) storage) (offsetValue : offset.val = pc % 64)
    (value : Option (alloc.vec.Vec U64)) (other : Nat) (different : other ≠ pc)
    (words : alloc.vec.Vec U64) :
    WordsAt (setSlot sets storage offset value) other words ↔ WordsAt sets other words := by
  have bounds := page_at_bounds invariant atPage
  let queryOffset : Usize := Usize.ofNatCore (other % 64) (by scalar_tac)
  have queryValue : queryOffset.val = other % 64 := by rfl
  rw [words_at_cell _ other queryOffset queryValue words, words_at_cell sets other queryOffset queryValue words]
  constructor
  · rintro ⟨address, otherPage, lookup⟩
    rw [page_at_set_slot] at otherPage
    have notTarget : ¬(address = storage ∧ queryOffset = offset) := by
      rintro ⟨sameStorage, sameOffset⟩
      have pageEqual := page_number_unique invariant (show PageAt sets (other / 64) storage from by
        simpa only [sameStorage] using otherPage) atPage
      have offsetEqual : other % 64 = pc % 64 := by
        rw [←queryValue, sameOffset, offsetValue]
      exact different (pc_eq_of_page_offset pageEqual offsetEqual)
    refine ⟨address, otherPage, ?_⟩
    simpa only [set_slot_cell sets storage offset value bounds.1
      (by rw [bounds.2, offsetValue]; omega), notTarget, ite_false] using lookup
  · rintro ⟨address, otherPage, lookup⟩
    have notTarget : ¬(address = storage ∧ queryOffset = offset) := by
      rintro ⟨sameStorage, sameOffset⟩
      have pageEqual := page_number_unique invariant (show PageAt sets (other / 64) storage from by
        simpa only [sameStorage] using otherPage) atPage
      have offsetEqual : other % 64 = pc % 64 := by
        rw [←queryValue, sameOffset, offsetValue]
      exact different (pc_eq_of_page_offset pageEqual offsetEqual)
    refine ⟨address, (page_at_set_slot sets storage offset value (other / 64) address).mpr otherPage, ?_⟩
    simpa only [set_slot_cell sets storage offset value bounds.1
      (by rw [bounds.2, offsetValue]; omega), notTarget, ite_false] using lookup

end MemoryWrite.MayWrite
