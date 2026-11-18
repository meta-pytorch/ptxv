/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexPageInsertion
import MemoryWrite.MayWriteSlots

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

def PageAt (pages : alloc.vec.Vec register_index.RegisterPage) (page : Nat) (position : Usize) : Prop :=
  position.val < pages.val.length ∧ (pages.val[position.val]!).key.val = page

def cell (pages : alloc.vec.Vec register_index.RegisterPage) (position offset : Usize) :
    Option (Option Usize) := pages.val[position.val]?.bind fun page => page.slots.val[offset.val]?

def setSlot (pages : alloc.vec.Vec register_index.RegisterPage) (position offset : Usize)
    (value : Option Usize) : alloc.vec.Vec register_index.RegisterPage :=
  let page := pages.val[position.val]!
  pages.set position { page with slots := page.slots.set offset value }

theorem set_slot_cell (pages : alloc.vec.Vec register_index.RegisterPage) (position offset : Usize)
    (value : Option Usize) (positionBound : position.val < pages.val.length)
    (offsetBound : offset.val < (pages.val[position.val]!).slots.val.length)
    (queryPosition queryOffset : Usize) :
    cell (setSlot pages position offset value) queryPosition queryOffset =
      if queryPosition = position ∧ queryOffset = offset then some value
      else cell pages queryPosition queryOffset := by
  by_cases positionEqual : queryPosition = position
  · subst queryPosition
    simp only [cell, setSlot, alloc.vec.Vec.set_val_eq,
      List.getElem?_set_self positionBound, Option.bind_some]
    by_cases offsetEqual : queryOffset = offset
    · subst queryOffset
      simp only [List.getElem?_set_self offsetBound, and_self, ite_true]
    · have different : offset.val ≠ queryOffset.val := by
        intro equal; exact offsetEqual (UScalar.eq_of_val_eq equal).symm
      simp only [List.getElem?_set_ne different, offsetEqual, and_false, ite_false]
      rw [list_some_at pages.val position.val positionBound]
      rfl
  · have different : position.val ≠ queryPosition.val := by
      intro equal; exact positionEqual (UScalar.eq_of_val_eq equal).symm
    simp only [cell, setSlot, alloc.vec.Vec.set_val_eq,
      List.getElem?_set_ne different, positionEqual, false_and, ite_false]

theorem set_slot_keys (pages : alloc.vec.Vec register_index.RegisterPage) (position offset : Usize)
    (value : Option Usize) (query : Nat) (inside : query < pages.val.length) :
    ((setSlot pages position offset value).val[query]!).key = (pages.val[query]!).key := by
  dsimp only [setSlot]
  simp only [alloc.vec.Vec.set_val_eq]
  rw [←List.Inhabited_getElem_eq_getElem! _ _ (by simpa only [List.length_set] using inside),
    List.getElem_set]
  split
  · rename_i equal
    simp only [←equal]
  · rw [List.Inhabited_getElem_eq_getElem! _ _ inside]

theorem set_slot_directory (pages : alloc.vec.Vec register_index.RegisterPage) (valid : Directory pages.val)
    (position offset : Usize) (value : Option Usize) (inside : position.val < pages.val.length) :
    Directory (setSlot pages position offset value).val := by
  have resultLength : (setSlot pages position offset value).val.length = pages.val.length := by
    simp only [setSlot, alloc.vec.Vec.set_val_eq, List.length_set]
  constructor
  · intro query queryBound
    have bound : query < pages.val.length := by omega
    dsimp only [setSlot]
    simp only [alloc.vec.Vec.set_val_eq]
    rw [←List.Inhabited_getElem_eq_getElem! _ _ (by simpa only [List.length_set] using bound),
      List.getElem_set]
    split
    · simpa only [alloc.vec.Vec.set_val_eq, List.length_set] using valid.1 position.val inside
    · simpa only [List.Inhabited_getElem_eq_getElem! _ _ bound] using valid.1 query bound
  · intro left right less bound
    have rightBound : right < pages.val.length := by omega
    rw [set_slot_keys pages position offset value left (by omega),
      set_slot_keys pages position offset value right rightBound]
    exact valid.2 left right less rightBound

theorem page_at_set_slot (pages : alloc.vec.Vec register_index.RegisterPage) (position offset : Usize)
    (value : Option Usize) (page : Nat) (query : Usize) :
    PageAt (setSlot pages position offset value) page query ↔ PageAt pages page query := by
  have length : (setSlot pages position offset value).val.length = pages.val.length := by
    simp only [setSlot, alloc.vec.Vec.set_val_eq, List.length_set]
  unfold PageAt
  constructor
  · rintro ⟨bound, key⟩
    have oldBound : query.val < pages.val.length := by omega
    exact ⟨oldBound, by simpa only [set_slot_keys pages position offset value query.val oldBound] using key⟩
  · rintro ⟨bound, key⟩
    exact ⟨by omega, by simpa only [set_slot_keys pages position offset value query.val bound] using key⟩

theorem page_position_unique (pages : alloc.vec.Vec register_index.RegisterPage)
    (valid : Directory pages.val) (page : Nat) (left right : Usize)
    (leftAt : PageAt pages page left) (rightAt : PageAt pages page right) : left = right := by
  apply UScalar.eq_of_val_eq
  exact directory_key_unique pages.val valid left.val right.val leftAt.1 rightAt.1
    (leftAt.2.trans rightAt.2.symm)

theorem stored_at_cell (pages : alloc.vec.Vec register_index.RegisterPage) (count : Usize)
    (key : symbolic.RegisterKey) (offset : Usize) (offsetValue : offset.val = keyOffset key) (id : Usize) :
    Stored ⟨pages, count⟩ key id ↔
      ∃ position, PageAt pages (keyPage key) position ∧ cell pages position offset = some (some id) := by
  constructor
  · rintro ⟨position, inside, samePage, slot⟩
    let address := Usize.ofNatCore position (by scalar_tac)
    refine ⟨address, ⟨inside, samePage⟩, ?_⟩
    simp only [cell, address, Usize.ofNatCore_val_eq, list_some_at pages.val position inside,
      Option.bind_some, offsetValue, slot]
  · rintro ⟨position, atPage, lookup⟩
    refine ⟨position.val, atPage.1, atPage.2, ?_⟩
    simpa only [cell, list_some_at pages.val position.val atPage.1, Option.bind_some, offsetValue] using lookup

theorem stored_at_known_page (pages : alloc.vec.Vec register_index.RegisterPage) (count : Usize)
    (valid : Directory pages.val) (key : symbolic.RegisterKey) (position offset : Usize)
    (atPage : PageAt pages (keyPage key) position) (offsetValue : offset.val = keyOffset key) (id : Usize) :
    Stored ⟨pages, count⟩ key id ↔ cell pages position offset = some (some id) := by
  rw [stored_at_cell pages count key offset offsetValue id]
  constructor
  · rintro ⟨other, otherPage, lookup⟩
    simpa only [page_position_unique pages valid (keyPage key) other position otherPage atPage] using lookup
  · intro lookup; exact ⟨position, atPage, lookup⟩

theorem set_slot_stored (pages : alloc.vec.Vec register_index.RegisterPage) (count : Usize)
    (valid : Directory pages.val) (target : symbolic.RegisterKey) (position offset : Usize)
    (atPage : PageAt pages (keyPage target) position) (offsetValue : offset.val = keyOffset target)
    (value : Usize) (query : symbolic.RegisterKey) (id : Usize) :
    Stored ⟨setSlot pages position offset (some value), count⟩ query id ↔
      (query = target ∧ id = value) ∨ (query ≠ target ∧ Stored ⟨pages, count⟩ query id) := by
  have offsetBound : offset.val < (pages.val[position.val]!).slots.val.length := by
    rw [valid.1 position.val atPage.1, offsetValue]
    exact Nat.mod_lt _ (by decide)
  have resultValid := set_slot_directory pages valid position offset (some value) atPage.1
  by_cases equal : query = target
  · subst query
    rw [stored_at_known_page _ count resultValid target position offset
      ((page_at_set_slot pages position offset (some value) (keyPage target) position).mpr atPage) offsetValue id,
      set_slot_cell pages position offset (some value) atPage.1 offsetBound]
    simp only [and_self, ite_true, Option.some.injEq, true_and, ne_eq, not_true_eq_false,
      false_and, or_false]
    exact eq_comm
  · let queryOffset := Usize.ofNatCore (keyOffset query) (by dsimp only [keyOffset]; scalar_tac)
    have queryOffsetValue : queryOffset.val = keyOffset query := rfl
    rw [stored_at_cell _ count query queryOffset queryOffsetValue id,
      stored_at_cell pages count query queryOffset queryOffsetValue id]
    have frame : ∀ address, PageAt pages (keyPage query) address →
        ¬(address = position ∧ queryOffset = offset) := by
      intro address queryPage
      rintro ⟨sameAddress, sameOffset⟩
      have samePage : keyPage query = keyPage target := by
        rw [sameAddress] at queryPage
        exact queryPage.2.symm.trans atPage.2
      have sameOff : keyOffset query = keyOffset target := by rw [←queryOffsetValue, sameOffset, offsetValue]
      exact equal (page_offset_injective query target samePage sameOff)
    simp only [equal, false_and, ne_eq, not_false_eq_true, true_and, false_or]
    constructor
    · rintro ⟨address, queryPage, lookup⟩
      have oldPage := (page_at_set_slot pages position offset (some value) (keyPage query) address).mp queryPage
      exact ⟨address, oldPage, by simpa only [set_slot_cell pages position offset (some value) atPage.1 offsetBound, frame address oldPage, ite_false] using lookup⟩
    · rintro ⟨address, queryPage, lookup⟩
      exact ⟨address, (page_at_set_slot pages position offset (some value) (keyPage query) address).mpr queryPage,
        by simpa only [set_slot_cell pages position offset (some value) atPage.1 offsetBound,
          frame address queryPage, ite_false] using lookup⟩

end MemoryWrite.RegisterIndex
