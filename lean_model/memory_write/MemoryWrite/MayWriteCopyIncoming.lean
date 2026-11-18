/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteSlots

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem lower_bound_present {length query : Nat} {key : Nat → Nat} {position : Usize}
    (ordered : OrderedKeys length key) (bound : LowerBound length query key position)
    (present : ∃ index < length, key index = query) :
    position.val < length ∧ key position.val = query := by
  obtain ⟨index, inside, equal⟩ := present
  have positionInside : position.val < length := by
    by_contra outside
    have missing := lower_bound_missing_key ordered bound (Or.inl (by have := bound.1; omega))
    exact missing index inside equal
  refine ⟨positionInside, ?_⟩
  by_contra different
  exact lower_bound_missing_key ordered bound (Or.inr different) index inside equal

@[step]
theorem rust_copy_incoming (sets : cache_graph.WorkSets) (pc : Usize)
    (invariant : DirectoryInvariant sets) (oldWords : alloc.vec.Vec U64)
    (oldAt : WordsAt sets pc.val oldWords) :
    cache_graph.copy_incoming sets pc ⦃ result => result.val = oldWords.val ⦄ := by
  obtain ⟨knownStorage, knownPage, knownSlots, knownLookup, knownWords⟩ := oldAt
  unfold cache_graph.copy_incoming cache_graph.PAGE_SIZE
  step as ⟨page, pageEq⟩
  have ordered := invariant.strict.ordered
  step with rust_page_position (alloc.vec.Vec.deref sets.directory) page
    (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using ordered) as ⟨position, lower⟩
  simp only [alloc.vec.Vec.deref, Slice.from_val] at lower
  have present : ∃ index < sets.directory.val.length,
      (sets.directory.val[index]!).page.val = page.val := by
    obtain ⟨entry, member, entryPage, _⟩ := knownPage
    obtain ⟨index, inside, atIndex⟩ := List.getElem_of_mem member
    refine ⟨index, inside, ?_⟩
    rw [←List.Inhabited_getElem_eq_getElem! _ _ inside, atIndex, entryPage, pageEq]
  have found := lower_bound_present ordered lower present
  step
  step with alloc.vec.Vec.index_usize_spec sets.directory position found.1 as ⟨entry, entryEq⟩
  have entryAt : entry = sets.directory.val[position.val]! := by
    rw [entryEq, List.Inhabited_getElem_eq_getElem! _ _ found.1]
  have pageEqual : entry.page = page := UScalar.eq_of_val_eq (by simpa only [entryAt] using found.2)
  step
  have entryMember : entry ∈ sets.directory.val := by
    rw [entryAt, ←List.Inhabited_getElem_eq_getElem! _ _ found.1]
    exact List.getElem_mem found.1
  have atPage : PageAt sets (pc.val / 64) entry.storage :=
    ⟨entry, entryMember, by rw [pageEqual, pageEq], rfl⟩
  have bounds := page_at_bounds invariant atPage
  step with alloc.vec.Vec.index_usize_spec sets.pages entry.storage bounds.1 as ⟨slots, slotsEq⟩
  have slotsAt : slots = sets.pages.val[entry.storage.val]! := by
    rw [slotsEq, List.Inhabited_getElem_eq_getElem! _ _ bounds.1]
  step as ⟨offset, offsetEq⟩
  have offsetBound : offset.val < slots.val.length := by rw [slotsAt, bounds.2, offsetEq]; omega
  step with alloc.vec.Vec.index_usize_spec slots offset offsetBound as ⟨previous, previousEq⟩
  have oldLookup := (words_at_known_page invariant atPage offsetEq oldWords).mp
    ⟨knownStorage, knownPage, knownSlots, knownLookup, knownWords⟩
  have previousEqual : previous = some oldWords := by
    simp only [cell, list_some_at sets.pages.val entry.storage.val bounds.1, Option.bind_some] at oldLookup
    rw [←slotsAt, List.getElem?_eq_getElem offsetBound] at oldLookup
    exact previousEq.trans (Option.some.inj oldLookup)
  rw [previousEqual]
  step with rust_copy_words (alloc.vec.Vec.deref oldWords) as ⟨result, copied⟩
  simpa only [alloc.vec.Vec.deref, Slice.from_val] using copied

end MemoryWrite.MayWrite
