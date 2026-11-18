/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteStorageInvariant

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem page_addition_preserves_invariant {before after : cache_graph.WorkSets}
    (invariant : DirectoryInvariant before) {page storage : Usize}
    (addition : PageAddition before after page storage)
    (missing : ∀ entry ∈ before.directory.val, entry.page ≠ page) : DirectoryInvariant after := by
  obtain ⟨storageAtEnd, position, positionBound, directoryEq, slots, slotsEq, pagesEq⟩ := addition
  have members (entry : cache_graph.PageIndex) :
      entry ∈ after.directory.val ↔ entry = ⟨page, storage⟩ ∨ entry ∈ before.directory.val := by
    rw [directoryEq, List.mem_insertIdx positionBound.1]
  have missingVal : ∀ index < before.directory.val.length,
      (before.directory.val[index]!).page.val ≠ page.val := by
    intro index inside equal
    have member : before.directory.val[index]! ∈ before.directory.val := by
      rw [←List.Inhabited_getElem_eq_getElem! _ _ inside]
      exact List.getElem_mem inside
    exact missing _ member (UScalar.eq_of_val_eq equal)
  refine ⟨?_, ?_, ?_, ?_⟩
  · simpa only [directoryEq] using
      insert_strict_keys before.directory.val position ⟨page, storage⟩ invariant.strict
        positionBound missingVal
  · intro entry member
    rcases (members entry).mp member with equal | old
    · subst entry
      simp only [pagesEq, List.length_append, List.length_singleton]
      exact Nat.lt_succ_of_le storageAtEnd.le
    · have oldBound := invariant.safe entry old
      simp only [pagesEq, List.length_append, List.length_singleton]
      omega
  · intro value member
    rw [pagesEq, List.mem_append, List.mem_singleton] at member
    rcases member with old | equal
    · exact invariant.sizes value old
    · subst value
      simp only [slotsEq, List.length_replicate]
  · intro left leftMember right rightMember equal
    rcases (members left).mp leftMember with leftNew | leftOld
    · subst left
      rcases (members right).mp rightMember with rightNew | rightOld
      · subst right; rfl
      · have oldBound := invariant.safe right rightOld
        have sameStorage := congrArg UScalar.val equal
        dsimp only at sameStorage
        omega
    · rcases (members right).mp rightMember with rightNew | rightOld
      · subst right
        have oldBound := invariant.safe left leftOld
        have sameStorage := congrArg UScalar.val equal
        dsimp only at sameStorage
        omega
      · exact invariant.separated left leftOld right rightOld equal

theorem page_addition_preserves_words {before after : cache_graph.WorkSets}
    (invariant : DirectoryInvariant before) {page storage : Usize}
    (addition : PageAddition before after page storage)
    (_missing : ∀ entry ∈ before.directory.val, entry.page ≠ page)
    (pc : Nat) (words : alloc.vec.Vec U64) : WordsAt after pc words ↔ WordsAt before pc words := by
  obtain ⟨storageAtEnd, position, positionBound, directoryEq, newSlots, newSlotsEq, pagesEq⟩ := addition
  constructor
  · rintro ⟨address, ⟨entry, member, pageEq, storageEq⟩, slots, slotsLookup, wordLookup⟩
    rw [directoryEq, List.mem_insertIdx positionBound.1] at member
    rcases member with equal | old
    · subst entry
      dsimp only at storageEq pageEq
      subst address
      have newLookup : after.pages.val[storage.val]? = some newSlots := by
        rw [pagesEq, storageAtEnd, List.getElem?_append_right (by omega)]
        simp
      have slotsEqual : slots = newSlots := Option.some.inj (slotsLookup.symm.trans newLookup)
      subst slots
      simp only [newSlotsEq, List.getElem?_replicate_of_lt (Nat.mod_lt pc (by decide : 0 < 64))] at wordLookup
      cases Option.some.inj wordLookup
    · have oldBound := invariant.safe entry old
      have lookupBefore : before.pages.val[address.val]? = some slots := by
        simpa only [pagesEq, ←storageEq, List.getElem?_append_left oldBound] using slotsLookup
      exact ⟨address, ⟨entry, old, pageEq, storageEq⟩, slots, lookupBefore, wordLookup⟩
  · rintro ⟨address, ⟨entry, member, pageEq, storageEq⟩, slots, slotsLookup, wordLookup⟩
    have oldBound := invariant.safe entry member
    refine ⟨address, ⟨entry, ?_, pageEq, storageEq⟩, slots, ?_, wordLookup⟩
    · rw [directoryEq, List.mem_insertIdx positionBound.1]
      exact Or.inr member
    · simpa only [pagesEq, ←storageEq, List.getElem?_append_left oldBound] using slotsLookup


theorem page_addition_page {before after : cache_graph.WorkSets} {page storage : Usize}
    (addition : PageAddition before after page storage) : PageAt after page.val storage := by
  obtain ⟨_, position, bound, directoryEq, _⟩ := addition
  refine ⟨⟨page, storage⟩, ?_, rfl, rfl⟩
  rw [directoryEq, List.mem_insertIdx bound.1]
  exact Or.inl rfl

theorem ensured_page_valid {before after : cache_graph.WorkSets}
    (invariant : DirectoryInvariant before) {page storage : Usize}
    (ensured : PageEnsured before after page storage) :
    DirectoryInvariant after ∧ PageAt after page.val storage ∧
      ∀ pc words, WordsAt after pc words ↔ WordsAt before pc words := by
  rcases ensured with ⟨equal, entry, member, pageEq, storageEq⟩ | ⟨addition, missing⟩
  · subst after
    exact ⟨invariant, ⟨entry, member, congrArg UScalar.val pageEq, storageEq⟩,
      fun _ _ => Iff.rfl⟩
  · exact ⟨page_addition_preserves_invariant invariant addition missing,
      page_addition_page addition, page_addition_preserves_words invariant addition missing⟩

@[step]
theorem rust_ensure_page_valid (sets : cache_graph.WorkSets) (page : Usize)
    (invariant : DirectoryInvariant sets)
    (directorySpace : sets.directory.val.length < Usize.max)
    (pagesSpace : sets.pages.val.length < Usize.max) :
    cache_graph.ensure_page sets page ⦃ storage result =>
      DirectoryInvariant result ∧ PageAt result page.val storage ∧
        ∀ pc words, WordsAt result pc words ↔ WordsAt sets pc words ⦄ := by
  apply WP.spec_mono (rust_ensure_page sets page invariant.strict.ordered directorySpace pagesSpace)
  intro result ensured
  exact ensured_page_valid invariant ensured

end MemoryWrite.MayWrite
