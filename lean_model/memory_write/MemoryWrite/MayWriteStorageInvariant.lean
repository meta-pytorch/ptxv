/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteStorage
import MemoryWrite.MayWriteWords

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

structure DirectoryInvariant (sets : cache_graph.WorkSets) : Prop where
  strict : StrictKeys sets.directory.val.length
    (fun index => (sets.directory.val[index]!).page.val)
  safe : ∀ entry ∈ sets.directory.val, entry.storage.val < sets.pages.val.length
  sizes : ∀ slots ∈ sets.pages.val, slots.val.length = 64
  separated : ∀ left ∈ sets.directory.val, ∀ right ∈ sets.directory.val,
    left.storage = right.storage → left.page = right.page

def PageAt (sets : cache_graph.WorkSets) (page : Nat) (storage : Usize) : Prop :=
  ∃ entry ∈ sets.directory.val, entry.page.val = page ∧ entry.storage = storage

def WordsAt (sets : cache_graph.WorkSets) (pc : Nat) (words : alloc.vec.Vec U64) : Prop :=
  ∃ storage, PageAt sets (pc / 64) storage ∧
    ∃ slots, sets.pages.val[storage.val]? = some slots ∧
      slots.val[pc % 64]? = some (some words)

def workSolution (sets : cache_graph.WorkSets) : Solution where
  present pc := ∃ words, WordsAt sets pc words
  incoming pc register := ∃ words, WordsAt sets pc words ∧ WordContains words.val register

theorem StrictKeys.injective {length : Nat} {key : Nat → Nat}
    (strict : StrictKeys length key) {left right : Nat}
    (leftLength : left < length) (rightLength : right < length) (equal : key left = key right) :
    left = right := by
  by_contra different
  rcases Nat.lt_or_gt_of_ne different with less | greater
  · have := strict left right less rightLength
    omega
  · have := strict right left greater leftLength
    omega

theorem page_storage_unique {sets : cache_graph.WorkSets} (invariant : DirectoryInvariant sets)
    {page : Nat} {left right : Usize} (leftAt : PageAt sets page left) (rightAt : PageAt sets page right) :
    left = right := by
  obtain ⟨leftEntry, leftMember, leftPage, leftStorage⟩ := leftAt
  obtain ⟨rightEntry, rightMember, rightPage, rightStorage⟩ := rightAt
  obtain ⟨leftIndex, leftBound, leftEq⟩ := List.getElem_of_mem leftMember
  obtain ⟨rightIndex, rightBound, rightEq⟩ := List.getElem_of_mem rightMember
  have keysEqual : (sets.directory.val[leftIndex]!).page.val =
      (sets.directory.val[rightIndex]!).page.val := by
    rw [←List.Inhabited_getElem_eq_getElem! _ _ leftBound,
      ←List.Inhabited_getElem_eq_getElem! _ _ rightBound, leftEq, rightEq, leftPage, rightPage]
  have indexEqual := invariant.strict.injective leftBound rightBound keysEqual
  have entryEqual : leftEntry = rightEntry := by simp only [←leftEq, ←rightEq, indexEqual]
  exact leftStorage.symm.trans ((congrArg cache_graph.PageIndex.storage entryEqual).trans rightStorage)

theorem words_at_unique {sets : cache_graph.WorkSets} (invariant : DirectoryInvariant sets)
    {pc : Nat} {left right : alloc.vec.Vec U64} (leftAt : WordsAt sets pc left)
    (rightAt : WordsAt sets pc right) : left = right := by
  obtain ⟨leftStorage, leftPage, leftSlots, leftLookup, leftWord⟩ := leftAt
  obtain ⟨rightStorage, rightPage, rightSlots, rightLookup, rightWord⟩ := rightAt
  have storageEqual := page_storage_unique invariant leftPage rightPage
  subst rightStorage
  have slotsEqual : leftSlots = rightSlots := Option.some.inj (leftLookup.symm.trans rightLookup)
  subst rightSlots
  exact Option.some.inj (Option.some.inj (leftWord.symm.trans rightWord))

end MemoryWrite.MayWrite
