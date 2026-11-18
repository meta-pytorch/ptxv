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

@[step]
theorem rust_merge_row (sets : cache_graph.WorkSets) (pc : Nat) (storage offset : Usize)
    (source : Slice U64) (oldWords : alloc.vec.Vec U64)
    (invariant : DirectoryInvariant sets) (atPage : PageAt sets (pc / 64) storage)
    (offsetValue : offset.val = pc % 64) (oldAt : WordsAt sets pc oldWords)
    (lengths : oldWords.val.length = source.val.length) :
    cache_graph.merge_row sets storage offset source ⦃ changed result =>
      ∃ words : alloc.vec.Vec U64,
        result = setSlot sets storage offset (some words) ∧
        words.val.length = oldWords.val.length ∧
        (∀ register, WordContains words.val register ↔
          WordContains oldWords.val register ∨ WordContains source.val register) ∧
        (changed = true ↔ words.val ≠ oldWords.val) ⦄ := by
  have bounds := page_at_bounds invariant atPage
  have offsetBound : offset.val < (sets.pages.val[storage.val]!).val.length := by
    rw [bounds.2, offsetValue]; omega
  have cellValue := (words_at_known_page invariant atPage offsetValue oldWords).mp oldAt
  unfold cache_graph.merge_row
  step with alloc.vec.Vec.index_mut_usize_spec sets.pages storage bounds.1
    as ⟨slots, outerRestore, outerValue, outerBack⟩
  have slotsAt : slots = sets.pages.val[storage.val]! := by
    rw [outerValue, List.Inhabited_getElem_eq_getElem! _ _ bounds.1]
  step with alloc.vec.Vec.index_mut_usize_spec slots offset
    (by simpa only [slotsAt] using offsetBound) as ⟨old, innerRestore, innerValue, innerBack⟩
  have oldEqual : old = some oldWords := by
    simp only [cell, list_some_at sets.pages.val storage.val bounds.1, Option.bind_some] at cellValue
    rw [←slotsAt, List.getElem?_eq_getElem (by simpa only [slotsAt] using offsetBound)] at cellValue
    exact (innerValue.trans (Option.some.inj cellValue))
  rw [oldEqual]
  simp only [alloc.vec.Vec.deref_mut, lift, bind_ok]
  step with rust_join_words oldWords.slice source lengths
    as ⟨changed, merged, mergedLength, mergedValues, changedIff⟩
  let words : alloc.vec.Vec U64 := ⟨merged⟩
  refine ⟨words, ?_, mergedLength, ?_, changedIff⟩
  · simp only [setSlot, outerBack, innerBack, slotsAt, words]
  · exact word_contains_union oldWords.val source.val words.val mergedLength.symm
      (lengths.symm.trans mergedLength.symm)
      (fun position bound => mergedValues position (by
        have lengthEq : words.val.length = oldWords.slice.val.length := mergedLength
        omega))

end MemoryWrite.MayWrite
