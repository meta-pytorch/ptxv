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

def PcEnsured (before after : cache_graph.WorkSets) (pc count : Usize)
    (answer : Usize × Usize × Bool) : Prop :=
  ∃ middle : cache_graph.WorkSets,
    DirectoryInvariant middle ∧ PageAt middle (pc.val / 64) answer.1 ∧
    (∃ page : Usize, page.val = pc.val / 64 ∧ PageEnsured before middle page answer.1) ∧
    (∀ address words, WordsAt middle address words ↔ WordsAt before address words) ∧
    answer.2.1.val = pc.val % 64 ∧
    ∃ previous : Option (alloc.vec.Vec U64),
      cell middle answer.1 answer.2.1 = some previous ∧
      answer.2.2 = previous.isNone ∧
      match previous with
      | none => ∃ words : alloc.vec.Vec U64,
          words.val = List.replicate count.val 0#u64 ∧
          after = setSlot middle answer.1 answer.2.1 (some words)
      | some _ => after = middle

@[step]
theorem rust_ensure_pc (sets : cache_graph.WorkSets) (pc count : Usize)
    (invariant : DirectoryInvariant sets)
    (directorySpace : sets.directory.val.length < Usize.max)
    (pagesSpace : sets.pages.val.length < Usize.max) :
    cache_graph.ensure_pc sets pc count ⦃ answer result => PcEnsured sets result pc count answer ⦄ := by
  unfold cache_graph.ensure_pc cache_graph.PAGE_SIZE
  step as ⟨page, pageEq⟩
  step with rust_ensure_page sets page invariant.strict.ordered directorySpace pagesSpace
    as ⟨storage, middle, pageEnsured⟩
  have ⟨middleInvariant, atPage, oldWords⟩ := ensured_page_valid invariant pageEnsured
  have atPcPage : PageAt middle (pc.val / 64) storage := by simpa only [pageEq] using atPage
  have pageBounds := page_at_bounds middleInvariant atPcPage
  step as ⟨offset, offsetEq⟩
  have offsetBound : offset.val < (middle.pages.val[storage.val]!).val.length := by
    rw [pageBounds.2, offsetEq]
    omega
  step with alloc.vec.Vec.index_usize_spec middle.pages storage pageBounds.1 as ⟨slots, slotsEq⟩
  have slotsAt : slots = middle.pages.val[storage.val]! := by
    rw [slotsEq, List.Inhabited_getElem_eq_getElem! _ _ pageBounds.1]
  step with alloc.vec.Vec.index_usize_spec slots offset (by simpa only [slotsAt] using offsetBound)
    as ⟨previous, previousEq⟩
  have cellValue : cell middle storage offset = some previous := by
    simp only [cell, list_some_at middle.pages.val storage.val pageBounds.1, Option.bind_some]
    rw [←slotsAt, List.getElem?_eq_getElem (by simpa only [slotsAt] using offsetBound), ←previousEq]
  cases previous with
  | none =>
      simp only [Option.isNone, ite_true]
      step with rust_zero_words count as ⟨words, wordsEq⟩
      step with alloc.vec.Vec.index_mut_usize_spec middle.pages storage pageBounds.1
        as ⟨borrowedSlots, outerRestore, outerValue, outerBack⟩
      have borrowedAt : borrowedSlots = middle.pages.val[storage.val]! := by
        rw [outerValue, List.Inhabited_getElem_eq_getElem! _ _ pageBounds.1]
      step with alloc.vec.Vec.index_mut_usize_spec borrowedSlots offset
        (by simpa only [borrowedAt] using offsetBound) as ⟨oldValue, innerRestore, innerValue, innerBack⟩
      refine ⟨middle, middleInvariant, atPcPage, ⟨page, pageEq, pageEnsured⟩,
        oldWords, offsetEq, none, cellValue, rfl, words, wordsEq, ?_⟩
      simp only [setSlot, outerBack, innerBack, borrowedAt]
  | some existing =>
      simp only [Option.isNone, Bool.false_eq_true, ite_false, WP.spec_ok, WP.uncurry'_pair]
      exact ⟨middle, middleInvariant, atPcPage, ⟨page, pageEq, pageEnsured⟩, oldWords, offsetEq,
        some existing, cellValue, rfl, rfl⟩

end MemoryWrite.MayWrite
