/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteInitialized
import MemoryWrite.MayWriteRowMerge
import MemoryWrite.MayWriteWorklist
import MemoryWrite.MayWriteResources

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

def RowsSized (sets : cache_graph.WorkSets) (count : Nat) : Prop :=
  ∀ pc words, WordsAt sets pc words → words.val.length = count

theorem InitializedPc.sized {before after : cache_graph.WorkSets} {pc count : Usize}
    {answer : Usize × Usize × Bool} (initialized : InitializedPc before after pc count answer)
    (beforeSized : RowsSized before count.val) : RowsSized after count.val := by
  intro address words atWords
  by_cases same : address = pc.val
  · subst address
    obtain ⟨current, currentAt, currentZero, currentOld⟩ := initialized.current
    have equal := words_at_unique initialized.invariant atWords currentAt
    subst words
    cases flag : answer.2.2 with
    | true => simp only [currentZero flag, List.length_replicate]
    | false => exact beforeSized pc.val current (currentOld flag)
  · exact beforeSized address words ((initialized.others address same words).mp atWords)

structure MergedPc (before after : cache_graph.WorkSets) (pc : Nat)
    (outgoing : List U64) (changed : Bool) : Prop where
  invariant : DirectoryInvariant after
  present : ∀ address, (workSolution after).present address ↔ (workSolution before).present address
  incoming : ∀ address register, (workSolution after).incoming address register ↔
    (workSolution before).incoming address register ∨
      (address = pc ∧ WordContains outgoing register)
  unchanged : changed = false → ∀ address words,
    WordsAt after address words ↔ WordsAt before address words
  sized : ∀ count, RowsSized before count → RowsSized after count
  resources : DirectoryResources before → DirectoryResources after

@[step]
theorem rust_merge_row_effect (sets : cache_graph.WorkSets) (pc : Nat) (storage offset : Usize)
    (source : Slice U64) (oldWords : alloc.vec.Vec U64)
    (invariant : DirectoryInvariant sets) (atPage : PageAt sets (pc / 64) storage)
    (offsetValue : offset.val = pc % 64) (oldAt : WordsAt sets pc oldWords)
    (lengths : oldWords.val.length = source.val.length) :
    cache_graph.merge_row sets storage offset source ⦃ changed result =>
      MergedPc sets result pc source.val changed ⦄ := by
  apply WP.spec_mono (rust_merge_row sets pc storage offset source oldWords invariant
    atPage offsetValue oldAt lengths)
  rintro ⟨changed, result⟩ ⟨words, resultEq, wordLength, unionBits, changedIff⟩
  subst result
  have bounds := page_at_bounds invariant atPage
  have afterInvariant := set_slot_preserves_invariant invariant storage offset (some words) bounds.1
  have newAt := (set_slot_words_at_target invariant atPage offsetValue (some words) words).mpr rfl
  have targetEqual : ∀ observed, WordsAt (setSlot sets storage offset (some words)) pc observed ↔ observed = words := by
    intro observed
    rw [set_slot_words_at_target invariant atPage offsetValue (some words) observed]
    simp only [Option.some.injEq, eq_comm]
  have others := set_slot_words_at_other invariant atPage offsetValue (some words)
  refine ⟨afterInvariant, ?_, ?_, ?_, ?_, ?_⟩
  · intro address
    by_cases same : address = pc
    · subst address
      exact ⟨fun _ => ⟨oldWords, oldAt⟩, fun _ => ⟨words, newAt⟩⟩
    · exact ⟨fun ⟨observed, atWords⟩ => ⟨observed, (others address same observed).mp atWords⟩,
        fun ⟨observed, atWords⟩ => ⟨observed, (others address same observed).mpr atWords⟩⟩
  · intro address register
    by_cases same : address = pc
    · subst address
      constructor
      · rintro ⟨observed, observedAt, bit⟩
        have sameWords := (targetEqual observed).mp observedAt
        subst observed
        rcases (unionBits register).mp bit with oldBit | outgoingBit
        · exact Or.inl ⟨oldWords, oldAt, oldBit⟩
        · exact Or.inr ⟨rfl, outgoingBit⟩
      · rintro (⟨observed, observedAt, bit⟩ | ⟨_, bit⟩)
        · have sameWords := words_at_unique invariant observedAt oldAt
          subst observed
          exact ⟨words, newAt, (unionBits register).mpr (Or.inl bit)⟩
        · exact ⟨words, newAt, (unionBits register).mpr (Or.inr bit)⟩
    · have convert := others address same
      constructor
      · rintro ⟨observed, observedAt, bit⟩
        exact Or.inl ⟨observed, (convert observed).mp observedAt, bit⟩
      · rintro (⟨observed, observedAt, bit⟩ | ⟨equal, _⟩)
        · exact ⟨observed, (convert observed).mpr observedAt, bit⟩
        · exact False.elim (same equal)
  · intro unchanged address observed
    have equalValues : words.val = oldWords.val := by
      by_contra different
      have changedTrue := changedIff.mpr different
      simp only [unchanged, Bool.false_eq_true] at changedTrue
    have equalWords : words = oldWords := alloc.vec.Vec.ext _ _ equalValues
    by_cases same : address = pc
    · subst address
      constructor
      · intro atObserved
        have observedEq := (targetEqual observed).mp atObserved
        simpa only [observedEq, equalWords] using oldAt
      · intro atObserved
        have observedEq := words_at_unique invariant atObserved oldAt
        simpa only [observedEq, equalWords] using newAt
    · exact others address same observed
  · intro count sized address observed atObserved
    by_cases same : address = pc
    · subst address
      have observedEq := (targetEqual observed).mp atObserved
      rw [observedEq, wordLength]
      exact sized pc oldWords oldAt
    · exact sized address observed ((others address same observed).mp atObserved)
  · intro resources
    exact set_slot_preserves_resources resources storage offset (some words)

end MemoryWrite.MayWrite
