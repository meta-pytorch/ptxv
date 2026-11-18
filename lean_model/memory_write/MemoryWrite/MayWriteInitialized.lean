/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteEnsurePc

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

structure InitializedPc (before after : cache_graph.WorkSets) (pc count : Usize)
    (answer : Usize × Usize × Bool) : Prop where
  invariant : DirectoryInvariant after
  page : PageAt after (pc.val / 64) answer.1
  offset : answer.2.1.val = pc.val % 64
  current : ∃ words : alloc.vec.Vec U64, WordsAt after pc.val words ∧
    (answer.2.2 = true → words.val = List.replicate count.val 0#u64) ∧
    (answer.2.2 = false → WordsAt before pc.val words)
  fresh : answer.2.2 = true ↔ ¬∃ words, WordsAt before pc.val words
  others : ∀ address, address ≠ pc.val → ∀ words,
    WordsAt after address words ↔ WordsAt before address words

theorem pc_ensured_initializes {before after : cache_graph.WorkSets} {pc count : Usize}
    {answer : Usize × Usize × Bool} (ensured : PcEnsured before after pc count answer) :
    InitializedPc before after pc count answer := by
  obtain ⟨middle, invariant, atPage, _, oldWords, offset, previous, lookup, flag, next⟩ := ensured
  have bounds := page_at_bounds invariant atPage
  cases previous with
  | none =>
      change answer.2.2 = true at flag
      obtain ⟨words, wordsEq, afterEq⟩ := next
      subst after
      have newInvariant := set_slot_preserves_invariant invariant answer.1 answer.2.1
        (some words) bounds.1
      have current : WordsAt (setSlot middle answer.1 answer.2.1 (some words)) pc.val words :=
        (set_slot_words_at_target invariant atPage offset (some words) words).mpr rfl
      have absent : ¬∃ old, WordsAt before pc.val old := by
        rintro ⟨old, oldAt⟩
        have oldLookup := (words_at_known_page invariant atPage offset old).mp
          ((oldWords pc.val old).mpr oldAt)
        have impossible := Option.some.inj (lookup.symm.trans oldLookup)
        cases impossible
      refine ⟨newInvariant, (page_at_set_slot _ _ _ _ _ _).mpr atPage, offset, ?_, ?_, ?_⟩
      · exact ⟨words, current, fun _ => wordsEq, fun oldFlag => by simp only [flag, reduceCtorEq] at oldFlag⟩
      · exact ⟨fun _ => absent, fun _ => flag⟩
      · intro address different old
        exact (set_slot_words_at_other invariant atPage offset (some words) address different old).trans
          (oldWords address old)
  | some existing =>
      change answer.2.2 = false at flag
      subst after
      have current : WordsAt middle pc.val existing :=
        (words_at_known_page invariant atPage offset existing).mpr lookup
      refine ⟨invariant, atPage, offset, ?_, ?_, ?_⟩
      · exact ⟨existing, current, fun freshFlag => by simp only [flag, reduceCtorEq] at freshFlag,
          fun _ => (oldWords pc.val existing).mp current⟩
      · constructor
        · intro freshFlag
          simp only [flag, reduceCtorEq] at freshFlag
        · intro absent
          exact False.elim (absent ⟨existing, (oldWords pc.val existing).mp current⟩)
      · intro address _ words
        exact oldWords address words


theorem InitializedPc.old_words_preserved {before after : cache_graph.WorkSets}
    {pc count : Usize} {answer : Usize × Usize × Bool}
    (initialized : InitializedPc before after pc count answer)
    (beforeInvariant : DirectoryInvariant before) {address : Nat} {words : alloc.vec.Vec U64}
    (oldAt : WordsAt before address words) : WordsAt after address words := by
  by_cases same : address = pc.val
  · subst address
    obtain ⟨current, currentAt, _, currentOld⟩ := initialized.current
    have notFresh : answer.2.2 ≠ true := by
      intro fresh
      exact initialized.fresh.mp fresh ⟨words, oldAt⟩
    have oldFlag : answer.2.2 = false := Bool.eq_false_iff.mpr notFresh
    have equal := words_at_unique beforeInvariant (currentOld oldFlag) oldAt
    simpa only [equal] using currentAt
  · exact (initialized.others address same words).mpr oldAt

theorem InitializedPc.present {before after : cache_graph.WorkSets}
    {pc count : Usize} {answer : Usize × Usize × Bool}
    (initialized : InitializedPc before after pc count answer)
    (beforeInvariant : DirectoryInvariant before) (address : Nat) :
    (workSolution after).present address ↔ (workSolution before).present address ∨ address = pc.val := by
  constructor
  · rintro ⟨words, atWords⟩
    by_cases same : address = pc.val
    · exact Or.inr same
    · exact Or.inl ⟨words, (initialized.others address same words).mp atWords⟩
  · rintro (⟨words, oldAt⟩ | same)
    · exact ⟨words, initialized.old_words_preserved beforeInvariant oldAt⟩
    · subst address
      obtain ⟨words, atWords, _⟩ := initialized.current
      exact ⟨words, atWords⟩

theorem InitializedPc.incoming {before after : cache_graph.WorkSets}
    {pc count : Usize} {answer : Usize × Usize × Bool}
    (initialized : InitializedPc before after pc count answer)
    (beforeInvariant : DirectoryInvariant before) (address register : Nat) :
    (workSolution after).incoming address register ↔ (workSolution before).incoming address register := by
  constructor
  · rintro ⟨words, atWords, bit⟩
    by_cases same : address = pc.val
    · subst address
      obtain ⟨current, currentAt, currentZero, currentOld⟩ := initialized.current
      have equal := words_at_unique initialized.invariant currentAt atWords
      subst words
      by_cases fresh : answer.2.2 = true
      · exact False.elim (zero_words_empty count.val register (by simpa only [currentZero fresh] using bit))
      · exact ⟨current, currentOld (Bool.eq_false_iff.mpr fresh), bit⟩
    · exact ⟨words, (initialized.others address same words).mp atWords, bit⟩
  · rintro ⟨words, oldAt, bit⟩
    exact ⟨words, initialized.old_words_preserved beforeInvariant oldAt, bit⟩

@[step]
theorem rust_ensure_pc_initialized (sets : cache_graph.WorkSets) (pc count : Usize)
    (invariant : DirectoryInvariant sets)
    (directorySpace : sets.directory.val.length < Usize.max)
    (pagesSpace : sets.pages.val.length < Usize.max) :
    cache_graph.ensure_pc sets pc count ⦃ answer result => InitializedPc sets result pc count answer ⦄ := by
  apply WP.spec_mono (rust_ensure_pc sets pc count invariant directorySpace pagesSpace)
  intro result ensured
  exact pc_ensured_initializes ensured

end MemoryWrite.MayWrite
