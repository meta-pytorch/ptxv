/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteMergeProgress

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem initialization_merge_growth {before middle after : cache_graph.WorkSets}
    {pc words : Usize} {answer : Usize × Usize × Bool} {outgoing : List U64} {changed : Bool}
    (beforeInvariant : DirectoryInvariant before)
    (initialized : InitializedPc before middle pc words answer)
    (merged : MergedPc middle after pc.val outgoing changed) :
    Grows (workSolution before) (workSolution after) := by
  constructor
  · intro address known
    exact (merged.present address).mpr ((initialized.present beforeInvariant address).mpr (Or.inl known))
  · intro address register bit
    exact (merged.incoming address register).mpr
      (Or.inl ((initialized.incoming beforeInvariant address register).mpr bit))

theorem queued_initialization_merge_strict {before middle after : cache_graph.WorkSets}
    {pc words : Usize} {answer : Usize × Usize × Bool} {outgoing : List U64} {changed : Bool}
    (count : Nat) (pcBound : pc.val < count)
    (beforeInvariant : DirectoryInvariant before)
    (initialized : InitializedPc before middle pc words answer)
    (merged : MergedPc middle after pc.val outgoing changed)
    (progress : changed = true → AddsIncoming middle after pc.val (words.val * 64))
    (queued : (answer.2.2 || changed) = true) :
    (facts count (words.val * 64) (workSolution before)).card <
      (facts count (words.val * 64) (workSolution after)).card := by
  have growth := initialization_merge_growth beforeInvariant initialized merged
  by_cases fresh : answer.2.2 = true
  · apply facts_card_strict count (words.val * 64) (workSolution before) (workSolution after) growth
      (⟨⟨pc.val, pcBound⟩, none⟩ : Fact count (words.val * 64))
    · obtain ⟨current, currentAt, _, _⟩ := initialized.current
      exact (merged.present pc.val).mpr ⟨current, currentAt⟩
    · exact initialized.fresh.mp fresh
  · have freshFalse : answer.2.2 = false := Bool.eq_false_iff.mpr fresh
    have changedTrue : changed = true := by simpa only [freshFalse, Bool.false_or] using queued
    obtain ⟨register, registerBound, added, missing⟩ := progress changedTrue
    apply facts_card_strict count (words.val * 64) (workSolution before) (workSolution after) growth
      (⟨⟨pc.val, pcBound⟩, some ⟨register, registerBound⟩⟩ : Fact count (words.val * 64)) added
    intro old
    exact missing ((initialized.incoming beforeInvariant pc.val register).mpr old)

end MemoryWrite.MayWrite
