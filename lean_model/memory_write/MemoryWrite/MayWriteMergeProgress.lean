/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteMergeEffect
import MemoryWrite.MayWriteWordProgress
import MemoryWrite.MayWriteFiniteFacts

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

def AddsIncoming (before after : cache_graph.WorkSets) (pc width : Nat) : Prop :=
  ∃ register, register < width ∧ (workSolution after).incoming pc register ∧
    ¬(workSolution before).incoming pc register

@[step]
theorem rust_merge_row_adds_incoming (sets : cache_graph.WorkSets) (pc : Nat) (storage offset : Usize)
    (source : Slice U64) (oldWords : alloc.vec.Vec U64)
    (invariant : DirectoryInvariant sets) (atPage : PageAt sets (pc / 64) storage)
    (offsetValue : offset.val = pc % 64) (oldAt : WordsAt sets pc oldWords)
    (lengths : oldWords.val.length = source.val.length) :
    cache_graph.merge_row sets storage offset source ⦃ changed result =>
      changed = true → AddsIncoming sets result pc (oldWords.val.length * 64) ⦄ := by
  apply WP.spec_mono (rust_merge_row sets pc storage offset source oldWords invariant
    atPage offsetValue oldAt lengths)
  rintro ⟨changed, result⟩ ⟨words, resultEq, wordLength, unionBits, changedIff⟩ changedTrue
  subst result
  have different : words.val ≠ oldWords.val := changedIff.mp changedTrue
  obtain ⟨register, bound, newBit, missing⟩ := growing_different_words_add_bit oldWords.val words.val
    wordLength.symm (fun register bit => (unionBits register).mpr (Or.inl bit)) different
  have afterAt := (set_slot_words_at_target invariant atPage offsetValue (some words) words).mpr rfl
  refine ⟨register, bound, ⟨words, afterAt, newBit⟩, ?_⟩
  rintro ⟨observed, observedAt, oldBit⟩
  have equal := words_at_unique invariant observedAt oldAt
  exact missing (by simpa only [equal] using oldBit)

end MemoryWrite.MayWrite
