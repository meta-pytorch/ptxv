/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.OwnerSymbolic
import MemoryWrite.StoreWordRefinement

open Aeneas Aeneas.Std Result PtxvIr PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace RegisterOwner.Symbolic

open MemoryWrite

def modelWordInput {Input : Type} (state : MemoryWrite.Symbolic.SourceState Input)
    (thread : Nat) (owners : List Nat) (source guard low high : Reg)
    (space : U8) (allocation : U64) : store.WordStoreInput :=
  { space, allocation
    source := state.metadata (resolvedKey thread owners source)
    guard := state.metadata (resolvedKey thread owners guard)
    address_low := state.metadata (resolvedKey thread owners low)
    address_high := state.metadata (resolvedKey thread owners high) }

def ownedWordInput {Input : Type} (state : MemoryWrite.Symbolic.SourceState Input)
    (thread : Usize) (owners : Slice U32) (source guard low high : Reg)
    (space : U8) (allocation : U64) : Result store.WordStoreInput := do
  let sourceKey ← register_owner.canonical_key thread owners source
  let guardKey ← register_owner.canonical_key thread owners guard
  let lowKey ← register_owner.canonical_key thread owners low
  let highKey ← register_owner.canonical_key thread owners high
  .ok
    { space := space
      allocation := allocation
      source := state.metadata (resolvedResult source sourceKey)
      guard := state.metadata (resolvedResult guard guardKey)
      address_low := state.metadata (resolvedResult low lowKey)
      address_high := state.metadata (resolvedResult high highKey) }

@[step]
theorem source_owned_word_input_exact {Input : Type}
    (state : MemoryWrite.Symbolic.SourceState Input)
    (thread : Usize) (owners : Slice U32) (source guard low high : Reg)
    (space : U8) (allocation : U64) :
    ownedWordInput state thread owners source guard low high space allocation
      ⦃ input => input = modelWordInput state thread.val (owners.val.map UScalar.val)
        source guard low high space allocation ⦄ := by
  unfold ownedWordInput
  step with rust_canonical_key thread owners source as ⟨sourceKey, sourceExact⟩
  step with rust_canonical_key thread owners guard as ⟨guardKey, guardExact⟩
  step with rust_canonical_key thread owners low as ⟨lowKey, lowExact⟩
  step with rust_canonical_key thread owners high as ⟨highKey, highExact⟩
  rw [resolved_result_exact _ _ _ _ sourceExact, resolved_result_exact _ _ _ _ guardExact,
    resolved_result_exact _ _ _ _ lowExact, resolved_result_exact _ _ _ _ highExact]
  rfl

def ownedWordCommit {Input : Type} (state : MemoryWrite.Symbolic.SourceState Input)
    (thread : Usize) (owners : Slice U32) (source guard low high : Reg)
    (space : U8) (allocation : U64) (memory : engine.Engine) :
    Result (core.result.Result Unit checked.EffectError × engine.Engine) := do
  let input ← ownedWordInput state thread owners source guard low high space allocation
  store.commit_word memory input

theorem source_owned_word_commit_preserves_dynamic {Input : Type} (sem : Semantics Input)
    (symbolicState : MemoryWrite.Symbolic.SourceState Input)
    (sound : (MemoryWrite.Symbolic.sourceView symbolicState).Sound sem)
    (thread : Usize) (owners : Slice U32) (source guard low high : Reg)
    (space : U8) (allocation : U64) (memory : Memory Input)
    (before : Storage) (state : engine.Engine)
    (wordMeaning : ∀ id input lane,
      sem.word id input lane = MemoryWrite.Symbolic.byteAt (symbolicState.meaning id input) lane)
    (represented : Storage.Represents sem memory before)
    (cells : state.cells.slice = before.toCells)
    (numericRoom : state.numeric_written.val.length + 4 ≤ Usize.max)
    (halfRoom : state.half_written.val.length + 4 ≤ Usize.max) :
    let input := modelWordInput symbolicState thread.val (owners.val.map UScalar.val)
      source guard low high space allocation
    ∃ result,
      ownedWordCommit symbolicState thread owners source guard low high space allocation state = .ok result ∧
      (∃ write, store.prepare_word input = .ok write ∧
        RustEngineRefines before state write result) ∧
      MemoryWrite.StoreWord.CommitOutcome sem memory before state input
        (symbolicState.values (resolvedKey thread.val (owners.val.map UScalar.val) source))
        (symbolicState.values (resolvedKey thread.val (owners.val.map UScalar.val) guard))
        (symbolicState.values (resolvedKey thread.val (owners.val.map UScalar.val) low))
        (symbolicState.values (resolvedKey thread.val (owners.val.map UScalar.val) high)) result := by
  dsimp only
  obtain ⟨input, inputRun, inputExact⟩ := spec_returns _ _
    (source_owned_word_input_exact symbolicState thread owners source guard low high space allocation)
  obtain ⟨result, committed, refines, outcome⟩ :=
    MemoryWrite.StoreWord.rust_commit_word_preserves_dynamic sem symbolicState.meaning memory
      before state input
      (symbolicState.values (resolvedKey thread.val (owners.val.map UScalar.val) source))
      (symbolicState.values (resolvedKey thread.val (owners.val.map UScalar.val) guard))
      (symbolicState.values (resolvedKey thread.val (owners.val.map UScalar.val) low))
      (symbolicState.values (resolvedKey thread.val (owners.val.map UScalar.val) high))
      (by simpa [inputExact, modelWordInput, MemoryWrite.Symbolic.sourceView] using
        sound (resolvedKey thread.val (owners.val.map UScalar.val) source))
      (by simpa [inputExact, modelWordInput, MemoryWrite.Symbolic.sourceView] using
        sound (resolvedKey thread.val (owners.val.map UScalar.val) guard))
      (by simpa [inputExact, modelWordInput, MemoryWrite.Symbolic.sourceView] using
        sound (resolvedKey thread.val (owners.val.map UScalar.val) low))
      (by simpa [inputExact, modelWordInput, MemoryWrite.Symbolic.sourceView] using
        sound (resolvedKey thread.val (owners.val.map UScalar.val) high))
      wordMeaning represented cells numericRoom halfRoom
  refine ⟨result, ?_, ?_, ?_⟩
  · simp only [ownedWordCommit, inputRun, bind_ok]
    exact committed
  · simpa only [inputExact] using refines
  · simpa only [inputExact] using outcome

end RegisterOwner.Symbolic
