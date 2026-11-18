/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ArrivalWalkCallbacks

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace ArrivalWalk

@[step]
theorem rust_restart {State W : Type} (walker : arrival_walk.Walker W State) (view : W → View State)
    (settled : W → Prop) (contracts : LocalContracts walker view settled) (world : W) (pending : arrival_walk.Frontier State)
    (empty : Represents pending []) :
    arrival_walk.restart walker world pending ⦃ success result frontier =>
      success = true → Invariant view result frontier true ⦄ := by
  unfold arrival_walk.restart
  step with contracts.begin world as ⟨initial, result, unseen, initialEq⟩
  step with rust_push pending [] empty initial as ⟨success, frontier, pushed⟩
  rcases pushed with ⟨failed, _⟩ | ⟨succeeded, rep⟩
  · simp only [failed, Bool.false_eq_true, false_implies]
  · intro _
    exact Invariant.begin view result frontier initial (by simpa using rep) unseen initialEq

end ArrivalWalk
