/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReachScan

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.BarrierReach

noncomputable def rank (count : Nat) (reach : List U32) (changed : Bool) : Nat :=
  2 * (count * 32 - (facts count reach).card) + if changed then 1 else 0

theorem scan_rank_decreases (count : Nat) (before after : List U32) (changed : Bool)
    (growth : Grows before after)
    (strict : changed = true → (facts count before).card < (facts count after).card) :
    rank count after changed < rank count before true := by
  have cardinality := facts_card_grow count before after growth
  have finite := facts_bound count after
  cases changed with
  | false =>
      simp only [rank, Bool.false_eq_true, ite_false, ite_true]
      omega
  | true =>
      have added := strict rfl
      simp only [rank, ite_true]
      omega

def PropagationInvariant (successors : List (alloc.vec.Vec Usize)) (initial reach : List U32)
    (changed : Bool) : Prop :=
  reach.length = initial.length ∧ Grows initial reach ∧ (changed = false → Closed successors reach)

def PropagationPost (successors : List (alloc.vec.Vec Usize)) (initial reach : List U32) : Prop :=
  reach.length = initial.length ∧ Grows initial reach ∧ Closed successors reach

@[step]
theorem rust_propagate (successors : Slice (alloc.vec.Vec Usize)) (initial : alloc.vec.Vec U32)
    (bounded : EdgesBounded successors.val initial.val.length) :
    barrier_reach.propagate successors initial
      ⦃ reach => PropagationPost successors.val initial.val reach.val ⦄ := by
  unfold barrier_reach.propagate barrier_reach.propagate_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U32 × Bool => rank initial.val.length state.1.val state.2)
    (fun state => PropagationInvariant successors.val initial.val state.1.val state.2)
  · rintro ⟨reach, changed⟩ ⟨sameLength, growth, stable⟩
    dsimp only at sameLength growth stable
    unfold barrier_reach.propagate_loop.body
    cases changed with
    | false =>
        simp only [Bool.false_eq_true, ite_false, WP.spec_ok]
        exact ⟨sameLength, growth, stable rfl⟩
    | true =>
        simp only [ite_true]
        step with rust_scan successors reach (by simpa only [sameLength] using bounded)
          as ⟨nextChanged, next, scanned⟩
        obtain ⟨nextLength, nextGrowth, strict, closed⟩ := scanned
        constructor
        · exact ⟨nextLength.trans sameLength, growth.trans nextGrowth, closed⟩
        · apply scan_rank_decreases initial.val.length reach.val next.val nextChanged nextGrowth
          intro nextTrue
          have added : (facts reach.val.length reach.val).card < (facts reach.val.length next.val).card := strict nextTrue
          rw [sameLength] at added
          exact added
  · exact ⟨rfl, Grows.refl initial.val, by intro impossible; cases impossible⟩

end MemoryWrite.BarrierReach
