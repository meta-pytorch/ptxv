/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWritePropagation

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

abbrev Fact (count width : Nat) := Fin count × Option (Fin width)

def HoldsFact (solution : Solution) {count width : Nat} (fact : Fact count width) : Prop :=
  match fact.2 with
  | none => solution.present fact.1.val
  | some register => solution.incoming fact.1.val register.val

noncomputable def facts (count width : Nat) (solution : Solution) : Finset (Fact count width) :=
  @Finset.filter _ (HoldsFact solution) (Classical.decPred _) Finset.univ

@[simp]
theorem mem_facts (count width : Nat) (solution : Solution) (fact : Fact count width) :
    fact ∈ facts count width solution ↔ HoldsFact solution fact := by
  simp only [facts, Finset.mem_filter, Finset.mem_univ, true_and]

theorem facts_grow (count width : Nat) (before after : Solution) (growth : Grows before after) :
    facts count width before ⊆ facts count width after := by
  intro fact member
  rw [mem_facts] at member ⊢
  rcases fact with ⟨pc, register⟩
  cases register with
  | none => exact growth.1 pc.val member
  | some register => exact growth.2 pc.val register.val member

theorem facts_card_bound (count width : Nat) (solution : Solution) :
    (facts count width solution).card ≤ count * (width + 1) := by
  simpa only [Fact, Fintype.card_prod, Fintype.card_fin, Fintype.card_option] using
    Finset.card_le_univ (facts count width solution)

theorem facts_card_grows (count width : Nat) (before after : Solution) (growth : Grows before after) :
    (facts count width before).card ≤ (facts count width after).card :=
  Finset.card_le_card (facts_grow count width before after growth)

theorem facts_card_strict (count width : Nat) (before after : Solution) (growth : Grows before after)
    (fact : Fact count width) (newFact : HoldsFact after fact) (absent : ¬HoldsFact before fact) :
    (facts count width before).card < (facts count width after).card := by
  apply Finset.card_lt_card
  apply Finset.ssubset_iff_subset_ne.mpr
  refine ⟨facts_grow count width before after growth, ?_⟩
  intro equal
  have member : fact ∈ facts count width before := by
    rw [equal, mem_facts]; exact newFact
  exact absent ((mem_facts count width before fact).mp member)

noncomputable def rank (count width : Nat) (solution : Solution) (pending : Nat) : Nat :=
  count * (width + 1) - (facts count width solution).card + pending

theorem rank_grows_no_queue (count width : Nat) (before after : Solution)
    (pending : Nat) (growth : Grows before after) : rank count width after pending ≤ rank count width before pending := by
  unfold rank
  have cards := facts_card_grows count width before after growth
  omega

theorem rank_grows_with_queue (count width : Nat) (before after : Solution)
    (pending : Nat) (_growth : Grows before after)
    (strict : (facts count width before).card < (facts count width after).card) :
    rank count width after (pending + 1) ≤ rank count width before pending := by
  unfold rank
  have bound := facts_card_bound count width after
  omega

theorem rank_pop (count width : Nat) (solution : Solution) (pending remaining : Nat)
    (positive : 0 < pending) (removed : remaining = pending - 1) :
    rank count width solution remaining < rank count width solution pending := by
  unfold rank
  omega

theorem pending_le_rank (count width : Nat) (solution : Solution) (pending : Nat) :
    pending ≤ rank count width solution pending := by unfold rank; omega

end MemoryWrite.MayWrite
