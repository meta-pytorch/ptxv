/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierReach

abbrev Bit := Fin 32

def HasBit (word : U32) (bit : Bit) : Prop := word.bv[bit.val] = true

def Includes (before after : U32) : Prop := ∀ bit, HasBit before bit → HasBit after bit

theorem Includes.refl (word : U32) : Includes word word := fun _ => id

theorem Includes.trans {before middle after : U32}
    (first : Includes before middle) (second : Includes middle after) : Includes before after :=
  fun bit present => second bit (first bit present)

@[step]
theorem rust_or (left right : U32) :
    lift (left ||| right) ⦃ word => word = left ||| right ⦄ := by
  simp [lift]

theorem includes_or_left (left right : U32) : Includes left (left ||| right) := by
  intro bit present
  simp only [HasBit, UScalar.bv_or, BitVec.getElem_or, Bool.or_eq_true] at present ⊢
  exact Or.inl present

theorem includes_or_right (left right : U32) : Includes right (left ||| right) := by
  intro bit present
  simp only [HasBit, UScalar.bv_or, BitVec.getElem_or, Bool.or_eq_true] at present ⊢
  exact Or.inr present

theorem word_eq_of_same_bits (left right : U32)
    (same : ∀ bit, HasBit left bit ↔ HasBit right bit) : left = right := by
  apply U32.bv_eq_imp_eq
  apply BitVec.eq_of_getElem_eq
  intro bit inside
  exact Bool.eq_iff_iff.mpr (same ⟨bit, inside⟩)

theorem different_adds_bit (before after : U32) (grows : Includes before after)
    (different : after ≠ before) : ∃ bit, HasBit after bit ∧ ¬HasBit before bit := by
  by_contra absent
  have reverse : Includes after before := by
    intro bit present
    by_contra missing
    exact absent ⟨bit, present, missing⟩
  apply different
  exact word_eq_of_same_bits after before (fun bit => ⟨reverse bit, grows bit⟩)

theorem shared_bit_intersection_nonzero (word mask : U32) (bit : Bit)
    (wordBit : HasBit word bit) (maskBit : HasBit mask bit) : word &&& mask ≠ 0#u32 := by
  intro zero
  have observation := congrArg (fun value : U32 => value.bv[bit.val]) zero
  change (word.bv &&& mask.bv)[bit.val] = (0#32)[bit.val] at observation
  simp only [HasBit] at wordBit maskBit
  simp only [BitVec.getElem_and, wordBit, maskBit, Bool.true_and, BitVec.getElem_zero,
    Bool.true_eq_false] at observation

def HasBitAt (reach : List U32) (pc : Nat) (bit : Bit) : Prop :=
  ∃ word, reach[pc]? = some word ∧ HasBit word bit

def Grows (before after : List U32) : Prop :=
  ∀ pc bit, HasBitAt before pc bit → HasBitAt after pc bit

theorem Grows.refl (reach : List U32) : Grows reach reach := fun _ _ => id

theorem Grows.trans {before middle after : List U32}
    (first : Grows before middle) (second : Grows middle after) : Grows before after :=
  fun pc bit present => second pc bit (first pc bit present)

theorem bit_at_bound (reach : List U32) (pc : Nat) (bit : Bit) (present : HasBitAt reach pc bit) :
    pc < reach.length := by
  obtain ⟨word, found, _⟩ := present
  exact (List.getElem?_eq_some_iff.mp found).1

theorem bit_at_row (reach : List U32) (pc : Nat) (inside : pc < reach.length) (bit : Bit) :
    HasBitAt reach pc bit ↔ HasBit reach[pc]! bit := by
  simp only [HasBitAt, List.getElem?_eq_getElem inside,
    List.Inhabited_getElem_eq_getElem! _ _ inside, Option.some.injEq, exists_eq_left']

theorem update_grows (reach : List U32) (pc : Nat) (word : U32)
    (inside : pc < reach.length) (included : Includes reach[pc]! word) :
    Grows reach (reach.set pc word) := by
  intro address bit present
  obtain ⟨previous, found, holds⟩ := present
  by_cases same : address = pc
  · subst address
    have previousEq : previous = reach[pc]! := by
      simpa only [List.getElem?_eq_getElem inside,
        List.Inhabited_getElem_eq_getElem! _ _ inside, Option.some.injEq] using found.symm
    exact ⟨word, List.getElem?_set_self inside, included bit (by simpa only [previousEq] using holds)⟩
  · exact ⟨previous, by simpa only [List.getElem?_set_ne (Ne.symm same)] using found, holds⟩

def EdgesBounded (successors : List (alloc.vec.Vec Usize)) (count : Nat) : Prop :=
  successors.length = count ∧
  ∀ pc < count, ∀ destination ∈ successors[pc]!.val, destination.val < count

def Closed (successors : List (alloc.vec.Vec Usize)) (reach : List U32) : Prop :=
  ∀ pc < reach.length, ∀ destination ∈ successors[pc]!.val,
    Includes reach[destination.val]! reach[pc]!

abbrev Fact (count : Nat) := Fin count × Bit

noncomputable def facts (count : Nat) (reach : List U32) : Finset (Fact count) :=
  @Finset.filter _ (fun fact => HasBitAt reach fact.1.val fact.2) (Classical.decPred _) Finset.univ

@[simp]
theorem mem_facts (count : Nat) (reach : List U32) (fact : Fact count) :
    fact ∈ facts count reach ↔ HasBitAt reach fact.1.val fact.2 := by
  simp only [facts, Finset.mem_filter, Finset.mem_univ, true_and]

theorem facts_bound (count : Nat) (reach : List U32) : (facts count reach).card ≤ count * 32 := by
  simpa only [Fact, Fintype.card_prod, Fintype.card_fin] using Finset.card_le_univ (facts count reach)

theorem facts_grow (count : Nat) (before after : List U32) (growth : Grows before after) :
    facts count before ⊆ facts count after := by
  intro fact present
  rw [mem_facts] at present ⊢
  exact growth _ _ present

theorem facts_card_grow (count : Nat) (before after : List U32) (growth : Grows before after) :
    (facts count before).card ≤ (facts count after).card :=
  Finset.card_le_card (facts_grow count before after growth)

theorem facts_card_strict (count : Nat) (before after : List U32) (growth : Grows before after)
    (fact : Fact count) (present : HasBitAt after fact.1.val fact.2)
    (absent : ¬HasBitAt before fact.1.val fact.2) :
    (facts count before).card < (facts count after).card := by
  apply Finset.card_lt_card
  apply Finset.ssubset_iff_subset_ne.mpr
  refine ⟨facts_grow count before after growth, ?_⟩
  intro equal
  apply absent
  rw [←mem_facts count before fact, equal, mem_facts]
  exact present

theorem changed_update_strict (count : Nat) (reach : List U32) (pc : Nat) (word : U32)
    (inside : pc < reach.length) (pcBound : pc < count)
    (included : Includes reach[pc]! word) (different : word ≠ reach[pc]!) :
    (facts count reach).card < (facts count (reach.set pc word)).card := by
  obtain ⟨bit, present, absent⟩ := different_adds_bit reach[pc]! word included different
  apply facts_card_strict count reach (reach.set pc word) (update_grows reach pc word inside included)
    (⟨pc, pcBound⟩, bit)
  · exact ⟨word, List.getElem?_set_self inside, present⟩
  · exact fun old => absent ((bit_at_row reach pc inside bit).mp old)

end MemoryWrite.BarrierReach
