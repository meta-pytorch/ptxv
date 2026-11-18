/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexEnsurePage
import MemoryWrite.RegisterIndexSlots

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

def Interned (before after : register_index.RegisterIndex) (key : symbolic.RegisterKey) (id : Usize) : Prop :=
  Valid after ∧ before.count.val ≤ after.count.val ∧ Stored after key id ∧
  ∀ query value, Stored after query value ↔ Stored before query value ∨ (query = key ∧ value = id)

theorem ready_stored (before after : alloc.vec.Vec register_index.RegisterPage) (count : Usize)
    (page : U64) (position : Usize) (ready : PageReady before after page position)
    (key : symbolic.RegisterKey) (id : Usize) :
    Stored ⟨after, count⟩ key id ↔ Stored ⟨before, count⟩ key id := by
  rw [stored_iff_rows, stored_iff_rows]
  exact ready.2.2.2 key id

theorem ready_valid (before after : alloc.vec.Vec register_index.RegisterPage) (count : Usize)
    (page : U64) (position : Usize) (ready : PageReady before after page position)
    (valid : Valid ⟨before, count⟩) : Valid ⟨after, count⟩ := by
  refine ⟨ready.1, ?_, ?_⟩
  · intro key id stored
    exact valid.2.1 key id ((ready_stored before after count page position ready key id).mp stored)
  · intro left right id leftStored rightStored
    exact valid.2.2 left right id
      ((ready_stored before after count page position ready left id).mp leftStored)
      ((ready_stored before after count page position ready right id).mp rightStored)

theorem fresh_slot_interned (index : register_index.RegisterIndex) (valid : Valid index)
    (key : symbolic.RegisterKey) (position offset next : Usize)
    (atPage : PageAt index.pages (keyPage key) position) (offsetValue : offset.val = keyOffset key)
    (empty : cell index.pages position offset = some none)
    (nextValue : next.val = index.count.val + 1) :
    Interned index ⟨setSlot index.pages position offset (some index.count), next⟩ key index.count := by
  have noOld : ∀ value, ¬Stored index key value := by
    intro value stored
    have lookup := (stored_at_known_page index.pages index.count valid.1 key position offset atPage offsetValue value).mp stored
    rw [empty] at lookup
    cases lookup
  have union : ∀ query value,
      Stored ⟨setSlot index.pages position offset (some index.count), next⟩ query value ↔
      Stored index query value ∨ (query = key ∧ value = index.count) := by
    intro query value
    have effect := set_slot_stored index.pages next valid.1 key position offset atPage offsetValue index.count query value
    constructor
    · intro stored
      rcases effect.mp stored with fresh | old
      · exact Or.inr fresh
      · exact Or.inl old.2
    · intro stored
      rcases stored with old | fresh
      · have different : query ≠ key := by intro equal; subst query; exact noOld value old
        exact effect.mpr (Or.inr ⟨different, old⟩)
      · exact effect.mpr (Or.inl fresh)
  refine ⟨⟨set_slot_directory index.pages valid.1 position offset (some index.count) atPage.1, ?_, ?_⟩,
    by dsimp only; omega, (union key index.count).mpr (Or.inr ⟨rfl, rfl⟩), union⟩
  · intro query value stored
    rcases (union query value).mp stored with old | fresh
    · have bound := valid.2.1 query value old
      dsimp only
      omega
    · dsimp only
      rw [fresh.2]
      omega
  · intro left right value leftStored rightStored
    rcases (union left value).mp leftStored with oldLeft | freshLeft <;>
      rcases (union right value).mp rightStored with oldRight | freshRight
    · exact valid.2.2 left right value oldLeft oldRight
    · have bound := valid.2.1 left value oldLeft
      rw [freshRight.2] at bound
      omega
    · have bound := valid.2.1 right value oldRight
      rw [freshLeft.2] at bound
      omega
    · exact freshLeft.1.trans freshRight.1.symm

theorem existing_slot_interned (index : register_index.RegisterIndex) (valid : Valid index)
    (key : symbolic.RegisterKey) (id : Usize) (stored : Stored index key id) :
    Interned index index key id := by
  refine ⟨valid, Nat.le_refl _, stored, ?_⟩
  intro query value
  constructor
  · exact Or.inl
  · rintro (old | ⟨equal, idEqual⟩)
    · exact old
    · subst query
      subst value
      exact stored

theorem interned_after_ready (before after : alloc.vec.Vec register_index.RegisterPage) (count : Usize)
    (page : U64) (position : Usize) (ready : PageReady before after page position)
    (result : register_index.RegisterIndex) (key : symbolic.RegisterKey) (id : Usize)
    (interned : Interned ⟨after, count⟩ result key id) : Interned ⟨before, count⟩ result key id := by
  refine ⟨interned.1, interned.2.1, interned.2.2.1, ?_⟩
  intro query value
  rw [interned.2.2.2 query value, ready_stored before after count page position ready query value]

end MemoryWrite.RegisterIndex
