/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexIntern

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex
deriving instance Inhabited for symbolic.RegisterKey

def Preserves (before after : register_index.RegisterIndex) : Prop :=
  ∀ key id, Stored before key id → Stored after key id

theorem Preserves.refl (index : register_index.RegisterIndex) : Preserves index index := fun _ _ => id

theorem Preserves.trans {before middle after : register_index.RegisterIndex}
    (first : Preserves before middle) (second : Preserves middle after) : Preserves before after :=
  fun key id stored => second key id (first key id stored)

theorem Interned.preserves {before after : register_index.RegisterIndex} {key : symbolic.RegisterKey} {id : Usize}
    (effect : Interned before after key id) : Preserves before after :=
  fun query value stored => (effect.2.2.2 query value).mpr (Or.inl stored)

def RowRepresents (index : register_index.RegisterIndex) (destinations : List symbolic.RegisterKey)
    (row : List Usize) : Prop :=
  row.length = destinations.length ∧
  ∀ position, position < destinations.length → Stored index destinations[position]! row[position]!

theorem RowRepresents.grows (before after : register_index.RegisterIndex)
    (preserves : Preserves before after) (destinations : List symbolic.RegisterKey) (row : List Usize)
    (represented : RowRepresents before destinations row) : RowRepresents after destinations row :=
  ⟨represented.1, fun position bound => preserves _ _ (represented.2 position bound)⟩

def RowPrefix (index : register_index.RegisterIndex) (destinations : List symbolic.RegisterKey)
    (row : List Usize) (position : Nat) : Prop :=
  position ≤ destinations.length ∧ row.length = position ∧
  ∀ earlier, earlier < position → Stored index destinations[earlier]! row[earlier]!

theorem row_prefix_append (before after : register_index.RegisterIndex)
    (destinations : List symbolic.RegisterKey) (row : List Usize) (position : Nat) (id : Usize)
    (beforePrefix : RowPrefix before destinations row position) (inside : position < destinations.length)
    (effect : Interned before after destinations[position]! id) :
    RowPrefix after destinations (row ++ [id]) (position + 1) := by
  refine ⟨by omega, by simp only [List.length_append, List.length_singleton, beforePrefix.2.1], ?_⟩
  have rowLength := beforePrefix.2.1
  intro earlier bound
  by_cases old : earlier < position
  · have rowBound : earlier < row.length := by omega
    rw [←List.Inhabited_getElem_eq_getElem! (row ++ [id]) earlier (by simp only [List.length_append, List.length_singleton]; omega),
      List.getElem_append_left rowBound, List.Inhabited_getElem_eq_getElem! _ _ rowBound]
    exact effect.preserves _ _ (beforePrefix.2.2 earlier old)
  · have atEnd : earlier = position := by omega
    subst earlier
    have rowLength := beforePrefix.2.1
    have appendedAt : (row ++ [id])[position]! = id := by
      rw [←List.Inhabited_getElem_eq_getElem! (row ++ [id]) position (by simp only [List.length_append, List.length_singleton]; omega),
        List.getElem_append_right (by omega)]
      have atZero : position - row.length = 0 := by omega
      simp only [atZero, List.getElem_singleton]
    rw [appendedAt]
    exact effect.2.2.1

end MemoryWrite.RegisterIndex
