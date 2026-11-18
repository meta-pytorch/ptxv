/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexAllocation

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

theorem insert_strict_keys (original : List register_index.RegisterPage) (position : Usize)
    (entry : register_index.RegisterPage)
    (strict : StrictKeys original.length (fun index => (original[index]!).key.val))
    (bound : LowerBound original.length entry.key.val (fun index => (original[index]!).key.val) position)
    (missing : ∀ index < original.length, (original[index]!).key.val ≠ entry.key.val) :
    StrictKeys (original.insertIdx position.val entry).length
      (fun index => ((original.insertIdx position.val entry)[index]!).key.val) := by
  intro left right less rightLength
  have positionBound := bound.1
  have resultLength := List.length_insertIdx_of_le_length bound.1 entry
  have leftLength : left < original.length + 1 := by omega
  have rightLength' : right < original.length + 1 := by omega
  dsimp only
  rw [get_inserted original position.val entry bound.1 left leftLength,
    get_inserted original position.val entry bound.1 right rightLength']
  by_cases rightBefore : right < position.val
  · have leftBefore : left < position.val := by omega
    simp only [rightBefore, leftBefore, ite_true]
    exact strict left right less (by omega)
  · simp only [rightBefore, ite_false]
    by_cases rightAt : right = position.val
    · have leftBefore : left < position.val := by omega
      simp only [rightAt, leftBefore, ite_true]
      exact bound.2.1 left leftBefore
    · simp only [rightAt, ite_false]
      have oldRight : right - 1 < original.length := by omega
      by_cases leftBefore : left < position.val
      · simp only [leftBefore, ite_true]
        exact strict left (right - 1) (by omega) oldRight
      · simp only [leftBefore, ite_false]
        by_cases leftAt : left = position.val
        · simp only [leftAt, ite_true]
          have upper := bound.2.2 (right - 1) (by omega) oldRight
          have different := missing (right - 1) oldRight
          dsimp only at upper
          omega
        · simp only [leftAt, ite_false]
          exact strict (left - 1) (right - 1) (by omega) oldRight


def StoredRows (pages : List register_index.RegisterPage) (key : symbolic.RegisterKey) (id : Usize) : Prop :=
  ∃ page ∈ pages, page.key.val = keyPage key ∧ page.slots.val[keyOffset key]? = some (some id)

theorem stored_iff_rows (index : register_index.RegisterIndex) (key : symbolic.RegisterKey) (id : Usize) :
    Stored index key id ↔ StoredRows index.pages.val key id := by
  constructor
  · rintro ⟨position, inside, sameKey, slot⟩
    refine ⟨index.pages.val[position]!, ?_, sameKey, slot⟩
    rw [←List.Inhabited_getElem_eq_getElem! _ _ inside]
    exact List.getElem_mem inside
  · rintro ⟨page, member, sameKey, slot⟩
    obtain ⟨position, inside, atPosition⟩ := List.getElem_of_mem member
    refine ⟨position, inside, ?_, ?_⟩ <;>
      simpa only [←List.Inhabited_getElem_eq_getElem! _ _ inside, atPosition]

theorem directory_insert (pages : List register_index.RegisterPage) (valid : Directory pages)
    (position : Usize) (page : register_index.RegisterPage)
    (slots : page.slots.val.length = 64)
    (bound : LowerBound pages.length page.key.val (fun index => (pages[index]!).key.val) position)
    (missing : ∀ index < pages.length, (pages[index]!).key.val ≠ page.key.val) :
    Directory (pages.insertIdx position.val page) := by
  have positionBound := bound.1
  constructor
  · intro index inside
    have length : index < pages.length + 1 := by
      simpa only [List.length_insertIdx_of_le_length bound.1] using inside
    rw [get_inserted pages position.val page bound.1 index length]
    split
    · exact valid.1 index (by omega)
    · split
      · exact slots
      · exact valid.1 (index - 1) (by omega)
  · exact insert_strict_keys pages position page valid.2 bound missing

theorem empty_insert_preserves_rows (pages : List register_index.RegisterPage)
    (position : Usize) (page : register_index.RegisterPage)
    (bound : position.val ≤ pages.length)
    (empty : page.slots.val = List.replicate 64 (none : Option Usize))
    (key : symbolic.RegisterKey) (id : Usize) :
    StoredRows (pages.insertIdx position.val page) key id ↔ StoredRows pages key id := by
  constructor
  · rintro ⟨row, member, sameKey, slot⟩
    rcases (List.mem_insertIdx bound).mp member with equal | old
    · subst row
      rw [empty, List.getElem?_replicate] at slot
      split at slot <;> cases slot
    · exact ⟨row, old, sameKey, slot⟩
  · rintro ⟨row, member, sameKey, slot⟩
    exact ⟨row, (List.mem_insertIdx bound).mpr (Or.inr member), sameKey, slot⟩

end MemoryWrite.RegisterIndex
