/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteAllocation
import MemoryWrite.MayWriteSearch

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

@[step]
theorem rust_insert_page (directory : alloc.vec.Vec cache_graph.PageIndex)
    (position : Usize) (entry : cache_graph.PageIndex)
    (bound : position.val ≤ directory.val.length) (space : directory.val.length < Usize.max) :
    cache_graph.insert_page directory position entry ⦃ result =>
      result.val = directory.val.insertIdx position.val entry ⦄ := by
  unfold cache_graph.insert_page
  by_cases atEnd : position.val = directory.val.length
  · have test : position = alloc.vec.Vec.len directory := by scalar_tac
    simp only [test, ite_true]
    step with alloc.vec.Vec.push_spec directory entry space as ⟨result, resultEq⟩
    simpa only [alloc.vec.Vec.len_val, atEnd, List.insertIdx_length_self] using resultEq
  · have test : position ≠ alloc.vec.Vec.len directory := by scalar_tac
    simp only [test, ite_false]
    exact alloc.vec.Vec.insert_spec directory position entry (by scalar_tac) space

def StrictKeys (length : Nat) (key : Nat → Nat) : Prop :=
  ∀ left right, left < right → right < length → key left < key right

theorem StrictKeys.ordered {length : Nat} {key : Nat → Nat}
    (strict : StrictKeys length key) : OrderedKeys length key := by
  intro left right le rightLength
  by_cases equal : left = right
  · simp [equal]
  · exact (strict left right (by omega) rightLength).le

theorem get_inserted {α : Type} [Inhabited α] (original : List α) (position : Nat) (entry : α)
    (bound : position ≤ original.length) (index : Nat) (inside : index < original.length + 1) :
    (original.insertIdx position entry)[index]! =
      if index < position then original[index]! else
      if index = position then entry else original[index - 1]! := by
  have insertedLength : index < (original.insertIdx position entry).length := by
    simpa only [List.length_insertIdx_of_le_length bound] using inside
  rw [←List.Inhabited_getElem_eq_getElem! _ index insertedLength]
  by_cases earlier : index < position
  · rw [ite_eq_left earlier, List.getElem_insertIdx_of_lt earlier insertedLength,
      List.Inhabited_getElem_eq_getElem! _ _ (by omega)]
  · rw [ite_eq_right earlier]
    by_cases equal : index = position
    · subst index
      rw [ite_eq_left rfl, List.getElem_insertIdx_self]
    · rw [ite_eq_right equal, List.getElem_insertIdx_of_gt (by omega) insertedLength,
        List.Inhabited_getElem_eq_getElem! _ _ (by omega)]


theorem insert_strict_keys (original : List cache_graph.PageIndex) (position : Usize)
    (entry : cache_graph.PageIndex)
    (strict : StrictKeys original.length (fun index => (original[index]!).page.val))
    (bound : LowerBound original.length entry.page.val (fun index => (original[index]!).page.val) position)
    (missing : ∀ index < original.length, (original[index]!).page.val ≠ entry.page.val) :
    StrictKeys (original.insertIdx position.val entry).length
      (fun index => ((original.insertIdx position.val entry)[index]!).page.val) := by
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


def PageAddition (before after : cache_graph.WorkSets) (page storage : Usize) : Prop :=
  storage.val = before.pages.val.length ∧
  ∃ position : Usize,
    LowerBound before.directory.val.length page.val (fun index => (before.directory.val[index]!).page.val) position ∧
    after.directory.val = before.directory.val.insertIdx position.val ⟨page, storage⟩ ∧
    ∃ slots : alloc.vec.Vec (Option (alloc.vec.Vec U64)),
      slots.val = List.replicate 64 none ∧ after.pages.val = before.pages.val ++ [slots]

@[step]
theorem rust_add_page (sets : cache_graph.WorkSets) (page position : Usize)
    (bound : LowerBound sets.directory.val.length page.val
      (fun index => (sets.directory.val[index]!).page.val) position)
    (directorySpace : sets.directory.val.length < Usize.max)
    (pagesSpace : sets.pages.val.length < Usize.max) :
    cache_graph.add_page sets page position ⦃ storage result =>
      PageAddition sets result page storage ⦄ := by
  unfold cache_graph.add_page
  step with rust_empty_page as ⟨slots, slotsEq⟩
  step with alloc.vec.Vec.push_spec sets.pages slots pagesSpace as ⟨pages, pagesEq⟩
  step with rust_insert_page sets.directory position ⟨page, alloc.vec.Vec.len sets.pages⟩
    bound.1 directorySpace as ⟨directory, directoryEq⟩
  exact ⟨by simp, position, bound, directoryEq, slots, slotsEq, pagesEq⟩


@[step]
theorem rust_vec_index {α : Type} (vector : alloc.vec.Vec α) (index : Usize)
    (bound : index.val < vector.val.length) :
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice α) vector index
      ⦃ value => value = vector.val[index.val] ⦄ := by
  exact Slice.index_usize_spec vector.slice index bound

def PageEnsured (before after : cache_graph.WorkSets) (page storage : Usize) : Prop :=
  (after = before ∧ ∃ entry ∈ before.directory.val,
    entry.page = page ∧ entry.storage = storage) ∨
  (PageAddition before after page storage ∧ ∀ entry ∈ before.directory.val, entry.page ≠ page)

theorem lower_bound_missing_page (directory : List cache_graph.PageIndex)
    (page position : Usize)
    (ordered : OrderedKeys directory.length (fun index => (directory[index]!).page.val))
    (bound : LowerBound directory.length page.val (fun index => (directory[index]!).page.val) position)
    (missing : position.val = directory.length ∨ (directory[position.val]!).page ≠ page) :
    ∀ entry ∈ directory, entry.page ≠ page := by
  have missingVal : position.val = directory.length ∨
      (directory[position.val]!).page.val ≠ page.val := by
    rcases missing with atEnd | different
    · exact Or.inl atEnd
    · right
      intro same
      exact different (UScalar.eq_of_val_eq same)
  have noMatch := lower_bound_missing_key ordered bound missingVal
  intro entry member same
  obtain ⟨index, indexLength, atIndex⟩ := List.getElem_of_mem member
  apply noMatch index indexLength
  rw [←List.Inhabited_getElem_eq_getElem! directory index indexLength, atIndex, same]

@[step]
theorem rust_ensure_page (sets : cache_graph.WorkSets) (page : Usize)
    (ordered : OrderedKeys sets.directory.val.length (fun index => (sets.directory.val[index]!).page.val))
    (directorySpace : sets.directory.val.length < Usize.max)
    (pagesSpace : sets.pages.val.length < Usize.max) :
    cache_graph.ensure_page sets page ⦃ storage result => PageEnsured sets result page storage ⦄ := by
  unfold cache_graph.ensure_page
  step with rust_page_position (alloc.vec.Vec.deref sets.directory) page
    (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using ordered) as ⟨position, bound⟩
  simp only [alloc.vec.Vec.deref, Slice.from_val] at bound
  by_cases inside : position.val < sets.directory.val.length
  · have test : position < alloc.vec.Vec.len sets.directory := by scalar_tac
    simp only [test, ite_true]
    step with alloc.vec.Vec.index_usize_spec sets.directory position inside as ⟨entry, entryEq⟩
    have entryAt : entry = sets.directory.val[position.val]! := by
      rw [entryEq, List.Inhabited_getElem_eq_getElem! _ _ inside]
    split
    · rename_i equal
      simp only [WP.spec_ok]
      left
      exact ⟨rfl, entry, by rw [entryEq]; exact List.getElem_mem inside, equal, rfl⟩
    · rename_i different
      step with rust_add_page sets page position bound directorySpace pagesSpace as ⟨storage, result, addition⟩
      right
      exact ⟨addition, lower_bound_missing_page _ page position ordered bound
        (Or.inr (by simpa only [entryAt] using different))⟩
  · have test : ¬position < alloc.vec.Vec.len sets.directory := by scalar_tac
    simp only [test, ite_false]
    step with rust_add_page sets page position bound directorySpace pagesSpace as ⟨storage, result, addition⟩
    right
    exact ⟨addition, lower_bound_missing_page _ page position ordered bound
      (Or.inl (by have := bound.1; omega))⟩

end MemoryWrite.MayWrite
