/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWritePaths

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

deriving instance Inhabited for cache_graph.PageIndex
deriving instance Inhabited for cache_graph.PcWrites

def OrderedKeys (length : Nat) (key : Nat → Nat) : Prop :=
  ∀ left right, left ≤ right → right < length → key left ≤ key right

def SearchInvariant (length query : Nat) (key : Nat → Nat)
    (state : Usize × Usize) : Prop :=
  state.1.val ≤ state.2.val ∧ state.2.val ≤ length ∧
  (∀ position < state.1.val, key position < query) ∧
  ∀ position, state.2.val ≤ position → position < length → query ≤ key position

def LowerBound (length query : Nat) (key : Nat → Nat) (position : Usize) : Prop :=
  position.val ≤ length ∧ (∀ index < position.val, key index < query) ∧
    ∀ index, position.val ≤ index → index < length → query ≤ key index

@[step]
theorem rust_page_position (directory : Slice cache_graph.PageIndex) (page : Usize)
    (ordered : OrderedKeys directory.val.length (fun index => (directory.val[index]!).page.val)) :
    cache_graph.page_position directory page ⦃ result =>
      LowerBound directory.val.length page.val (fun index => (directory.val[index]!).page.val) result ⦄ := by
  unfold cache_graph.page_position cache_graph.page_position_loop
  apply loop.spec_decr_nat
    (fun state : Usize × Usize => state.2.val - state.1.val)
    (SearchInvariant directory.val.length page.val (fun index => (directory.val[index]!).page.val))
  · rintro ⟨left, right⟩ ⟨leftRight, rightLength, lower, upper⟩
    dsimp only at leftRight rightLength lower upper
    unfold cache_graph.page_position_loop.body
    by_cases hlt : left.val < right.val
    · have test : left < right := by scalar_tac
      simp only [test, ite_true]
      step as ⟨distance, distanceEq⟩
      step as ⟨half, halfEq⟩
      step as ⟨middle, middleEq⟩
      have middleBounds : left.val ≤ middle.val ∧ middle.val < right.val := by scalar_tac
      have middleLength : middle.val < directory.val.length := by omega
      step with Slice.index_usize_spec directory middle middleLength as ⟨entry, entryEq⟩
      have entryKey : entry.page.val = (directory.val[middle.val]!).page.val := by
        rw [entryEq, List.Inhabited_getElem_eq_getElem! directory.val middle.val middleLength]
      split
      · rename_i less
        have lessKey : (directory.val[middle.val]!).page.val < page.val := by scalar_tac
        step with Usize.add_spec (x := middle) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
        refine ⟨?_, by scalar_tac⟩
        dsimp only [SearchInvariant]
        refine ⟨by scalar_tac, rightLength, ?_, upper⟩
        intro position hposition
        by_cases earlier : position < left.val
        · exact lower position earlier
        · exact lt_of_le_of_lt (ordered position middle.val (by omega) middleLength) lessKey
      · rename_i notLess
        have notLessKey : page.val ≤ (directory.val[middle.val]!).page.val := by scalar_tac
        simp only [WP.spec_ok]
        refine ⟨?_, by scalar_tac⟩
        dsimp only [SearchInvariant]
        refine ⟨middleBounds.1, by omega, lower, ?_⟩
        intro position hposition positionLength
        exact notLessKey.trans (ordered middle.val position hposition positionLength)
    · have test : ¬left < right := by scalar_tac
      have equal : left.val = right.val := by omega
      simp only [test, ite_false, WP.spec_ok]
      exact ⟨by omega, lower, by simpa only [equal] using upper⟩
  · exact ⟨by scalar_tac, by scalar_tac, by intro position hposition; scalar_tac,
      by intro position hposition hlength; scalar_tac⟩

@[step]
theorem rust_find_pc_lower_bound (rows : Slice cache_graph.PcWrites) (pc : Usize)
    (ordered : OrderedKeys rows.val.length (fun index => (rows.val[index]!).pc.val)) :
    cache_graph.find_pc_loop rows pc 0#usize (Slice.len rows) ⦃ result =>
      LowerBound rows.val.length pc.val (fun index => (rows.val[index]!).pc.val) result ⦄ := by
  unfold cache_graph.find_pc_loop
  apply loop.spec_decr_nat
    (fun state : Usize × Usize => state.2.val - state.1.val)
    (SearchInvariant rows.val.length pc.val (fun index => (rows.val[index]!).pc.val))
  · rintro ⟨left, right⟩ ⟨leftRight, rightLength, lower, upper⟩
    dsimp only at leftRight rightLength lower upper
    unfold cache_graph.find_pc_loop.body
    by_cases hlt : left.val < right.val
    · have test : left < right := by scalar_tac
      simp only [test, ite_true]
      step as ⟨distance, distanceEq⟩
      step as ⟨half, halfEq⟩
      step as ⟨middle, middleEq⟩
      have middleBounds : left.val ≤ middle.val ∧ middle.val < right.val := by scalar_tac
      have middleLength : middle.val < rows.val.length := by omega
      step with Slice.index_usize_spec rows middle middleLength as ⟨entry, entryEq⟩
      have entryKey : entry.pc.val = (rows.val[middle.val]!).pc.val := by
        rw [entryEq, List.Inhabited_getElem_eq_getElem! rows.val middle.val middleLength]
      split
      · rename_i less
        have lessKey : (rows.val[middle.val]!).pc.val < pc.val := by scalar_tac
        step with Usize.add_spec (x := middle) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
        refine ⟨?_, by scalar_tac⟩
        dsimp only [SearchInvariant]
        refine ⟨by scalar_tac, rightLength, ?_, upper⟩
        intro position hposition
        by_cases earlier : position < left.val
        · exact lower position earlier
        · exact lt_of_le_of_lt (ordered position middle.val (by omega) middleLength) lessKey
      · rename_i notLess
        have notLessKey : pc.val ≤ (rows.val[middle.val]!).pc.val := by scalar_tac
        simp only [WP.spec_ok]
        refine ⟨?_, by scalar_tac⟩
        dsimp only [SearchInvariant]
        refine ⟨middleBounds.1, by omega, lower, ?_⟩
        intro position hposition positionLength
        exact notLessKey.trans (ordered middle.val position hposition positionLength)
    · have test : ¬left < right := by scalar_tac
      have equal : left.val = right.val := by omega
      simp only [test, ite_false, WP.spec_ok]
      exact ⟨by omega, lower, by simpa only [equal] using upper⟩
  · exact ⟨by scalar_tac, by scalar_tac, by intro position hposition; scalar_tac,
      by intro position hposition hlength; scalar_tac⟩


theorem lower_bound_missing_key {length query : Nat} {key : Nat → Nat} {position : Usize}
    (ordered : OrderedKeys length key) (bound : LowerBound length query key position)
    (missing : position.val = length ∨ key position.val ≠ query) :
    ∀ index < length, key index ≠ query := by
  intro index indexLength same
  by_cases earlier : index < position.val
  · have := bound.2.1 index earlier
    omega
  · have positionLength : position.val < length := by omega
    have comparison := ordered position.val index (by omega) indexLength
    have upper := bound.2.2 position.val (by omega) positionLength
    rcases missing with equal | different
    · omega
    · apply different
      omega

@[step]
theorem rust_find_pc (rows : Slice cache_graph.PcWrites) (pc : Usize)
    (ordered : OrderedKeys rows.val.length (fun index => (rows.val[index]!).pc.val)) :
    cache_graph.find_pc rows pc ⦃ result =>
      match result with
      | none => ∀ row ∈ rows.val, row.pc ≠ pc
      | some index => index.val < rows.val.length ∧ (rows.val[index.val]!).pc = pc ⦄ := by
  unfold cache_graph.find_pc
  step with rust_find_pc_lower_bound rows pc ordered as ⟨left, lowerBound⟩
  by_cases inside : left.val < rows.val.length
  · have test : left < Slice.len rows := by scalar_tac
    simp only [test, ite_true]
    step with Slice.index_usize_spec rows left inside as ⟨entry, entryEq⟩
    have entryAt : entry = rows.val[left.val]! := by
      rw [entryEq, List.Inhabited_getElem_eq_getElem! rows.val left.val inside]
    split
    · rename_i matched
      simp only [WP.spec_ok]
      exact ⟨inside, by simpa only [entryAt] using matched⟩
    · rename_i mismatch
      simp only [WP.spec_ok]
      intro row member same
      obtain ⟨index, indexLength, atIndex⟩ := List.getElem_of_mem member
      have missing : (rows.val[left.val]!).pc.val ≠ pc.val := by
        intro equal
        apply mismatch
        apply UScalar.eq_of_val_eq
        simpa only [entryAt] using equal
      have noneEqual := lower_bound_missing_key ordered lowerBound (Or.inr missing)
      apply noneEqual index indexLength
      rw [←List.Inhabited_getElem_eq_getElem! rows.val index indexLength, atIndex, same]
  · have test : ¬left < Slice.len rows := by scalar_tac
    simp only [test, ite_false, WP.spec_ok]
    intro row member same
    obtain ⟨index, indexLength, atIndex⟩ := List.getElem_of_mem member
    have endPosition : left.val = rows.val.length := by have := lowerBound.1; omega
    have noneEqual := lower_bound_missing_key ordered lowerBound (Or.inl endPosition)
    apply noneEqual index indexLength
    rw [←List.Inhabited_getElem_eq_getElem! rows.val index indexLength, atIndex, same]

end MemoryWrite.MayWrite
