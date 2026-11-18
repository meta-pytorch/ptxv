/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustSpan
import MemoryWrite.Effects

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite

def wordPatchView (patch : words.WordPatch) : WordUpdate :=
  ⟨patch.key, patch.history⟩

def wordPatchedCell (cell : Cell) (patch : words.WordPatch) : Cell :=
  { cell with word := if patch.key = cell.key then patch.history else cell.word }

@[step]
theorem rust_patch_cell_spec (cell : Cell) (patch : words.WordPatch) :
    words.patch_cell cell patch ⦃ updated => updated = wordPatchedCell cell patch ⦄ := by
  unfold words.patch_cell
  step with rust_same_key_spec cell.key patch.key as ⟨matched, hmatched⟩
  by_cases heq : patch.key = cell.key
  · have htrue : matched = true := hmatched.mpr heq.symm
    simp [htrue, wordPatchedCell, heq]
  · have hfalse : matched = false := by
      simpa using (show matched ≠ true from fun h => heq (hmatched.mp h).symm)
    simp [hfalse, wordPatchedCell, heq]

def WordMapPrefix (original current : Slice Cell) (patch : words.WordPatch)
    (cursor : Nat) : Prop :=
  current.val.length = original.val.length ∧
    ∀ position, current.val[position]? = if position < cursor then
      original.val[position]?.map (fun cell => wordPatchedCell cell patch)
    else original.val[position]?

@[step]
theorem rust_install_word_spec (original : Slice Cell) (patch : words.WordPatch) :
    words.install_word original patch
      ⦃ updated => updated.val = original.val.map (fun cell => wordPatchedCell cell patch) ⦄ := by
  unfold words.install_word words.install_word_loop
  apply loop.spec_decr_nat
    (fun current : Slice Cell × Usize => original.val.length - current.2.val)
    (fun current : Slice Cell × Usize => current.2.val ≤ original.val.length ∧
      WordMapPrefix original current.1 patch current.2.val)
  · intro current hcurrent
    rcases current with ⟨cells, cursor⟩
    dsimp only [WordMapPrefix] at hcurrent
    obtain ⟨hcursor, hlength, hprefix⟩ := hcurrent
    unfold words.install_word_loop.body
    by_cases hlt : cursor.val < cells.val.length
    · have htest : cursor < Slice.len cells := by scalar_tac
      simp only [htest, ite_true]
      step with Slice.index_usize_spec cells cursor hlt as ⟨cell, hcell⟩
      step with rust_patch_cell_spec cell patch as ⟨after, hafter⟩
      step with Slice.update_spec cells cursor after hlt as ⟨updated, hupdated⟩
      step as ⟨next, hnext⟩
      have hnextVal : next.val = cursor.val + 1 := by scalar_tac
      refine ⟨?_, ⟨?_, ?_⟩, ?_⟩
      · scalar_tac
      · rw [hupdated]
        simpa using hlength
      · intro position
        rw [hupdated]
        simp only [Slice.set, Slice.setAtNat, Slice.from_val]
        by_cases heq : position = cursor.val
        · subst position
          rw [List.getElem?_set_self hlt, ite_eq_left (by omega : cursor.val < next.val)]
          have horiginal : original.val[cursor.val]? = some cell := by
            have hread := hprefix cursor.val
            simp only [Nat.lt_irrefl, ite_false] at hread
            rw [←hread]
            simp only [List.getElem?_eq_getElem hlt, hcell]
          simp only [horiginal, Option.map_some, hafter]
        · rw [List.getElem?_set_ne (Ne.symm heq), hprefix]
          have hsame : position < next.val ↔ position < cursor.val := by omega
          simp only [hsame]
      · scalar_tac
    · have htest : ¬cursor < Slice.len cells := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have hfinish : cursor.val = original.val.length := by omega
      apply List.ext_getElem?
      intro position
      rw [hprefix, List.getElem?_map]
      by_cases hposition : position < original.val.length
      · simp [hfinish, hposition]
      · simp [hfinish, hposition]
  · exact ⟨by scalar_tac, rfl, by intro position; simp⟩

theorem wordPatchedCell_getCell (state : Storage) (patch : words.WordPatch) (key : Key) :
    wordPatchedCell (state.getCell key) patch =
      (state.setWord (wordPatchView patch)).getCell key := by
  rfl

@[step]
theorem rust_install_word_refines (state : Storage) (patch : words.WordPatch) :
    words.install_word state.toCells patch
      ⦃ next => next = (state.setWord (wordPatchView patch)).toCells ⦄ := by
  step with rust_install_word_spec state.toCells patch as ⟨next, hnext⟩
  apply Slice.ext
  rw [hnext]
  simp only [Storage.toCells, Storage.setWord_keys, Slice.from_val, List.map_map]
  congr 1

@[step]
theorem rust_install_words_refines (state : Storage) (patches : Slice words.WordPatch) :
    words.install_words state.toCells patches
      ⦃ next => next = (state.installWords (patches.val.map wordPatchView)).toCells ⦄ := by
  unfold words.install_words words.install_words_loop
  apply loop.spec_decr_nat
    (fun current : Slice Cell × Usize => patches.val.length - current.2.val)
    (fun current : Slice Cell × Usize => current.2.val ≤ patches.val.length ∧
      current.1 = (state.installWords ((patches.val.take current.2.val).map wordPatchView)).toCells)
  · intro current hcurrent
    rcases current with ⟨cells, cursor⟩
    change cursor.val ≤ patches.val.length ∧
      cells = (state.installWords ((patches.val.take cursor.val).map wordPatchView)).toCells at hcurrent
    obtain ⟨hcursor, hcells⟩ := hcurrent
    unfold words.install_words_loop.body
    by_cases hlt : cursor.val < patches.val.length
    · have htest : cursor < Slice.len patches := by scalar_tac
      simp only [htest, ite_true]
      step with Slice.index_usize_spec patches cursor hlt as ⟨patch, hpatch⟩
      rw [hcells]
      step with rust_install_word_refines
        (state.installWords ((patches.val.take cursor.val).map wordPatchView)) patch as ⟨updated, hupdated⟩
      step as ⟨next, hnext⟩
      have hnextVal : next.val = cursor.val + 1 := by scalar_tac
      refine ⟨by omega, ?_, by omega⟩
      rw [hupdated, hnextVal, List.take_succ_eq_append_getElem hlt, List.map_append]
      simp only [Storage.installWords, List.foldl_append, List.map_cons, List.map_nil,
        List.foldl_cons, List.foldl_nil, hpatch]
    · have htest : ¬cursor < Slice.len patches := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      simpa only [List.take_of_length_le (by omega : patches.val.length ≤ cursor.val)] using hcells
  · exact ⟨by scalar_tac, by simp [Storage.installWords]⟩

end MemoryWrite
