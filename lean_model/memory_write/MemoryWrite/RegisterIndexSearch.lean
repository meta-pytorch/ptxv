/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexKey
import MemoryWrite.MayWriteSearch

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

deriving instance Inhabited for register_index.RegisterPage

@[step]
theorem rust_page_position (directory : Slice register_index.RegisterPage) (page : U64)
    (ordered : OrderedKeys directory.val.length (fun index => (directory.val[index]!).key.val)) :
    register_index.page_position directory page ⦃ result =>
      LowerBound directory.val.length page.val (fun index => (directory.val[index]!).key.val) result ⦄ := by
  unfold register_index.page_position register_index.page_position_loop
  apply loop.spec_decr_nat
    (fun state : Usize × Usize => state.2.val - state.1.val)
    (SearchInvariant directory.val.length page.val (fun index => (directory.val[index]!).key.val))
  · rintro ⟨left, right⟩ ⟨leftRight, rightLength, lower, upper⟩
    dsimp only at leftRight rightLength lower upper
    unfold register_index.page_position_loop.body
    by_cases hlt : left.val < right.val
    · have test : left < right := by scalar_tac
      simp only [test, ite_true]
      step as ⟨distance, distanceEq⟩
      step as ⟨half, halfEq⟩
      step as ⟨middle, middleEq⟩
      have middleBounds : left.val ≤ middle.val ∧ middle.val < right.val := by scalar_tac
      have middleLength : middle.val < directory.val.length := by omega
      step with Slice.index_usize_spec directory middle middleLength as ⟨entry, entryEq⟩
      have entryKey : entry.key.val = (directory.val[middle.val]!).key.val := by
        rw [entryEq, List.Inhabited_getElem_eq_getElem! directory.val middle.val middleLength]
      split
      · rename_i less
        have lessKey : (directory.val[middle.val]!).key.val < page.val := by scalar_tac
        step with Usize.add_spec (x := middle) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
        refine ⟨?_, by scalar_tac⟩
        dsimp only [SearchInvariant]
        refine ⟨by scalar_tac, rightLength, ?_, upper⟩
        intro position hposition
        by_cases earlier : position < left.val
        · exact lower position earlier
        · exact lt_of_le_of_lt (ordered position middle.val (by omega) middleLength) lessKey
      · rename_i notLess
        have notLessKey : page.val ≤ (directory.val[middle.val]!).key.val := by scalar_tac
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


end MemoryWrite.RegisterIndex
