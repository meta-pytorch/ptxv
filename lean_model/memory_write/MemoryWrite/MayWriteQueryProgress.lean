/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

@[step] theorem rust_find_pc_loop_total (rows : Slice cache_graph.PcWrites) (pc left right : Usize)
    (ordered : left.val ≤ right.val) (bounded : right.val ≤ rows.val.length) :
    cache_graph.find_pc_loop rows pc left right ⦃ result => result.val ≤ rows.val.length ⦄ := by
  unfold cache_graph.find_pc_loop
  apply loop.spec_decr_nat
    (fun current : Usize × Usize => current.2.val - current.1.val)
    (fun current : Usize × Usize => current.1.val ≤ current.2.val ∧ current.2.val ≤ rows.val.length)
  · intro current invariant
    rcases current with ⟨left, right⟩
    change left.val ≤ right.val ∧ right.val ≤ rows.val.length at invariant
    obtain ⟨ordered, bounded⟩ := invariant
    unfold cache_graph.find_pc_loop.body
    by_cases inside : left.val < right.val
    · have test : left < right := by scalar_tac
      simp only [test, ite_true]
      step as ⟨difference, difference_exact⟩ ; try scalar_tac
      step as ⟨half, half_exact⟩ ; try scalar_tac
      have half_value : half.val = (right.val - left.val) / 2 := by scalar_tac
      have half_lt : half.val < right.val - left.val := by
        rw [half_value]
        exact Nat.div_lt_self (by omega) (by omega)
      step as ⟨middle, middle_exact⟩ ; try scalar_tac
      have middle_value : middle.val = left.val + half.val := by scalar_tac
      have middle_lt : middle.val < right.val := by omega
      have index_bound : middle.val < rows.val.length := by omega
      step with Slice.index_usize_spec rows middle index_bound as ⟨row, row_exact⟩
      split
      · step as ⟨next, next_exact⟩
        all_goals try scalar_tac

      · simp only [WP.spec_ok]
        all_goals omega
    · have test : ¬left < right := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      omega
  · exact ⟨ordered, bounded⟩

@[step] theorem rust_find_pc_total (rows : Slice cache_graph.PcWrites) (pc : Usize) :
    cache_graph.find_pc rows pc ⦃ result =>
      ∀ index, result = some index → index.val < rows.val.length ⦄ := by
  unfold cache_graph.find_pc
  step with rust_find_pc_loop_total rows pc 0#usize (Slice.len rows)
    (by scalar_tac) (by simp [Slice.len]) as ⟨left, left_bound⟩
  split
  · rename_i inside
    have index_bound : left.val < rows.val.length := by scalar_tac
    step with Slice.index_usize_spec rows left index_bound as ⟨row, row_exact⟩
    split <;> simp_all [WP.spec_ok]
  · simp [WP.spec_ok]


@[step] theorem rust_may_at_total (sets : cache_graph.MayWriteSets) (pc : Usize)
    (register : Option Usize) :
    cache_graph.may_at sets pc register ⦃ _ => True ⦄ := by
  unfold cache_graph.may_at
  step with rust_find_pc_total (alloc.vec.Vec.deref sets.rows) pc as ⟨found, found_bound⟩
  cases found with
  | none => simp [WP.spec_ok]
  | some position =>
    have index_bound : position.val < sets.rows.val.length := by
      simpa [alloc.vec.Vec.deref, Slice.from_val] using found_bound position rfl
    step with alloc.vec.Vec.index_usize_spec sets.rows position index_bound as ⟨row, row_exact⟩
    cases register with
    | none => simp [WP.spec_ok]
    | some register =>
      step with Usize.div_spec (x := register) (y := 64#usize) (by scalar_tac)
        as ⟨word_index, word_index_exact⟩
      split
      · simp [WP.spec_ok]
      · rename_i inside
        have word_bound : word_index.val < row.words.val.length := by scalar_tac
        step with alloc.vec.Vec.index_usize_spec row.words word_index word_bound as ⟨word, word_exact⟩
        step with Usize.rem_spec (x := register) (y := 64#usize) (by scalar_tac)
          as ⟨shift, shift_exact⟩
        step with U64.ShiftRight_spec word shift (by scalar_tac) as ⟨shifted, shifted_exact⟩
        simp [lift, bind_ok, WP.spec_ok]

end MemoryWrite.MayWrite
