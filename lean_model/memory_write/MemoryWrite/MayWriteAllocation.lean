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

@[step]
theorem rust_zero_words (count : Usize) :
    cache_graph.zero_words count ⦃ result => result.val = List.replicate count.val 0#u64 ⦄ := by
  unfold cache_graph.zero_words cache_graph.zero_words_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U64 × Usize => count.val - state.2.val)
    (fun state : alloc.vec.Vec U64 × Usize => state.2.val ≤ count.val ∧
      state.1.val = List.replicate state.2.val 0#u64)
  · rintro ⟨words, index⟩ ⟨bound, values⟩
    dsimp only at bound values
    unfold cache_graph.zero_words_loop.body
    by_cases hlt : index.val < count.val
    · have test : index < count := by scalar_tac
      simp only [test, ite_true]
      step with alloc.vec.Vec.push_spec words 0#u64 (by scalar_tac) as ⟨appended, appendedEq⟩
      step with Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
      exact ⟨by scalar_tac, by simp only [appendedEq, values, nextEq, List.replicate_succ'],
        by scalar_tac⟩
    · have test : ¬index < count := by scalar_tac
      have equal : index.val = count.val := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [equal] using values
  · exact ⟨by scalar_tac, rfl⟩

@[step]
theorem rust_copy_words (source : Slice U64) :
    cache_graph.copy_words source ⦃ result => result.val = source.val ⦄ := by
  unfold cache_graph.copy_words cache_graph.copy_words_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U64 × Usize => source.val.length - state.2.val)
    (fun state : alloc.vec.Vec U64 × Usize => state.2.val ≤ source.val.length ∧
      state.1.val = source.val.take state.2.val)
  · rintro ⟨words, index⟩ ⟨bound, values⟩
    dsimp only at bound values
    unfold cache_graph.copy_words_loop.body
    by_cases hlt : index.val < source.val.length
    · have test : index < Slice.len source := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec source index hlt as ⟨value, valueEq⟩
      step with alloc.vec.Vec.push_spec words value (by scalar_tac) as ⟨appended, appendedEq⟩
      step with Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      simp only [appendedEq, values, nextEq, valueEq, List.take_succ_eq_append_getElem hlt]
    · have test : ¬index < Slice.len source := by scalar_tac
      have equal : index.val = source.val.length := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [equal, List.take_length] using values
  · exact ⟨by scalar_tac, rfl⟩

@[step]
theorem rust_push_work (work : alloc.vec.Vec Usize) (pending pc : Usize)
    (bound : pending.val ≤ work.val.length) (space : pending.val < Usize.max) :
    cache_graph.push_work work pending pc ⦃ next result =>
      next.val = pending.val + 1 ∧ result.val.take next.val = work.val.take pending.val ++ [pc] ∧
      next.val ≤ result.val.length ⦄ := by
  unfold cache_graph.push_work
  by_cases full : pending.val = work.val.length
  · have test : pending = alloc.vec.Vec.len work := by scalar_tac
    simp only [test, ite_true]
    step with alloc.vec.Vec.push_spec work pc (by omega) as ⟨result, resultEq⟩
    step with Usize.add_spec (x := alloc.vec.Vec.len work) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
    refine ⟨nextEq, ?_, ?_⟩
    · simp [resultEq, nextEq, List.take_append]

    · simp only [resultEq, List.length_append, List.length_singleton]; scalar_tac
  · have test : pending ≠ alloc.vec.Vec.len work := by scalar_tac
    simp only [test, ite_false]
    step as ⟨previous, previousEq, backEq⟩
    rcases previous with ⟨old, back⟩
    dsimp only at backEq
    step with Usize.add_spec (x := pending) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
    refine ⟨nextEq, ?_, ?_⟩
    · rw [backEq]
      simp only [alloc.vec.Vec.set_val_eq, nextEq]
      rw [List.take_succ_eq_append_getElem (by simp; omega)]
      simp only [List.take_set, List.getElem_set_self, List.append_cancel_right_eq]
      exact List.set_eq_of_length_le (by simp)
    · rw [backEq]
      simp only [alloc.vec.Vec.set_val_eq, List.length_set]; omega

@[step]
theorem rust_empty_page :
    cache_graph.empty_page ⦃ result => result.val = List.replicate 64 (none : Option (alloc.vec.Vec U64)) ⦄ := by
  unfold cache_graph.empty_page cache_graph.empty_page_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (Option (alloc.vec.Vec U64)) × Usize => 64 - state.2.val)
    (fun state : alloc.vec.Vec (Option (alloc.vec.Vec U64)) × Usize => state.2.val ≤ 64 ∧
      state.1.val = List.replicate state.2.val (none : Option (alloc.vec.Vec U64)))
  · rintro ⟨words, index⟩ ⟨bound, values⟩
    dsimp only at bound values
    unfold cache_graph.empty_page_loop.body cache_graph.PAGE_SIZE
    by_cases hlt : index.val < 64
    · have test : index < 64#usize := by scalar_tac
      simp only [test, ite_true]
      step with alloc.vec.Vec.push_spec words (none : Option (alloc.vec.Vec U64)) (by scalar_tac) as ⟨appended, appendedEq⟩
      step with Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
      exact ⟨by scalar_tac, by simp only [appendedEq, values, nextEq, List.replicate_succ'],
        by scalar_tac⟩
    · have test : ¬index < 64#usize := by scalar_tac
      have equal : index.val = 64 := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [equal] using values
  · exact ⟨by scalar_tac, rfl⟩

end MemoryWrite.MayWrite
