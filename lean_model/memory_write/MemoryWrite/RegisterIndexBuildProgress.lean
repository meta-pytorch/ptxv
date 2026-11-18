/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexBudget
import MemoryWrite.RegisterIndexRowProgress

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

private theorem empty_index_valid : Valid
    ({ pages := alloc.vec.Vec.new register_index.RegisterPage, count := 0#usize } : register_index.RegisterIndex) := by
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · intro position inside; simp at inside
  · intro left right less inside; simp at inside
  · rintro key id ⟨position, inside, _, _⟩; simp at inside
  · rintro left right id ⟨position, inside, _, _⟩; simp at inside

@[step]
theorem rust_build_writes_returns (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (fits : BuildFits destinations.val) :
    register_index.build_writes destinations ⦃ result =>
      Valid result.1 ∧ result.1.count.val ≤ keyCount destinations.val ∧
      result.1.pages.val.length ≤ keyCount destinations.val ⦄ := by
  have initialValid := empty_index_valid
  unfold register_index.build_writes register_index.RegisterIndex.new
  simp only [bind_ok]
  unfold register_index.build_writes_loop0
  apply loop.spec_decr_nat
    (fun state : register_index.RegisterIndex × alloc.vec.Vec (alloc.vec.Vec Usize) × Usize =>
      destinations.val.length - state.2.2.val)
    (fun state : register_index.RegisterIndex × alloc.vec.Vec (alloc.vec.Vec Usize) × Usize =>
      Valid state.1 ∧ state.2.2.val ≤ destinations.val.length ∧
      state.2.1.val.length = state.2.2.val ∧
      state.1.count.val ≤ usedKeys destinations.val state.2.2.val ∧
      state.1.pages.val.length ≤ usedKeys destinations.val state.2.2.val)
  · rintro ⟨current, writes, pc⟩ ⟨currentValid, pcBound, writesLength, countBound, pageBound⟩
    dsimp only at pcBound writesLength countBound pageBound
    unfold register_index.build_writes_loop0.body
    by_cases inside : pc.val < destinations.val.length
    · have test : pc < Slice.len destinations := by scalar_tac
      simp only [test, ite_true]
      have nextBudget := used_keys_next destinations.val pc.val inside
      have budgetBound := used_keys_bound destinations.val (pc.val + 1)
      have countSpace : current.count.val + (destinations.val[pc.val]!).val.length ≤ Usize.max := by
        dsimp only [BuildFits] at fits
        omega
      have pageSpace : current.pages.val.length + (destinations.val[pc.val]!).val.length ≤ Usize.max := by
        dsimp only [BuildFits] at fits
        omega
      step with rust_build_row_returns destinations pc current inside currentValid countSpace pageSpace
        as ⟨updated, row, updatedValid, nextCount, nextPages⟩
      step with alloc.vec.Vec.push_spec writes row (by scalar_tac) as ⟨appended, appendedEq⟩
      step with Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac) as ⟨next, nextValue⟩
      refine ⟨updatedValid, by scalar_tac, ?_, ?_, ?_, by scalar_tac⟩
      · simp only [appendedEq, List.length_append, List.length_singleton]
        scalar_tac
      · rw [nextValue, nextBudget]
        omega
      · rw [nextValue, nextBudget]
        omega
    · have test : ¬pc < Slice.len destinations := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      have budgetBound := used_keys_bound destinations.val pc.val
      exact ⟨currentValid, by omega, by omega⟩
  · refine ⟨initialValid, by scalar_tac, by simp, ?_, ?_⟩ <;>
      simp [used_keys_zero]


end MemoryWrite.RegisterIndex
