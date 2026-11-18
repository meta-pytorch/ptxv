/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexModel
import MemoryWrite.MayWriteStorage
import MemoryWrite.SuccessfulPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

@[step]
theorem rust_empty_page_slots :
    register_index.empty_page_loop (alloc.vec.Vec.with_capacity (Option Usize) register_index.PAGE_SIZE) 0#usize ⦃ result => result.val = List.replicate 64 (none : Option Usize) ⦄ := by
  unfold register_index.empty_page_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (Option Usize) × Usize => 64 - state.2.val)
    (fun state : alloc.vec.Vec (Option Usize) × Usize => state.2.val ≤ 64 ∧
      state.1.val = List.replicate state.2.val (none : Option Usize))
  · rintro ⟨words, index⟩ ⟨bound, values⟩
    dsimp only at bound values
    unfold register_index.empty_page_loop.body register_index.PAGE_SIZE
    by_cases hlt : index.val < 64
    · have test : index < 64#usize := by scalar_tac
      simp only [test, ite_true]
      step with alloc.vec.Vec.push_spec words (none : Option Usize) (by scalar_tac) as ⟨appended, appendedEq⟩
      step with Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
      exact ⟨by scalar_tac, by simp only [appendedEq, values, nextEq, List.replicate_succ'],
        by scalar_tac⟩
    · have test : ¬index < 64#usize := by scalar_tac
      have equal : index.val = 64 := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [equal] using values
  · exact ⟨by scalar_tac, rfl⟩


@[step]
theorem rust_empty_page (key : U64) :
    register_index.empty_page key ⦃ result => result.key = key ∧
      result.slots.val = List.replicate 64 (none : Option Usize) ⦄ := by
  unfold register_index.empty_page
  step with rust_empty_page_slots as ⟨slots, slotsEqual⟩
  exact slotsEqual

@[step]
theorem rust_insert_page (pages : alloc.vec.Vec register_index.RegisterPage)
    (position : Usize) (key : U64)
    (bound : position.val ≤ pages.val.length) (space : pages.val.length < Usize.max) :
    register_index.insert_page pages position key ⦃ result =>
      ∃ page, page.key = key ∧ page.slots.val = List.replicate 64 (none : Option Usize) ∧
        result.val = pages.val.insertIdx position.val page ⦄ := by
  unfold register_index.insert_page
  step with rust_empty_page key as ⟨page, pageKey, pageSlots⟩
  by_cases atEnd : position.val = pages.val.length
  · have test : position = alloc.vec.Vec.len pages := by scalar_tac
    simp only [test, ite_true]
    step with alloc.vec.Vec.push_spec pages page space as ⟨result, resultEq⟩
    exact ⟨page, pageKey, pageSlots, by
      simpa only [alloc.vec.Vec.len_val, atEnd, List.insertIdx_length_self] using resultEq⟩
  · have test : position ≠ alloc.vec.Vec.len pages := by scalar_tac
    simp only [test, ite_false]
    step with alloc.vec.Vec.insert_spec pages position page (by scalar_tac) space
      as ⟨result, resultEq⟩
    exact ⟨page, pageKey, pageSlots, resultEq⟩

theorem successful_vec_insert {α : Type} (before : alloc.vec.Vec α) (position : Usize)
    (value : α) (after : alloc.vec.Vec α)
    (success : alloc.vec.Vec.insert before position value = .ok after) :
    after.val = before.val.insertIdx position.val value := by
  unfold alloc.vec.Vec.insert at success
  split at success
  · split at success
    · have equal := Result.ok_injective success
      rw [←equal]
      simp only [alloc.vec.Vec.from_val]
    · simp only [fail_not_ok] at success
  · simp only [fail_not_ok] at success

theorem successful_insert_page (pages result : alloc.vec.Vec register_index.RegisterPage)
    (position : Usize) (key : U64) (_bound : position.val ≤ pages.val.length)
    (success : register_index.insert_page pages position key = .ok result) :
    ∃ page, page.key = key ∧ page.slots.val = List.replicate 64 (none : Option Usize) ∧
      result.val = pages.val.insertIdx position.val page := by
  unfold register_index.insert_page at success
  obtain ⟨page, pageOk, tailOk⟩ := MemoryWrite.bind_success _ _ _ success
  have pageSpec := rust_empty_page key
  simp only [pageOk, WP.spec_ok] at pageSpec
  refine ⟨page, pageSpec.1, pageSpec.2, ?_⟩
  by_cases atEnd : position.val = pages.val.length
  · have test : position = alloc.vec.Vec.len pages := by scalar_tac
    simp only [test, ite_true] at tailOk
    simpa only [atEnd, List.insertIdx_length_self] using
      MemoryWrite.successful_vec_push pages page result tailOk
  · have test : position ≠ alloc.vec.Vec.len pages := by scalar_tac
    simp only [test, ite_false] at tailOk
    exact successful_vec_insert pages position page result tailOk

end MemoryWrite.RegisterIndex
