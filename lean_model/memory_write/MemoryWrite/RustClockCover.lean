/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustSparseClock

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

@[simp] theorem baseModel_drop (base : List (U32 × U32)) (index : Nat) :
    baseModel (base.drop index) = (baseModel base).drop index := by
  simp [baseModel]

theorem located_present (base : List Entry) (actor index : Nat) (found : Bool)
    (h : Located base actor index found) :
    base.any (fun entry => decide (entry.1 = actor)) = found := by
  cases found with
  | false =>
    apply List.any_eq_false.mpr
    intro entry member
    obtain ⟨position, bound, rfl⟩ := List.mem_iff_getElem.mp member
    by_cases before : position < index
    · have less := h.2.1 position bound before
      simp [show base[position].1 ≠ actor by omega]
    · have less := h.2.2 position bound (by omega)
      simp [show base[position].1 ≠ actor by omega]
  | true =>
    obtain ⟨bound, key, _⟩ := h.2.2
    apply List.any_eq_true.mpr
    exact ⟨base[index], List.getElem_mem bound, by simp [key]⟩

@[step] theorem rust_entry_covered (base : Slice (U32 × U32)) (entry : U32 × U32)
    (hs : Sorted (baseModel base.val)) :
    clock.entry_covered base entry ⦃ result =>
      result = entryCovered (baseModel base.val) (entryModel entry) ⦄ := by
  rcases entry with ⟨actor, requested⟩
  unfold clock.entry_covered
  step with rust_locate base actor hs as ⟨index, found, hlocated⟩
  have hp := located_present (baseModel base.val) actor.val index.val found hlocated
  have hv := located_value (baseModel base.val) actor.val index.val found hlocated
  cases found with
  | false =>
    simp only [Bool.false_eq_true, ite_false, WP.spec_ok]
    change false = entryCovered (baseModel base.val) (actor.val, requested.val)
    unfold entryCovered
    rw [hp]
    rfl
  | true =>
    obtain ⟨hi, _, _⟩ := hlocated.2.2
    have hib : index.val < base.val.length := by simpa using hi
    simp only [ite_true]
    step with Slice.index_usize_spec base index hib as ⟨key, stamp, hread⟩
    have hstamp : stamp = base.val[index.val].2 := congrArg Prod.snd hread
    have hvalue : entriesValue (baseModel base.val) actor.val = stamp.val := by
      rw [hv, hstamp]
      simp only [ite_true, List.getElem?_eq_getElem hi, Option.getD_some]
      exact congrArg Prod.snd (baseModel_getElem base.val index.val hib)
    change decide (stamp >= requested) =
      entryCovered (baseModel base.val) (actor.val, requested.val)
    unfold entryCovered
    rw [hp, hvalue]
    simp only [Bool.true_and]
    scalar_tac

@[step] theorem rust_base_covers_sparse_loop (left right : Slice (U32 × U32))
    (covered : Bool) (index : Usize) (hl : Sorted (baseModel left.val))
    (hi : index.val ≤ right.val.length) :
    clock.base_covers_sparse_loop left right covered index ⦃ result =>
      result = (covered && baseCovers (baseModel left.val)
        (baseModel (right.val.drop index.val))) ⦄ := by
  let expected := covered && baseCovers (baseModel left.val)
    (baseModel (right.val.drop index.val))
  change clock.base_covers_sparse_loop left right covered index ⦃ result => result = expected ⦄
  unfold clock.base_covers_sparse_loop
  apply loop.spec_decr_nat
    (fun current : Bool × Usize => right.val.length - current.2.val)
    (fun current : Bool × Usize => current.2.val ≤ right.val.length ∧
      (current.1 && baseCovers (baseModel left.val)
        (baseModel (right.val.drop current.2.val))) = expected)
  · intro current hc
    rcases current with ⟨active, cursor⟩
    obtain ⟨hbound, he⟩ := hc
    change cursor.val ≤ right.val.length at hbound
    change (active && baseCovers (baseModel left.val) (baseModel (right.val.drop cursor.val))) = expected at he
    unfold clock.base_covers_sparse_loop.body
    cases active with
    | false => simpa using he
    | true =>
      simp only [Bool.true_and] at he
      simp only [ite_true]
      by_cases hlt : cursor.val < right.val.length
      · have htest : cursor < Slice.len right := by scalar_tac
        simp only [htest, ite_true]
        step with Slice.index_usize_spec right cursor hlt as ⟨entry, hentry⟩
        step with rust_entry_covered left entry hl as ⟨known, hknown⟩
        step as ⟨next, hnext⟩; try scalar_tac
        have hn : next.val = cursor.val + 1 := by scalar_tac
        have hdrop : right.val.drop cursor.val =
            entry :: right.val.drop (cursor.val + 1) := by
          rw [List.drop_eq_getElem_cons hlt]
          rw [hentry]
        refine ⟨by omega, ?_, by omega⟩
        simpa [hn, hknown, hdrop, baseModel, baseCovers] using he
      · have htest : ¬cursor < Slice.len right := by scalar_tac
        simp only [htest, ite_false, WP.spec_ok]
        have heq : cursor.val = right.val.length := by omega
        simpa [heq, baseModel, baseCovers] using he
  · exact ⟨hi, rfl⟩

@[step] theorem rust_base_covers_sparse (left right : Slice (U32 × U32))
    (hl : Sorted (baseModel left.val)) :
    clock.base_covers_sparse left right ⦃ result =>
      result = baseCovers (baseModel left.val) (baseModel right.val) ⦄ := by
  unfold clock.base_covers_sparse
  step with rust_base_covers_sparse_loop left right true 0#usize hl (by simp) as ⟨result, hr⟩
  simpa using hr

end MemoryWrite.SparseClock
