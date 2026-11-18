/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteRowOrder
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.MayWrite

@[step]
theorem rust_compact_page_total (sets : cache_graph.WorkSets) (invariant : DirectoryInvariant sets)
    (count : Nat) (countBound : count ≤ Usize.max)
    (knownBound : ∀ pc, (workSolution sets).present pc → pc < count)
    (entry : cache_graph.PageIndex) (member : entry ∈ sets.directory.val)
    (rows : alloc.vec.Vec cache_graph.PcWrites) (ordered : RowsBefore rows.val (entry.page.val * 64)) :
    cache_graph.compact_loop0_loop0 sets.pages rows entry 0#usize
      ⦃ result => RowsBefore result.val (entry.page.val * 64 + 64) ⦄ := by
  unfold cache_graph.compact_loop0_loop0
  apply loop.spec_decr_nat (fun state : alloc.vec.Vec cache_graph.PcWrites × Usize => 64 - state.2.val)
    (fun state => state.2.val ≤ 64 ∧ RowsBefore state.1.val (entry.page.val * 64 + state.2.val))
  · rintro ⟨currentRows, offset⟩ ⟨offsetBound, currentOrder⟩
    have offsetLe : offset.val ≤ 64 := offsetBound
    have rowOrder : RowsBefore currentRows.val (entry.page.val * 64 + offset.val) := currentOrder
    unfold cache_graph.compact_loop0_loop0.body cache_graph.PAGE_SIZE
    by_cases inside : offset.val < 64
    · have test : offset < 64#usize := by scalar_tac
      simp only [test, ite_true]
      have storageBound := invariant.safe entry member
      step with alloc.vec.Vec.index_usize_spec sets.pages entry.storage storageBound as ⟨slots, slotsEq⟩
      have slotsAt : sets.pages.val[entry.storage.val]? = some slots := by
        rw [List.getElem?_eq_getElem storageBound, ←slotsEq]
      have slotsMember : slots ∈ sets.pages.val := by rw [slotsEq]; exact List.getElem_mem storageBound
      have slotsSize := invariant.sizes slots slotsMember
      have offsetInside : offset.val < slots.val.length := by omega
      step with alloc.vec.Vec.index_usize_spec slots offset offsetInside as ⟨previous, previousEq⟩
      apply WP.spec_bind (Pₘ := fun nextRows => RowsBefore nextRows.val (entry.page.val * 64 + offset.val + 1))
      · cases previous with
        | none =>
            simp only [WP.spec_ok]
            exact rowOrder.mono (by omega)
        | some words =>
            have wordLookup : slots.val[offset.val]? = some (some words) := by
              rw [List.getElem?_eq_getElem offsetInside, ←previousEq]
            have atWords := words_at_page_slot sets entry member slots slotsAt offset inside words wordLookup
            have addressBound := knownBound (entry.page.val * 64 + offset.val) ⟨words, atWords⟩
            step with Usize.mul_spec (x := entry.page) (y := 64#usize) (by scalar_tac)
              as ⟨base, baseValue⟩
            step with Usize.add_spec (x := base) (y := offset) (by scalar_tac)
              as ⟨address, addressValue⟩
            step with rust_copy_words (alloc.vec.Vec.deref words) as ⟨copy, copied⟩
            have rowSpace : currentRows.val.length < Usize.max := by
              have := rowOrder.length_le
              omega
            step with alloc.vec.Vec.push_spec currentRows (⟨address, copy⟩ : cache_graph.PcWrites) rowSpace
              as ⟨nextRows, nextEq⟩
            rw [nextEq]
            exact rowOrder.append ⟨address, copy⟩ (by scalar_tac)
      · intro nextRows nextOrder
        step with Usize.add_spec (x := offset) (y := 1#usize) (by scalar_tac)
          as ⟨nextOffset, nextValue⟩
        refine ⟨?_, ?_, ?_⟩
        · scalar_tac
        · have nextLimit : entry.page.val * 64 + nextOffset.val = entry.page.val * 64 + offset.val + 1 := by scalar_tac
          simpa only [nextLimit] using nextOrder
        · scalar_tac
    · have test : ¬offset < 64#usize := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      have finalOffset : offset.val = 64 := by omega
      simpa only [finalOffset] using rowOrder
  · exact ⟨by scalar_tac, by simpa using ordered⟩


theorem rust_compact_returns (sets : cache_graph.WorkSets) (invariant : DirectoryInvariant sets)
    (count : Nat) (countBound : count ≤ Usize.max)
    (knownBound : ∀ pc, (workSolution sets).present pc → pc < count) :
    cache_graph.compact sets ⦃ _ => True ⦄ := by
  unfold cache_graph.compact
  apply WP.spec_bind (Pₘ := fun _ => True)
  · unfold cache_graph.compact_loop0
    apply loop.spec_decr_nat
      (fun state : alloc.vec.Vec cache_graph.PcWrites × Usize => sets.directory.val.length - state.2.val)
      (fun state => state.2.val ≤ sets.directory.val.length ∧
        ∀ position, state.2.val ≤ position → position < sets.directory.val.length →
          RowsBefore state.1.val (sets.directory.val[position]!.page.val * 64))
    · rintro ⟨currentRows, page⟩ ⟨pageBound, futureBounds⟩
      have pageLe : page.val ≤ sets.directory.val.length := pageBound
      have future : ∀ position, page.val ≤ position → position < sets.directory.val.length →
          RowsBefore currentRows.val (sets.directory.val[position]!.page.val * 64) := futureBounds
      unfold cache_graph.compact_loop0.body
      by_cases inside : page.val < sets.directory.val.length
      · have test : page < alloc.vec.Vec.len sets.directory := by scalar_tac
        simp only [test, ite_true]
        step with alloc.vec.Vec.index_usize_spec sets.directory page inside as ⟨entry, entryEq⟩
        have entryAt : entry = sets.directory.val[page.val]! := by
          rw [entryEq, List.Inhabited_getElem_eq_getElem! _ _ inside]
        have member : entry ∈ sets.directory.val := by rw [entryEq]; exact List.getElem_mem inside
        have ordered : RowsBefore currentRows.val (entry.page.val * 64) := by
          simpa only [entryAt] using future page.val (Nat.le_refl _) inside
        step with rust_compact_page_total sets invariant count countBound knownBound entry member currentRows ordered
          as ⟨nextRows, nextOrder⟩
        step with Usize.add_spec (x := page) (y := 1#usize) (by scalar_tac) as ⟨nextPage, nextValue⟩
        refine ⟨?_, ?_, ?_⟩
        · scalar_tac
        · intro position atOrAfter positionBound
          have nextPosition : nextPage.val ≤ position := atOrAfter
          have increasing : page.val < position := by scalar_tac
          have greaterPage := invariant.strict page.val position increasing positionBound
          change sets.directory.val[page.val]!.page.val < sets.directory.val[position]!.page.val at greaterPage
          have increase : entry.page.val * 64 + 64 ≤ sets.directory.val[position]!.page.val * 64 := by
            rw [entryAt]
            omega
          exact nextOrder.mono increase
        · scalar_tac
      · have test : ¬page < alloc.vec.Vec.len sets.directory := by scalar_tac
        simp only [test, ite_false, WP.spec_ok]
    · refine ⟨by scalar_tac, ?_⟩
      intro position after positionBound
      constructor
      · intro left right increasing rightBound
        simp only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.length_nil] at rightBound
        omega
      · intro row member
        simp only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.not_mem_nil] at member
  · intro rows _
    simp only [WP.spec_ok]

@[step]
theorem rust_compact_total (sets : cache_graph.WorkSets) (invariant : DirectoryInvariant sets)
    (count : Nat) (countBound : count ≤ Usize.max)
    (knownBound : ∀ pc, (workSolution sets).present pc → pc < count) :
    cache_graph.compact sets ⦃ result => RowsCoherent sets result.rows.val ⦄ := by
  obtain ⟨result, executed, _⟩ := MemoryWrite.spec_returns _ _
    (rust_compact_returns sets invariant count countBound knownBound)
  simp only [executed, WP.spec_ok]
  exact successful_compact sets invariant result executed

end MemoryWrite.MayWrite
