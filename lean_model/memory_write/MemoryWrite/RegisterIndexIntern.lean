/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexInternModel
import MemoryWrite.RegisterIndexLookup

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

theorem successful_intern (before after : register_index.RegisterIndex) (key : symbolic.RegisterKey)
    (id : Usize) (valid : Valid before)
    (success : register_index.RegisterIndex.intern before key = .ok (id, after)) :
    Interned before after key id := by
  unfold register_index.RegisterIndex.intern at success
  obtain ⟨page, pageOk, ensureOk⟩ := MemoryWrite.bind_success _ _ _ success
  have pageSpec := rust_page_key key
  simp only [pageOk, WP.spec_ok] at pageSpec
  obtain ⟨prepared, preparedOk, remainderOk⟩ := MemoryWrite.bind_success _ _ _ ensureOk
  rcases prepared with ⟨position, pages⟩
  have ready := successful_ensure_page before.pages pages page position valid.1 preparedOk
  have preparedValid := ready_valid before.pages pages before.count page position ready valid
  have atPage : PageAt pages (keyPage key) position :=
    ⟨ready.2.1, (congrArg UScalar.val ready.2.2.1).trans pageSpec⟩
  obtain ⟨remainder, remainderOk, rowOk⟩ := MemoryWrite.bind_success _ _ _ remainderOk
  have remainderSpec := U16.rem_spec (x := key.index) (y := 64#u16) (by scalar_tac)
  simp only [remainderOk, WP.spec_ok] at remainderSpec
  simp only [lift, bind_ok] at rowOk
  let offset := core.convert.num.FromUsizeU16.from remainder
  have offsetValue : offset.val = keyOffset key := by dsimp only [offset, keyOffset]; scalar_tac
  have offsetBound : offset.val < (pages.val[position.val]!).slots.val.length := by
    rw [preparedValid.1.1 position.val ready.2.1, offsetValue]
    exact Nat.mod_lt _ (by decide)
  obtain ⟨row, indexed, slotOk⟩ := MemoryWrite.bind_success _ _ _ rowOk
  have rowSpec := rust_vec_index pages position ready.2.1
  simp only [indexed, WP.spec_ok] at rowSpec
  have rowAt : row = pages.val[position.val]! := by
    rw [rowSpec, List.Inhabited_getElem_eq_getElem! _ _ ready.2.1]
  obtain ⟨slot, slotIndexed, completed⟩ := MemoryWrite.bind_success _ _ _ slotOk
  change alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (Option Usize)) row.slots offset = .ok slot at slotIndexed
  have slotSpec := rust_vec_index row.slots offset (by simpa only [rowAt] using offsetBound)
  simp only [slotIndexed, WP.spec_ok] at slotSpec
  have slotCell : cell pages position offset = some slot := by
    simp only [cell, list_some_at pages.val position.val ready.2.1, Option.bind_some]
    rw [←rowAt, List.getElem?_eq_getElem (by simpa only [rowAt] using offsetBound), ←slotSpec]
  cases slot with
  | some existing =>
      have equal := Result.ok_injective completed
      have produced : Stored ⟨pages, before.count⟩ key existing :=
        (stored_at_known_page pages before.count preparedValid.1 key position offset atPage offsetValue existing).mpr slotCell
      have localResult := existing_slot_interned ⟨pages, before.count⟩ preparedValid key existing produced
      have result := interned_after_ready before.pages pages before.count page position ready
        ⟨pages, before.count⟩ key existing localResult
      have idEqual : existing = id := congrArg Prod.fst equal
      have stateEqual : ({ before with pages := pages } : register_index.RegisterIndex) = after := congrArg Prod.snd equal
      simpa only [idEqual, stateEqual] using result
  | none =>
      obtain ⟨next, nextOk, outerOk⟩ := MemoryWrite.bind_success _ _ _ completed
      have nextValue := MemoryWrite.successful_add before.count 1#usize next nextOk
      obtain ⟨outer, outerIndexed, innerOk⟩ := MemoryWrite.bind_success _ _ _ outerOk
      rcases outer with ⟨borrowed, restoreOuter⟩
      have outerSpec := alloc.vec.Vec.index_mut_usize_spec pages position ready.2.1
      simp only [alloc.vec.Vec.index_mut_slice_index] at outerIndexed
      simp only [outerIndexed, WP.spec_ok] at outerSpec
      have borrowedAt : borrowed = pages.val[position.val]! := by
        rw [outerSpec.1, List.Inhabited_getElem_eq_getElem! _ _ ready.2.1]
      obtain ⟨inner, innerIndexed, returned⟩ := MemoryWrite.bind_success _ _ _ innerOk
      rcases inner with ⟨previous, restoreInner⟩
      have innerSpec := alloc.vec.Vec.index_mut_usize_spec borrowed.slots offset
        (by simpa only [borrowedAt] using offsetBound)
      change alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice (Option Usize)) borrowed.slots offset = .ok (previous, restoreInner) at innerIndexed
      simp only [alloc.vec.Vec.index_mut_slice_index] at innerIndexed
      simp only [innerIndexed, WP.spec_ok] at innerSpec
      have pagesEqual : restoreOuter { borrowed with slots := restoreInner (some before.count) } =
          setSlot pages position offset (some before.count) := by
        simp only [setSlot, outerSpec.2, innerSpec.2, borrowedAt]
      have equal := Result.ok_injective returned
      have idEqual : before.count = id := congrArg Prod.fst equal
      have stateEqual : (⟨setSlot pages position offset (some before.count), next⟩ : register_index.RegisterIndex) = after := by
        simpa only [pagesEqual] using congrArg Prod.snd equal
      have localResult := fresh_slot_interned ⟨pages, before.count⟩ preparedValid key position offset next
        atPage offsetValue slotCell (by scalar_tac)
      have result := interned_after_ready before.pages pages before.count page position ready
        ⟨setSlot pages position offset (some before.count), next⟩ key before.count localResult
      change Interned before ⟨setSlot pages position offset (some before.count), next⟩ key before.count at result
      rw [stateEqual] at result
      simpa only [idEqual] using result

end MemoryWrite.RegisterIndex
