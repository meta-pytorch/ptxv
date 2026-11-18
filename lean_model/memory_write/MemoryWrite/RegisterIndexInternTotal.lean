/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexEnsureTotal

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

private theorem rust_intern_progress (before : register_index.RegisterIndex) (key : symbolic.RegisterKey)
    (valid : Valid before) (pageSpace : before.pages.val.length < Usize.max)
    (countSpace : before.count.val < Usize.max) :
    register_index.RegisterIndex.intern before key ⦃ result =>
      result.2.count.val ≤ before.count.val + 1 ∧
      result.2.pages.val.length ≤ before.pages.val.length + 1 ⦄ := by
  unfold register_index.RegisterIndex.intern
  step with rust_page_key key as ⟨page, pageValue⟩
  step with rust_ensure_page_total before.pages page valid.1 pageSpace
    as ⟨position, pages, ready, lengthBound⟩
  step with U16.rem_spec (x := key.index) (y := 64#u16) (by scalar_tac)
    as ⟨remainder, remainderValue⟩
  simp only [lift, bind_ok]
  let offset := core.convert.num.FromUsizeU16.from remainder
  have offsetBound : offset.val < (pages.val[position.val]!).slots.val.length := by
    rw [ready.1.1 position.val ready.2.1]
    dsimp only [offset]
    scalar_tac
  step with alloc.vec.Vec.index_usize_spec pages position ready.2.1 as ⟨row, rowEq⟩
  have rowAt : row = pages.val[position.val]! := by
    rw [rowEq, List.Inhabited_getElem_eq_getElem! _ _ ready.2.1]
  step with alloc.vec.Vec.index_usize_spec row.slots offset (by simpa only [rowAt] using offsetBound)
    as ⟨slot, slotEq⟩
  cases slot with
  | some existing =>
      simp only [WP.spec_ok]
      exact ⟨by omega, lengthBound⟩
  | none =>
      step with Usize.add_spec (x := before.count) (y := 1#usize) (by scalar_tac)
        as ⟨next, nextValue⟩
      step with alloc.vec.Vec.index_mut_usize_spec pages position ready.2.1
        as ⟨borrowed, restoreOuter, borrowedEq, outerEq⟩
      have borrowedAt : borrowed = pages.val[position.val]! := by
        rw [borrowedEq, List.Inhabited_getElem_eq_getElem! _ _ ready.2.1]
      step with alloc.vec.Vec.index_mut_usize_spec borrowed.slots offset
        (by simpa only [borrowedAt] using offsetBound)
        as ⟨old, restoreInner, oldEq, innerEq⟩
      refine ⟨by scalar_tac, ?_⟩
      simpa only [outerEq, alloc.vec.Vec.set_val_eq, List.length_set] using lengthBound

@[step]
theorem rust_intern_total (before : register_index.RegisterIndex) (key : symbolic.RegisterKey)
    (valid : Valid before) (pageSpace : before.pages.val.length < Usize.max)
    (countSpace : before.count.val < Usize.max) :
    register_index.RegisterIndex.intern before key ⦃ result =>
      Interned before result.2 key result.1 ∧ result.2.count.val ≤ before.count.val + 1 ∧
      result.2.pages.val.length ≤ before.pages.val.length + 1 ⦄ := by
  obtain ⟨⟨id, after⟩, executed, bounds⟩ := MemoryWrite.spec_returns _ _
    (rust_intern_progress before key valid pageSpace countSpace)
  rw [executed, WP.spec_ok]
  exact ⟨successful_intern before after key id valid executed, bounds⟩

end MemoryWrite.RegisterIndex
