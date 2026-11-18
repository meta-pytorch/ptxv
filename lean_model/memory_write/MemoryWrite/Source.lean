/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs
import MemoryWrite.Model

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite

@[step]
theorem cell_after_write_spec (cell : Cell) (write : Key) (payload : Slice PayloadByte) :
    cell_after_write cell write payload ⦃ result => result = modelCell cell write payload ⦄ := by
  unfold cell_after_write
  simp only [bne_iff_ne]
  split
  · rename_i hs
    have hspace : write.space ≠ cell.key.space := by scalar_tac
    simp [modelCell, wordAfter, Touches, CoversWord, Overlaps, SameRegion, hspace]
  · rename_i hs
    have hspace : write.space = cell.key.space := by scalar_tac
    split
    · rename_i ha
      have halloc : write.allocation ≠ cell.key.allocation := by scalar_tac
      simp [modelCell, wordAfter, Touches, CoversWord, Overlaps, SameRegion, halloc]
    · rename_i ha
      have halloc : write.allocation = cell.key.allocation := by scalar_tac
      have hsize : payload.val.length ≤ U64.max := by
        have := Slice.property payload
        have := Usize.bounds_eq
        scalar_tac
      step with UScalar.cast_inBounds_spec .U128 write.offset (by scalar_tac) as ⟨start, hstart⟩
      step with UScalar.cast_inBounds_spec .U128 (Slice.len payload) (by simpa using (show payload.val.length ≤ UScalar.max .U128 by scalar_tac)) as ⟨width, hwidth⟩
      have hwidthVal : width.val = payload.val.length := by simpa using hwidth
      step as ⟨stop, hstop⟩ ; try scalar_tac
      step with UScalar.cast_inBounds_spec .U128 cell.key.offset (by scalar_tac) as ⟨address, haddress⟩
      step as ⟨empty, hempty⟩
      have hclear :
          (if empty then ok cell.word else
            if address < stop then do
              let after ← address + 4#u128
              if start < after then ok none else ok cell.word
            else ok cell.word)
          ⦃ result => result = if Overlaps write payload.val.length cell.key
            then none else cell.word ⦄ := by
        by_cases he : empty = true
        · simp only [he, ite_true, WP.spec_ok]
          have hz : payload.val.length = 0 := by simpa using hempty.mp he
          simp [Overlaps, hz]
        · simp only [he, Bool.false_eq_true, ite_false]
          split
          · rename_i hbefore
            step as ⟨after, hafter⟩ ; try scalar_tac
            split
            · have ho : Overlaps write payload.val.length cell.key := by
                refine ⟨⟨hspace, halloc⟩, ?_, ?_, ?_⟩ <;> scalar_tac
              simp [ho]
            · have ho : ¬Overlaps write payload.val.length cell.key := by
                intro h
                obtain ⟨_, _, _, hlast⟩ := h
                scalar_tac
              simp [ho]
          · have ho : ¬Overlaps write payload.val.length cell.key := by
              intro h
              obtain ⟨_, _, hfirst, _⟩ := h
              scalar_tac
            simp [ho]
      step with hclear as ⟨oldWord, holdWord⟩
      split
      · rename_i hlo
        split
        · rename_i hhi
          step as ⟨difference, hdifference⟩ ; try scalar_tac
          have hi : difference.val < payload.val.length := by scalar_tac
          have hbound : difference.val ≤ UScalar.max .Usize := by
            have := Slice.property payload
            scalar_tac
          step with UScalar.cast_inBounds_spec .Usize difference hbound as ⟨index, hindex⟩
          have hindexBound : index.val < payload.val.length := by omega
          step with Slice.index_usize_spec payload index hindexBound as ⟨byte, hbyte⟩
          have hbyteAt : byte = payloadAt payload (cell.key.offset.val - write.offset.val) := by
            have hoffset : index.val = cell.key.offset.val - write.offset.val := by scalar_tac
            simp only [payloadAt, ←hoffset, List.getElem?_eq_getElem hindexBound,
              Option.getD_some, hbyte]
          step as ⟨after, hafter⟩ ; try scalar_tac
          have ht : Touches write payload.val.length cell.key := by
            exact ⟨⟨hspace, halloc⟩, by scalar_tac, by scalar_tac⟩
          split
          · have hc : CoversWord write payload.val.length cell.key := by
              exact ⟨⟨hspace, halloc⟩, by scalar_tac, by scalar_tac⟩
            simp [modelCell, ht, wordAfter, hc, hbyteAt]
          · have hc : ¬CoversWord write payload.val.length cell.key := by
              intro h
              obtain ⟨_, _, hlast⟩ := h
              scalar_tac
            simp [modelCell, ht, wordAfter, hc, hbyteAt, holdWord]
        · have ht : ¬Touches write payload.val.length cell.key := by
            intro h
            obtain ⟨_, _, hlast⟩ := h
            scalar_tac
          have hc : ¬CoversWord write payload.val.length cell.key := by
            intro h
            obtain ⟨_, _, hlast⟩ := h
            scalar_tac
          simp [modelCell, ht, wordAfter, hc, holdWord]
      · have ht : ¬Touches write payload.val.length cell.key := by
          intro h
          obtain ⟨_, hfirst, _⟩ := h
          scalar_tac
        have hc : ¬CoversWord write payload.val.length cell.key := by
          intro h
          obtain ⟨_, hfirst, _⟩ := h
          scalar_tac
        simp [modelCell, ht, wordAfter, hc, holdWord]

def MapPrefix (original current : Slice Cell) (write : Key) (payload : Slice PayloadByte)
    (cursor : Nat) : Prop :=
  current.val.length = original.val.length ∧
    ∀ position, current.val[position]? = if position < cursor then
      original.val[position]?.map (fun cell => modelCell cell write payload)
    else original.val[position]?

@[step]
theorem apply_write_effect_spec (original : Slice Cell) (write : Key) (payload : Slice PayloadByte) :
    apply_write_effect original write payload
      ⦃ updated => updated.val = original.val.map (fun cell => modelCell cell write payload) ⦄ := by
  unfold apply_write_effect apply_write_effect_loop
  apply loop.spec_decr_nat
    (fun current : Slice Cell × Usize => original.val.length - current.2.val)
    (fun current : Slice Cell × Usize => current.2.val ≤ original.val.length ∧
      MapPrefix original current.1 write payload current.2.val)
  · intro current hcurrent
    rcases current with ⟨cells, cursor⟩
    dsimp only [MapPrefix] at hcurrent
    obtain ⟨hcursor, hlength, hprefix⟩ := hcurrent
    unfold apply_write_effect_loop.body
    by_cases hlt : cursor.val < cells.val.length
    · have htest : cursor < Slice.len cells := by scalar_tac
      simp only [htest, ite_true]
      step with Slice.index_usize_spec cells cursor hlt as ⟨value, hvalue⟩
      step with cell_after_write_spec value write payload as ⟨after, hafter⟩
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
          have horiginal : original.val[cursor.val]? = some value := by
            have hread := hprefix cursor.val
            simp only [Nat.lt_irrefl, ite_false] at hread
            rw [←hread]
            simp only [List.getElem?_eq_getElem hlt, hvalue]
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

@[step]
theorem snapshot_cells_spec (source : Slice Cell) :
    snapshot_cells source ⦃ payload => payload.val = source.val.map snapshotByte ⦄ := by
  unfold snapshot_cells snapshot_cells_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec PayloadByte × Usize => source.val.length - current.2.val)
    (fun current : alloc.vec.Vec PayloadByte × Usize =>
      current.2.val ≤ source.val.length ∧
        current.1.val = (source.val.take current.2.val).map snapshotByte)
  · intro current hcurrent
    rcases current with ⟨payload, cursor⟩
    change cursor.val ≤ source.val.length ∧
      payload.val = (source.val.take cursor.val).map snapshotByte at hcurrent
    obtain ⟨hcursor, hprefix⟩ := hcurrent
    have hlength : payload.val.length = cursor.val := by
      rw [hprefix, List.length_map, List.length_take]
      omega
    unfold snapshot_cells_loop.body
    by_cases hlt : cursor.val < source.val.length
    · have htest : cursor < Slice.len source := by scalar_tac
      simp only [htest, ite_true]
      step with Slice.index_usize_spec source cursor hlt as ⟨cell, hcell⟩
      step with alloc.vec.Vec.push_spec payload
        ⟨cell.value, cell.origin, cell.tainted, cell.defined, cell.word⟩ (by
        have := source.property
        omega) as ⟨updated, hupdated⟩
      step as ⟨next, hnext⟩
      have hnextVal : next.val = cursor.val + 1 := by scalar_tac
      refine ⟨?_, ?_, ?_⟩
      · omega
      · rw [hupdated, hprefix, hnextVal, List.take_succ_eq_append_getElem hlt,
          List.map_append]
        simp [hcell, snapshotByte]
      · omega
    · have htest : ¬cursor < Slice.len source := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      rw [hprefix, List.take_of_length_le (by omega)]
  · exact ⟨by scalar_tac, by simp⟩

@[step]
theorem apply_copy_effect_spec (state source : Slice Cell) (write : Key) :
    apply_copy_effect state write source
      ⦃ updated => updated.val = state.val.map
        (fun cell => modelCell cell write (snapshotPayload source)) ⦄ := by
  unfold apply_copy_effect
  step with snapshot_cells_spec source as ⟨payload, hpayload⟩
  have hslice : alloc.vec.Vec.deref payload = snapshotPayload source := by
    apply Slice.ext
    simpa [snapshotPayload, alloc.vec.Vec.deref, alloc.vec.Vec.val] using hpayload
  rw [hslice]
  exact apply_write_effect_spec state write (snapshotPayload source)

end MemoryWrite
