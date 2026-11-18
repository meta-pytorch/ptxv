/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Footprint

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.Commutation

open Footprint

def DisjointComplete (first second : WriteEffect) : Prop :=
  ∀ key, ¬(Complete first key ∧ Complete second key)

theorem DisjointComplete.symm (first second : WriteEffect)
    (hdisjoint : DisjointComplete first second) : DisjointComplete second first := by
  intro key ⟨hsecond, hfirst⟩
  exact hdisjoint key ⟨hfirst, hsecond⟩

theorem effectCell_commute (first second : WriteEffect) (cell : Cell)
    (hdisjoint : DisjointComplete first second) :
    effectCell second (effectCell first cell) = effectCell first (effectCell second cell) := by
  by_cases hfirst : Complete first cell.key
  · have hsecond : ¬Complete second cell.key := fun h => hdisjoint cell.key ⟨hfirst, h⟩
    rw [effectCell_frame second (effectCell first cell) (by simpa using hsecond),
      effectCell_frame second cell hsecond]
  · rw [effectCell_frame first cell hfirst,
      effectCell_frame first (effectCell second cell) (by simpa using hfirst)]

theorem modelCell_determined (left right : Cell) (destination : Key) (payload : Slice PayloadByte)
    (hkey : left.key = right.key) (htouched : Touches destination payload.val.length left.key) :
    modelCell left destination payload = modelCell right destination payload := by
  have hoverlap := Storage.touches_overlaps destination left.key payload.val.length htouched
  cases left with
  | mk leftKey leftValue leftOrigin leftTainted leftDefined leftWritten leftWord =>
    cases right with
    | mk rightKey rightValue rightOrigin rightTainted rightDefined rightWritten rightWord =>
      dsimp only at hkey htouched hoverlap
      subst rightKey
      by_cases hfull : CoversWord destination payload.val.length leftKey <;>
        simp [modelCell, wordAfter, htouched, hoverlap, hfull]

theorem effectCell_determined (effect : WriteEffect) (left right : Cell)
    (hkey : left.key = right.key) (htouched : ByteAffected effect left.key) :
    effectCell effect left = effectCell effect right := by
  have hmodel := modelCell_determined left right effect.destination effect.payload hkey htouched.2
  simp only [effectCell, htouched.1, ↓reduceIte, hmodel]

theorem introduced_byteAffected (layout : MemoryLayout) (before after : Storage)
    (write : CheckedWrite) (hrun : applyChecked layout before write = .ok after)
    (key : Key) (hintroduced : Introduced write key) : ByteAffected write.normalizedEffect key := by
  have hpermission := applyChecked_permission layout before after write hintroduced.1 hrun
  have hwrap := read_valid_no_wrap layout write.location write.effect.payload.val.length hpermission.2
  refine ⟨hintroduced.1, ?_⟩
  simpa only [CheckedWrite.normalizedEffect, Slice.len_val] using
    Storage.spanKeys_touches write.normalizedEffect.destination key (Slice.len write.effect.payload)
      (by simpa [CheckedWrite.normalizedEffect] using hwrap) hintroduced.2.2

structure ObservationalEq (left right : Storage) : Prop where
  inventory : ∀ key, key ∈ left.keys.val ↔ key ∈ right.keys.val
  cells : ∀ key ∈ left.keys.val, left.getCell key = right.getCell key
  backing : ∀ key, (left.getCell key).value = (right.getCell key).value

/-- Prepared effects keep their payloads and postword IDs fixed across the two executions. -/
theorem checked_commute (layout : MemoryLayout) (before firstOnly secondOnly firstSecond secondFirst : Storage)
    (first second : CheckedWrite)
    (hfirst : applyChecked layout before first = .ok firstOnly)
    (hsecond : applyChecked layout before second = .ok secondOnly)
    (hfirstSecond : applyChecked layout firstOnly second = .ok firstSecond)
    (hsecondFirst : applyChecked layout secondOnly first = .ok secondFirst)
    (hdisjoint : DisjointComplete first.normalizedEffect second.normalizedEffect) :
    ObservationalEq firstSecond secondFirst := by
  have hsfirst := applyChecked_spec layout before firstOnly first hfirst
  have hssecond := applyChecked_spec layout before secondOnly second hsecond
  have hsfirstSecond := applyChecked_spec layout firstOnly firstSecond second hfirstSecond
  have hssecondFirst := applyChecked_spec layout secondOnly secondFirst first hsecondFirst
  have hinventory : ∀ key, key ∈ firstSecond.keys.val ↔ key ∈ secondFirst.keys.val := by
    intro key
    rw [hsfirstSecond.keys, hsfirst.keys, hssecondFirst.keys, hssecond.keys]
    tauto
  have hcells : ∀ key ∈ firstSecond.keys.val, firstSecond.getCell key = secondFirst.getCell key := by
    intro key hkey
    have hother := (hinventory key).mp hkey
    by_cases hbefore : key ∈ before.keys.val
    · have hf := (hsfirst.keys key).mpr (Or.inl hbefore)
      have hs := (hssecond.keys key).mpr (Or.inl hbefore)
      rw [hsfirstSecond.cells key hkey, hssecondFirst.cells key hother,
        hsfirst.cells key hf, hssecond.cells key hs]
      exact effectCell_commute _ _ _ hdisjoint
    · have hnew : Introduced first key ∨ Introduced second key := by
        rw [hsfirstSecond.keys, hsfirst.keys] at hkey
        tauto
      rcases hnew with hnew | hnew
      · have hfirstTouched := introduced_byteAffected layout before firstOnly first hfirst key hnew
        have hsecondFrame : ¬Complete second.normalizedEffect key := by
          intro h
          exact hdisjoint key ⟨Or.inl hfirstTouched, h⟩
        have hf := (hsfirst.keys key).mpr (Or.inr hnew)
        rw [hsfirstSecond.cells key hkey,
          effectCell_frame _ _ (by simpa using hsecondFrame), hsfirst.cells key hf,
          hssecondFirst.cells key hother]
        exact effectCell_determined _ _ _ (by simp) (by simpa using hfirstTouched)
      · have hsecondTouched := introduced_byteAffected layout before secondOnly second hsecond key hnew
        have hfirstFrame : ¬Complete first.normalizedEffect key := by
          intro h
          exact hdisjoint key ⟨h, Or.inl hsecondTouched⟩
        have hs := (hssecond.keys key).mpr (Or.inr hnew)
        rw [hssecondFirst.cells key hother,
          effectCell_frame _ _ (by simpa using hfirstFrame), hssecond.cells key hs,
          hsfirstSecond.cells key hkey]
        exact effectCell_determined _ _ _ (by simp) (by simpa using hsecondTouched)
  refine ⟨hinventory, hcells, ?_⟩
  intro key
  by_cases hkey : key ∈ firstSecond.keys.val
  · exact congrArg Cell.value (hcells key hkey)
  · have hother : key ∉ secondFirst.keys.val := fun h => hkey ((hinventory key).mpr h)
    have hnotfirst : key ∉ firstOnly.keys.val := by
      intro h
      exact hkey ((hsfirstSecond.keys key).mpr (Or.inl h))
    have hnotsecond : key ∉ secondOnly.keys.val := by
      intro h
      exact hother ((hssecondFirst.keys key).mpr (Or.inl h))
    rw [hsfirstSecond.inactive key hkey, hsfirst.inactive key hnotfirst,
      hssecondFirst.inactive key hother, hssecond.inactive key hnotsecond]

def eraseOpaque (cell : Cell) : Cell :=
  { cell with word := cell.word.map fun history =>
      match history with
      | .Known id => .Known id
      | .Opaque _ => .Opaque 0#u32 }

theorem ObservationalEq.opaque_insensitive (left right : Storage)
    (heq : ObservationalEq left right) (key : Key) (hkey : key ∈ left.keys.val) :
    eraseOpaque (left.getCell key) = eraseOpaque (right.getCell key) :=
  congrArg eraseOpaque (heq.cells key hkey)

def MarkerEq (left right : WrittenMarkers) : Prop :=
  (∀ key, key ∈ left.numericWritten ↔ key ∈ right.numericWritten) ∧
    (∀ key, key ∈ left.halfWritten ↔ key ∈ right.halfWritten)

theorem marker_records_commute (markers : WrittenMarkers) (tracking : WrittenTracking)
    (first second : WriteEffect) :
    MarkerEq ((markers.record tracking first).record tracking second)
      ((markers.record tracking second).record tracking first) := by
  by_cases hfirst : (first.enabled && first.destination.space == 1#u8) = true <;>
    by_cases hsecond : (second.enabled && second.destination.space == 1#u8) = true <;>
    cases hn : tracking.numericEnabled <;> cases hh : tracking.halfEnabled <;>
    simp [MarkerEq, WrittenMarkers.record, hfirst, hsecond, hn, hh,
      List.mem_append, or_assoc, or_left_comm, or_comm]

theorem record_numeric_commutes (markers : WrittenMarkers) (tracking : WrittenTracking)
    (effect : WriteEffect) (key : Key) :
    MarkerEq ((markers.record tracking effect).recordNumeric tracking key)
      ((markers.recordNumeric tracking key).record tracking effect) := by
  cases heffect : effect.enabled <;>
    by_cases hspace : effect.destination.space = 1#u8 <;>
    by_cases hkey : key.space = 1#u8 <;>
    cases hn : tracking.numericEnabled <;> cases hh : tracking.halfEnabled <;>
    simp [MarkerEq, WrittenMarkers.record, WrittenMarkers.recordNumeric,
      heffect, hspace, hkey, hn, hh, List.mem_append, or_assoc, or_comm]


theorem checked_commute_with_markers (layout : MemoryLayout)
    (before firstOnly secondOnly firstSecond secondFirst : Storage)
    (first second : CheckedWrite) (markers : WrittenMarkers) (tracking : WrittenTracking)
    (hfirst : applyChecked layout before first = .ok firstOnly)
    (hsecond : applyChecked layout before second = .ok secondOnly)
    (hfirstSecond : applyChecked layout firstOnly second = .ok firstSecond)
    (hsecondFirst : applyChecked layout secondOnly first = .ok secondFirst)
    (hdisjoint : DisjointComplete first.normalizedEffect second.normalizedEffect) :
    ObservationalEq firstSecond secondFirst ∧
      MarkerEq ((markers.record tracking first.normalizedEffect).record tracking second.normalizedEffect)
        ((markers.record tracking second.normalizedEffect).record tracking first.normalizedEffect) := by
  exact ⟨checked_commute layout before firstOnly secondOnly firstSecond secondFirst first second
    hfirst hsecond hfirstSecond hsecondFirst hdisjoint,
    marker_records_commute markers tracking first.normalizedEffect second.normalizedEffect⟩

end MemoryWrite.Commutation
