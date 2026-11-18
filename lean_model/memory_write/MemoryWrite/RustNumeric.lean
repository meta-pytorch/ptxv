/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustWords
import MemoryWrite.Production

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite

def numericMarkerView (state : engine.Engine) : WrittenMarkers :=
  ⟨state.numeric_written.val, state.half_written.val⟩

def numericTrackingView (state : engine.Engine) : WrittenTracking :=
  ⟨state.track_numeric, state.track_half⟩

def NumericMarkerEq (left right : WrittenMarkers) : Prop :=
  (∀ key, key ∈ left.numericWritten ↔ key ∈ right.numericWritten) ∧
    left.halfWritten = right.halfWritten

structure RustNumericRefines (before : Storage) (state : engine.Engine)
    (key : Key) (next : engine.Engine) : Prop where
  cells : next.cells.slice = (before.setWord ⟨key, none⟩).toCells
  numeric : next.numeric_written.val = state.numeric_written.val ++
    if state.track_numeric = true ∧ key.space = 1#u8 then [key] else []
  half : next.half_written = state.half_written
  numericTracking : next.track_numeric = state.track_numeric
  halfTracking : next.track_half = state.track_half

@[step]
theorem rust_forget_numeric_spec (before : Storage) (state : engine.Engine) (key : Key)
    (hcells : state.cells.slice = before.toCells)
    (hcapacity : state.track_numeric = true → key.space = 1#u8 →
      state.numeric_written.val.length < Usize.max) :
    numeric.forget_numeric state key ⦃ next => RustNumericRefines before state key next ⦄ := by
  unfold numeric.forget_numeric
  simp only [alloc.vec.Vec.deref_mut, lift, bind_ok]
  rw [hcells]
  apply WP.spec_bind (rust_install_word_refines before ⟨key, none⟩)
  intro cells hcleared
  cases hnumeric : state.track_numeric with
  | false =>
    simp only [Bool.false_eq_true, ite_false, WP.spec_ok]
    refine ⟨?_, ?_, rfl, hnumeric.symm, rfl⟩
    · exact hcleared
    · simp [hnumeric]
  | true =>
    simp only [ite_true]
    by_cases hglobal : key.space = 1#u8
    · simp only [hglobal, ite_true]
      step with alloc.vec.Vec.push_spec state.numeric_written key
        (hcapacity hnumeric hglobal) as ⟨written, hwritten⟩
      refine ⟨?_, ?_, rfl, hnumeric.symm, rfl⟩
      · exact hcleared
      · simpa [hnumeric, hglobal] using hwritten
    · simp only [hglobal, ite_false, WP.spec_ok]
      refine ⟨?_, ?_, rfl, hnumeric.symm, rfl⟩
      · exact hcleared
      · simp [hglobal]

theorem RustNumericRefines.markers (before : Storage) (state next : engine.Engine) (key : Key)
    (hnext : RustNumericRefines before state key next) :
    NumericMarkerEq (numericMarkerView next)
      ((numericMarkerView state).recordNumeric (numericTrackingView state) key) := by
  rcases hnext with ⟨hcells, hnumeric, hhalf, htrackNumeric, htrackHalf⟩
  cases hn : state.track_numeric <;> by_cases hglobal : key.space = 1#u8 <;>
    simp [NumericMarkerEq, numericMarkerView, numericTrackingView,
      WrittenMarkers.recordNumeric, hnumeric, hhalf, hn, hglobal, List.mem_append, or_comm]

def numericByteView (cell : Cell) :=
  (cell.key, cell.value, cell.origin, cell.tainted, cell.defined, cell.written)

theorem RustNumericRefines.byte_frame (before : Storage) (state next : engine.Engine) (key : Key)
    (hcells : state.cells.slice = before.toCells)
    (hnext : RustNumericRefines before state key next) :
    next.cells.slice.val.map numericByteView = state.cells.slice.val.map numericByteView := by
  rw [hnext.cells, hcells]
  simp [Storage.toCells, Storage.getCell_setWord, numericByteView, List.map_map]

theorem RustNumericRefines.word_effect (before : Storage) (state next : engine.Engine) (key : Key)
    (hnext : RustNumericRefines before state key next) :
    ∀ cell ∈ next.cells.slice.val,
      cell.word = if key = cell.key then none else (before.getCell cell.key).word := by
  intro cell hcell
  rw [hnext.cells] at hcell
  simp only [Storage.toCells, Slice.from_val, List.mem_map] at hcell
  obtain ⟨address, _, rfl⟩ := hcell
  simp [Storage.getCell_setWord]

theorem rust_forget_numeric_preserves_representation {Input : Type}
    (sem : Semantics Input) (memory : Memory Input) (before : Storage)
    (state : engine.Engine) (key : Key)
    (hcells : state.cells.slice = before.toCells)
    (hcapacity : state.track_numeric = true → key.space = 1#u8 →
      state.numeric_written.val.length < Usize.max)
    (hstate : Storage.Represents sem memory before) :
    ∃ next, numeric.forget_numeric state key = .ok next ∧
      RustNumericRefines before state key next ∧
      Represents sem memory next.cells.slice ∧
      NumericMarkerEq (numericMarkerView next)
        ((numericMarkerView state).recordNumeric (numericTrackingView state) key) ∧
      next.half_written = state.half_written := by
  obtain ⟨next, hrun, hnext⟩ := spec_returns _ _
    (rust_forget_numeric_spec before state key hcells hcapacity)
  refine ⟨next, hrun, hnext, ?_, hnext.markers before state next key, hnext.half⟩
  rw [hnext.cells]
  apply Storage.setWord_preserves_representation sem memory before ⟨key, none⟩ hstate
  intro id hid
  cases hid

end MemoryWrite
