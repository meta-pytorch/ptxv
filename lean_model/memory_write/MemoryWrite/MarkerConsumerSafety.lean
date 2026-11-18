/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustEngine
import MemoryWrite.Consumers

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite
namespace MarkerConsumerSafety

/-- A consumer of the extracted engine's byte markers, preserving explicit current origins. -/
def halfOriginWithFallback (state : engine.Engine) (cell : Cell)
    (initialOrigin : Key → Option Origin) : Option Origin :=
  match cell.origin with
  | some origin => some origin
  | none => if cell.key ∈ state.half_written.val then none else initialOrigin cell.key

/-- Any touched byte blocks an initial identity for its aligned numeric word. -/
def initialNumericFallback (state : engine.Engine) (byte : Key)
    (initialHistory : Option WordHistory) : Option WordHistory :=
  if alignedWordKey byte ∈ state.numeric_written.val then none else initialHistory

theorem apply_success_refines (before : Storage) (state next : engine.Engine)
    (write : checked.PreparedWrite) (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (hrun : engine.apply state write = .ok (.Ok (), next)) :
    RustEngineRefines before state write (.Ok (), next) := by
  have hspec := rust_engine_apply_refines before state write hcells hnumeric hhalf
  rw [hrun] at hspec
  simpa only [WP.spec_ok] using hspec

theorem apply_half_membership (before : Storage) (state next : engine.Engine)
    (write : checked.PreparedWrite) (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (hrun : engine.apply state write = .ok (.Ok (), next))
    (henabled : write.enabled = true) (hglobal : write.destination.space = 1#u8)
    (htracking : state.track_half = true) (key : Key) :
    key ∈ next.half_written.val ↔
      key ∈ Storage.spanKeys write.destination (alloc.vec.Vec.len write.payload) ∨
        key ∈ state.half_written.val := by
  have hmarkers := (apply_success_refines before state next write hcells hnumeric hhalf hrun).2.2
  have hmember := hmarkers.2 key
  have hrecord := WrittenMarkers.record_half_mem (rustMarkerView state)
    (rustTrackingView state) (rustEffect write) key henabled hglobal htracking
  exact hmember.trans (by simpa [rustEffect, rust_payload_len, rustMarkerView] using hrecord)

theorem apply_numeric_membership (before : Storage) (state next : engine.Engine)
    (write : checked.PreparedWrite) (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (hrun : engine.apply state write = .ok (.Ok (), next))
    (henabled : write.enabled = true) (hglobal : write.destination.space = 1#u8)
    (htracking : state.track_numeric = true) (word : Key) :
    word ∈ next.numeric_written.val ↔
      (∃ byte ∈ Storage.spanKeys write.destination (alloc.vec.Vec.len write.payload),
        alignedWordKey byte = word) ∨ word ∈ state.numeric_written.val := by
  have hmarkers := (apply_success_refines before state next write hcells hnumeric hhalf hrun).2.2
  have hmember := hmarkers.1 word
  have hrecord := WrittenMarkers.record_numeric_mem (rustMarkerView state)
    (rustTrackingView state) (rustEffect write) word henabled hglobal htracking
  exact hmember.trans (by simpa [rustEffect, rust_payload_len, rustMarkerView] using hrecord)

theorem apply_half_prevents_initial_origin (before : Storage) (state next : engine.Engine)
    (write : checked.PreparedWrite) (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (hrun : engine.apply state write = .ok (.Ok (), next))
    (henabled : write.enabled = true) (hglobal : write.destination.space = 1#u8)
    (htracking : state.track_half = true) (cell : Cell)
    (htouched : cell.key ∈ Storage.spanKeys write.destination (alloc.vec.Vec.len write.payload))
    (hunknown : cell.origin = none) (initialOrigin : Key → Option Origin) :
    halfOriginWithFallback next cell initialOrigin = none := by
  have hmarked := (apply_half_membership before state next write hcells hnumeric hhalf
    hrun henabled hglobal htracking cell.key).mpr (Or.inl htouched)
  simp [halfOriginWithFallback, hunknown, hmarked]

theorem apply_partial_word_prevents_initial_numeric (before : Storage)
    (state next : engine.Engine) (write : checked.PreparedWrite)
    (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (hrun : engine.apply state write = .ok (.Ok (), next))
    (henabled : write.enabled = true) (hglobal : write.destination.space = 1#u8)
    (htracking : state.track_numeric = true) (byte : Key)
    (htouched : byte ∈ Storage.spanKeys write.destination (alloc.vec.Vec.len write.payload))
    (initialHistory : Option WordHistory) :
    initialNumericFallback next byte initialHistory = none := by
  have hmarked := (apply_numeric_membership before state next write hcells hnumeric hhalf
    hrun henabled hglobal htracking (alignedWordKey byte)).mpr
    (Or.inl ⟨byte, htouched, rfl⟩)
  simp [initialNumericFallback, hmarked]

/-- Progress, actual marker membership, and fallback safety, independently for both trackers.
The source layout and vector capacities are explicit; no after-state marker is assumed. -/
theorem enabled_global_apply_consumer_safety (before : Storage) (state : engine.Engine)
    (write : checked.PreparedWrite) (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (haddress : write.address_dependent = false) (hguard : write.guard_dependent = false)
    (henabled : write.enabled = true) (hglobal : write.destination.space = 1#u8)
    (hfamily : (rustFamily write.class).allows write.destination.space = true)
    (hspan : Storage.SpanFits before write.destination (alloc.vec.Vec.len write.payload)) :
    ∃ next, engine.apply state write = .ok (.Ok (), next) ∧
      (∀ index : Fin write.payload.val.length,
        let byte := Storage.offsetKey write.destination index.val
        (state.track_half = true → byte ∈ next.half_written.val) ∧
        (state.track_numeric = true → alignedWordKey byte ∈ next.numeric_written.val)) ∧
      (∀ cell : Cell, cell ∈ next.cells.slice.val →
        cell.key ∈ Storage.spanKeys write.destination (alloc.vec.Vec.len write.payload) →
        cell.origin = none → state.track_half = true →
        ∀ initialOrigin, halfOriginWithFallback next cell initialOrigin = none) ∧
      (∀ byte ∈ Storage.spanKeys write.destination (alloc.vec.Vec.len write.payload),
        state.track_numeric = true →
        ∀ initialHistory, initialNumericFallback next byte initialHistory = none) := by
  obtain ⟨next, hrun, _⟩ := rust_engine_apply_progress before state write hcells hnumeric hhalf
    haddress hguard henabled hfamily hspan
  refine ⟨next, hrun, ?_, ?_, ?_⟩
  · intro index
    have htouched : Storage.offsetKey write.destination index.val ∈
        Storage.spanKeys write.destination (alloc.vec.Vec.len write.payload) := by
      apply List.mem_ofFn.mpr
      exact ⟨⟨index.val, by simp [alloc.vec.Vec.len]⟩, rfl⟩
    constructor
    · intro htracking
      exact (apply_half_membership before state next write hcells hnumeric hhalf
        hrun henabled hglobal htracking _).mpr (Or.inl htouched)
    · intro htracking
      exact (apply_numeric_membership before state next write hcells hnumeric hhalf
        hrun henabled hglobal htracking _).mpr (Or.inl ⟨_, htouched, rfl⟩)
  · intro cell _ htouched hunknown htracking initialOrigin
    exact apply_half_prevents_initial_origin before state next write hcells hnumeric hhalf
      hrun henabled hglobal htracking cell htouched hunknown initialOrigin
  · intro byte htouched htracking initialHistory
    exact apply_partial_word_prevents_initial_numeric before state next write hcells hnumeric hhalf
      hrun henabled hglobal htracking byte htouched initialHistory

theorem apply_success_has_independent_address_and_guard (before : Storage)
    (state next : engine.Engine) (write : checked.PreparedWrite)
    (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (hrun : engine.apply state write = .ok (.Ok (), next)) :
    write.address_dependent = false ∧ write.guard_dependent = false := by
  obtain ⟨after, hmodel, _⟩ :=
    (apply_success_refines before state next write hcells hnumeric hhalf hrun).1
  cases haddress : write.address_dependent with
  | true => simp [applyEffect, rustEffect, haddress] at hmodel
  | false =>
    cases hguard : write.guard_dependent with
    | true => simp [applyEffect, rustEffect, haddress, hguard] at hmodel
    | false => exact ⟨rfl, rfl⟩

end MarkerConsumerSafety
end MemoryWrite
