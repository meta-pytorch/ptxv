/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustSpan

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite

/-- Every position is the first matching live cell at its source address. -/
def CapturedSpan (cells : Slice Cell) (start : Key) (width : Nat)
    (source : Slice Cell) : Prop :=
  source.val.length = width ∧
    ∀ index (hindex : index < source.val.length),
      findCellModel cells (Storage.offsetKey start index) = some source.val[index]

theorem capturedSpan_represents {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (cells source : Slice Cell) (start : Key) (width : Nat)
    (hstate : Represents sem memory cells) (hsource : CapturedSpan cells start width source) :
    Represents sem memory source := by
  intro cell hcell
  obtain ⟨index, hindex, hread⟩ := List.getElem_of_mem hcell
  have hfind := hsource.2 index hindex
  rw [hread] at hfind
  exact hstate cell (findCellModel_some cells _ cell hfind).1

theorem capturedSpan_contiguous (cells source : Slice Cell) (start : Key) (width : Nat)
    (hsource : CapturedSpan cells start width source)
    (hfit : start.offset.val + width ≤ U64.max + 1) : Contiguous source start := by
  intro index hindex
  have hkey := (findCellModel_some cells _ _ (hsource.2 index hindex)).2
  rw [hkey]
  have hlength := hsource.1
  exact Storage.offsetKey_location start index (by omega)

theorem capturedSpan_payload_sound {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (cells source : Slice Cell) (start : Key) (width : Nat)
    (hstate : Represents sem memory cells) (hsource : CapturedSpan cells start width source)
    (hfit : start.offset.val + width ≤ U64.max + 1) :
    PayloadSound sem (snapshotPayload source) (snapshotValues memory start) := by
  exact snapshotPayload_sound sem memory source start
    (capturedSpan_represents sem memory cells source start width hstate hsource)
    (capturedSpan_contiguous cells source start width hsource hfit)

def CaptureResult (cells : Slice Cell) (start : Key) (width : Nat)
    (result : Option (alloc.vec.Vec PayloadByte)) : Prop :=
  ∃ source : Slice Cell, ∃ payload : alloc.vec.Vec PayloadByte,
    result = some payload ∧ CapturedSpan cells start width source ∧
      payload.val = source.val.map snapshotByte

@[step]
theorem rust_capture_span_loop_spec (cells : Slice Cell) (start : Key) (width : Usize)
    (source : alloc.vec.Vec Cell) (cursor : Usize)
    (hspan : ValidWriteSpan cells start width.val) (hcursor : cursor.val ≤ width.val)
    (hsource : CapturedSpan cells start cursor.val source.slice) :
    checked.capture_span_loop cells start width source cursor
      ⦃ result => CaptureResult cells start width.val result ⦄ := by
  unfold checked.capture_span_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec Cell × Usize => width.val - current.2.val)
    (fun current : alloc.vec.Vec Cell × Usize => current.2.val ≤ width.val ∧
      CapturedSpan cells start current.2.val current.1.slice)
  · intro current hcurrent
    rcases current with ⟨source, cursor⟩
    change cursor.val ≤ width.val ∧ CapturedSpan cells start cursor.val source.slice at hcurrent
    obtain ⟨hcursor, hsource⟩ := hcurrent
    have hlength : source.val.length = cursor.val := hsource.1
    unfold checked.capture_span_loop.body
    by_cases hlt : cursor.val < width.val
    · have htest : cursor < width := by scalar_tac
      simp only [htest, ite_true]
      have hbound : start.offset.val + cursor.val ≤ U64.max := by
        have := hspan.no_wrap
        omega
      step with UScalar.cast_inBounds_spec .U64 cursor (by scalar_tac) as ⟨index, hindex⟩
      step as ⟨offset, hoffset⟩ ; try scalar_tac
      have hkey : { start with offset := offset } = Storage.offsetKey start cursor.val := by
        apply key_eq_of_location_eq
        rw [Storage.offsetKey_location start cursor.val hbound]
        simp only [location]
        congr 1
        scalar_tac
      rw [hkey]
      step with rust_find_cell_spec cells (Storage.offsetKey start cursor.val) as ⟨found, hfound⟩
      obtain ⟨cell, hcell⟩ := validSpan_findCellModel_some cells start width cursor.val hspan hlt
      rw [hcell] at hfound
      rw [hfound]
      step with alloc.vec.Vec.push_spec source cell (by scalar_tac) as ⟨updated, hupdated⟩
      step as ⟨next, hnext⟩
      have hnextVal : next.val = cursor.val + 1 := by scalar_tac
      refine ⟨?_, ⟨?_, ?_⟩, ?_⟩
      · omega
      · change updated.val.length = next.val
        simp [hupdated, hlength, hnextVal]
      · intro position hposition
        change position < updated.val.length at hposition
        change findCellModel cells (Storage.offsetKey start position) = some updated.val[position]
        by_cases hbefore : position < source.val.length
        · have hread := hsource.2 position hbefore
          change findCellModel cells (Storage.offsetKey start position) = some source.val[position]
            at hread
          simpa only [hupdated, List.getElem_append_left hbefore] using hread
        · have heq : position = cursor.val := by simp [hupdated, hlength] at hposition; omega
          subst position
          simpa [hupdated, List.getElem_append_right (by omega : source.val.length ≤ cursor.val),
            hlength] using hcell
      · omega
    · have htest : ¬cursor < width := by scalar_tac
      simp only [htest, ite_false]
      step with snapshot_cells_spec (alloc.vec.Vec.deref source) as ⟨payload, hpayload⟩
      refine ⟨source.slice, payload, rfl, ?_, ?_⟩
      · have heq : cursor.val = width.val := by omega
        simpa only [heq] using hsource
      · simpa only [alloc.vec.Vec.deref, alloc.vec.Vec.val, Slice.from_val] using hpayload
  · exact ⟨hcursor, hsource⟩

@[step]
theorem rust_capture_span_spec (cells : Slice Cell) (start : Key) (width : Usize) :
    checked.capture_span cells start width ⦃ result =>
      (ValidWriteSpan cells start width.val → CaptureResult cells start width.val result) ∧
      (¬ValidWriteSpan cells start width.val → result = none) ⦄ := by
  unfold checked.capture_span
  step with rust_span_fits_spec cells start width as ⟨fits, hfits⟩
  by_cases hfit : fits = true
  · simp only [hfit, ite_true]
    have hspan := hfits.mp hfit
    step with rust_capture_span_loop_spec cells start width (alloc.vec.Vec.new Cell) 0#usize
      hspan (by scalar_tac) (by
        constructor <;> simp [alloc.vec.Vec.new, alloc.vec.Vec.from, Slice.from_val])
      as ⟨result, hresult⟩
    exact ⟨by intro _; exact hresult, by intro h; exact False.elim (h hspan)⟩
  · simp only [hfit, Bool.false_eq_true, ite_false, WP.spec_ok]
    exact ⟨by intro h; exact False.elim (hfit (hfits.mpr h)), by intro _; trivial⟩

theorem rust_capture_span_sound {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (cells : Slice Cell) (start : Key) (width : Usize)
    (hstate : Represents sem memory cells) (hspan : ValidWriteSpan cells start width.val) :
    ∃ payload : alloc.vec.Vec PayloadByte,
      checked.capture_span cells start width = .ok (some payload) ∧
      payload.val.length = width.val ∧
      PayloadSound sem payload.slice (snapshotValues memory start) := by
  obtain ⟨result, hrun, hspec⟩ := spec_returns _ _ (rust_capture_span_spec cells start width)
  obtain ⟨source, payload, hresult, hsource, hpayload⟩ := hspec.1 hspan
  subst result
  have hslice : payload.slice = snapshotPayload source := by
    apply Slice.ext
    simpa only [snapshotPayload, Slice.from_val, alloc.vec.Vec.val] using hpayload
  refine ⟨payload, hrun, ?_, ?_⟩
  · simp only [hpayload, List.length_map, hsource.1]
  · rw [hslice]
    exact capturedSpan_payload_sound sem memory cells source start width.val hstate hsource
      hspan.no_wrap

theorem rust_capture_span_rejects_invalid (cells : Slice Cell) (start : Key) (width : Usize)
    (hspan : ¬ValidWriteSpan cells start width.val) :
    checked.capture_span cells start width = .ok none := by
  obtain ⟨result, hrun, hspec⟩ := spec_returns _ _ (rust_capture_span_spec cells start width)
  simpa only [hspec.2 hspan] using hrun

theorem rust_capture_span_zero (cells : Slice Cell) (start : Key) :
    ∃ payload : alloc.vec.Vec PayloadByte,
      checked.capture_span cells start 0#usize = .ok (some payload) ∧ payload.val = [] := by
  have hspan : ValidWriteSpan cells start (0#usize).val := by
    constructor
    · scalar_tac
    · intro index hindex
      scalar_tac
  obtain ⟨result, hrun, hspec⟩ := spec_returns _ _ (rust_capture_span_spec cells start 0#usize)
  obtain ⟨source, payload, hresult, hsource, hpayload⟩ := hspec.1 hspan
  subst result
  have hempty : source.val = [] := List.eq_nil_of_length_eq_zero hsource.1
  exact ⟨payload, hrun, by simpa only [hempty, List.map_nil] using hpayload⟩

theorem rust_capture_span_frozen_sound {Input : Type} (sem : Semantics Input)
    (issueMemory : Memory Input) (cells : Slice Cell) (start : Key) (width : Usize)
    (payload : alloc.vec.Vec PayloadByte) (hstate : Represents sem issueMemory cells)
    (hspan : ValidWriteSpan cells start width.val)
    (hread : checked.capture_span cells start width = .ok (some payload)) :
    payload.val.length = width.val ∧
      PayloadSound sem payload.slice (snapshotValues issueMemory start) := by
  obtain ⟨snapshot, hrun, hlength, hsound⟩ :=
    rust_capture_span_sound sem issueMemory cells start width hstate hspan
  have heq : snapshot = payload := by
    simpa only [hread, Result.ok.injEq, Option.some.injEq] using hrun.symm
  simpa only [heq] using And.intro hlength hsound

/-- Completion retains the issue snapshot even if the source changed or overlaps the destination. -/
theorem complete_rust_capture_preserves_representation {Input : Type}
    (sem : Semantics Input) (issueMemory completionMemory : Memory Input)
    (source state : Slice Cell) (start write : Key) (width : Usize)
    (payload : alloc.vec.Vec PayloadByte) (hsource : Represents sem issueMemory source)
    (hsourceSpan : ValidWriteSpan source start width.val)
    (hread : checked.capture_span source start width = .ok (some payload))
    (hstate : Represents sem completionMemory state)
    (hspan : ValidWriteSpan state write width.val) :
    ∃ next,
      apply_write_effect state write payload.slice = .ok next ∧
      WriteSpec state next write payload.slice ∧
      Represents sem (storeBytes completionMemory write width.val
        (snapshotValues issueMemory start)) next ∧
      ValidWriteSpan next write width.val ∧
      next.val.map Cell.key = state.val.map Cell.key := by
  obtain ⟨hlength, hpayload⟩ :=
    rust_capture_span_frozen_sound sem issueMemory source start width payload hsource hsourceSpan hread
  have hsliceLength : payload.slice.val.length = width.val := hlength
  simpa only [hsliceLength] using
    apply_write_effect_preserves_representation sem completionMemory state write payload.slice
      (snapshotValues issueMemory start) hstate (by simpa only [hsliceLength] using hspan) hpayload

end MemoryWrite
