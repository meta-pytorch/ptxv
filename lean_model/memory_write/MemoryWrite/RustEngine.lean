/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Commutation
import MemoryWrite.RustRefinement

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite

def markerPrefix (start : Key) (count : Nat) : List Key :=
  (List.range count).map (Storage.offsetKey start)

@[simp] theorem markerPrefix_length (start : Key) (count : Nat) :
    (markerPrefix start count).length = count := by simp [markerPrefix]

@[simp] theorem markerPrefix_zero (start : Key) : markerPrefix start 0 = [] := rfl

theorem markerPrefix_succ (start : Key) (count : Nat) :
    markerPrefix start (count + 1) = markerPrefix start count ++ [Storage.offsetKey start count] := by
  simp [markerPrefix, List.range_succ, List.map_append]

theorem markerPrefix_spanKeys (start : Key) (width : Usize) :
    markerPrefix start width.val = Storage.spanKeys start width := by
  apply List.ext_getElem
  · simp [Storage.spanKeys]
  · intro index hleft hright
    simp [markerPrefix, Storage.spanKeys]

def markerAppend (original : List Key) (enabled : Bool) (additions : List Key) : List Key :=
  original ++ if enabled then additions else []

@[simp] theorem markerAppend_length (original additions : List Key) (enabled : Bool) :
    (markerAppend original enabled additions).length =
      original.length + if enabled then additions.length else 0 := by
  cases enabled <;> simp [markerAppend]

theorem markerAppend_succ (original additions : List Key) (enabled : Bool) (key : Key) :
    markerAppend original enabled (additions ++ [key]) =
      markerAppend original enabled additions ++ if enabled then [key] else [] := by
  cases enabled <;> simp [markerAppend, List.append_assoc]

@[step]
theorem rust_record_write_loop_spec (numeric half : alloc.vec.Vec Key)
    (trackNumeric trackHalf : Bool) (start : Key) (width : Usize)
    (hnumeric : numeric.val.length + width.val ≤ Usize.max)
    (hhalf : half.val.length + width.val ≤ Usize.max) :
    engine.record_write_loop numeric half trackNumeric trackHalf
      start.space start.allocation start.offset width 0#usize
      ⦃ result =>
        result.1.val = markerAppend numeric.val trackNumeric ((markerPrefix start width.val).map alignedWordKey) ∧
        result.2.val = markerAppend half.val trackHalf (markerPrefix start width.val) ⦄ := by
  unfold engine.record_write_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec Key × alloc.vec.Vec Key × Usize => width.val - current.2.2.val)
    (fun current : alloc.vec.Vec Key × alloc.vec.Vec Key × Usize =>
      current.2.2.val ≤ width.val ∧
      current.1.val = markerAppend numeric.val trackNumeric
        ((markerPrefix start current.2.2.val).map alignedWordKey) ∧
      current.2.1.val = markerAppend half.val trackHalf (markerPrefix start current.2.2.val))
  · intro current hcurrent
    rcases current with ⟨currentNumeric, currentHalf, index⟩
    obtain ⟨hindex, hnumericPrefix, hhalfPrefix⟩ := hcurrent
    dsimp only at hindex hnumericPrefix hhalfPrefix
    unfold engine.record_write_loop.body
    by_cases hlt : index.val < width.val
    · have htest : index < width := by scalar_tac
      simp only [htest, ite_true]
      step with UScalar.cast_inBounds_spec .U64 index (by have := Usize.bounds_eq; scalar_tac) as ⟨index64, hindex64⟩
      simp only [lift, bind_ok]
      let offset := core.num.U64.wrapping_add start.offset index64
      have hrawkey : (⟨start.space, start.allocation, offset⟩ : Key) = Storage.offsetKey start index.val := by
        congr 1
        apply UScalar.eq_of_val_eq
        simp [offset, hindex64, U64.size, U64.max]
      have hnumericStep :
          (if trackNumeric then do
            let quotient ← offset / 4#u64
            let aligned ← quotient * 4#u64
            alloc.vec.Vec.push currentNumeric (⟨start.space, start.allocation, aligned⟩ : Key)
          else ok currentNumeric)
          ⦃ updated => updated.val = currentNumeric.val ++
            if trackNumeric then [alignedWordKey (Storage.offsetKey start index.val)] else [] ⦄ := by
        cases htrack : trackNumeric with
        | false => simp
        | true =>
          simp only [ite_true]
          step as ⟨quotient, hquotient⟩
          step as ⟨aligned, haligned⟩ ; try scalar_tac
          have hkey : (⟨start.space, start.allocation, aligned⟩ : Key) =
              alignedWordKey (Storage.offsetKey start index.val) := by
            rw [←hrawkey]
            simp only [alignedWordKey]
            congr 1
            apply UScalar.eq_of_val_eq
            scalar_tac
          step with alloc.vec.Vec.push_spec currentNumeric
            (⟨start.space, start.allocation, aligned⟩ : Key) (by
              rw [hnumericPrefix]
              simp only [markerAppend_length, htrack, ↓reduceIte, List.length_map, markerPrefix_length]
              omega) as ⟨updated, hupdated⟩
          rw [hkey] at hupdated
          exact hupdated
      step with hnumericStep as ⟨updatedNumeric, hupdatedNumeric⟩
      have hhalfStep :
          (if trackHalf then alloc.vec.Vec.push currentHalf (⟨start.space, start.allocation, offset⟩ : Key)
          else ok currentHalf)
          ⦃ updated => updated.val = currentHalf.val ++
            if trackHalf then [Storage.offsetKey start index.val] else [] ⦄ := by
        cases htrack : trackHalf with
        | false => simp
        | true =>
          simp only [ite_true]
          step with alloc.vec.Vec.push_spec currentHalf
            (⟨start.space, start.allocation, offset⟩ : Key) (by
              rw [hhalfPrefix]
              simp only [markerAppend_length, htrack, ↓reduceIte, markerPrefix_length]
              omega) as ⟨updated, hupdated⟩
          rw [hrawkey] at hupdated
          exact hupdated
      step with hhalfStep as ⟨updatedHalf, hupdatedHalf⟩
      step as ⟨next, hnext⟩
      have hnextVal : next.val = index.val + 1 := by scalar_tac
      refine ⟨?_, ?_, ?_, ?_⟩
      · omega
      · rw [hupdatedNumeric, hnumericPrefix, hnextVal, markerPrefix_succ, List.map_append]
        simpa using (markerAppend_succ numeric.val
          ((markerPrefix start index.val).map alignedWordKey) trackNumeric
          (alignedWordKey (Storage.offsetKey start index.val))).symm
      · rw [hupdatedHalf, hhalfPrefix, hnextVal, markerPrefix_succ]
        exact (markerAppend_succ half.val (markerPrefix start index.val) trackHalf
          (Storage.offsetKey start index.val)).symm
      · omega
    · have htest : ¬index < width := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have hfinish : index.val = width.val := by omega
      simpa [hfinish] using And.intro hnumericPrefix hhalfPrefix
  · simp [markerAppend]

@[step]
theorem rust_record_write_spec (numeric half : alloc.vec.Vec Key)
    (trackNumeric trackHalf : Bool) (start : Key) (width : Usize)
    (hnumeric : numeric.val.length + width.val ≤ Usize.max)
    (hhalf : half.val.length + width.val ≤ Usize.max) :
    engine.record_write numeric half trackNumeric trackHalf start width
      ⦃ result =>
        result.1.val = (if start.space = 1#u8 then
          markerAppend numeric.val trackNumeric (numericTouchedKeys start width) else numeric.val) ∧
        result.2.val = (if start.space = 1#u8 then
          markerAppend half.val trackHalf (Storage.spanKeys start width) else half.val) ⦄ := by
  unfold engine.record_write
  split
  · rename_i hspace
    have hnot : start.space ≠ 1#u8 := by simpa using hspace
    simp [hnot]
  · rename_i hspace
    have hglobal : start.space = 1#u8 := by scalar_tac
    step with rust_record_write_loop_spec numeric half trackNumeric trackHalf start width hnumeric hhalf
      as ⟨result, hnumericResult, hhalfResult⟩
    simpa [hglobal, numericTouchedKeys, markerPrefix_spanKeys] using And.intro hnumericResult hhalfResult

@[step]
theorem rust_record_write_membership (numeric half : alloc.vec.Vec Key)
    (trackNumeric trackHalf : Bool) (effect : WriteEffect) (henabled : effect.enabled = true)
    (hnumeric : numeric.val.length + (Slice.len effect.payload).val ≤ Usize.max)
    (hhalf : half.val.length + (Slice.len effect.payload).val ≤ Usize.max) :
    engine.record_write numeric half trackNumeric trackHalf effect.destination (Slice.len effect.payload)
      ⦃ result => Commutation.MarkerEq
        { numericWritten := result.1.val, halfWritten := result.2.val }
        (({ numericWritten := numeric.val, halfWritten := half.val } : WrittenMarkers).record
          { numericEnabled := trackNumeric, halfEnabled := trackHalf } effect) ⦄ := by
  step with rust_record_write_spec numeric half trackNumeric trackHalf effect.destination
    (Slice.len effect.payload) hnumeric hhalf as ⟨result, hnumericResult, hhalfResult⟩
  cases hn : trackNumeric <;> cases hh : trackHalf <;>
    by_cases hspace : effect.destination.space = 1#u8 <;>
    simp [Commutation.MarkerEq, WrittenMarkers.record, henabled, hspace, hn, hh,
      hnumericResult, hhalfResult, markerAppend, List.mem_append, or_comm]

def rustMarkerView (state : engine.Engine) : WrittenMarkers :=
  { numericWritten := state.numeric_written.val, halfWritten := state.half_written.val }

def rustTrackingView (state : engine.Engine) : WrittenTracking :=
  { numericEnabled := state.track_numeric, halfEnabled := state.track_half }

def RustEngineRefines (before : Storage) (state : engine.Engine) (write : checked.PreparedWrite)
    (result : core.result.Result Unit checked.EffectError × engine.Engine) : Prop :=
  RustWriteRefines before (rustEffect write) (result.1, result.2.cells.slice) ∧
    rustTrackingView result.2 = rustTrackingView state ∧
    match result.1 with
    | .Err _ => rustMarkerView result.2 = rustMarkerView state
    | .Ok _ => Commutation.MarkerEq (rustMarkerView result.2)
        ((rustMarkerView state).record (rustTrackingView state) (rustEffect write))

@[step]
theorem rust_engine_apply_refines (before : Storage) (state : engine.Engine)
    (write : checked.PreparedWrite) (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max) :
    engine.apply state write ⦃ result => RustEngineRefines before state write result ⦄ := by
  unfold engine.apply
  simp only [alloc.vec.Vec.deref_mut, lift, bind_ok]
  rw [hcells]
  apply WP.spec_bind (rust_checked_write_refines before write)
  rintro ⟨status, cells⟩ hchecked
  cases status with
  | Ok value =>
    cases value
    cases henabled : write.enabled with
    | true =>
      simp only [ite_true]
      have hspec := rust_record_write_membership state.numeric_written state.half_written
        state.track_numeric state.track_half (rustEffect write) henabled
        (by simpa [rustEffect, alloc.vec.Vec.deref] using hnumeric)
        (by simpa [rustEffect, alloc.vec.Vec.deref] using hhalf)
      simp only [rustEffect, rust_payload_len] at hspec
      apply WP.spec_bind hspec
      rintro ⟨nextNumeric, nextHalf⟩ hmarkers
      change (ok (core.result.Result.Ok (),
        { state with cells := ⟨cells⟩, numeric_written := nextNumeric, half_written := nextHalf }))
        ⦃ result => RustEngineRefines before state write result ⦄
      simp only [WP.spec_ok]
      exact ⟨hchecked, rfl, hmarkers⟩
    | false =>
      simp only [Bool.false_eq_true, ite_false]
      change (ok (core.result.Result.Ok (), { state with cells := ⟨cells⟩ }))
        ⦃ result => RustEngineRefines before state write result ⦄
      simp only [WP.spec_ok]
      refine ⟨hchecked, rfl, ?_⟩
      simp [rustMarkerView, Commutation.MarkerEq, WrittenMarkers.record, rustEffect, henabled]
  | Err error =>
    change (ok (core.result.Result.Err error, { state with cells := ⟨cells⟩ }))
      ⦃ result => RustEngineRefines before state write result ⦄
    simp only [WP.spec_ok]
    exact ⟨hchecked, rfl, rfl⟩

theorem rust_engine_apply_progress (before : Storage) (state : engine.Engine)
    (write : checked.PreparedWrite) (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (haddress : write.address_dependent = false) (hguard : write.guard_dependent = false)
    (henabled : write.enabled = true)
    (hfamily : (rustFamily write.class).allows write.destination.space = true)
    (hspan : Storage.SpanFits before write.destination (alloc.vec.Vec.len write.payload)) :
    ∃ next, engine.apply state write = .ok (.Ok (), next) ∧
      RustEngineRefines before state write (.Ok (), next) := by
  have hmodel : applyEffect before (rustEffect write) =
      .ok ((before.write write.destination (alloc.vec.Vec.deref write.payload)).installWords
        (rustEffect write).words) := by
    simp [applyEffect, rustEffect, Storage.installWords, haddress, hguard, henabled, hfamily, hspan]
  obtain ⟨⟨status, next⟩, hrun, hrefines⟩ := spec_returns _ _
    (rust_engine_apply_refines before state write hcells hnumeric hhalf)
  cases status with
  | Err error =>
    have herror : applyEffect before (rustEffect write) = .error (rustError error) :=
      (show applyEffect before (rustEffect write) = .error (rustError error) ∧
        next.cells.slice = before.toCells from hrefines.1).1
    rw [hmodel] at herror
    contradiction
  | Ok value =>
    cases value
    exact ⟨next, hrun, hrefines⟩

theorem rust_engine_error_unchanged (before : Storage) (state : engine.Engine)
    (write : checked.PreparedWrite)
    (result : core.result.Result Unit checked.EffectError × engine.Engine)
    (hcells : state.cells.slice = before.toCells)
    (hrefines : RustEngineRefines before state write result) (error : checked.EffectError)
    (hstatus : result.1 = .Err error) : result.2 = state := by
  have hslice : result.2.cells.slice = before.toCells := (show
    applyEffect before (rustEffect write) = .error (rustError error) ∧
      result.2.cells.slice = before.toCells from by
    simpa [RustWriteRefines, hstatus] using hrefines.1).2
  have hmarkers : rustMarkerView result.2 = rustMarkerView state := by
    simpa [hstatus] using hrefines.2.2
  have htracking := hrefines.2.1
  rcases result with ⟨status, next⟩
  cases state
  cases next
  simp_all [rustMarkerView, rustTrackingView, alloc.vec.Vec.eq_iff, alloc.vec.Vec.val]

/-- Includes Rust termination, checked rejection, all live value/history planes, and both marker sets. -/
theorem rust_engine_apply_preserves_representation {Input : Type}
    (sem : Semantics Input) (memory : Memory Input) (before : Storage) (state : engine.Engine)
    (write : checked.PreparedWrite) (values : Input → Nat → Byte)
    (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + write.payload.val.length ≤ Usize.max)
    (hhalf : state.half_written.val.length + write.payload.val.length ≤ Usize.max)
    (hstate : Storage.Represents sem memory before)
    (hsound : EffectSound sem memory (rustEffect write) values) :
    ∃ result, engine.apply state write = .ok result ∧ RustEngineRefines before state write result ∧
      match result.1 with
      | .Err _ => result.2 = state
      | .Ok _ => Represents sem (effectMemory memory (rustEffect write) values) result.2.cells.slice := by
  obtain ⟨result, hrun, hrefines⟩ := spec_returns _ _
    (rust_engine_apply_refines before state write hcells hnumeric hhalf)
  refine ⟨result, hrun, hrefines, ?_⟩
  cases hstatus : result.1 with
  | Err error => exact rust_engine_error_unchanged before state write result hcells hrefines error hstatus
  | Ok value =>
    obtain ⟨after, hafter, hresult⟩ := (show ∃ after,
      applyEffect before (rustEffect write) = .ok after ∧ result.2.cells.slice = after.toCells from by
      simpa [RustWriteRefines, hstatus] using hrefines.1)
    rw [hresult]
    exact (applyEffect_preserves_representation sem memory before after (rustEffect write)
      values hstate hsound hafter).1

end MemoryWrite
