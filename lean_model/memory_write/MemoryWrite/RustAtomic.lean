/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Theorem

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite

def AtomicHistoryMember (effect : atomic.AtomicFootprint) (word : U64) : Prop :=
  effect.history_start.val ≤ word.val ∧ word.val < effect.history_end.val

def AtomicMarkerMember (effect : atomic.AtomicFootprint) (word : U64) : Prop :=
  effect.marker_start.val ≤ word.val ∧ word.val < effect.marker_end.val ∧
    (word.val - effect.marker_start.val) % effect.marker_stride.val = 0

instance (effect : atomic.AtomicFootprint) (word : U64) :
    Decidable (AtomicHistoryMember effect word) := inferInstanceAs (Decidable (_ ∧ _))

instance (effect : atomic.AtomicFootprint) (word : U64) :
    Decidable (AtomicMarkerMember effect word) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

structure AtomicFootprintBounds (offset : U64) (bytes : U8) (didWrite : Bool)
    (effect : atomic.AtomicFootprint) : Prop where
  historyStart : effect.history_start.val =
    if didWrite = true ∧ bytes.val > 0 then offset.val - 3 else offset.val
  historyEnd : effect.history_end.val =
    if didWrite = true then min (offset.val + bytes.val) (U64.max + 1) else offset.val + 1
  markerStart : effect.marker_start.val =
    if didWrite = true ∧ bytes.val > 0 then offset.val / 4 * 4 else offset.val
  markerEnd : effect.marker_end = effect.history_end
  markerStride : effect.marker_stride.val = if didWrite = true then 4 else 1

@[step]
theorem rust_atomic_footprint_bounds (offset : U64) (bytes : U8) (didWrite : Bool) :
    atomic.atomic_footprint offset bytes didWrite
      ⦃ effect => AtomicFootprintBounds offset bytes didWrite effect ⦄ := by
  unfold atomic.atomic_footprint
  cases hdid : didWrite with
  | false =>
    simp only [Bool.false_eq_true, ite_false]
    step with UScalar.cast_inBounds_spec .U128 offset (by scalar_tac) as ⟨first, hfirst⟩
    step as ⟨stop, hstop⟩; try scalar_tac
    step with UScalar.cast_inBounds_spec .U128 offset (by scalar_tac) as ⟨second, hsecond⟩
    step as ⟨stop2, hstop2⟩; try scalar_tac
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simp
    · simp only [Bool.false_eq_true, ite_false]
      scalar_tac
    · simp
    · apply UScalar.eq_of_val_eq
      scalar_tac
    · simp
  | true =>
    simp only [ite_true]
    split
    · rename_i hzero
      step with UScalar.cast_inBounds_spec .U128 offset (by scalar_tac) as ⟨first, hfirst⟩
      step with UScalar.cast_inBounds_spec .U128 offset (by scalar_tac) as ⟨second, hsecond⟩
      have hz : bytes.val = 0 := by scalar_tac
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · simp [hz]
      · simp only [hz, Nat.add_zero, ↓reduceIte]
        rw [hfirst]
        exact (Nat.min_eq_left (by scalar_tac)).symm
      · simp [hz]
      · apply UScalar.eq_of_val_eq
        scalar_tac
      · simp
    · rename_i hnonzero
      have hpositive : bytes.val > 0 := by scalar_tac
      step with UScalar.cast_inBounds_spec .U128 offset (by scalar_tac) as ⟨start, hstart⟩
      step with UScalar.cast_inBounds_spec .U128 bytes (by scalar_tac) as ⟨width, hwidth⟩
      step as ⟨stop, hstop⟩; try scalar_tac
      step as ⟨limit, hlimit⟩; try scalar_tac
      have hlimitVal : limit.val = U64.max + 1 := by scalar_tac
      have hend :
          (if stop > limit then ok limit else ok stop)
            ⦃ result => result.val = min (offset.val + bytes.val) (U64.max + 1) ⦄ := by
        split <;> simp only [WP.spec_ok] <;> scalar_tac
      step with hend as ⟨last, hlast⟩
      have hbegin :
          (if offset < 3#u64 then ok 0#u64 else offset - 3#u64)
            ⦃ result => result.val = offset.val - 3 ⦄ := by
        split
        · simp only [WP.spec_ok]
          scalar_tac
        · step as ⟨first, hfirst⟩; scalar_tac
      step with hbegin as ⟨first, hfirst⟩
      step as ⟨quotient, hquotient⟩; try scalar_tac
      step as ⟨aligned, haligned⟩; try scalar_tac
      refine ⟨?_, ?_, ?_, rfl, ?_⟩
      · simpa [hpositive] using hfirst
      · simpa using hlast
      · simpa [hpositive, hquotient] using haligned
      · simp

theorem AtomicFootprintBounds.history (offset : U64) (bytes : U8) (didWrite : Bool)
    (effect : atomic.AtomicFootprint) (h : AtomicFootprintBounds offset bytes didWrite effect)
    (word : U64) :
    AtomicHistoryMember effect word ↔
      if didWrite = true then
        bytes.val > 0 ∧ word.val < offset.val + bytes.val ∧ offset.val < word.val + 4
      else word = offset := by
  obtain ⟨hstart, hend, _, _, _⟩ := h
  unfold AtomicHistoryMember
  rw [hstart, hend]
  cases didWrite <;> simp only [Bool.false_eq_true, false_and, ite_false, ite_true, true_and]
  · constructor
    · intro hw
      apply UScalar.eq_of_val_eq
      omega
    · intro hw
      subst word
      omega
  · have hword : word.val ≤ U64.max := by scalar_tac
    split <;> omega

theorem AtomicFootprintBounds.safe (offset : U64) (bytes : U8) (didWrite : Bool)
    (effect : atomic.AtomicFootprint) (h : AtomicFootprintBounds offset bytes didWrite effect) :
    effect.history_end.val ≤ U64.max + 1 ∧ effect.marker_end.val ≤ U64.max + 1 ∧
      (effect.marker_stride.val = 1 ∨ effect.marker_stride.val = 4) ∧
      0 < effect.marker_stride.val := by
  obtain ⟨_, hend, _, hsame, hstride⟩ := h
  have hoffset : offset.val ≤ U64.max := by scalar_tac
  have hendBound : effect.history_end.val ≤ U64.max + 1 := by
    rw [hend]
    cases didWrite <;> simp only [Bool.false_eq_true, ite_false, ite_true] <;> omega
  refine ⟨hendBound, ?_, ?_, ?_⟩
  · simpa only [hsame] using hendBound
  · rw [hstride]
    cases didWrite <;> decide
  · rw [hstride]
    cases didWrite <;> decide

theorem AtomicFootprintBounds.range_bound (offset : U64) (bytes : U8) (didWrite : Bool)
    (effect : atomic.AtomicFootprint) (h : AtomicFootprintBounds offset bytes didWrite effect)
    (point : Nat) (hpoint : point < effect.history_end.val ∨ point < effect.marker_end.val) :
    point ≤ U64.max := by
  have hsafety := h.safe offset bytes didWrite effect
  omega

theorem AtomicFootprintBounds.markers (offset : U64) (bytes : U8) (didWrite : Bool)
    (effect : atomic.AtomicFootprint) (h : AtomicFootprintBounds offset bytes didWrite effect)
    (word : U64) :
    AtomicMarkerMember effect word ↔
      if didWrite = true then
        bytes.val > 0 ∧ word.val % 4 = 0 ∧ word.val < offset.val + bytes.val ∧
          offset.val < word.val + 4
      else word = offset := by
  obtain ⟨_, hend, hstart, hsameEnd, hstride⟩ := h
  unfold AtomicMarkerMember
  rw [hsameEnd, hend, hstart, hstride]
  cases didWrite <;> simp only [Bool.false_eq_true, false_and, ite_false, ite_true, true_and]
  · constructor
    · intro hw
      apply UScalar.eq_of_val_eq
      omega
    · intro hw
      subst word
      simp
  · have hword : word.val ≤ U64.max := by scalar_tac
    split <;> omega

theorem AtomicFootprintBounds.overlaps (destination key : Key) (bytes : U8)
    (didWrite : Bool) (effect : atomic.AtomicFootprint)
    (h : AtomicFootprintBounds destination.offset bytes didWrite effect) :
    (SameRegion destination key ∧ AtomicHistoryMember effect key.offset) ↔
      if didWrite = true then Overlaps destination bytes.val key else key = destination := by
  rw [h.history destination.offset bytes didWrite effect key.offset]
  cases didWrite
  · simp only [Bool.false_eq_true, ite_false]
    cases destination with
    | mk ds da doff =>
      cases key with
      | mk ks ka koff => simp [SameRegion, Key.mk.injEq, eq_comm, and_comm, and_left_comm]
  · simp [Overlaps, Nat.pos_iff_ne_zero]

theorem AtomicFootprintBounds.wordAfter_unknown (destination : Key) (bytes : U8)
    (effect : atomic.AtomicFootprint)
    (h : AtomicFootprintBounds destination.offset bytes true effect)
    (cell : Cell) (payload : Slice PayloadByte)
    (hwidth : payload.val.length = bytes.val)
    (hunknown : ∀ index, (payloadAt payload index).word = none) :
    wordAfter cell destination payload =
      if SameRegion destination cell.key ∧ AtomicHistoryMember effect cell.key.offset
      then none else cell.word := by
  have hfootprint := h.overlaps destination cell.key bytes true effect
  simp only [ite_true] at hfootprint
  simp only [hfootprint]
  by_cases hcover : CoversWord destination payload.val.length cell.key
  · have hoverlap : Overlaps destination bytes.val cell.key := by
      obtain ⟨hregion, hstart, hend⟩ := hcover
      refine ⟨hregion, ?_, ?_, ?_⟩ <;> omega
    simp [MemoryWrite.wordAfter, hcover, hunknown, hoverlap]
  · have hcover' : ¬CoversWord destination bytes.val cell.key := by
      simpa only [hwidth] using hcover
    simp [MemoryWrite.wordAfter, hwidth, hcover']

theorem rust_atomic_footprint_spec (offset : U64) (bytes : U8) (didWrite : Bool) :
    ∃ effect, atomic.atomic_footprint offset bytes didWrite = .ok effect ∧
      AtomicFootprintBounds offset bytes didWrite effect ∧
      (∀ word, AtomicHistoryMember effect word ↔
        if didWrite = true then
          bytes.val > 0 ∧ word.val < offset.val + bytes.val ∧ offset.val < word.val + 4
        else word = offset) ∧
      (∀ word, AtomicMarkerMember effect word ↔
        if didWrite = true then
          bytes.val > 0 ∧ word.val % 4 = 0 ∧ word.val < offset.val + bytes.val ∧
            offset.val < word.val + 4
        else word = offset) := by
  obtain ⟨effect, hactual, hbounds⟩ := spec_returns _ _
    (rust_atomic_footprint_bounds offset bytes didWrite)
  exact ⟨effect, hactual, hbounds,
    hbounds.history offset bytes didWrite effect,
    hbounds.markers offset bytes didWrite effect⟩

end MemoryWrite
