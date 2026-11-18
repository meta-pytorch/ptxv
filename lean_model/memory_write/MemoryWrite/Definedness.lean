/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Model

set_option autoImplicit false

namespace MemoryWrite.Definedness

def consumes (start width bit : Nat) : Prop := start ≤ bit ∧ bit < start + width

def selected (value start width : Nat) : Nat := (value >>> start) % 2 ^ width

def defined (mask start width : Nat) : Prop :=
  ∀ bit, consumes start width bit → mask.testBit bit = false

theorem selected_bit (value start width bit : Nat) :
    (selected value start width).testBit bit =
      (decide (bit < width) && value.testBit (start + bit)) := by
  simp [selected, Nat.testBit_mod_two_pow]

theorem selected_zero_iff_defined (mask start width : Nat) :
    selected mask start width = 0 ↔ defined mask start width := by
  constructor
  · intro h bit hb
    have ht := congrArg (fun value : Nat => value.testBit (bit - start)) h
    have hlt : bit - start < width := by unfold consumes at hb; omega
    have heq : start + (bit - start) = bit := by unfold consumes at hb; omega
    simpa [selected_bit, hlt, heq] using ht
  · intro h
    apply Nat.eq_of_testBit_eq
    intro bit
    rw [selected_bit]
    by_cases hb : bit < width
    · have hm := h (start + bit) (by unfold consumes; omega)
      simp [hb, hm]
    · simp [hb]

theorem selected_eq_of_consumed_bits (left right start width : Nat)
    (h : ∀ bit, consumes start width bit → left.testBit bit = right.testBit bit) :
    selected left start width = selected right start width := by
  apply Nat.eq_of_testBit_eq
  intro bit
  rw [selected_bit, selected_bit]
  by_cases hb : bit < width
  · rw [h (start + bit) (by unfold consumes; omega)]
  · simp [hb]

theorem defined_eq_of_consumed_bits (left right start width : Nat)
    (h : ∀ bit, consumes start width bit → left.testBit bit = right.testBit bit) :
    defined left start width ↔ defined right start width := by
  rw [←selected_zero_iff_defined, ←selected_zero_iff_defined,
    selected_eq_of_consumed_bits left right start width h]

def consistent (captured mask actual : Nat) : Prop :=
  ∀ bit, mask.testBit bit = false → actual.testBit bit = captured.testBit bit

theorem defined_selection_agrees (captured mask actual start width : Nat)
    (hc : consistent captured mask actual) (hd : defined mask start width) :
    selected actual start width = selected captured start width := by
  apply selected_eq_of_consumed_bits
  intro bit hb
  exact hc bit (hd bit hb)

theorem conversion_of_defined_bits {Output : Type} (convert : Nat → Output)
    (captured mask actual start width : Nat)
    (hc : consistent captured mask actual) (hd : defined mask start width) :
    convert (selected actual start width) = convert (selected captured start width) := by
  rw [defined_selection_agrees captured mask actual start width hc hd]

def byteUndefined (mask byte : Nat) : Bool := decide (selected mask (8 * byte) 8 ≠ 0)

theorem byte_defined_iff (mask byte : Nat) :
    byteUndefined mask byte = false ↔ defined mask (8 * byte) 8 := by
  simp [byteUndefined, selected_zero_iff_defined]

theorem byte_ignores_other_source_bits (left right byte : Nat)
    (h : ∀ bit, consumes (8 * byte) 8 bit → left.testBit bit = right.testBit bit) :
    byteUndefined left byte = byteUndefined right byte := by
  unfold byteUndefined
  rw [selected_eq_of_consumed_bits left right (8 * byte) 8 h]

def byteMask (mask : Nat) : Nat → Nat
  | 0 => 0
  | count + 1 =>
    if byteUndefined mask count then byteMask mask count ||| 2 ^ count else byteMask mask count

@[simp]
theorem byte_mask_bit (mask count bit : Nat) :
    (byteMask mask count).testBit bit = (decide (bit < count) && byteUndefined mask bit) := by
  induction count with
  | zero => simp [byteMask]
  | succ count ih =>
    unfold byteMask
    split
    · rename_i hu
      rw [Nat.testBit_or, ih, Nat.testBit_two_pow]
      by_cases hb : bit = count
      · subst bit
        simp [hu]
      · have hc : (bit < count + 1) ↔ bit < count := by omega
        simp [Ne.symm hb, hc]
    · rename_i hu
      rw [ih]
      by_cases hb : bit = count
      · subst bit
        simp_all
      · have hc : (bit < count + 1) ↔ bit < count := by omega
        simp [hc]

theorem byte_mask_bound (mask count : Nat) : byteMask mask count < 2 ^ count := by
  apply Nat.lt_pow_two_of_testBit
  intro bit hb
  simp [show ¬bit < count by omega]

theorem byte_mask_locality (left right count byte : Nat)
    (h : ∀ bit, consumes (8 * byte) 8 bit → left.testBit bit = right.testBit bit) :
    (byteMask left count).testBit byte = (byteMask right count).testBit byte := by
  rw [byte_mask_bit, byte_mask_bit, byte_ignores_other_source_bits left right byte h]

def mergeByteMask (previous source first count : Nat) : Nat :=
  if first < 64 then
    previous ||| ((byteMask source (min count 8) <<< first) % 2 ^ 64)
  else previous

abbrev suppliesByte (first count byte : Nat) : Prop :=
  first < 64 ∧ first ≤ byte ∧ byte < 64 ∧ byte - first < count ∧ byte - first < 8

def placedUndefined (source first count byte : Nat) : Bool :=
  decide (suppliesByte first count byte) && byteUndefined source (byte - first)

theorem merge_byte_mask_bit (previous source first count byte : Nat) :
    (mergeByteMask previous source first count).testBit byte =
      (previous.testBit byte || placedUndefined source first count byte) := by
  by_cases hf : first < 64
  · simp only [mergeByteMask, hf, ite_true, Nat.testBit_or,
      Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, byte_mask_bit]
    by_cases hb : byte < 64 <;> by_cases hl : first ≤ byte <;>
      by_cases hc : byte - first < count <;> by_cases h8 : byte - first < 8 <;>
      simp [placedUndefined, suppliesByte, hf, hb, hl, hc, h8]
  · simp [mergeByteMask, hf, placedUndefined, suppliesByte]

theorem merge_byte_mask_frame (previous source first count byte : Nat)
    (outside : ¬suppliesByte first count byte) :
    (mergeByteMask previous source first count).testBit byte = previous.testBit byte := by
  simp [merge_byte_mask_bit, placedUndefined, outside]

structure RegisterBytes where
  mask : Nat
  first : Nat
  count : Nat

def combineByteMasks (previous : Nat) (sources : List RegisterBytes) : Nat :=
  sources.foldl (fun result source =>
    mergeByteMask result source.mask source.first source.count) previous

theorem combine_byte_masks_bit (previous : Nat) (sources : List RegisterBytes) (byte : Nat) :
    (combineByteMasks previous sources).testBit byte =
      (previous.testBit byte || sources.any (fun source =>
        placedUndefined source.mask source.first source.count byte)) := by
  induction sources generalizing previous with
  | nil => simp [combineByteMasks]
  | cons source sources ih =>
    simp only [combineByteMasks, List.foldl_cons] at *
    rw [ih, merge_byte_mask_bit]
    simp [Bool.or_assoc]

theorem vector_store_definedness_is_local (sources : List RegisterBytes) (byte : Nat) :
    (combineByteMasks 0 sources).testBit byte = true ↔
      ∃ source ∈ sources, suppliesByte source.first source.count byte ∧
        byteUndefined source.mask (byte - source.first) = true := by
  simp [combine_byte_masks_bit, placedUndefined, Bool.and_eq_true]

def writtenUndefined (source previous unordered : Bool) : Bool :=
  source || unordered && previous

theorem ordered_write_exact (source previous : Bool) :
    writtenUndefined source previous false = source := by
  simp [writtenUndefined]

theorem unordered_write_retains_possible_undefinedness (source previous : Bool) :
    writtenUndefined source previous true = (source || previous) := by
  rfl

end MemoryWrite.Definedness
