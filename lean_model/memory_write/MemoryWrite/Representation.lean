/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Model

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite

structure Address where
  space : U8
  allocation : U64
  offset : Nat
  deriving DecidableEq

def location (key : Key) (lane : Nat := 0) : Address :=
  ⟨key.space, key.allocation, key.offset.val + lane⟩

structure Byte where
  bits : U8
  defined : Bool
  deriving DecidableEq

abbrev Memory (Input : Type) := Input → Address → Byte

/-- Histories denote rounded bits, including signed zeros and NaN encodings. -/
structure Semantics (Input : Type) where
  domain : Input → Prop
  capture : Input
  capture_valid : domain capture
  origin : Origin → Input → U8
  zero_origin : ∀ input, domain input → origin .Zero input = 0#u8
  word : U32 → Input → Fin 4 → U8
  initial : Memory Input

structure DatumSound {Input : Type} (sem : Semantics Input) (values : Input → Byte)
    (value : U8) (origin : Option Origin) (tainted defined : Bool) : Prop where
  captured : value = (values sem.capture).bits
  defined_sound : defined = true → ∀ input, sem.domain input → (values input).defined = true
  origin_sound : ∀ id, origin = some id → ∀ input, sem.domain input →
    (values input).defined = true ∧ (values input).bits = sem.origin id input
  constant : tainted = false → ∀ input, sem.domain input →
    (values input).bits = (values sem.capture).bits

structure CellRepresents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (cell : Cell) : Prop where
  byte : DatumSound sem (fun input => memory input (location cell.key))
    cell.value cell.origin cell.tainted cell.defined
  word : ∀ id, cell.word = some (.Known id) → ∀ input, sem.domain input → ∀ lane : Fin 4,
    (memory input (location cell.key lane.val)).defined = true ∧
    (memory input (location cell.key lane.val)).bits = sem.word id input lane
  initial_fallback : cell.written = false → ∀ input, sem.domain input →
    memory input (location cell.key) = sem.initial input (location cell.key)

def Represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (cells : Slice Cell) : Prop :=
  ∀ cell ∈ cells.val, CellRepresents sem memory cell

def InWrite (write : Key) (width : Nat) (address : Address) : Prop :=
  write.space = address.space ∧ write.allocation = address.allocation ∧
    write.offset.val ≤ address.offset ∧ address.offset < write.offset.val + width

instance (write : Key) (width : Nat) (address : Address) : Decidable (InWrite write width address) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def storeBytes {Input : Type} (memory : Memory Input) (write : Key) (width : Nat)
    (values : Input → Nat → Byte) : Memory Input :=
  fun input address => if InWrite write width address
    then values input (address.offset - write.offset.val) else memory input address

structure ValidWriteSpan (cells : Slice Cell) (write : Key) (width : Nat) : Prop where
  no_wrap : write.offset.val + width ≤ U64.max + 1
  coverage : ∀ index, index < width →
    ∃ cell ∈ cells.val, location cell.key = location write index

/-- The payload is frozen before the write; its histories include control/address dependencies. -/
structure PayloadSound {Input : Type} (sem : Semantics Input)
    (payload : Slice PayloadByte) (values : Input → Nat → Byte) : Prop where
  bytes : ∀ index, index < payload.val.length →
    let byte := payloadAt payload index
    DatumSound sem (fun input => values input index)
      byte.value byte.origin byte.tainted byte.defined
  words : ∀ index, index + 4 ≤ payload.val.length → ∀ id,
    (payloadAt payload index).word = some (.Known id) → ∀ input, sem.domain input → ∀ lane : Fin 4,
      (values input (index + lane.val)).defined = true ∧
      (values input (index + lane.val)).bits = sem.word id input lane

theorem inWrite_location (write key : Key) (width : Nat) :
    InWrite write width (location key) ↔ Touches write width key := by
  simp [InWrite, location, Touches, SameRegion, and_assoc]

theorem covered_lane (write key : Key) (width : Nat)
    (h : CoversWord write width key) (lane : Fin 4) :
    InWrite write width (location key lane.val) := by
  obtain ⟨⟨hs, ha⟩, hlo, hhi⟩ := h
  exact ⟨hs, ha, by dsimp [location]; omega, by dsimp [location]; omega⟩

theorem disjoint_lane (write key : Key) (width : Nat)
    (h : ¬Overlaps write width key) (lane : Fin 4) :
    ¬InWrite write width (location key lane.val) := by
  intro ⟨hs, ha, hlo, hhi⟩
  apply h
  refine ⟨⟨hs, ha⟩, ?_, ?_, ?_⟩ <;> dsimp [location] at * <;> omega

theorem covers_touches (write key : Key) (width : Nat)
    (h : CoversWord write width key) : Touches write width key := by
  obtain ⟨hr, hlo, hhi⟩ := h
  exact ⟨hr, hlo, by omega⟩

theorem modelCell_represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (cell : Cell) (write : Key) (payload : Slice PayloadByte) (values : Input → Nat → Byte)
    (hcell : CellRepresents sem memory cell) (hpayload : PayloadSound sem payload values) :
    CellRepresents sem (storeBytes memory write payload.val.length values)
      (modelCell cell write payload) := by
  have hkey := modelCell_key cell write payload
  constructor
  · by_cases ht : Touches write payload.val.length cell.key
    · have hi : cell.key.offset.val - write.offset.val < payload.val.length := by
        obtain ⟨_, hlo, hhi⟩ := ht
        omega
      have hin := (inWrite_location write cell.key payload.val.length).mpr ht
      simp only [modelCell, ite_eq_left ht, storeBytes, ite_eq_left hin]
      simpa only [location, Nat.add_zero] using hpayload.bytes (cell.key.offset.val - write.offset.val) hi
    · have hout := mt (inWrite_location write cell.key payload.val.length).mp ht
      simpa [modelCell, ht, storeBytes, hout] using hcell.byte
  · intro id hid input hinput lane
    rw [hkey]
    have hword := (modelCell_spec cell write payload).word_effect
    rw [hword] at hid
    unfold wordAfter at hid
    split at hid
    · rename_i hfull
      have hin := covered_lane write cell.key payload.val.length hfull lane
      have hindex : cell.key.offset.val - write.offset.val + 4 ≤ payload.val.length := by
        obtain ⟨_, hlo, hhi⟩ := hfull
        omega
      have hoff : cell.key.offset.val + lane.val - write.offset.val =
          (cell.key.offset.val - write.offset.val) + lane.val := by
        obtain ⟨_, hlo, _⟩ := hfull
        omega
      simp only [storeBytes, ite_eq_left hin]
      change (values input (cell.key.offset.val + lane.val - write.offset.val)).defined = true ∧
        (values input (cell.key.offset.val + lane.val - write.offset.val)).bits = sem.word id input lane
      rw [hoff]
      exact hpayload.words _ hindex id hid input hinput lane
    · split at hid
      · contradiction
      · rename_i hdisjoint
        have hout := disjoint_lane write cell.key payload.val.length hdisjoint lane
        simpa [storeBytes, hout] using hcell.word id hid input hinput lane
  · intro hnot input hinput
    rw [hkey]
    by_cases ht : Touches write payload.val.length cell.key
    · simp [modelCell, ht] at hnot
    · have hout := mt (inWrite_location write cell.key payload.val.length).mp ht
      have hbefore : cell.written = false := by simpa [modelCell, ht] using hnot
      simpa [storeBytes, hout] using hcell.initial_fallback hbefore input hinput

end MemoryWrite
