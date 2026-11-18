/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Types

open Aeneas Aeneas.Std PtxvMemoryWrite

namespace MemoryWrite

deriving instance DecidableEq for Origin, WordHistory, Key, Cell, PayloadByte

def SameRegion (write key : Key) : Prop :=
  write.space = key.space ∧ write.allocation = key.allocation

def Touches (write : Key) (width : Nat) (key : Key) : Prop :=
  SameRegion write key ∧ write.offset.val ≤ key.offset.val ∧
    key.offset.val < write.offset.val + width

def Overlaps (write : Key) (width : Nat) (key : Key) : Prop :=
  SameRegion write key ∧ width ≠ 0 ∧ key.offset.val < write.offset.val + width ∧
    write.offset.val < key.offset.val + 4

def CoversWord (write : Key) (width : Nat) (key : Key) : Prop :=
  SameRegion write key ∧ write.offset.val ≤ key.offset.val ∧
    key.offset.val + 4 ≤ write.offset.val + width

instance (write key : Key) : Decidable (SameRegion write key) :=
  inferInstanceAs (Decidable (_ ∧ _))

instance (write key : Key) (width : Nat) : Decidable (Touches write width key) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

instance (write key : Key) (width : Nat) : Decidable (Overlaps write width key) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

instance (write key : Key) (width : Nat) : Decidable (CoversWord write width key) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def payloadAt (payload : Slice PayloadByte) (index : Nat) : PayloadByte :=
  payload.val[index]?.getD ⟨0#u8, none, true, false, none⟩

def snapshotByte (cell : Cell) : PayloadByte :=
  ⟨cell.value, cell.origin, cell.tainted, cell.defined, cell.word⟩

def snapshotPayload (source : Slice Cell) : Slice PayloadByte :=
  Slice.from (source.val.map snapshotByte) (by simp)

@[simp]
theorem snapshotPayload_length (source : Slice Cell) :
    (snapshotPayload source).val.length = source.val.length := by
  simp [snapshotPayload]

theorem snapshotPayload_at (source : Slice Cell) (index : Nat)
    (hindex : index < source.val.length) :
    payloadAt (snapshotPayload source) index = snapshotByte source.val[index] := by
  simp [payloadAt, snapshotPayload, List.getElem?_eq_getElem hindex]

def wordAfter (cell : Cell) (write : Key) (payload : Slice PayloadByte) : Option WordHistory :=
  if CoversWord write payload.val.length cell.key then
    (payloadAt payload (cell.key.offset.val - write.offset.val)).word
  else if Overlaps write payload.val.length cell.key then none
  else cell.word

def modelCell (cell : Cell) (write : Key) (payload : Slice PayloadByte) : Cell :=
  let word := wordAfter cell write payload
  if Touches write payload.val.length cell.key then
    let byte := payloadAt payload (cell.key.offset.val - write.offset.val)
    { cell with
      value := byte.value
      origin := byte.origin
      tainted := byte.tainted
      defined := byte.defined
      written := true
      word := word }
  else { cell with word := word }

structure CellWriteSpec (before after : Cell) (write : Key) (payload : Slice PayloadByte) : Prop where
  key : after.key = before.key
  byte_effect :
    if Touches write payload.val.length before.key then
      let byte := payloadAt payload (before.key.offset.val - write.offset.val)
      after.value = byte.value ∧ after.origin = byte.origin ∧
        after.tainted = byte.tainted ∧ after.defined = byte.defined ∧ after.written = true
    else after.value = before.value ∧ after.origin = before.origin ∧
      after.tainted = before.tainted ∧ after.defined = before.defined ∧
      after.written = before.written
  word_effect : after.word = wordAfter before write payload

def WriteSpec (before after : Slice Cell) (write : Key) (payload : Slice PayloadByte) : Prop :=
  List.Forall₂ (fun a b => CellWriteSpec a b write payload) before.val after.val

theorem modelCell_spec (cell : Cell) (write : Key) (payload : Slice PayloadByte) :
    CellWriteSpec cell (modelCell cell write payload) write payload := by
  unfold modelCell
  split <;> constructor <;> simp_all

theorem modelCell_key (cell : Cell) (write : Key) (payload : Slice PayloadByte) :
    (modelCell cell write payload).key = cell.key := by
  exact (modelCell_spec cell write payload).key

end MemoryWrite
