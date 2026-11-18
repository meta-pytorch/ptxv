/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreWindowModel

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

def payloadByte (windows : List ByteWindow) (read : Key → Nat) (index : Nat) : Nat :=
  match windows.find? (fun window =>
      decide (window.first ≤ index ∧ index < window.first + window.count)) with
  | none => 0
  | some window => (read window.source >>> (8 * (index - window.first))) % 256

def payloadBytes (total : Nat) (windows : List ByteWindow) (read : Key → Nat) : List Nat :=
  (List.range total).map (payloadByte windows read)

theorem payload_byte_bound (windows : List ByteWindow) (read : Key → Nat) (index : Nat) :
    payloadByte windows read index < 256 := by
  unfold payloadByte
  split
  · omega
  · exact Nat.mod_lt _ (by omega)

theorem payload_unselected_byte_zero (windows : List ByteWindow) (read : Key → Nat)
    (index : Nat) (h : ∀ window ∈ windows, index < window.first ∨
      window.first + window.count ≤ index) : payloadByte windows read index = 0 := by
  unfold payloadByte
  have hn : windows.find? (fun window =>
      decide (window.first ≤ index ∧ index < window.first + window.count)) = none := by
    apply List.find?_eq_none.mpr
    intro window hw
    have := h window hw
    simp only [decide_eq_true_eq]
    omega
  rw [hn]

def wordByte (value byte : Nat) : Nat := (value >>> (8 * byte)) % 256

def overwriteBytes (initial : List Nat) (first count value : Nat) : List Nat :=
  initial.mapIdx fun index old =>
    if first ≤ index ∧ index < first + count then wordByte value (index - first) else old

@[simp] theorem overwrite_bytes_length (initial : List Nat) (first count value : Nat) :
    (overwriteBytes initial first count value).length = initial.length := by
  simp [overwriteBytes]

@[simp] theorem overwrite_bytes_zero (initial : List Nat) (first value : Nat) :
    overwriteBytes initial first 0 value = initial := by
  apply List.ext_getElem
  · simp
  · intro index hi hj
    simp only [overwriteBytes, List.getElem_mapIdx]
    split
    · omega
    · rfl

theorem overwrite_bytes_succ (initial : List Nat) (first count value : Nat)
    (hbound : first + count < initial.length) :
    overwriteBytes initial first (count + 1) value =
      (overwriteBytes initial first count value).set (first + count) (wordByte value count) := by
  apply List.ext_getElem
  · simp
  · intro index hi hj
    by_cases heq : index = first + count
    · subst index
      simp [overwriteBytes, List.getElem_set_self]
    · have hsame : (first ≤ index ∧ index < first + (count + 1)) ↔
          (first ≤ index ∧ index < first + count) := by omega
      simp [overwriteBytes, List.getElem_set, heq, Ne.symm heq, hsame]

end Instruction
