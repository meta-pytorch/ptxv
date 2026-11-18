/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Snapshot

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite

inductive MemoryLocation where
  | shared (cta offset : U64)
  | global (allocation offset : U64)
  | param (grid offset : U64)
  | local (thread offset : U64)
  | tmem (cta : U64) (lane : Fin 128) (column : Fin 512) (byte : Fin 4)
  deriving DecidableEq

def tmemWordIndex (lane : Fin 128) (column : Fin 512) : Nat :=
  column.val * 128 + lane.val

def tmemByteOffset (lane : Fin 128) (column : Fin 512) (byte : Fin 4) : Nat :=
  4 * tmemWordIndex lane column + byte.val

@[simp] theorem tmemWordIndex_bound (lane : Fin 128) (column : Fin 512) :
    tmemWordIndex lane column < 65536 := by
  unfold tmemWordIndex
  omega

@[simp] theorem tmemByteOffset_bound (lane : Fin 128) (column : Fin 512) (byte : Fin 4) :
    tmemByteOffset lane column byte < 262144 := by
  have := tmemWordIndex_bound lane column
  unfold tmemByteOffset
  omega

def MemoryLocation.key : MemoryLocation → Key
  | .shared cta offset => ⟨0#u8, cta, offset⟩
  | .global allocation offset => ⟨1#u8, allocation, offset⟩
  | .param grid offset => ⟨2#u8, grid, offset⟩
  | .local thread offset => ⟨4#u8, thread, offset⟩
  | .tmem cta lane column byte =>
      ⟨3#u8, cta, U64.ofNat (tmemByteOffset lane column byte)
        (by have := tmemByteOffset_bound lane column byte; change _ ≤ 18446744073709551615; omega)⟩

/-- Production numeric and taint TMEM keys count words; half-origin keys count bytes. -/
def MemoryLocation.numericKey : MemoryLocation → Key
  | .tmem cta lane column _ =>
      ⟨3#u8, cta, U64.ofNat (tmemWordIndex lane column)
        (by have := tmemWordIndex_bound lane column; change _ ≤ 18446744073709551615; omega)⟩
  | loc => loc.key

def MemoryLocation.address (loc : MemoryLocation) : Address := location loc.key

@[simp] theorem key_location (loc : MemoryLocation) :
    location loc.key = loc.address := rfl

@[simp] theorem tmem_key_offset (cta : U64) (lane : Fin 128)
    (column : Fin 512) (byte : Fin 4) :
    (MemoryLocation.tmem cta lane column byte).key.offset.val =
      4 * (column.val * 128 + lane.val) + byte.val := rfl

@[simp] theorem tmem_numeric_offset (cta : U64) (lane : Fin 128)
    (column : Fin 512) (byte : Fin 4) :
    (MemoryLocation.tmem cta lane column byte).numericKey.offset.val =
      column.val * 128 + lane.val := rfl

@[simp] theorem tmem_byte_to_word (cta : U64) (lane : Fin 128)
    (column : Fin 512) (byte : Fin 4) :
    (MemoryLocation.tmem cta lane column byte).key.offset.val / 4 =
      (MemoryLocation.tmem cta lane column byte).numericKey.offset.val := by
  simp only [tmem_key_offset, tmem_numeric_offset]
  omega

@[simp] theorem tmem_byte_lane (lane : Fin 128) (column : Fin 512) (byte : Fin 4) :
    tmemByteOffset lane column byte % 4 = byte.val := by
  unfold tmemByteOffset
  omega

theorem tmemWordIndex_injective {lane₁ lane₂ : Fin 128} {column₁ column₂ : Fin 512}
    (h : tmemWordIndex lane₁ column₁ = tmemWordIndex lane₂ column₂) :
    lane₁ = lane₂ ∧ column₁ = column₂ := by
  have hl : lane₁.val = lane₂.val := by unfold tmemWordIndex at h; omega
  have hc : column₁.val = column₂.val := by unfold tmemWordIndex at h; omega
  exact ⟨Fin.ext hl, Fin.ext hc⟩

theorem tmemByteOffset_injective {lane₁ lane₂ : Fin 128}
    {column₁ column₂ : Fin 512} {byte₁ byte₂ : Fin 4}
    (h : tmemByteOffset lane₁ column₁ byte₁ = tmemByteOffset lane₂ column₂ byte₂) :
    lane₁ = lane₂ ∧ column₁ = column₂ ∧ byte₁ = byte₂ := by
  have hw : tmemWordIndex lane₁ column₁ = tmemWordIndex lane₂ column₂ := by
    unfold tmemByteOffset at h
    omega
  obtain ⟨hl, hc⟩ := tmemWordIndex_injective hw
  have hb : byte₁.val = byte₂.val := by unfold tmemByteOffset at h; omega
  exact ⟨hl, hc, Fin.ext hb⟩

structure MemoryLayout where
  sharedBytes : U64 → Nat
  globalBytes : U64 → Nat
  paramBytes : U64 → Nat
  localBytes : U64 → Nat
  tmemAllocated : U64 → Nat → Bool

def MemoryLayout.capacity (layout : MemoryLayout) : MemoryLocation → Nat
  | .shared cta _ => layout.sharedBytes cta
  | .global allocation _ => layout.globalBytes allocation
  | .param grid _ => layout.paramBytes grid
  | .local thread _ => layout.localBytes thread
  | .tmem .. => 262144

def MemoryLocation.writable : MemoryLocation → Bool
  | .param .. => false
  | _ => true

def ReadValid (layout : MemoryLayout) (loc : MemoryLocation) (width : Nat) : Prop :=
  loc.key.offset.val + width ≤ layout.capacity loc ∧
  loc.key.offset.val + width ≤ U64.max + 1 ∧
  match loc with
  | .tmem cta _ _ _ => ∀ index, index < width →
      layout.tmemAllocated cta ((loc.key.offset.val + index) / 4) = true
  | _ => True

def WriteValid (layout : MemoryLayout) (loc : MemoryLocation) (width : Nat) : Prop :=
  loc.writable = true ∧ ReadValid layout loc width

@[simp] theorem param_not_writable (grid offset : U64) :
    (MemoryLocation.param grid offset).writable = false := rfl

@[simp] theorem param_write_invalid (layout : MemoryLayout) (grid offset : U64)
    (width : Nat) : ¬WriteValid layout (.param grid offset) width := by
  simp [WriteValid, MemoryLocation.writable]

theorem write_valid_read_valid (layout : MemoryLayout) (loc : MemoryLocation) (width : Nat)
    (h : WriteValid layout loc width) : ReadValid layout loc width := h.2

theorem read_valid_no_wrap (layout : MemoryLayout) (loc : MemoryLocation) (width : Nat)
    (h : ReadValid layout loc width) : loc.key.offset.val + width ≤ U64.max + 1 := h.2.1

theorem tmem_read_requires_allocation (layout : MemoryLayout) (cta : U64)
    (lane : Fin 128) (column : Fin 512) (byte : Fin 4) (width : Nat)
    (h : ReadValid layout (.tmem cta lane column byte) width)
    (index : Nat) (hi : index < width) :
    layout.tmemAllocated cta ((tmemByteOffset lane column byte + index) / 4) = true :=
  h.2.2 index hi

theorem tmem_unallocated_read_invalid (layout : MemoryLayout) (cta : U64)
    (lane : Fin 128) (column : Fin 512) (byte : Fin 4) (width : Nat)
    (hwidth : 0 < width)
    (hunallocated : layout.tmemAllocated cta (tmemWordIndex lane column) = false) :
    ¬ReadValid layout (.tmem cta lane column byte) width := by
  intro h
  have halloc := tmem_read_requires_allocation layout cta lane column byte width h 0 hwidth
  have hword : (tmemByteOffset lane column byte + 0) / 4 = tmemWordIndex lane column := by
    unfold tmemByteOffset
    omega
  rw [hword, hunallocated] at halloc
  contradiction

end MemoryWrite
