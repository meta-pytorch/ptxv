/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

set_option autoImplicit false

namespace ValueFlow

structure Pointer where
  allocation : Nat
  offset : Nat
  deriving DecidableEq

structure Object where
  bytes : Nat
  alignment : Nat
  deriving DecidableEq

def supportedWidth (bytes : Nat) : Prop :=
  bytes = 1 ∨ bytes = 2 ∨ bytes = 4 ∨ bytes = 8

instance (bytes : Nat) : Decidable (supportedWidth bytes) :=
  inferInstanceAs (Decidable (bytes = 1 ∨ bytes = 2 ∨ bytes = 4 ∨ bytes = 8))

def ObjectAccess (object : Object) (offset bytes : Nat) : Prop :=
  supportedWidth bytes ∧ object.alignment ≠ 0 ∧
  bytes ∣ object.alignment ∧ bytes ∣ offset ∧ offset + bytes ≤ object.bytes

def LegalPlacement (object : Object) (base : Nat) : Prop :=
  object.alignment ∣ base ∧ base + object.bytes ≤ 2 ^ 64

def PhysicalAccess (object : Object) (base offset bytes : Nat) : Prop :=
  bytes ∣ base + offset ∧ base ≤ base + offset ∧
  base + offset + bytes ≤ base + object.bytes ∧ base + offset + bytes ≤ 2 ^ 64

theorem object_access_all_placements (object : Object) (offset bytes : Nat)
    (access : ObjectAccess object offset bytes) :
    ∀ base, LegalPlacement object base → PhysicalAccess object base offset bytes := by
  intro base placement
  rcases access with ⟨_, _, widthAlignment, widthOffset, bounds⟩
  rcases placement with ⟨alignmentBase, addressSpace⟩
  refine ⟨Nat.dvd_add (dvd_trans widthAlignment alignmentBase) widthOffset, ?_, ?_, ?_⟩ <;> omega

def addOffset (pointer : Pointer) (amount : Nat) : Pointer :=
  { pointer with offset := pointer.offset + amount }

theorem add_offset_preserves_origin (pointer : Pointer) (amount : Nat) :
    (addOffset pointer amount).allocation = pointer.allocation := rfl

theorem base_equality_does_not_establish_origin (first second : Nat) (different : first ≠ second)
    (offset otherOffset : Nat) :
    (Pointer.mk first offset) ≠ (Pointer.mk second otherOffset) := by
  intro equality
  exact different (congrArg Pointer.allocation equality)

theorem insufficient_declared_alignment (object : Object) (offset bytes : Nat)
    (insufficient : ¬ bytes ∣ object.alignment) : ¬ ObjectAccess object offset bytes := by
  intro access
  exact insufficient access.2.2.1

theorem crossing_object_end_rejected (object : Object) (offset bytes : Nat)
    (outside : object.bytes < offset + bytes) : ¬ ObjectAccess object offset bytes := by
  intro access
  exact (Nat.not_lt_of_ge access.2.2.2.2) outside

end ValueFlow
