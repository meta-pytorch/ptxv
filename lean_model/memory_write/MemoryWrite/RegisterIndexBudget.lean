/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexInternTotal

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.RegisterIndex

def keyCount : List (alloc.vec.Vec symbolic.RegisterKey) → Nat
  | [] => 0
  | row :: rows => row.val.length + keyCount rows

def usedKeys : List (alloc.vec.Vec symbolic.RegisterKey) → Nat → Nat
  | [], _ => 0
  | _ :: _, 0 => 0
  | row :: rows, position + 1 => row.val.length + usedKeys rows position

def BuildFits (rows : List (alloc.vec.Vec symbolic.RegisterKey)) : Prop := keyCount rows ≤ Usize.max

theorem used_keys_zero (rows : List (alloc.vec.Vec symbolic.RegisterKey)) : usedKeys rows 0 = 0 := by
  cases rows <;> rfl

theorem used_keys_bound (rows : List (alloc.vec.Vec symbolic.RegisterKey)) (position : Nat) :
    usedKeys rows position ≤ keyCount rows := by
  induction rows generalizing position with
  | nil => simp only [usedKeys, keyCount, Nat.le_refl]
  | cons row rows ih =>
      cases position with
      | zero => simp only [usedKeys]; omega
      | succ position =>
          simp only [usedKeys, keyCount]
          exact Nat.add_le_add_left (ih position) row.val.length

theorem used_keys_next (rows : List (alloc.vec.Vec symbolic.RegisterKey)) (position : Nat)
    (inside : position < rows.length) :
    usedKeys rows (position + 1) = usedKeys rows position + rows[position]!.val.length := by
  induction rows generalizing position with
  | nil => simp at inside
  | cons row rows ih =>
      cases position with
      | zero => cases rows <;> simp [usedKeys]
      | succ position =>
          simp only [usedKeys, List.getElem!_cons_succ]
          have prior := ih position (by simpa using inside)
          omega

end MemoryWrite.RegisterIndex
