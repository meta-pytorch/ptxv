/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexSearch

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

def Directory (pages : List register_index.RegisterPage) : Prop :=
  (∀ position < pages.length, (pages[position]!).slots.val.length = 64) ∧
  ∀ left right, left < right → right < pages.length →
    (pages[left]!).key.val < (pages[right]!).key.val

def Stored (index : register_index.RegisterIndex) (key : symbolic.RegisterKey) (id : Usize) : Prop :=
  ∃ position, position < index.pages.val.length ∧
    (index.pages.val[position]!).key.val = keyPage key ∧
    (index.pages.val[position]!).slots.val[keyOffset key]? = some (some id)

def IdsBounded (index : register_index.RegisterIndex) : Prop :=
  ∀ key id, Stored index key id → id.val < index.count.val

def IdsInjective (index : register_index.RegisterIndex) : Prop :=
  ∀ left right id, Stored index left id → Stored index right id → left = right

def Valid (index : register_index.RegisterIndex) : Prop :=
  Directory index.pages.val ∧ IdsBounded index ∧ IdsInjective index

theorem directory_ordered (pages : List register_index.RegisterPage)
    (valid : Directory pages) : OrderedKeys pages.length (fun position => (pages[position]!).key.val) := by
  intro left right order inside
  by_cases same : left = right
  · subst left; exact Nat.le_refl _
  · exact Nat.le_of_lt (valid.2 left right (by omega) inside)

theorem directory_key_unique (pages : List register_index.RegisterPage) (valid : Directory pages)
    (left right : Nat) (leftBound : left < pages.length) (rightBound : right < pages.length)
    (sameKey : (pages[left]!).key.val = (pages[right]!).key.val) : left = right := by
  by_cases earlier : left < right
  · have := valid.2 left right earlier rightBound; omega
  · by_cases later : right < left
    · have := valid.2 right left later leftBound; omega
    · omega

theorem stored_offset_bound (index : register_index.RegisterIndex) (valid : Directory index.pages.val)
    (key : symbolic.RegisterKey) (position : Nat) (inside : position < index.pages.val.length) :
    keyOffset key < (index.pages.val[position]!).slots.val.length := by
  rw [valid.1 position inside]
  exact Nat.mod_lt _ (by decide : 0 < 64)

end MemoryWrite.RegisterIndex
