/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.MatrixLoad

def byte (words : Array Nat) (index : Nat) : Nat :=
  (words[index / 4]! / 2^(8*(index%4))) % 256

def field (words : Array Nat) (row width col : Nat) : Nat :=
  let bit := width*col
  let offset := row*16+bit/8
  ((byte words offset + 256*byte words (offset+1)) / 2^(bit%8)) % 2^width

def signedField (words : Array Nat) (row col : Nat) : Nat :=
  let value := (byte words (row*8+col/2) / 2^(4*(col%2))) % 16
  if value < 8 then value else value+240

def value (shape width signed shift : Nat) (words : Array Nat)
    (lane reg component : Nat) : Nat :=
  let row := if shape = 8 then reg*8+lane/4
    else (reg/2)*16+2*(lane%4)+component%2+8*(component/2)
  let col := if shape = 8 then 4*(lane%4)+component
    else lane/4+8*(reg%2)
  if signed = 1 then signedField words row col
  else if width = 8 then byte words (row*16+col)
  else field words row width col * 2^shift

def word (shape matrices width signed shift : Nat) (words : Array Nat)
    (lane reg : Nat) : Nat :=
  let registers := matrices*(if shape = 8 then 1 else 2)
  if reg >= registers then 0xdeadbeef
  else (List.range 4).foldl (fun acc b =>
    acc + value shape width signed shift words lane reg b * 2^(8*b)) 0

theorem byte_bound (words : Array Nat) (index : Nat) : byte words index < 256 := by
  exact Nat.mod_lt _ (by decide)

theorem field_bound (words : Array Nat) (row width col : Nat) :
    field words row width col < 2^width := by
  exact Nat.mod_lt _ (Nat.two_pow_pos _)

theorem signedField_bound (words : Array Nat) (row col : Nat) :
    signedField words row col < 256 := by
  unfold signedField
  have h : (byte words (row*8+col/2) / 2^(4*(col%2))) % 16 < 16 := Nat.mod_lt _ (by decide)
  dsimp only
  split <;> omega

theorem signed_zero : signedField #[0] 0 0 = 0 := by decide

theorem signed_minus_one : signedField #[15] 0 0 = 255 := by decide

theorem signed_minimum : signedField #[8] 0 0 = 248 := by decide

theorem value_in_bounds (shape width signed : Nat) (words : Array Nat)
    (lane reg component : Nat) (hw : width = 4 ∨ width = 6 ∨ width = 8) :
    value shape width signed 0 words lane reg component < 256 := by
  unfold value
  dsimp only
  split
  · exact signedField_bound _ _ _
  · split
    · exact byte_bound _ _
    · simp only [Nat.pow_zero, Nat.mul_one]
      have h := field_bound words
      rcases hw with h4 | h6 | h8
      · subst width; exact Nat.lt_trans (h _ 4 _) (by decide)
      · subst width; exact Nat.lt_trans (h _ 6 _) (by decide)
      · contradiction

#print axioms byte_bound
#print axioms field_bound
#print axioms signedField_bound
#print axioms signed_zero
#print axioms signed_minus_one
#print axioms signed_minimum
#print axioms value_in_bounds
end GB300Math.MatrixLoad
