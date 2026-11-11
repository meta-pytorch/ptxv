/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Fswzadd

namespace GB300Math.FswzaddGeneral

def laneCode (pattern lane : Nat) : Nat :=
  pattern / 2 ^ (2 * (3 - lane % 4)) % 4

def operands (code a b : Nat) : Nat × Nat :=
  if code = 3 then (0, b)
  else if code = 2 then (a ^^^ 0x80000000, b)
  else if code = 1 then (a, b ^^^ 0x80000000)
  else (a, b)

def evaluate (pattern lane a b : Nat) : Option Nat :=
  if pattern < 256 then
    let (x, y) := operands (laneCode pattern lane) a b
    some (Fswzadd.add x y)
  else none

theorem code_bound (pattern lane : Nat) : laneCode pattern lane < 4 := by
  exact Nat.mod_lt _ (by decide)

theorem periodic (pattern lane a b : Nat) :
    evaluate pattern (lane + 4) a b = evaluate pattern lane a b := by
  simp [evaluate, laneCode]

theorem positive (a b : Nat) : operands 0 a b = (a, b) := by rfl

theorem negate_right (a b : Nat) : operands 1 a b = (a, b ^^^ 0x80000000) := by rfl

theorem negate_left (a b : Nat) : operands 2 a b = (a ^^^ 0x80000000, b) := by rfl

theorem zero_left (a b : Nat) : operands 3 a b = (0, b) := by rfl

theorem zero_left_ignores_source (a a' b : Nat) :
    operands 3 a b = operands 3 a' b := by rfl

end GB300Math.FswzaddGeneral
