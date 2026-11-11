/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.Fswzadd

open FloatLib.Floats.Formats.BinaryInterchange

def supported (pattern : Nat) : Bool := pattern = 0xcc || pattern = 0xf0

def maskA (pattern lane : Nat) : Bool :=
  if pattern = 0xcc then lane % 2 = 0 else lane % 4 < 2

def add (a b : Nat) : Nat :=
  let x : Model FloatFormat.binary32 := Model.ofNatBits a
  let y : Model FloatFormat.binary32 := Model.ofNatBits b
  let r := Model.addWithRounding .nearestEven x y
  if Model.isNaN r then 0x7fffffff else Model.toNatBits r

def evaluate (pattern lane a b : Nat) : Option Nat :=
  if supported pattern then some (add (if maskA pattern lane then 0 else a) b)
  else none

theorem cc_even (lane a b : Nat) (h : lane % 2 = 0) :
    evaluate 0xcc lane a b = some (add 0 b) := by
  simp [evaluate, supported, maskA, h]

theorem cc_odd (lane a b : Nat) (h : lane % 2 ≠ 0) :
    evaluate 0xcc lane a b = some (add a b) := by
  simp [evaluate, supported, maskA, h]

theorem f0_low (lane a b : Nat) (h : lane % 4 < 2) :
    evaluate 0xf0 lane a b = some (add 0 b) := by
  simp [evaluate, supported, maskA, h]

theorem f0_high (lane a b : Nat) (h : ¬ lane % 4 < 2) :
    evaluate 0xf0 lane a b = some (add a b) := by
  simp [evaluate, supported, maskA, h]

theorem periodic (pattern lane a b : Nat) :
    evaluate pattern (lane + 4) a b = evaluate pattern lane a b := by
  simp [evaluate, maskA, Nat.add_mod]

end GB300Math.Fswzadd
