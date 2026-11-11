/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.TensorDoubleShapes
open FloatLib.Floats.Formats.BinaryInterchange

def rounding (mode : String) : Model.IEEERoundingMode :=
  match mode with
  | "rz" => .towardZero
  | "rm" => .towardNegativeInfinity
  | "rp" => .towardPositiveInfinity
  | _ => .nearestEven

def fma (mode : String) (a b c : Nat) : Nat :=
  let fmt := FloatFormat.binary64
  let x : Model fmt := Model.ofNatBits a
  let y : Model fmt := Model.ofNatBits b
  let z : Model fmt := Model.ofNatBits c
  let value := Model.fmaWithRounding (rounding mode) x y z
  if !Model.isNaN value then Model.toNatBits value else
  match [y,z,x].find? Model.isNaN with
  | some nan => Model.toNatBits nan ||| 0x0008000000000000
  | none => 0xfff8000000000000

def dot (mode : String) (a b : Array Nat) (c : Nat) : Nat := Id.run do
  let mut word := c
  for i in [:a.size] do word := fma mode a[i]! b[i]! word
  return word

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem positive_half_ulp :
    fma "rn" 0x3ca0000000000000 0x3ff0000000000000 0x3ff0000000000000 = 0x3ff0000000000000 ∧
    fma "rp" 0x3ca0000000000000 0x3ff0000000000000 0x3ff0000000000000 = 0x3ff0000000000001 := by decide +kernel

theorem negative_half_ulp :
    fma "rz" 0xbca0000000000000 0x3ff0000000000000 0xbff0000000000000 = 0xbff0000000000000 ∧
    fma "rm" 0xbca0000000000000 0x3ff0000000000000 0xbff0000000000000 = 0xbff0000000000001 := by decide +kernel

theorem exact_cancellation_sign :
    fma "rm" 0x3ff0000000000000 0x3ff0000000000000 0xbff0000000000000 = 0x8000000000000000 ∧
    fma "rp" 0x3ff0000000000000 0x3ff0000000000000 0xbff0000000000000 = 0 := by decide +kernel

theorem overflow_rounding :
    fma "rz" 0x7fefffffffffffff 0x4000000000000000 0 = 0x7fefffffffffffff ∧
    fma "rp" 0x7fefffffffffffff 0x4000000000000000 0 = 0x7ff0000000000000 := by decide +kernel

#print axioms positive_half_ulp
#print axioms negative_half_ulp
#print axioms exact_cancellation_sign
#print axioms overflow_rounding

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset+j]!.toNat * 2^(8*j)
  return word
end GB300Math.TensorDoubleShapes
