/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

namespace GB300Math.Tensor

structure Dyadic where
  significand : Int
  exponent : Int
  deriving Inhabited

def signed (x width : Nat) : Int :=
  let y := x % 2 ^ width
  if y < 2 ^ (width - 1) then y else (y : Int) - 2 ^ width

def integerDot (a b : Array Nat) (c : Nat) (signedInputs saturate : Bool) : Nat := Id.run do
  let mut total := signed c 32
  for i in [:a.size] do
    let x : Int := if signedInputs then signed a[i]! 8 else a[i]! % 256
    let y : Int := if signedInputs then signed b[i]! 8 else b[i]! % 256
    total := total + x * y
  if saturate then total := max (-(2 ^ 31)) (min (2 ^ 31 - 1) total)
  return (total % 2 ^ 32).toNat

def decode (x fw ew : Nat) : Dyadic :=
  let e := x / 2 ^ fw % 2 ^ ew
  let m := x % 2 ^ fw + if e = 0 then 0 else 2 ^ fw
  ⟨if x / 2 ^ (fw + ew) % 2 = 0 then (m : Int) else -(m : Int),
   (max 1 e : Int) - (2 ^ (ew - 1) - 1 : Nat) - fw⟩

def shiftTruncate (x : Int) (shift : Int) : Int :=
  let m := if shift ≥ 0 then x.natAbs * 2 ^ shift.toNat
           else x.natAbs / 2 ^ (-shift).toNat
  if x < 0 then -(m : Int) else m

def encodeRz (x : Dyadic) : Nat :=
  if x.significand = 0 then 0 else
  let sign := if x.significand < 0 then 2 ^ 31 else 0
  let shift := max ((x.significand.natAbs.log2 : Int) - 23) (-149 - x.exponent)
  let m := (shiftTruncate x.significand (-shift)).natAbs
  if m = 0 then 0 else
  let e := x.exponent + shift + (m.log2 : Int) + 127
  if e ≥ 255 then sign + 0x7f800000
  else if e ≤ 0 then sign + m
  else sign + e.toNat * 2 ^ 23 + m % 2 ^ 23

def inputTop (x : Dyadic) : Int :=
  if x.significand = 0 then -10000 else x.exponent + (x.significand.natAbs.log2 : Int)

def isNaN (x fw ew : Nat) : Bool :=
  x / 2 ^ fw % 2 ^ ew == 2 ^ ew - 1 && x % 2 ^ fw != 0

def isInf (x fw ew : Nat) : Bool :=
  x % 2 ^ (fw + ew) == (2 ^ ew - 1) * 2 ^ fw

def isZero (x fw ew : Nat) : Bool := x % 2 ^ (fw + ew) == 0

end GB300Math.Tensor
