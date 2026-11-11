/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureBits

def roundWeight (n : Nat) : Nat := (n + 128) / 256

def plane (x y total : Nat) : Array Nat :=
  let highX := roundWeight (x * total)
  let lowX := total - highX
  let lowLow := roundWeight (lowX * (256 - y))
  let highHigh := roundWeight (highX * y)
  #[lowLow, highX - highHigh, lowX - lowLow, highHigh]

def coefficients (x y z : Nat) : Array Nat :=
  plane x y (256 - z) ++ plane x y z

def unitBits (n : Nat) : Nat :=
  if n = 0 then 0 else
    let exponent := n.log2
    (119 + exponent) * 8388608 + (n - 2 ^ exponent) * 2 ^ (23 - exponent)

theorem roundWeight_le (total fraction : Nat) (h : fraction ≤ 256) :
    roundWeight (fraction * total) ≤ total := by
  have hm := Nat.mul_le_mul_right total h
  unfold roundWeight
  omega

theorem plane_sum (x y total initial : Nat) (hx : x ≤ 256) (hy : y ≤ 256) :
    (plane x y total).foldl (· + ·) initial = initial + total := by
  have h1 := roundWeight_le total x hx
  have h2 := roundWeight_le (total - roundWeight (x * total)) (256 - y) (by omega)
  have h3 := roundWeight_le (roundWeight (x * total)) y hy
  rw [Nat.mul_comm (256 - y)] at h2
  rw [Nat.mul_comm y] at h3
  change initial + roundWeight ((total - roundWeight (x * total)) * (256 - y)) +
    (roundWeight (x * total) - roundWeight (roundWeight (x * total) * y)) +
    (total - roundWeight (x * total) - roundWeight ((total - roundWeight (x * total)) * (256 - y))) +
    roundWeight (roundWeight (x * total) * y) = initial + total
  omega

theorem coefficients_sum (x y z : Nat) (hx : x ≤ 256) (hy : y ≤ 256) (hz : z ≤ 256) :
    (coefficients x y z).foldl (· + ·) 0 = 256 := by
  rw [coefficients, Array.foldl_append, plane_sum x y (256 - z) 0 hx hy,
    plane_sum x y z (0 + (256 - z)) hx hy]
  omega

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut value := 0
  for j in [:4] do value := value + data[offset + j]!.toNat * 2 ^ (8 * j)
  return value

end GB300Math.TextureBits

open GB300Math.TextureBits

def main (args : List String) : IO UInt32 := do
  let [input, output, cornerText, errorsText] := args |
    throw (IO.userError "Expected weights.u16x3 output.f32 corner expected-errors")
  let some corner := cornerText.toNat? | throw (IO.userError "Invalid corner")
  let some expectedErrors := errorsText.toNat? | throw (IO.userError "Invalid expected errors")
  unless corner < 8 do throw (IO.userError "Corner must be in [0,8)")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % 6 = 0 ∧ ys.size = xs.size / 6 * 4 do
    throw (IO.userError "Invalid capture sizes")
  let mut mismatches := 0
  for i in [:xs.size / 6] do
    let vals := (Array.range 3).map fun a =>
      xs[i * 6 + 2 * a]!.toNat + 256 * xs[i * 6 + 2 * a + 1]!.toNat
    unless vals.all (· ≤ 256) do throw (IO.userError s!"Out-of-cell input at {i}")
    let x := vals[0]!
    let coeffs := coefficients (if x = 256 then 0 else x) vals[1]! vals[2]!
    let effectiveCorner := if x = 256 then corner - corner % 2 else corner
    let n := if x = 256 && corner % 2 = 0 then 0 else coeffs[effectiveCorner]!
    let predicted := unitBits n
    let actual := readWord ys (i * 4)
    if predicted != actual then
      if mismatches < 3 then
        IO.println s!"witness row={i} xyz={vals} corner={corner} model={predicted} hardware={actual}"
      mismatches := mismatches + 1
  IO.println s!"result corner={corner} count={xs.size / 6} mismatches={mismatches} expected={expectedErrors}"
  return if mismatches = expectedErrors then 0 else 1
