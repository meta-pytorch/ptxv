/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.SmallConversions

namespace GB300Math.SmallConversionNative

open GB300Math.SmallConversions

def roundNative (mode : String) (negative : Bool) (m : Nat) (shift : Int) : Nat :=
  if shift ≤ 0 then m * 2^(-shift).toNat else
    let d := shift.toNat
    if mode = "rn" then nearestEven m d
    else m / 2^d + if mode = "rp" ∧ !negative ∧ m % 2^d ≠ 0 then 1 else 0

def downNative (fmt : Format) (mode : String) (relu sat : Bool) (x : Nat) : Nat :=
  let mag := x % 2^31
  let negative := x / 2^31 % 2 = 1
  let sign := if negative then 2^(fmt.exponentBits + fmt.fractionBits) else 0
  let overflow := if sat ∨ fmt.exponentBits < 4 then sign + fmt.limit
    else if fmt.exponentBits = 4 then 127 else sign + 124
  if mag > 0x7f800000 then if fmt.exponentBits ≥ 4 then 127 else fmt.limit
  else if relu ∧ negative then 0
  else if mag = 0x7f800000 then overflow
  else
    let e := mag / 2^23
    let m := mag % 2^23 + if e = 0 then 0 else 2^23
    let target : Int := (max e 1 : Nat) - 127 + (fmt.bias : Int)
    let shift := 23 - (fmt.fractionBits : Int) + max (1-target) 0
    let value := (max (target-1) 0).toNat * 2^fmt.fractionBits + roundNative mode negative m shift
    if value > fmt.limit then overflow else sign + value

def downE8Native (mode : String) (sat : Bool) (x : Nat) : Nat :=
  if mode = "rz" ∨ mode = "rp" then downE8m0 (mode = "rp") sat x else
    let mag := x % 2^31
    if mag > 0x7f800000 then 255 else
      let e := mag / 2^23
      let f := mag % 2^23
      let rounded := if e = 0 then (if f > 0x600000 then 1 else 0)
        else e + if f > 0x400000 ∨ (f = 0x400000 ∧ e % 2 = 1) then 1 else 0
      min rounded (if sat then 254 else 255)

def evaluate (name mode : String) (relu sat : Bool) (a b : Nat) : Option Nat := do
  unless ["rn", "rp", "rz"].contains mode do none
  if name = "ue8m0" then
    return downE8Native mode sat a * 256 + downE8Native mode sat b
  let fmt ← format name
  let stride := if name = "e2m1" then 16 else 256
  return downNative fmt mode relu sat a * stride + downNative fmt mode relu sat b

def evaluateSource (name other mode : String) (relu sat : Bool) (a b : Nat) : Option Nat := do
  if other = "f32" then evaluate name mode relu sat a b else
    unless other = "f16x2" ∨ other = "bf16x2" do none
    evaluate name mode relu sat
      (widen16 (other = "bf16x2") (a / 65536)) (widen16 (other = "bf16x2") a)

theorem exact_shift (mode : String) (negative : Bool) (m : Nat) (shift : Int)
    (h : shift ≤ 0) : roundNative mode negative m shift = m * 2^(-shift).toNat := by
  simp [roundNative, h]

theorem nearest_matches (negative : Bool) (m : Nat) (shift : Int) :
    roundNative "rn" negative m shift = roundShift m shift := by
  simp [roundNative, roundShift]

#print axioms exact_shift
#print axioms nearest_matches

end GB300Math.SmallConversionNative

open GB300Math.SmallConversions

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected native conversion manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failures := 0
  for row in rows do
    let [name, other, mode, relu, sat, input, output, errors] := row.splitOn "\t" |
      throw (IO.userError "Malformed native conversion row")
    unless [relu, sat].all (fun s => s = "0" ∨ s = "1") do throw (IO.userError "Invalid flag")
    let some expected := errors.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size / 2 do
      throw (IO.userError "Invalid record sizes")
    let mut mismatches := 0
    for i in [:xs.size/16] do
      let a := readWord xs (16*i) 4
      let b := readWord xs (16*i+4) 4
      let some target := GB300Math.SmallConversionNative.evaluateSource name other mode (relu = "1") (sat = "1") a b |
        throw (IO.userError "Unsupported configuration")
      let actual := readWord ys (8*i) 8
      if actual ≠ target then
        if mismatches < 5 then IO.println s!"witness row={i} a={a} b={b} expected={target} actual={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} records={xs.size/16} mismatches={mismatches} expected={expected}"
    if mismatches ≠ expected then failures := failures + 1
  return if failures = 0 then 0 else 1
