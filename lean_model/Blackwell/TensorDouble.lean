/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.TensorDouble
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
end GB300Math.TensorDouble
open GB300Math.TensorDouble

def main (args : List String) : IO UInt32 := do
  let [mode,input,output,expected] := args | throw (IO.userError "Expected mode inputs outputs expected-mismatches")
  unless mode ∈ ["rn","rz","rm","rp"] do throw (IO.userError "Invalid rounding mode")
  let some expected := expected.toNat? | throw (IO.userError "Invalid expected mismatch count")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % 1024 = 0 ∧ ys.size = xs.size / 2 do
    throw (IO.userError "Incorrect 8x4 A, 4x8 B, 8x8 C capture extent")
  let mut errors := 0
  for sample in [:xs.size/1024] do
    let base := sample*1024
    for row in [:8] do
      let a := (Array.range 4).map (fun k => readWord xs (base+8*(row*4+k)))
      for col in [:8] do
        let b := (Array.range 4).map (fun k => readWord xs (base+8*(32+col*4+k)))
        let c := readWord xs (base+8*(64+row*8+col))
        let model := dot mode a b c
        let actual := readWord ys ((sample*64+row*8+col)*8)
        if model ≠ actual then
          if errors < 8 then IO.println s!"witness sample={sample} row={row} col={col} a={a} b={b} c={c} model={model} actual={actual}"
          errors := errors+1
  IO.println s!"result mode={mode} cells={xs.size/1024*64} mismatches={errors} expected={expected}"
  return if errors = expected then 0 else 1
