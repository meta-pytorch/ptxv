/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.MixedScalar

open FloatLib.Floats.Formats.BinaryInterchange

def rounding : String → Option Model.IEEERoundingMode
  | "rn" => some .nearestEven
  | "rz" => some .towardZero
  | "rp" => some .towardPositiveInfinity
  | "rm" => some .towardNegativeInfinity
  | _ => none

def widen (fmt : FloatFormat) (word : Nat) : Model FloatFormat.binary32 :=
  Model.castWithRounding fmt FloatFormat.binary32 (Model.ofNatBits word) .nearestEven

def evaluate (precision op : String) (mode : Model.IEEERoundingMode)
    (sat : Bool) (a b c : Nat) : Option Nat := do
  if a ≥ 65536 || b ≥ 65536 || c ≥ 2^32 then none else do
    let fmt ← if precision = "f16" then some FloatFormat.binary16
      else if precision = "bf16" then some FloatFormat.bfloat16 else none
    let x := widen fmt a
    let y := widen fmt b
    let z : Model FloatFormat.binary32 := Model.ofNatBits c
    let result ← match op with
      | "add" => some (Model.addWithRounding mode x z)
      | "sub" => some (Model.addWithRounding mode x (Model.ofNatBits (c ^^^ 0x80000000)))
      | "fma" => some (Model.fmaWithRounding mode x y z)
      | _ => none
    let bits := Model.toNatBits result
    if sat then
      if Model.isNaN result ∨ bits ≥ 0x80000000 then return 0
      return min bits 0x3f800000
    if Model.isNaN result then return 0x7fffffff
    return bits

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem bf16_smallest_subnormal_preserved :
    Model.toNatBits (widen FloatFormat.bfloat16 1) = 0x10000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem f16_smallest_subnormal_preserved :
    Model.toNatBits (widen FloatFormat.binary16 1) = 0x33800000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem saturated_nan :
    evaluate "bf16" "fma" .nearestEven true 0x7fc0 0x3f80 0 = some 0 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem directed_cancellation :
    evaluate "f16" "sub" .towardNegativeInfinity false 0x3c00 0 0x3f800000 =
      some 0x80000000 := by decide

#print axioms bf16_smallest_subnormal_preserved
#print axioms f16_smallest_subnormal_preserved
#print axioms saturated_nan
#print axioms directed_cancellation

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.MixedScalar

open GB300Math.MixedScalar

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, op, modeName, saturation, input, output, errors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mode := rounding modeName | throw (IO.userError "Invalid rounding mode")
    let sat ← match saturation with
      | "0" => pure false
      | "1" => pure true
      | _ => throw (IO.userError "Invalid saturation mode")
    let some expected := errors.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32*i)
      let b := readWord xs (32*i+8)
      let c := readWord xs (32*i+16)
      let some model := evaluate precision op mode sat a b c |
        throw (IO.userError s!"Unsupported input: {line} row={i}")
      let actual := readWord ys (8*i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {precision} {op} {modeName} sat={sat} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size/32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
