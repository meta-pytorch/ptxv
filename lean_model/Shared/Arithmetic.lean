/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.Arithmetic

open FloatLib.Floats.Formats.BinaryInterchange

def rounding (name : String) : Option Model.IEEERoundingMode :=
  match name with
  | "rn" => some .nearestEven
  | "rz" => some .towardZero
  | "ru" => some .towardPositiveInfinity
  | "rd" => some .towardNegativeInfinity
  | _ => none

def lessThan (width a b : Nat) : Bool :=
  let sign := 2 ^ (width - 1)
  let ma := a % sign
  let mb := b % sign
  if ma = 0 ∧ mb = 0 then false
  else if (a ≥ sign) != (b ≥ sign) then a ≥ sign
  else if a ≥ sign then ma > mb else ma < mb

def select (fmt : FloatFormat) (maximum propagate : Bool) (a b : Nat) : Nat :=
  let nx := Model.isNaN (Model.ofNatBits a : Model fmt)
  let ny := Model.isNaN (Model.ofNatBits b : Model fmt)
  if nx ∨ ny then
    if propagate ∨ (nx ∧ ny) then 2 ^ (fmt.bitWidth - 1) - 1
    else if nx then b else a
  else if a % 2 ^ (fmt.bitWidth - 1) = 0 ∧ b % 2 ^ (fmt.bitWidth - 1) = 0 then
    if maximum then a &&& b else a ||| b
  else if lessThan fmt.bitWidth a b = maximum then b else a

def evaluate (fmt : FloatFormat) (op : String) (mode : Model.IEEERoundingMode)
    (a b c : Nat) : Option Nat := do
  let x : Model fmt := Model.ofNatBits a
  let y : Model fmt := Model.ofNatBits b
  let z : Model fmt := Model.ofNatBits c
  if op ∈ ["min", "max", "min_nan", "max_nan"] then
    return select fmt (op.startsWith "max") (op.endsWith "_nan") (Model.toNatBits x) (Model.toNatBits y)
  let baseOp := (op.replace "_sat" "").replace "_relu" ""
  let result ← match baseOp with
    | "add" => some (Model.addWithRounding mode x y)
    | "sub" => some (Model.subWithRounding mode x y)
    | "mul" => some (Model.mulWithRounding mode x y)
    | "div" => some (Model.divWithRounding mode x y)
    | "rcp" => some (Model.divWithRounding mode (Model.ofNatBits (fmt.exponentBias * 2 ^ fmt.fracWidth)) x)
    | "sqrt" => some (Model.sqrtWithRounding mode x)
    | "fma" => some (Model.fmaWithRounding mode x y z)
    | _ => none
  let bits := Model.toNatBits result
  if op.endsWith "_sat" then
    if Model.isNaN result ∨ bits ≥ 2 ^ (fmt.bitWidth - 1) then return 0
    return min bits (fmt.exponentBias * 2 ^ fmt.fracWidth)
  if op.endsWith "_relu" ∧ !Model.isNaN result ∧ bits ≥ 2 ^ (fmt.bitWidth - 1) then return 0
  if !Model.isNaN result then return Model.toNatBits result
  if fmt.bitWidth ≠ 64 then return 2 ^ (fmt.bitWidth - 1) - 1
  let operands := if op = "sqrt" ∨ op = "rcp" then [x]
                  else if op = "fma" then [x, z, y] else [x, y]
  match operands.find? Model.isNaN with
  | some nan => return Model.toNatBits nan ||| 0x0008000000000000
  | none => return 0xfff8000000000000

def apply (precision op : String) (mode : Model.IEEERoundingMode) (a b c : Nat) : Option Nat := do
  let fmt ← match precision.replace "x2" "" with
    | "f32" => some FloatFormat.binary32
    | "f64" => some FloatFormat.binary64
    | "f16" => some FloatFormat.binary16
    | "bf16" => some FloatFormat.bfloat16
    | _ => none
  if precision.endsWith "x2" then
    let width := fmt.bitWidth
    let lo ← evaluate fmt op mode (a % 2 ^ width) (b % 2 ^ width) (c % 2 ^ width)
    let hi ← evaluate fmt op mode (a / 2 ^ width) (b / 2 ^ width) (c / 2 ^ width)
    return lo + hi * 2 ^ width
  if op.startsWith "to_" then
    let dst ← match op with
      | "to_f16" => some FloatFormat.binary16
      | "to_bf16" => some FloatFormat.bfloat16
      | "to_f32" => some FloatFormat.binary32
      | _ => none
    let value := Model.castWithRounding fmt dst (Model.ofNatBits a) mode
    if Model.isNaN value then
      if fmt.bitWidth = 64 then
        return ((a / 2 ^ 32) &&& 0x80000000) ||| 0x7fc00000 ||| ((a % 2 ^ 52) / 2 ^ 29)
      return 2 ^ (dst.bitWidth - 1) - 1
    return Model.toNatBits value
  evaluate fmt op mode a b c

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.Arithmetic

open GB300Math.Arithmetic

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, op, modeName, input, output] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mode := rounding modeName | throw (IO.userError s!"Invalid rounding: {modeName}")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let b := readWord xs (32 * i + 8)
      let c := readWord xs (32 * i + 16)
      let some model := apply precision op mode a b c |
        throw (IO.userError s!"Unsupported: {precision} {op}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {precision} {op} {modeName} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {precision} {op} {modeName} cases={xs.size / 32} mismatches={mismatches}"
    failures := failures + mismatches
  return if failures = 0 then 0 else 1
