/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.ScaledMultiply

open FloatLib.Floats.Formats.BinaryInterchange

def rounding : String → Option Model.IEEERoundingMode
  | "rn" => some .nearestEven
  | "rz" => some .towardZero
  | "ru" => some .towardPositiveInfinity
  | "rd" => some .towardNegativeInfinity
  | _ => none

def modify (width modifier input : Nat) : Nat :=
  let word := input % 2 ^ width
  let value := if modifier / 2 % 2 = 1 then word % 2 ^ (width - 1) else word
  if modifier % 2 = 1 then value ^^^ 2 ^ (width - 1) else value

def flush (x : Nat) : Nat :=
  if x % 0x80000000 < 0x00800000 then x &&& 0x80000000 else x

def extended : FloatFormat := FloatFormat.ieee 11 23

def exactFormat : FloatFormat := FloatFormat.ieee 11 48

set_option maxHeartbeats 4000000 in
def evaluate (precision op : String) (mode : Model.IEEERoundingMode)
    (flags parameter a b _c : Nat) : Option Nat := do
  let scale := parameter / 64
  let modifiers := parameter % 64
  if precision != "f32" || op != "mul" || flags ≥ 4 || scale < 1 || scale > 7 then none else do
    let a := modify 32 (modifiers % 4) a
    let b := modify 32 (modifiers / 4 % 4) b
    let ftz := flags % 2 = 1
    let load := fun x => Model.castWithRounding FloatFormat.binary32 exactFormat
      (Model.ofNatBits (if ftz then flush x else x)) .nearestEven
    let product := Model.mulWithRounding .nearestEven (load a) (load b)
    let factor : Model exactFormat := Model.ofNatBits ((1023 + scale - 4) * 2^48)
    let scaled := Model.mulWithRounding .nearestEven product factor
    let rounded := Model.castWithRounding exactFormat extended scaled mode
    let bits := Model.toNatBits rounded
    let output := if ftz then
        if bits % 2^34 < (1023 - 126) * 2^23 then bits / 2^34 * 2^31
        else Model.toNatBits (Model.castWithRounding extended FloatFormat.binary32 rounded mode)
      else Model.toNatBits (Model.castWithRounding exactFormat FloatFormat.binary32 scaled mode)
    if flags / 2 = 1 then
      if Model.isNaN scaled ∨ output ≥ 0x80000000 then return 0
      return min output 0x3f800000
    if Model.isNaN scaled then return 0x7fffffff
    return output

set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
theorem scaling_rescues_overflow :
    evaluate "f32" "mul" .nearestEven 1 192 0x7f7fffff 0x40000000 0 = some 0x7f7fffff := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
theorem scaling_rescues_flushed_product :
    evaluate "f32" "mul" .nearestEven 1 448 0x00800000 0x3e000000 0 = some 0x00800000 := by decide

#print axioms scaling_rescues_overflow
#print axioms scaling_rescues_flushed_product

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.ScaledMultiply

open GB300Math.ScaledMultiply

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, op, modeName, flagsText, modsText, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mode := rounding modeName | throw (IO.userError "Invalid rounding mode")
    let some flags := flagsText.toNat? | throw (IO.userError "Invalid flags")
    let some mods := modsText.toNat? | throw (IO.userError "Invalid modifiers")
    let some expected := errorsText.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let b := readWord xs (32 * i + 8)
      let c := readWord xs (32 * i + 16)
      let some model := evaluate precision op mode flags mods a b c |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {precision} {op} {modeName} flags={flags} mods={mods} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
