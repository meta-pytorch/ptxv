/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.AluModifiers

open FloatLib.Floats.Formats.BinaryInterchange

def rounding (name : String) : Option Model.IEEERoundingMode :=
  match name with
  | "rn" => some .nearestEven
  | "rz" => some .towardZero
  | "ru" => some .towardPositiveInfinity
  | "rd" => some .towardNegativeInfinity
  | _ => none

def flush (x : Nat) : Nat :=
  if x % 0x80000000 < 0x00800000 then x &&& 0x80000000 else x

-- The wider exponent prevents IEEE subnormal rounding before the hardware flush.
def extended : FloatFormat := FloatFormat.ieee 11 23

def arithmetic (fmt : FloatFormat) (op : String) (mode : Model.IEEERoundingMode)
    (a b c : Model fmt) : Option (Model fmt) :=
  match op with
  | "add" => some (Model.addWithRounding mode a b)
  | "sub" => some (Model.subWithRounding mode a b)
  | "mul" => some (Model.mulWithRounding mode a b)
  | "fma" => some (Model.fmaWithRounding mode a b c)
  | _ => none

def evaluate (op : String) (mode : Model.IEEERoundingMode) (ftz sat : Bool)
    (a b c : Nat) : Option Nat := do
  let fmt := if ftz then extended else FloatFormat.binary32
  let load := fun x => Model.castWithRounding FloatFormat.binary32 fmt
    (Model.ofNatBits (if ftz then flush x else x)) .nearestEven
  let result ← arithmetic fmt op mode (load a) (load b) (load c)
  let bits := Model.toNatBits result
  let output := if ftz ∧ bits % 2^34 < (1023 - 126) * 2^23 then
      (bits / 2^34) * 2^31
    else Model.toNatBits (Model.castWithRounding fmt FloatFormat.binary32 result mode)
  if sat then
    if Model.isNaN result ∨ output ≥ 0x80000000 then return 0
    return min output 0x3f800000
  if Model.isNaN result then return 0x7fffffff
  return output

def fchk (a b : Nat) : Bool :=
  let ea := (a / 2^23) % 256
  let eb := (b / 2^23) % 256
  !(25 ≤ ea ∧ ea ≤ 254 ∧ 2 ≤ eb ∧ eb ≤ 251 ∧ eb ≤ ea + 124 ∧ ea ≤ eb + 126)

theorem fchk_lower_numerator_boundary :
    fchk (24 * 2^23) (2 * 2^23) = true ∧
    fchk (25 * 2^23) (2 * 2^23) = false := by decide

theorem fchk_exponent_difference_boundary :
    fchk (25 * 2^23) (149 * 2^23) = false ∧
    fchk (25 * 2^23) (150 * 2^23) = true := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem ftz_round_before_flush :
    evaluate "mul" .nearestEven true false 0x00800000 0x3f7fffff 0 = some 0 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem ieee_round_then_flush_differs :
    evaluate "mul" .nearestEven false false 0x00800000 0x3f7fffff 0 = some 0x00800000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem ftz_rounding_can_reach_normal :
    evaluate "mul" .nearestEven true false 0x00800001 0x3f7ffffe 0 = some 0x00800000 := by decide

#print axioms fchk_lower_numerator_boundary
#print axioms fchk_exponent_difference_boundary
#print axioms ftz_round_before_flush
#print axioms ieee_round_then_flush_differs
#print axioms ftz_rounding_can_reach_normal

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2^(8*j)
  return word

end GB300Math.AluModifiers

open GB300Math.AluModifiers

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected case manifest")
  let cases ← IO.FS.readFile manifest
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [op, modeName, modifier, input, output] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mode := rounding modeName | throw (IO.userError s!"Invalid mode {modeName}")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32*i)
      let b := readWord xs (32*i+8)
      let c := readWord xs (32*i+16)
      let expected ← if op = "fchk" then pure (if fchk a b then 0 else 1)
        else match evaluate op mode (modifier.startsWith "ftz") (modifier.endsWith "sat") a b c with
          | some value => pure value
          | none => throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8*i)
      if expected ≠ actual then
        if mismatches < 4 then
          IO.println s!"witness {op} {modeName} {modifier} row={i} a={a} b={b} c={c} model={expected} actual={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {op} {modeName} {modifier} cases={xs.size / 32} mismatches={mismatches}"
    (← IO.getStdout).flush
    failures := failures + mismatches
  return if failures = 0 then 0 else 1
