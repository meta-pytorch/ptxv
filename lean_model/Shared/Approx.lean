/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu
import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.Approx

open FloatLib.Floats.Formats.BinaryInterchange

def asFloat (x : Nat) : Model FloatFormat.binary32 := Model.ofNatBits x
def asWord (x : Model FloatFormat.binary32) : Nat :=
  if Model.isNaN x then 0x7fffffff else Model.toNatBits x

def mul (a b : Nat) (rz : Bool := false) : Nat :=
  asWord (Model.mulWithRounding (if rz then .towardZero else .nearestEven) (asFloat a) (asFloat b))

def flush (x : Nat) : Nat :=
  if Mufu.magnitude x < 0x00800000 then Mufu.sign x else x

def mulFtz (a b : Nat) : Nat :=
  let extended := FloatFormat.ieee 11 23
  let load := fun x => Model.castWithRounding FloatFormat.binary32 extended
    (asFloat (flush x)) .nearestEven
  let result := Model.mulWithRounding .nearestEven (load a) (load b)
  let bits := Model.toNatBits result
  if Model.isNaN result then 0x7fffffff
  else if bits % 2^34 < (1023 - 126) * 2^23 then (bits / 2^34) * 2^31
  else asWord (Model.castWithRounding extended FloatFormat.binary32 result .nearestEven)

def division (full ftz : Bool) (x y : Nat) : Nat :=
  if ftz then
    let large := full ∧ Mufu.magnitude y > 0x7e800000 ∧ Mufu.magnitude y ≤ 0x7f800000
    let x := if large then mulFtz x 0x3e800000 else x
    let y := if large then mulFtz y 0x3e800000 else y
    mulFtz x (Mufu.rcp y)
  else
    let scale := if Mufu.magnitude y < 0x00800000 then 0x4b800000
      else if full ∧ Mufu.magnitude y > 0x7e800000 ∧ Mufu.magnitude y ≤ 0x7f800000 then 0x3e800000
      else 0x3f800000
    mul (Mufu.rcp (mul y scale)) (mul x scale)

def evaluate (op : String) (ftz : Bool) (x : Nat) (y : Nat := 0) : Option Nat :=
  let tiny := Mufu.magnitude x < 0x00800000
  if op = "div" then some (division false ftz x y)
  else if op = "div_full" then some (division true ftz x y)
  else if op = "sin" ∨ op = "cos" then
    some (Mufu.sincos (mul x 0x3e22f983 true) (op = "cos"))
  else if op = "tanh" then Mufu.tanh x
  else if ftz then Mufu.evaluate op x
  else match op with
    | "ex2" =>
      if Mufu.sign x ≠ 0 ∧ Mufu.magnitude x > 0x42fc0000 ∧ Mufu.magnitude x ≤ 0x7f800000 then
        let y := Mufu.ex2 (mul x 0x3f000000)
        some (mul y y)
      else some (Mufu.ex2 x)
    | "rcp" =>
      let s := if Mufu.magnitude x > 0x7e800000 then 0x3e800000
               else if tiny then 0x4b800000 else 0x3f800000
      some (mul s (Mufu.rcp (mul x s)))
    | "rsq" => some (if tiny then mul (Mufu.rsq (mul x 0x4b800000)) 0x45800000 else Mufu.rsq x)
    | "sqrt" => some (if tiny then mul (Mufu.sqrt (mul x 0x4b800000)) 0x39800000 else Mufu.sqrt x)
    | "lg2" => some (if tiny then
        asWord (Model.addWithRounding .nearestEven (asFloat (Mufu.lg2 (mul x 0x4b800000))) (asFloat 0xc1c00000))
        else Mufu.lg2 x)
    | _ => none

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.Approx

open GB300Math.Approx

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected tab-separated manifest")
  let cases ← IO.FS.readFile manifest
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [op, ftz, input, output] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    unless ftz = "true" ∨ ftz = "false" do throw (IO.userError "Invalid FTZ mode")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError "Invalid capture size")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let x := readWord xs (32 * i)
      let y := readWord xs (32 * i + 8)
      let some expected := evaluate op (ftz = "true") x y |
        throw (IO.userError s!"Unsupported: {op} {x}")
      let actual := readWord ys (8 * i)
      if expected ≠ actual then
        if mismatches < 3 then IO.println s!"witness {op} {ftz} input={x},{y} model={expected} actual={actual}"
        mismatches := mismatches + 1
    failures := failures + mismatches
    IO.println s!"result {op} ftz={ftz} cases={xs.size / 32} mismatches={mismatches}"
  return if failures = 0 then 0 else 1
