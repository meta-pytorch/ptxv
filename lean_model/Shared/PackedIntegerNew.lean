/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.PackedIntegerNew

def decode (signed : Bool) (width word : Nat) : Int :=
  let value := word % 2^width
  if signed && value ≥ 2^(width-1) then (value : Int) - 2^width else value

def encode (width : Nat) (value : Int) : Nat := (value % (2^width : Nat)).toNat

def clamp (signed : Bool) (width : Nat) (value : Int) : Int :=
  let lower : Int := if signed then -(2^(width-1) : Nat) else 0
  let upper : Int := if signed then 2^(width-1)-1 else 2^width-1
  max lower (min upper value)

def pack (signed : Bool) (width a b c : Nat) : Nat :=
  encode width (clamp signed width (decode true 32 b)) +
  encode width (clamp signed width (decode true 32 a)) * 2^width +
  (c % 2^(32-2*width)) * 2^(2*width)

def lane (operation : String) (signed relu : Bool) (width a b : Nat) : Nat :=
  let x := decode signed width a
  let y := decode signed width b
  let value := if operation = "add" then x+y else if operation = "min" then min x y else max x y
  encode width (if relu then max 0 value else value)

def evaluate (operation : String) (signed relu : Bool) (width a b c : Nat) : Nat :=
  if operation = "pack" then pack signed width a b c
  else if width = 32 then lane operation signed relu width a b
  else lane operation signed relu 16 a b + lane operation signed relu 16 (a/65536) (b/65536) * 65536

theorem signed_minimum : decode true 16 32768 = -32768 := by decide
theorem lane_wrap : evaluate "add" false false 16 65535 1 0 = 0 := by decide
theorem lane_does_not_carry : evaluate "add" false false 16 65535 65537 0 = 65536 := by decide
theorem negative_saturates : pack false 2 4294967295 7 0 = 3 := by decide
theorem signed_saturates : pack true 2 2147483647 2147483648 0 = 6 := by decide
theorem payload_shift : pack false 2 0 0 305419896 = 591751040 := by decide

#print axioms signed_minimum
#print axioms lane_wrap
#print axioms lane_does_not_carry
#print axioms negative_saturates
#print axioms signed_saturates
#print axioms payload_shift

def readWord (bytes : ByteArray) (offset count : Nat) : Nat :=
  (List.range count).foldl (fun x i => x + bytes[offset+i]!.toNat * 2^(8*i)) 0

end GB300Math.PackedIntegerNew

open GB300Math.PackedIntegerNew

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected packed integer manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [operation, signed, relu, width, input, output, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed packed integer row")
    unless operation ∈ ["add", "min", "max", "pack"] ∧ signed ∈ ["0", "1"] ∧ relu ∈ ["0", "1"] do
      throw (IO.userError "Invalid operation flags")
    let some width := width.toNat? | throw (IO.userError "Invalid width")
    unless (operation = "pack" ∧ width ∈ [2,4,8,16]) ∨ (operation ≠ "pack" ∧ width ∈ [16,32]) do
      throw (IO.userError "Unsupported width")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size/2 do
      throw (IO.userError "Invalid record sizes")
    let mut errors := 0
    for i in [:xs.size/16] do
      let target := evaluate operation (signed = "1") (relu = "1") width
        (readWord xs (16*i) 4) (readWord xs (16*i+4) 4) (readWord xs (16*i+8) 4)
      let actual := readWord ys (8*i) 8
      if actual ≠ target then
        if errors < 5 then IO.println s!"witness row={i} expected={target} actual={actual}"
        errors := errors+1
    IO.println s!"result {output} records={xs.size/16} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
