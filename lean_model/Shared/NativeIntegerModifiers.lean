/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.NativeIntegerModifiers

def decode (signed : Bool) (width word : Nat) : Int :=
  let value := word % 2^width
  if signed && value ≥ 2^(width-1) then (value : Int) - 2^width else value

def encode (width : Nat) (value : Int) : Nat := (value % (2^width : Nat)).toNat

def clamp (signed : Bool) (width : Nat) (value : Int) : Int :=
  let lower : Int := if signed then -(2^(width-1) : Nat) else 0
  let upper : Int := if signed then 2^(width-1)-1 else 2^width-1
  max lower (min upper value)

def convert (signed sat : Bool) (width word : Nat) : Nat :=
  let value := decode true 32 word
  encode width (if sat then clamp signed width value else value)

def pack (signed sat : Bool) (width selector a b c : Nat) : Nat :=
  convert signed sat width b + convert signed sat width a * 2^width +
  ((c / (if width = 8 ∧ selector = 1 then 65536 else 1)) % 2^(32-2*width)) * 2^(2*width)

def lane (operation : String) (signed relu neg : Bool) (width a b : Nat) : Nat :=
  let x := decode signed width a
  let y := decode signed width b
  let value := if operation = "add" then x+(if neg then -y else y)
    else if operation = "min" then min x y else max x y
  encode width (if relu then max 0 value else value)

def evaluate (operation : String) (signed relu sat neg : Bool) (width selector a b c : Nat) : Nat :=
  if operation = "pack" then pack signed sat width selector a b c
  else if width = 32 then lane operation signed relu neg width a b
  else lane operation signed relu neg 16 a b + lane operation signed relu neg 16 (a/65536) (b/65536) * 65536

theorem subtraction_does_not_borrow : evaluate "add" false false false true 16 0 65536 1 0 = 131071 := by decide
theorem upper_merge_payload : pack false true 8 1 1 2 305419896 = 305398018 := by decide
theorem unsaturated_negative_wrap : pack false false 2 3 4294967295 0 0 = 12 := by decide
#print axioms subtraction_does_not_borrow
#print axioms upper_merge_payload
#print axioms unsaturated_negative_wrap

def readWord (bytes : ByteArray) (offset count : Nat) : Nat :=
  (List.range count).foldl (fun x i => x + bytes[offset+i]!.toNat * 2^(8*i)) 0

end GB300Math.NativeIntegerModifiers

open GB300Math.NativeIntegerModifiers

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected packed integer manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [operation, signed, relu, sat, neg, width, selector, input, output, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed packed integer row")
    unless operation ∈ ["add", "min", "max", "pack"] ∧ signed ∈ ["0", "1"] ∧ relu ∈ ["0", "1"] ∧ sat ∈ ["0", "1"] ∧ neg ∈ ["0", "1"] do
      throw (IO.userError "Invalid operation flags")
    let some width := width.toNat? | throw (IO.userError "Invalid width")
    unless (operation = "pack" ∧ width ∈ [2,4,8,16]) ∨ (operation ≠ "pack" ∧ width ∈ [16,32]) do
      throw (IO.userError "Unsupported width")
    let some selector := selector.toNat? | throw (IO.userError "Invalid selector")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size/2 do
      throw (IO.userError "Invalid record sizes")
    let mut errors := 0
    for i in [:xs.size/16] do
      let target := evaluate operation (signed = "1") (relu = "1") (sat = "1") (neg = "1") width selector
        (readWord xs (16*i) 4) (readWord xs (16*i+4) 4) (readWord xs (16*i+8) 4)
      let actual := readWord ys (8*i) 8
      if actual ≠ target then
        if errors < 5 then IO.println s!"witness row={i} expected={target} actual={actual}"
        errors := errors+1
    IO.println s!"result {output} records={xs.size/16} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
