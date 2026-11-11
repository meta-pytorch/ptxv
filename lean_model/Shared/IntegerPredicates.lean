/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.IntegerPredicates

def signedWord (bits x : Nat) : Int :=
  if x < 2 ^ (bits - 1) then x else (x : Int) - 2 ^ bits

def combine (operation : Nat) (a b : Bool) : Bool :=
  if operation = 0 then a && b else if operation = 1 then a || b else a != b

def bit (value : Bool) : Nat := if value then 1 else 0

def evaluate (op : String) (signed : Bool) (parameter a b c d : Nat) : Option (Nat × Nat) := do
  if a ≥ 2 ^ 64 || b ≥ 2 ^ 64 || c ≥ 2 ^ 64 || d ≥ 2 ^ 64 then none else do
    let mask32 := 0xffffffff
    let predicate := c % 2 ^ 32 != 0
    if op = "setp32" || op = "setp64" then
      if parameter ≥ 36 then none else do
        let bits := if op = "setp32" then 32 else 64
        let aa := a % 2 ^ bits
        let bb := b % 2 ^ bits
        let x : Int := if signed then signedWord bits aa else aa
        let y : Int := if signed then signedWord bits bb else bb
        let comparison := match parameter / 6 with
          | 0 => x == y
          | 1 => x != y
          | 2 => x < y
          | 3 => x ≤ y
          | 4 => x > y
          | _ => x ≥ y
        let third := if parameter % 2 = 1 then !predicate else predicate
        let p := combine (parameter / 2 % 3) comparison third
        let q := combine (parameter / 2 % 3) (!comparison) third
        return (bit p + 2 * bit q, 0)
    else if op = "sel32" || op = "sel64" then
      if parameter > 1 then none else do
        let choose := if parameter = 1 then !predicate else predicate
        let bits := if op = "sel32" then 32 else 64
        return ((if choose then a else b) % 2 ^ bits, 0)
    else if op = "plop" then
      if parameter ≥ 256 then none else do
        let pa := bit (a % 2 ^ 32 != 0)
        let pb := bit (b % 2 ^ 32 != 0)
        let pc := bit predicate
        return (parameter / 2 ^ (4 * pa + 2 * pb + pc) % 2, pa + 2 * pb + 4 * pc)
    else if op = "transfer_update" || op = "transfer_read" then
      if parameter ≥ 256 then none else do
        let writeMask := if op = "transfer_update" then parameter % 128 else 0
        let readMask := if op = "transfer_update" then 255 else parameter
        let predicates := (a &&& (127 ^^^ writeMask)) ||| (b &&& writeMask)
        return ((c &&& (mask32 ^^^ readMask)) ||| (predicates &&& readMask), 0)
    else none

theorem signed_unsigned_boundary :
    evaluate "setp32" true 12 0x80000000 0 1 0 = some (1, 0) ∧
    evaluate "setp32" false 12 0x80000000 0 1 0 = some (2, 0) := by decide

theorem and_can_clear_both_destinations :
    evaluate "setp32" false 0 1 1 0 0 = some (0, 0) := by decide

theorem select_preserves_bits :
    evaluate "sel64" false 0 0xffff0000ffff0000 0 1 0 = some (0xffff0000ffff0000, 0) := by decide

theorem plop_xor :
    evaluate "plop" false 0x96 1 1 1 0 = some (1, 7) := by decide

theorem packed_predicate_bit_seven_is_zero :
    evaluate "transfer_update" false 255 255 255 0 0 = some (127, 0) := by decide

#print axioms packed_predicate_bit_seven_is_zero
#print axioms signed_unsigned_boundary
#print axioms and_can_clear_both_destinations
#print axioms select_preserves_bits
#print axioms plop_xor

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.IntegerPredicates

open GB300Math.IntegerPredicates

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [op, mode, parameter, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    unless mode = "0" || mode = "1" do throw (IO.userError "Invalid signedness")
    let some shift := parameter.toNat? | throw (IO.userError "Invalid shift")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 2 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let b := readWord xs (32 * i + 8)
      let c := readWord xs (32 * i + 16)
      let d := readWord xs (32 * i + 24)
      let some model := evaluate op (mode = "1") shift a b c d |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := (readWord ys (16 * i), readWord ys (16 * i + 8))
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {op} signed={mode} shift={shift} row={i} a={a} b={b} c={c} d={d} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
