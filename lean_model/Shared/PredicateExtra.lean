/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.PredicateExtra

def bit (value : Bool) : Nat := if value then 1 else 0

def lut (table a b c : Nat) : Nat := table / 2 ^ (4 * a + 2 * b + c) % 2

def evaluate (op : String) (_signed : Bool) (parameter a b c d : Nat) : Option (Nat × Nat) := do
  if a ≥ 2 ^ 64 || b ≥ 2 ^ 64 || c ≥ 2 ^ 64 || d ≥ 2 ^ 64 then none else do
    if op = "plop_pair" then
      if parameter ≥ 65536 then none else do
        let pa := bit (a % 2 ^ 32 != 0)
        let pb := bit (b % 2 ^ 32 != 0)
        let pc := bit (c % 2 ^ 32 != 0)
        let primary := lut (parameter % 256) pa pb pc
        let secondary := lut (parameter / 256) pa pb pc
        return (primary, pa + 2 * pb + 4 * secondary)
    else if op = "r2p" || op = "p2r" then
      if parameter ≥ 1024 then none else do
        let position := parameter / 256
        let mask := parameter % 256
        let lowMask := mask % 128
        let predicates := if op = "r2p" then
          (a &&& (127 ^^^ lowMask)) ||| ((b / 2 ^ (8 * position)) &&& lowMask)
          else a % 128
        let readMask := if op = "r2p" then 255 else mask * 2 ^ (8 * position)
        let shifted := if op = "r2p" then predicates else predicates * 2 ^ (8 * position)
        return ((c &&& (0xffffffff ^^^ readMask)) ||| (shifted &&& readMask), 0)
    else none

theorem distinct_lut_results :
    evaluate "plop_pair" false (0x80 + 0x7f * 256) 1 1 1 0 = some (1, 3) := by decide

theorem r2p_reads_selected_byte :
    evaluate "r2p" false (256 + 255) 0 0x1200 0 0 = some (0x12, 0) := by decide

theorem p2r_preserves_other_bytes :
    evaluate "p2r" false (3 * 256 + 255) 0x12 0 0xffffffff 0 = some (0x12ffffff, 0) := by decide

theorem p2r_clears_selected_bit_seven :
    evaluate "p2r" false (2 * 256 + 128) 255 0 0xffffffff 0 = some (0xff7fffff, 0) := by decide

#print axioms distinct_lut_results
#print axioms r2p_reads_selected_byte
#print axioms p2r_preserves_other_bytes
#print axioms p2r_clears_selected_bit_seven

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.PredicateExtra

open GB300Math.PredicateExtra

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
