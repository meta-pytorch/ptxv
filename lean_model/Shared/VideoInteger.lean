/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.VideoInteger

def base : Nat := 4294967296

def byteValue (signed : Bool) (word : Nat) : Int :=
  let value := word % 256
  if signed && value ≥ 128 then (value : Int) - 256 else value

def difference (signed : Bool) (a b : Nat) : Nat :=
  (byteValue signed a - byteValue signed b).natAbs

def encode (total : Nat) : Nat × Nat :=
  (total % base, if total ≥ base then 1 else 0)

def evaluate (signed accumulate : Bool) (a b c : Nat) : Nat × Nat := Id.run do
  let mut packed := 0
  let mut sum := 0
  for lane in [:4] do
    let value := difference signed (a / 2 ^ (8 * lane)) (b / 2 ^ (8 * lane))
    packed := packed + value * 2 ^ (8 * lane)
    sum := sum + value
  return if accumulate then encode (c % base + sum) else (packed, 0)

def evaluateScalar (signed : Bool) (a b c : Nat) : Nat × Nat :=
  let value := fun (x : Nat) =>
    let word := x % base
    if signed && word ≥ 2147483648 then (word : Int) - base else (word : Int)
  encode ((value a - value b).natAbs + c % base)

theorem scalar_signed_carry :
    evaluateScalar true 0x80000000 0x7fffffff 1 = (0, 1) ∧
    evaluateScalar false 0x80000000 0x7fffffff 1 = (2, 0) := by decide +kernel

theorem reconstruct_accumulate (total : Nat) (h : total < 2 * base) :
    (encode total).1 + base * (encode total).2 = total := by
  unfold base at h
  by_cases h1 : 4294967296 ≤ total <;> simp [encode, base, h1] <;> omega

theorem packed_ignores_accumulator :
    evaluate false false 0xff00807f 0x007fff80 0xffffffff = (0xff7f7f01, 0) := by decide +kernel

theorem signed_bytes_differ :
    evaluate true false 0x7f7f7f7f 0x80808080 0 = (0xffffffff, 0) ∧
    evaluate false false 0x7f7f7f7f 0x80808080 0 = (0x01010101, 0) := by decide +kernel

theorem accumulator_carries :
    evaluate false true 0xffffffff 0 0xffffffff = (1019, 1) := by decide +kernel

theorem packed_never_sets_carry :
    evaluate false false 0xffffffff 0 0xffffffff = (0xffffffff, 0) := by decide +kernel

#print axioms scalar_signed_carry
#print axioms reconstruct_accumulate
#print axioms packed_ignores_accumulator
#print axioms signed_bytes_differ
#print axioms accumulator_carries
#print axioms packed_never_sets_carry

def readWord (data : ByteArray) (offset width : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:width] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.VideoInteger

open GB300Math.VideoInteger

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [mode, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some parameter := mode.toNat? | throw (IO.userError "Invalid mode")
    unless parameter < 6 do throw (IO.userError "Unsupported mode")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size / 2 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 16] do
      let a := readWord xs (16 * i) 4
      let b := readWord xs (16 * i + 4) 4
      let c := readWord xs (16 * i + 8) 4
      let model := if parameter < 4 then evaluate (parameter / 2 = 1) (parameter % 2 = 1) a b c
        else evaluateScalar (parameter = 5) a b c
      let actual := (readWord ys (8 * i) 4, readWord ys (8 * i + 4) 4)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness mode={parameter} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 16} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
