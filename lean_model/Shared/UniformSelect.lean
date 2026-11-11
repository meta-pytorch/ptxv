/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.UniformSelect

def evaluate (mode a b d : Nat) : Option Nat :=
  if mode ≥ 2 || d > 3 then none else
    some (if (d % 2 = 1) != (mode = 1) then a % 2^32 else b % 2^32)

theorem true_selects_first (a b : Nat) : evaluate 0 a b 1 = some (a % 2^32) := by simp [evaluate]
theorem false_selects_second (a b : Nat) : evaluate 0 a b 0 = some (b % 2^32) := by simp [evaluate]
theorem negated_true_selects_second (a b : Nat) : evaluate 1 a b 1 = some (b % 2^32) := by simp [evaluate]
theorem selected_bits_are_preserved : evaluate 0 0x7fa12345 0 1 = some 0x7fa12345 := by decide
#print axioms true_selects_first
#print axioms false_selects_second
#print axioms negated_true_selects_second
#print axioms selected_bits_are_preserved

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.UniformSelect

open GB300Math.UniformSelect

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
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 16] do
      let a := readWord xs (16 * i)
      let b := readWord xs (16 * i + 4)
      let some model := evaluate parameter a b (readWord xs (16 * i + 12)) |
        throw (IO.userError s!"Unsupported mode: {parameter}")
      let actual := readWord ys (4 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness mode={parameter} row={i} a={a} b={b} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 16} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
