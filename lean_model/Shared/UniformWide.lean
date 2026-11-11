/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.UniformWide

def base : Nat := 4294967296

def signed32 (x : Nat) : Int := if x < 2147483648 then x else (x : Int) - base

def evaluate (mode a0 b0 c0 d0 : Nat) : Option Nat := do
  if mode ≥ 2 then none else do
    let a : Int := if mode = 1 then signed32 (a0 % base) else a0 % base
    let b : Int := if mode = 1 then signed32 (b0 % base) else b0 % base
    let c := c0 % base + (d0 % base) * base
    return ((a * b + c) % ((base : Int) * base)).toNat

theorem full_product : evaluate 0 0xffffffff 0xffffffff 0 0 = some 0xfffffffe00000001 := by decide
theorem signed_product : evaluate 1 0xffffffff 0xffffffff 0 0 = some 1 := by decide
theorem carry_to_upper_word : evaluate 0 1 1 0xffffffff 0 = some 0x100000000 := by decide
theorem full_width_wrap : evaluate 0 1 1 0xffffffff 0xffffffff = some 0 := by decide
theorem sign_extended_product : evaluate 1 0x80000000 1 0 0 = some 0xffffffff80000000 := by decide
#print axioms full_product
#print axioms signed_product
#print axioms carry_to_upper_word
#print axioms full_width_wrap
#print axioms sign_extended_product

def readWord (data : ByteArray) (offset count : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:count] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.UniformWide

open GB300Math.UniformWide

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
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size / 2 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 16] do
      let a := readWord xs (16 * i) 4
      let b := readWord xs (16 * i + 4) 4
      let some model := evaluate parameter a b (readWord xs (16 * i + 8) 4) (readWord xs (16 * i + 12) 4) |
        throw (IO.userError s!"Unsupported mode: {parameter}")
      let actual := readWord ys (8 * i) 8
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness mode={parameter} row={i} a={a} b={b} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 16} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
