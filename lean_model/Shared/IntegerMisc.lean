/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.IntegerMisc

def base : Nat := 4294967296

def wrap (x : Int) : Nat := (x % (base : Int)).toNat

def signed32 (x : Nat) : Int :=
  if x % base < 2147483648 then x % base else (x % base : Int) - base

def evaluate (mode a0 b0 : Nat) : Option Nat := do
  if mode ≥ 10 then none else do
    let a := a0 % base
    let b := b0 % base
    if mode < 2 then
      let start := if mode = 1 then a % 32 else min a 32
      let width := if mode = 1 then b % 32 else min b 32
      return ((2 ^ width - 1) * 2 ^ start) % base
    else if mode < 6 then
      let width := if mode ≥ 4 then b % 32 else min b 32
      let low := a % 2 ^ width
      if mode % 2 = 1 && width > 0 && low ≥ 2 ^ (width - 1) then
        return wrap ((low : Int) - 2 ^ width)
      else return low
    else
      let bits := if mode < 8 then 8 else 16
      let signed := mode % 2 = 1
      let minimum : Int := if signed then -(2 ^ (bits - 1) : Nat) else 0
      let maximum : Int := (2 ^ (bits - if signed then 1 else 0) : Nat) - 1
      return wrap (max minimum (min maximum (signed32 a)))

theorem mask_count_32_clamps_or_wraps :
    evaluate 0 0 32 = some 0xffffffff ∧ evaluate 1 0 32 = some 0 := by decide

theorem mask_clips_at_word_end :
    evaluate 0 31 32 = some 0x80000000 ∧ evaluate 0 32 32 = some 0 := by decide

theorem extension_zero_width :
    evaluate 2 0xffffffff 0 = some 0 ∧ evaluate 3 0xffffffff 0 = some 0 := by decide

theorem extension_count_32_clamps_or_wraps :
    evaluate 3 0x80000000 32 = some 0x80000000 ∧ evaluate 5 0x80000000 32 = some 0 := by decide

theorem signed_extension_preserves_sign :
    evaluate 2 0x80 8 = some 0x80 ∧ evaluate 3 0x80 8 = some 0xffffff80 := by decide

theorem saturating_narrow_signed :
    evaluate 7 0x80000000 0 = some 0xffffff80 ∧ evaluate 9 0x7fffffff 0 = some 32767 := by decide

theorem saturating_narrow_unsigned :
    evaluate 6 0xffffffff 0 = some 0 ∧ evaluate 8 65536 0 = some 65535 := by decide

#print axioms mask_count_32_clamps_or_wraps
#print axioms mask_clips_at_word_end
#print axioms extension_zero_width
#print axioms extension_count_32_clamps_or_wraps
#print axioms signed_extension_preserves_sign
#print axioms saturating_narrow_signed
#print axioms saturating_narrow_unsigned

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.IntegerMisc

open GB300Math.IntegerMisc

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
      let some model := evaluate parameter a b |
        throw (IO.userError s!"Unsupported mode: {parameter}")
      let actual := readWord ys (4 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness mode={parameter} row={i} a={a} b={b} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 16} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
