/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu
import GB300Math.MufuSm103
import GB300Math.MufuPacked

namespace GB300Math.MufuSelector

def modify (width modifier input : Nat) : Nat :=
  let word := input % 2^width
  let value := if modifier / 2 % 2 = 1 then word % 2^(width-1) else word
  if modifier % 2 = 1 then value ^^^ 2^(width-1) else value

def evaluate (precision op : String) (modifier selector input : Nat) : Option Nat := do
  if precision = "f16x2" then
    if op = "ex2" then some (MufuPacked.evaluate modifier input) else none
  else if precision = "f16" then
    Mufu.evaluate (op ++ ".f16") (modify 16 modifier (input / 2^(16*selector)))
  else if precision = "bf16" then
    MufuSm103.evaluateBf16 op (modify 16 modifier (input / 2^(16*selector)))
  else if precision = "f32" then Mufu.evaluate op (modify 32 modifier input)
  else if precision = "f64h" then
    if op = "rcp64h" then some (Fp64Seed.highWord false (modify 32 modifier input))
    else if op = "rsq64h" then some (Fp64Seed.highWord true (modify 32 modifier input))
    else none
  else none

theorem packed_selector_independent (modifier selector input : Nat) :
    evaluate "f16x2" "ex2" modifier selector input =
    evaluate "f16x2" "ex2" modifier 0 input := by
  change some (MufuPacked.evaluate modifier input) = some (MufuPacked.evaluate modifier input)
  rfl

theorem scalar_selector_independent (op : String) (modifier selector input : Nat) :
    evaluate "f32" op modifier selector input =
    evaluate "f32" op modifier 0 input := by
  change Mufu.evaluate op (modify 32 modifier input) = Mufu.evaluate op (modify 32 modifier input)
  rfl

theorem high_half_ignores_low (lo hi : Nat) (h : lo < 65536) :
    (lo + hi * 65536) / 65536 = hi := by omega

theorem high_one : evaluate "f16" "ex2" 0 1 0x3c00ffff = some 0x4000 := by decide +kernel
theorem high_negated : evaluate "f16" "ex2" 1 1 0x3c000000 = some 0x3800 := by decide +kernel

#print axioms packed_selector_independent
#print axioms scalar_selector_independent
#print axioms high_half_ignores_low
#print axioms high_one
#print axioms high_negated

end GB300Math.MufuSelector

open GB300Math.MufuPacked

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected MUFU selector manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [precision, op, modifier, selector, input, actualOutput, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed MUFU selector row")
    let some modifier := modifier.toNat? | throw (IO.userError "Invalid modifier")
    let some selector := selector.toNat? | throw (IO.userError "Invalid selector")
    unless modifier < 4 ∧ selector < 4 do throw (IO.userError "Unsupported modifier/selector")
    if precision ∈ ["f16", "bf16"] ∧ selector > 1 then
      throw (IO.userError "Invalid half selector")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile actualOutput
    unless xs.size > 0 ∧ xs.size % 8 = 0 ∧ ys.size = xs.size do
      throw (IO.userError "Invalid record sizes")
    let mut errors := 0
    for i in [:xs.size/8] do
      let a := readWord xs (8*i)
      let some target := GB300Math.MufuSelector.evaluate precision op modifier selector (a % 2^32) |
        throw (IO.userError "Unsupported MUFU form")
      let actual := readWord ys (8*i)
      if actual ≠ target then
        if errors < 5 then IO.println s!"witness row={i} a={a} expected={target} actual={actual}"
        errors := errors+1
    IO.println s!"result {actualOutput} records={xs.size/8} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
