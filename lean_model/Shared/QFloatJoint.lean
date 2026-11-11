/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.QFloatModifiers

namespace GB300Math.QFloatJoint

open GB300Math.AlternateFloat GB300Math.QFloatModifiers

def operand (fmt : String) (word lane modifier : Nat) : Option Value := do
  let value ← decode fmt (select (defaultSelector fmt) word lane)
  return modify modifier value

def evaluate (op dst af bf : String) (sat : Bool) (ma mb mc a b c : Nat) : Option Nat := do
  unless legal op dst af bf ∧ op ≠ "sub" ∧ ma < 4 ∧ mb < 4 ∧ mc < 4 do none
  unless a < 2^32 ∧ b < 2^32 ∧ c < 2^32 do none
  let mut output := 0
  for lane in [:4] do
    let x ← operand af a lane ma
    let y ← operand bf b lane mb
    let z ← operand dst c lane mc
    let value := if op = "add" then add x y else if op = "mul" then mul x y else add (mul x y) z
    output := output + encode dst sat value * 2^(8*lane)
  return output

theorem identity_modifier (value : Value) : modify 0 value = value := by rfl

theorem double_negation (value : Value) : modify 2 (modify 2 value) = value := by
  exact negate_involution value

theorem abs_negation (value : Value) : modify 3 value = negate (absolute value) := by rfl

#print axioms identity_modifier
#print axioms double_negation
#print axioms abs_negation

end GB300Math.QFloatJoint

open GB300Math.AlternateFloat

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected Q modifier manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failures := 0
  for row in rows do
    let [op, dst, af, bf, sat, ma, mb, mc, input, output, errors] := row.splitOn "\t" |
      throw (IO.userError "Malformed Q modifier row")
    let some ma := ma.toNat? | throw (IO.userError "Invalid A modifier")
    let some mb := mb.toNat? | throw (IO.userError "Invalid B modifier")
    let some mc := mc.toNat? | throw (IO.userError "Invalid C modifier")
    let some expected := errors.toNat? | throw (IO.userError "Invalid mismatch count")
    unless sat = "0" ∨ sat = "1" do throw (IO.userError "Invalid saturation flag")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size do
      throw (IO.userError "Invalid input/output record sizes")
    let mut mismatches := 0
    for i in [:xs.size/16] do
      let a := read32 xs (16*i)
      let b := read32 xs (16*i+4)
      let c := read32 xs (16*i+8)
      unless read32 xs (16*i+12) = 0x93b471ea do throw (IO.userError "Invalid sentinel")
      let some target := GB300Math.QFloatJoint.evaluate op dst af bf (sat = "1") ma mb mc a b c |
        throw (IO.userError "Unsupported native configuration")
      let actual := read32 ys (16*i)
      if actual ≠ target ∨ read32 ys (16*i+4) ≠ a ∨ read32 ys (16*i+8) ≠ b ∨ read32 ys (16*i+12) ≠ c then
        if mismatches < 5 then IO.println s!"witness row={i} a={a} b={b} c={c} expected={target} actual={actual}"
        mismatches := mismatches+1
    IO.println s!"result {output} records={xs.size/16} mismatches={mismatches} expected={expected}"
    if mismatches ≠ expected then failures := failures+1
  return if failures = 0 then 0 else 1
