/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.UniformWideCarry

def base : Nat := 18446744073709551616

def encode (total : Nat) : Nat × Nat := (total % base, if total ≥ base then 1 else 0)

def evaluate (mode a b c d carry : Nat) : Option (Nat × Nat) := do
  if mode ≥ 4 || carry > 1 then none else do
    let operand : Nat → Int := fun (x : Nat) => if mode ≥ 2 && x % 2^32 ≥ 2^31 then ((x % 2^32 : Nat) : Int) - 2^32 else (x % 2^32 : Nat)
    let product := ((operand a * operand b) % (base : Int)).toNat
    return encode (product + c % 2^32 + (d % 2^32) * 2^32 + (carry ^^^ (mode % 2)))

theorem reconstruct_total (total : Nat) (h : total < 2 * base) :
    total = (encode total).1 + base * (encode total).2 := by
  unfold base at h
  by_cases hx : 18446744073709551616 ≤ total <;> simp [encode, base, hx] <;> omega

theorem carry_is_a_bit (total : Nat) : (encode total).2 ≤ 1 := by
  simp [encode]; split <;> omega

theorem carry_crosses_low_word : evaluate 0 1 1 0xfffffffe 0 1 = some (0x100000000, 0) := by decide
theorem carry_crosses_full_word : evaluate 0 0 0 0xffffffff 0xffffffff 1 = some (0, 1) := by decide
theorem signed_product_carry : evaluate 2 0xffffffff 1 1 0 0 = some (0, 1) := by decide
theorem inverted_input_carry : evaluate 1 0 0 0 0 0 = some (1, 0) := by decide
#print axioms reconstruct_total
#print axioms carry_is_a_bit
#print axioms carry_crosses_low_word
#print axioms carry_crosses_full_word
#print axioms signed_product_carry
#print axioms inverted_input_carry

def readWord (data : ByteArray) (offset count : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:count] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.UniformWideCarry

open GB300Math.UniformWideCarry

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [parameter, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some param := parameter.toNat? | throw (IO.userError "Invalid instruction parameter")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 2 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i) 4
      let b := readWord xs (32 * i + 4) 4
      let c := readWord xs (32 * i + 8) 4
      let d := readWord xs (32 * i + 12) 4
      let some model := evaluate param a b c d (readWord xs (32 * i + 16) 4) |
        throw (IO.userError s!"Unsupported parameter: {param}")
      let actual := (readWord ys (16 * i) 8, readWord ys (16 * i + 8) 8)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness parameter={param} row={i} a={a} b={b} c={c} d={d} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
