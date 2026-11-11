/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.Carryless

def productPrefix (a b : BitVec 64) : Nat → BitVec 128
  | 0 => 0
  | n + 1 => productPrefix a b n ^^^
      (if a.toNat.testBit n then (BitVec.ofNat 128 b.toNat) <<< n else 0)

def product (a b : BitVec 64) : BitVec 128 := productPrefix a b 64

def evaluate (high : Bool) (a b c : BitVec 64) : BitVec 64 :=
  BitVec.ofNat 64 ((product a b).toNat >>> (if high then 64 else 0)) ^^^ c

theorem productPrefix_zero_left (b : BitVec 64) (n : Nat) :
    productPrefix 0 b n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [productPrefix, ih]; simp

theorem productPrefix_zero_right (a : BitVec 64) (n : Nat) :
    productPrefix a 0 n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [productPrefix, ih]; simp

theorem zero_left (high : Bool) (b c : BitVec 64) :
    evaluate high 0 b c = c := by
  rw [evaluate, product, productPrefix_zero_left]
  simp

theorem zero_right (high : Bool) (a c : BitVec 64) :
    evaluate high a 0 c = c := by
  rw [evaluate, product, productPrefix_zero_right]
  simp

theorem accumulator_xor (high : Bool) (a b c delta : BitVec 64) :
    evaluate high a b (c ^^^ delta) = evaluate high a b c ^^^ delta := by
  simp [evaluate, BitVec.xor_assoc]

theorem high_bit_product : evaluate true 0x8000000000000000 2 0 = 1 := by decide
theorem discarded_low_bit_product : evaluate false 0x8000000000000000 2 0 = 0 := by decide
theorem xor_is_not_integer_add : evaluate false 3 3 1 = 4 := by decide

#print axioms zero_left
#print axioms zero_right
#print axioms accumulator_xor
#print axioms high_bit_product
#print axioms discarded_low_bit_product
#print axioms xor_is_not_integer_add

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset+j]!.toNat * 2^(8*j)
  return word

end GB300Math.Carryless

open GB300Math.Carryless

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected carryless case manifest")
  let cases ← IO.FS.readFile manifest
  unless (cases.splitOn "\n").any (! ·.isEmpty) do
    throw (IO.userError "Empty carryless manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [mode, input, output, errors] := line.splitOn "\t" |
      throw (IO.userError "Malformed carryless case")
    unless mode = "hi" ∨ mode = "lo" do throw (IO.userError "Invalid carryless mode")
    let some expectedErrors := errors.toNat? | throw (IO.userError "Invalid expected error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 24 = 0 ∧ ys.size = xs.size / 24 * 8 do
      throw (IO.userError "Invalid carryless record sizes")
    let mut mismatches := 0
    for i in [:xs.size / 24] do
      let a := BitVec.ofNat 64 (readWord xs (24*i))
      let b := BitVec.ofNat 64 (readWord xs (24*i+8))
      let c := BitVec.ofNat 64 (readWord xs (24*i+16))
      let actual := readWord ys (8*i)
      let model := (evaluate (mode == "hi") a b c).toNat
      if model ≠ actual then
        if mismatches < 3 then IO.println s!"witness {mode} row={i} model={model} actual={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {mode} records={xs.size / 24} mismatches={mismatches} expected={expectedErrors}"
    if mismatches ≠ expectedErrors then failures := failures + 1
  return if failures = 0 then 0 else 1
