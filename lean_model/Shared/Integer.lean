/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.Integer

def modulus : Nat := 4294967296

def signed (x : Nat) : Int :=
  let word := x % modulus
  if word < 2147483648 then word else (word : Int) - modulus

def wrap (bits : Nat) (x : Int) : Nat := (x % (2 ^ bits : Nat)).toNat

def lop3 (a b c table : Nat) : Nat := Id.run do
  let mut result := 0
  for i in [:8] do
    if table / 2 ^ i % 2 = 1 then
      let x := if i / 4 % 2 = 1 then a else a ^^^ 0xffffffff
      let y := if i / 2 % 2 = 1 then b else b ^^^ 0xffffffff
      let z := if i % 2 = 1 then c else c ^^^ 0xffffffff
      result := result ||| (x &&& y &&& z)
  return result

def permute (a b selector : Nat) : Nat := Id.run do
  let pair := b * modulus + a
  let mut result := 0
  for lane in [:4] do
    let choice := selector / 2 ^ (4 * lane) % 16
    let byte := pair / 2 ^ (8 * (choice % 8)) % 256
    let value := if choice ≥ 8 then (if byte ≥ 128 then 255 else 0) else byte
    result := result + value * 2 ^ (8 * lane)
  return result

def leading (x : Nat) : Nat := Id.run do
  let mut count := 0
  for i in [:32] do
    if x / 2 ^ (31 - i) % 2 = 0 && count = i then count := count + 1
  return count

def evaluate (op mode : String) (param a0 b0 c0 : Nat) : Option Nat := do
  let a := a0 % modulus
  let b := b0 % modulus
  let c := c0 % modulus
  let x : Int := if mode = "s32" then signed a else a
  let y : Int := if mode = "s32" then signed b else b
  match op with
  | "add" => return (a + b) % modulus
  | "sub" => return wrap 32 ((a : Int) - b)
  | "add_sat" => return wrap 32 (max (-2147483648) (min 2147483647 (signed a + signed b)))
  | "add3" =>
    let arg := fun (v index : Nat) => if param / 2 ^ index % 2 = 1 then -(v : Int) else (v : Int)
    return wrap 32 (arg a 0 + arg b 1 + arg c 2)
  | "neg" => return wrap 32 (-(a : Int))
  | "abs" => return (signed a).natAbs
  | "min" => return wrap 32 (min x y)
  | "max" => return wrap 32 (max x y)
  | "mul_lo" => return wrap 32 (x * y)
  | "mul_hi" => return wrap 32 ((x * y) / modulus)
  | "mul_wide" => return wrap 64 (x * y)
  | "mad_lo" => return wrap 32 (x * y + c)
  | "mad_hi" => return wrap 32 ((x * y) / modulus + c)
  | "mad_wide" => return wrap 64 (x * y + c0)
  | "sad" => return ((x - y).natAbs + c) % modulus
  | "shl" => return (a * 2 ^ (min b 32)) % modulus
  | "shr" => return a / 2 ^ (min b 32)
  | "shra" => return wrap 32 (signed a / 2 ^ (min b 32))
  | "shfl_wrap" | "shfl_clamp" | "shfr_wrap" | "shfr_clamp" =>
    let shift := if op.endsWith "wrap" then c % 32 else min c 32
    let pair := a + b * modulus
    if op.startsWith "shfl" then return pair / 2 ^ (32 - shift) % modulus
    else return pair / 2 ^ shift % modulus
  | "lop3" => return lop3 a b c param
  | "prmt" => return permute a b c
  | "clz" => return leading a
  | "bfind" | "bfind_shift" =>
    let word := if mode = "s32" && a ≥ 2147483648 then a ^^^ 0xffffffff else a
    if word = 0 then return 0xffffffff
    return if op = "bfind_shift" then leading word else 31 - leading word
  | "brev" =>
    let mut result := 0
    for i in [:32] do result := result + a / 2 ^ i % 2 * 2 ^ (31 - i)
    return result
  | "popc" =>
    let mut result := 0
    for i in [:32] do result := result + a / 2 ^ i % 2
    return result
  | _ => none

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.Integer

open GB300Math.Integer

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [op, mode, parameter, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some param := parameter.toNat? | throw (IO.userError "Invalid instruction parameter")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let b := readWord xs (32 * i + 8)
      let c := readWord xs (32 * i + 16)
      let some model := evaluate op mode param a b c | throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {op} {mode} param={param} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
