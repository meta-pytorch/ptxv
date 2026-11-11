/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.Predicates

structure Format where
  width : Nat
  fraction : Nat
  deriving Repr

def format (name : String) : Option Format :=
  match name with
  | "f16" | "f16x2" => some ⟨16, 10⟩
  | "bf16x2" => some ⟨16, 7⟩
  | "f32" => some ⟨32, 23⟩
  | "f64" => some ⟨64, 52⟩
  | _ => none

def Format.sign (f : Format) : Nat := 2 ^ (f.width - 1)
def Format.infinity (f : Format) : Nat := f.sign - 2 ^ f.fraction

def flush (f : Format) (ftz : Bool) (a : Nat) : Nat :=
  let x := a % 2 ^ f.width
  if ftz && x % f.sign < 2 ^ f.fraction then x / f.sign * f.sign else x

def isNaN (f : Format) (a : Nat) : Bool := a % f.sign > f.infinity

def less (f : Format) (a b : Nat) : Bool :=
  let ma := a % f.sign
  let mb := b % f.sign
  if ma = 0 && mb = 0 then false
  else if (a ≥ f.sign) != (b ≥ f.sign) then a ≥ f.sign
  else if a ≥ f.sign then ma > mb else ma < mb

def totalLess (f : Format) (a b : Nat) : Bool :=
  if (a ≥ f.sign) != (b ≥ f.sign) then a ≥ f.sign
  else if a ≥ f.sign then a % f.sign > b % f.sign else a % f.sign < b % f.sign

def compare (f : Format) (op : String) (a b : Nat) : Option Bool := do
  let unordered := isNaN f a || isNaN f b
  let equal := a == b || (a % f.sign == 0 && b % f.sign == 0)
  let lt := less f a b
  let gt := less f b a
  match op with
  | "min_select" => return !isNaN f a && (isNaN f b || !totalLess f b a)
  | "max_select" => return !isNaN f a && (isNaN f b || totalLess f b a)
  | "false" => return false
  | "true" => return true
  | "eq" => return !unordered && equal
  | "ne" => return !unordered && !equal
  | "lt" => return !unordered && lt
  | "le" => return !unordered && !gt
  | "gt" => return !unordered && gt
  | "ge" => return !unordered && !lt
  | "equ" => return unordered || equal
  | "neu" => return unordered || !equal
  | "ltu" => return unordered || lt
  | "leu" => return unordered || !gt
  | "gtu" => return unordered || gt
  | "geu" => return unordered || !lt
  | "num" => return !unordered
  | "nan" => return unordered
  | _ => none

def combine (op : String) (p q : Bool) : Option Bool :=
  match op with
  | "and" => some (p && q)
  | "or" => some (p || q)
  | "xor" => some (p != q)
  | _ => none

def select (f : Format) (maximum propagate : Bool) (a b : Nat) : Nat :=
  let nx := isNaN f a
  let ny := isNaN f b
  if nx || ny then
    if propagate || (nx && ny) then f.sign - 1 else if nx then b else a
  else if a % f.sign = 0 && b % f.sign = 0 then
    if maximum then a &&& b else a ||| b
  else if less f a b = maximum then b else a

def signModifier (f : Format) (op : String) (a : Nat) : Option Nat :=
  match op with
  | "id" => some (a % 2 ^ f.width)
  | "neg" => some ((a % 2 ^ f.width) ^^^ f.sign)
  | "abs" => some (a % f.sign)
  | "negabs" => some (a % f.sign + f.sign)
  | _ => none

def bit (p : Bool) : Nat := if p then 1 else 0

def evaluate (kind fmt op comb : String) (neg ftz bits propagate : Bool)
    (a b c pred : Nat) : Option Nat := do
  let f ← format fmt
  let p := (pred % 2 ^ 32 != 0) != neg
  if kind = "fsel" then
    return ← signModifier f (if p then op else comb) (if p then a else b)
  if kind = "selp" then return (if p then a else b) % 2 ^ f.width
  if kind = "slct" then
    let z := flush f ftz c
    return (if !isNaN f z && !less f z 0 then a else b) % 2 ^ f.width
  let x := flush f ftz a
  let y := flush f ftz b
  if kind = "min" || kind = "max" then
    let lo := select f (kind = "max") propagate x y
    if !fmt.endsWith "x2" then return lo
    return lo + 2 ^ f.width * select f (kind = "max") propagate
      (flush f ftz (a / 2 ^ f.width)) (flush f ftz (b / 2 ^ f.width))
  let comparison ← compare f op x y
  let low ← combine comb comparison p
  if kind = "setp" then
    if fmt = "f16" then return bit low
    let other ← if fmt.endsWith "x2" then
      compare f op (flush f ftz (a / 2 ^ f.width)) (flush f ftz (b / 2 ^ f.width))
      else some (!comparison)
    let high ← combine comb other p
    return bit low + 2 * bit high
  if kind != "set" then none else do
    let truth := if bits then 2 ^ f.width - 1
      else ((2 ^ (f.width - f.fraction - 2)) - 1) * 2 ^ f.fraction
    if !fmt.endsWith "x2" then return bit low * truth
    let comparisonHigh ← compare f op (flush f ftz (a / 2 ^ f.width)) (flush f ftz (b / 2 ^ f.width))
    let high ← combine comb comparisonHigh p
    return bit low * truth + 2 ^ f.width * bit high * truth

theorem ordered_nan_false (f : Format) (a b : Nat) (h : isNaN f a = true) :
    compare f "eq" a b = some false := by simp [compare, h]

theorem unordered_nan_true (f : Format) (a b : Nat) (h : isNaN f a = true) :
    compare f "equ" a b = some true := by simp [compare, h]

theorem and_false (p : Bool) : combine "and" p false = some false := by
  cases p <;> rfl

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.Predicates

open GB300Math.Predicates

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [kind, fmt, op, comb, neg, ftz, bits, propagate, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let b := readWord xs (32 * i + 8)
      let c := readWord xs (32 * i + 16)
      let p := readWord xs (32 * i + 24)
      let some model := evaluate kind fmt op comb (neg = "1") (ftz = "1") (bits = "1")
        (propagate = "1") a b c p | throw (IO.userError s!"Unsupported case: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {kind} {fmt} {op} {comb} row={i} a={a} b={b} c={c} p={p} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
