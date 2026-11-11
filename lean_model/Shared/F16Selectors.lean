/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.F16Selectors

structure Format where
  exponent : Nat
  fraction : Nat
  deriving Repr

def Format.width (f : Format) : Nat := f.exponent + f.fraction + 1
def Format.bias (f : Format) : Nat := 2 ^ (f.exponent - 1) - 1
def Format.infinity (f : Format) : Nat := (2 ^ f.exponent - 1) * 2 ^ f.fraction

def format (name : String) : Option Format :=
  match name with
  | "f16" => some ⟨5, 10⟩
  | "bf16" => some ⟨8, 7⟩
  | "f32" => some ⟨8, 23⟩
  | "f64" => some ⟨11, 52⟩
  | _ => none

def integerFormat (name : String) : Option (Nat × Bool) :=
  match name with
  | "s8" => some (8, true)
  | "u8" => some (8, false)
  | "s16" => some (16, true)
  | "u16" => some (16, false)
  | "s32" => some (32, true)
  | "u32" => some (32, false)
  | "s64" => some (64, true)
  | "u64" => some (64, false)
  | _ => none

def roundMagnitude (mode : String) (negative : Bool) (n shift : Nat) : Nat :=
  let unit := 2 ^ shift
  let q := n / unit
  let r := n % unit
  let up : Bool := match mode with
    | "rn" => 2 * r > unit ∨ (2 * r = unit ∧ q % 2 = 1)
    | "ra" => 2 * r ≥ unit
    | "ru" => !negative ∧ r ≠ 0
    | "rd" => negative ∧ r ≠ 0
    | _ => false
  q + if up then 1 else 0

def integerBits (negative : Bool) (magnitude : Nat) : Nat :=
  if negative ∧ magnitude ≠ 0 then (2 ^ 64 - magnitude) % 2 ^ 64 else magnitude

def decode (f : Format) (a : Nat) : Bool × Nat × Int :=
  let negative := a % 2 ^ f.width ≥ 2 ^ (f.width - 1)
  let exponent := a / 2 ^ f.fraction % 2 ^ f.exponent
  let fraction := a % 2 ^ f.fraction
  (negative, fraction + if exponent = 0 then 0 else 2 ^ f.fraction,
    (max 1 exponent : Int) - f.bias - f.fraction)

def roundedInteger (mode : String) (negative : Bool) (mantissa : Nat) (exponent : Int) : Nat :=
  if exponent ≥ 0 then mantissa * 2 ^ exponent.toNat
  else roundMagnitude mode negative mantissa (-exponent).toNat

def toInteger (f : Format) (width : Nat) (signed : Bool) (mode : String) (a : Nat)
    (ntz : Bool := true) : Nat := Id.run do
  let magnitude := a % 2 ^ (f.width - 1)
  if magnitude > f.infinity then
    return if ntz ∧ width ≠ 64 ∧ f.width ≠ 64 then 0 else integerBits signed (2 ^ (width - 1))
  let (negative, mantissa, exponent) := decode f a
  let limit := if signed then 2 ^ (width - 1) - if negative then 0 else 1
               else if negative then 0 else 2 ^ width - 1
  let rounded := if magnitude = f.infinity then limit
                 else min limit (roundedInteger mode negative mantissa exponent)
  return integerBits negative rounded

def encodeInteger (f : Format) (mode : String) (negative : Bool) (magnitude : Nat) : Nat := Id.run do
  let sign := if negative then 2 ^ (f.width - 1) else 0
  if magnitude = 0 then return sign
  let top := magnitude.log2
  let mantissa := if top ≤ f.fraction then magnitude * 2 ^ (f.fraction - top)
                  else roundMagnitude mode negative magnitude (top - f.fraction)
  let exponent := top + f.bias + if mantissa ≥ 2 ^ (f.fraction + 1) then 1 else 0
  let normalized := if mantissa ≥ 2 ^ (f.fraction + 1) then mantissa / 2 else mantissa
  if exponent ≥ 2 ^ f.exponent - 1 then
    let toInfinity : Bool := mode = "rn" ∨ mode = "ra" ∨ (mode = "ru" ∧ !negative) ∨ (mode = "rd" ∧ negative)
    return sign + f.infinity - if toInfinity then 0 else 1
  return sign + exponent * 2 ^ f.fraction + normalized - 2 ^ f.fraction

def fromInteger (f : Format) (width : Nat) (signed : Bool) (mode : String) (a : Nat) : Nat :=
  let bits := a % 2 ^ width
  let negative := signed ∧ bits ≥ 2 ^ (width - 1)
  encodeInteger f mode negative (if negative then 2 ^ width - bits else bits)

def integral (f : Format) (mode : String) (a : Nat) : Nat := Id.run do
  let bits := a % 2 ^ f.width
  let magnitude := bits % 2 ^ (f.width - 1)
  if magnitude > f.infinity then
    return if f.width = 64 then bits ||| 2 ^ (f.fraction - 1) else 2 ^ (f.width - 1) - 1
  if magnitude = f.infinity then return bits
  let (negative, mantissa, exponent) := decode f bits
  if exponent ≥ 0 then return bits
  return encodeInteger f "rn" negative (roundedInteger mode negative mantissa exponent)

def apply (src dst mode : String) (a : Nat) : Option Nat := do
  if dst = "integral" ∨ dst = "integral_ftz" then
    if src.endsWith "x2" then
      let f ← format (src.replace "x2" "")
      return integral f mode (a % 2 ^ f.width) + integral f mode (a / 2 ^ f.width) * 2 ^ f.width
    let f ← format src
    let value := if dst = "integral_ftz" ∧ a % 2 ^ (f.width - 1) < 2 ^ f.fraction
                 then a / 2 ^ (f.width - 1) * 2 ^ (f.width - 1) else a
    return integral f mode value
  if let some f := format (src.replace "_ftz" "") then
    let (width, signed) ← integerFormat (dst.replace "_raw" "")
    let value := if src.endsWith "_ftz" ∧ a % 2 ^ (f.width - 1) < 2 ^ f.fraction
                 then a / 2 ^ (f.width - 1) * 2 ^ (f.width - 1) else a
    return toInteger f width signed mode value (!dst.endsWith "_raw")
  let (width, signed) ← integerFormat src
  return fromInteger (← format dst) width signed mode a

def modify (width modifier input : Nat) : Nat :=
  let word := input % 2 ^ width
  let value := if modifier / 2 % 2 = 1 then word % 2 ^ (width - 1) else word
  if modifier % 2 = 1 then value ^^^ 2 ^ (width - 1) else value

def sourceBits (f : Format) (dst : String) (modifier input : Nat) : Nat :=
  if f.width = 64 ∧ dst = "integral" ∧ input % 2^63 > f.infinity then input
  else modify f.width modifier input

def widenHalf (raw modifier : Nat) : Nat :=
  let bits := modify 16 modifier raw
  let sign := bits / 32768 * 2^63
  let exponent := bits / 1024 % 32
  let fraction := bits % 1024
  if raw % 32768 > 0x7c00 then
    raw / 32768 * 2^63 + 0x7ff0000000000000 + ((raw % 1024 * 2^42) ||| 2^51)
  else if exponent = 31 then sign + 0x7ff0000000000000
  else if exponent > 0 then sign + (exponent + 1008) * 2^52 + fraction * 2^42
  else if fraction = 0 then sign
  else let top := fraction.log2
       sign + (top+999)*2^52 + (fraction*2^(52-top))%2^52

def evaluate (dst mode : String) (modifier selector ntz input : Nat) : Option Nat := do
  if modifier ≥ 4 || selector ≥ 2 || ntz ≥ 2 || mode ∉ ["rn", "rz", "ru", "rd"] then none else do
    let raw := input / 2^(16*selector) % 65536
    let bits := modify 16 modifier raw
    if dst = "f64" then
      if mode != "rn" || ntz != 0 then none else some (widenHalf raw modifier)
    else if dst = "integral" then
      if ntz != 0 then none else some (integral ⟨5,10⟩ mode bits)
    else do
      let (width, signed) ← integerFormat dst
      if width = 64 && ntz != 0 then none else do
        if bits % 32768 > 0x7c00 && ntz = 0 then return 2^(width-1)
        return toInteger ⟨5,10⟩ width signed mode bits (ntz = 1) % 2^(max 32 width)

theorem high_half_signed_minimum : evaluate "s8" "rn" 0 1 0 0xd8003c00 = some 0xffffff80 := by decide

theorem narrow_NaN_NTZ : evaluate "s8" "rn" 0 1 1 0x7c013c00 = some 0 := by decide

theorem narrow_NaN_without_NTZ : evaluate "s8" "rn" 0 1 0 0x7c013c00 = some 0x80 := by decide

theorem half_subnormal_to_double : evaluate "f64" "rn" 0 1 0 0x0001ffff = some 0x3e70000000000000 := by decide

theorem nan_forwarding_ignores_negation : evaluate "f64" "rn" 1 1 0 0x7c01ffff = some 0x7ff8040000000000 := by decide

#print axioms high_half_signed_minimum
#print axioms narrow_NaN_NTZ
#print axioms narrow_NaN_without_NTZ
#print axioms half_subnormal_to_double
#print axioms nan_forwarding_ignores_negation

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.F16Selectors

open GB300Math.F16Selectors

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [op, modeName, modsText, selectorText, ntzText, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    unless modeName ∈ ["rn", "rz", "ru", "rd"] do
      throw (IO.userError "Invalid rounding mode")
    let some mods := modsText.toNat? | throw (IO.userError "Invalid modifiers")
    let some selector := selectorText.toNat? | throw (IO.userError "Invalid selector")
    let some ntz := ntzText.toNat? | throw (IO.userError "Invalid NTZ flag")
    let some expected := errorsText.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let some model := evaluate op modeName mods selector ntz a |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {op} {modeName} mods={mods} selector={selector} ntz={ntz} row={i} a={a} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
