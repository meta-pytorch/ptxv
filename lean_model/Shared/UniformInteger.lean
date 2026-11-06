import Std

namespace GB300Math.UniformInteger

def base : Nat := 4294967296

def signed32 (x : Nat) : Int := if x < 2147483648 then x else (x : Int) - base

def lutGo (table a b c : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 =>
    let previous := lutGo table a b c i
    if table / 2 ^ i % 2 = 0 then previous else
      let av := if i / 4 = 1 then a else a ^^^ 0xffffffff
      let bv := if i / 2 % 2 = 1 then b else b ^^^ 0xffffffff
      let cv := if i % 2 = 1 then c else c ^^^ 0xffffffff
      previous ||| (av &&& bv &&& cv)

def lut (table a b c : Nat) : Nat := lutGo table a b c 8

def evaluate (mode a0 b0 c0 : Nat) : Option Nat := do
  if mode ≥ 414 || mode = 313 || mode = 317 then none else do
    let a := a0 % base
    let b := b0 % base
    let c := c0 % base
    if mode = 0 then return (a + b + c) % base
    else if mode < 3 then return (a * b + c) % base
    else if mode < 259 then return lut (mode - 3) a b c
    else if mode < 291 then
      let variant := mode - 259
      let typ := variant / 8
      let wrapping := variant / 4 % 2 = 1
      let right := variant / 2 % 2 = 1
      let high := variant % 2 = 1
      let width := if typ < 2 then 64 else 32
      let shift := if wrapping then b % width else min b width
      let pair := a + c * base
      let value : Int := if right && typ % 2 = 0 && c ≥ 2147483648 then (pair : Int) - base * base else pair
      let shifted := if right then value / (2 ^ shift : Nat) else value * (2 ^ shift : Nat)
      return ((shifted / (if high then (base : Int) else 1)) % base).toNat
    else if mode < 295 then
      let variant := mode - 291
      let bits := if variant / 2 = 1 && a ≥ 2147483648 then (a ^^^ 0xffffffff) else a
      if bits = 0 then return 0xffffffff
      else return if variant % 2 = 1 then 31 - bits.log2 else bits.log2
    else if mode ≥ 311 && mode < 318 then
      let flags := mode - 310
      let arg := fun (value bit : Nat) => if flags / 2 ^ bit % 2 = 1 then -(value : Int) else (value : Int)
      return ((arg a 0 + arg b 1 + arg c 2) % base).toNat
    else
      let variant := if mode < 318 then mode - 295 else (mode - 318) / 6
      let av : Int := if variant / 8 = 1 then signed32 a else a
      let bv : Int := if variant / 8 = 1 then signed32 b else b
      let predicate := match variant % 8 with
        | 0 => false
        | 1 => av < bv
        | 2 => av = bv
        | 3 => av ≤ bv
        | 4 => av > bv
        | 5 => av ≠ bv
        | 6 => av ≥ bv
        | _ => true
      if mode < 318 then return if predicate then 1 else 0
      else
        let third := if (mode - 318) % 2 = 1 then c = 0 else c ≠ 0
        let combined := match (mode - 318) / 2 % 3 with
          | 0 => predicate && third
          | 1 => predicate || third
          | _ => predicate != third
        return if combined then 1 else 0

theorem sum_wrap : evaluate 0 0xffffffff 1 1 = some 1 := by decide
theorem signed_unsigned_low_imad_agree (a b c : Nat) : evaluate 1 a b c = evaluate 2 a b c := by rfl
theorem xor_table : evaluate (3 + 0x96) 0x12345678 0xffffffff 0x12345678 = some 0xffffffff := by decide
theorem unsigned_comparison : evaluate (295 + 1) 0xffffffff 1 0 = some 0 := by decide
theorem signed_comparison : evaluate (295 + 9) 0xffffffff 1 0 = some 1 := by decide
theorem right_shift_width_changes_wrap :
    evaluate (259 + 8 + 4 + 2) 1 32 2 = some 2 ∧
    evaluate (259 + 24 + 4 + 2) 1 32 2 = some 1 := by decide
theorem signed_shift_extends : evaluate (259 + 2 + 1) 0 32 0x80000000 = some 0xffffffff := by decide
theorem flo_signed_all_ones : evaluate 293 0xffffffff 0 0 = some 0xffffffff := by decide
theorem negated_first_and_third_sum : evaluate 315 1 2 3 = some 0xfffffffe := by decide
theorem runtime_predicate_or : evaluate (318 + 2) 0 0 1 = some 1 := by decide
#print axioms right_shift_width_changes_wrap
#print axioms signed_shift_extends
#print axioms flo_signed_all_ones
#print axioms negated_first_and_third_sum
#print axioms runtime_predicate_or

#print axioms sum_wrap
#print axioms signed_unsigned_low_imad_agree
#print axioms xor_table
#print axioms unsigned_comparison
#print axioms signed_comparison

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.UniformInteger

open GB300Math.UniformInteger

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
      let some model := evaluate parameter a b (readWord xs (16 * i + 8)) |
        throw (IO.userError s!"Unsupported mode: {parameter}")
      let actual := readWord ys (4 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness mode={parameter} row={i} a={a} b={b} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 16} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
