import Std

namespace GB300Math.I2FSelectors

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

def evaluate (src dst mode : String) (selector input : Nat) : Option Nat := do
  if !(mode ∈ ["rn", "rz", "ru", "rd"]) || !(dst ∈ ["f16", "f32", "f64"]) then none else do
    let (width, signed) ← integerFormat src
    if !(width = 8 || width = 16) || selector ≥ 32 / width then none else do
      return fromInteger (← format dst) width signed mode
        (input % 2^32 / 2^(width*selector))

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

theorem signed_high_byte :
    evaluate "s8" "f32" "rn" 3 0x80000001 = some 0xc3000000 := by decide

theorem unsigned_high_half :
    evaluate "u16" "f32" "rn" 1 0xffff0001 = some 0x477fff00 := by decide

theorem half_overflow_rounding :
    evaluate "u16" "f16" "rn" 1 0xffff0001 = some 0x7c00 ∧
    evaluate "u16" "f16" "rz" 1 0xffff0001 = some 0x7bff := by decide

theorem invalid_half_selector :
    evaluate "u16" "f32" "rn" 2 0x12345678 = none := by decide

#print axioms signed_high_byte
#print axioms unsigned_high_half
#print axioms half_overflow_rounding
#print axioms invalid_half_selector

end GB300Math.I2FSelectors

open GB300Math.I2FSelectors

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected tab-separated conversion manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [src, dst, mode, selectorText, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some selector := selectorText.toNat? | throw (IO.userError "Invalid selector")
    let some expected := errorsText.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let some model := evaluate src dst mode selector a |
        throw (IO.userError s!"Unsupported: {src} {dst}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {src} {dst} {mode} row={i} a={a} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {src} {dst} {mode} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
