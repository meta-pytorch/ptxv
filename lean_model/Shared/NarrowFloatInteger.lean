import Std

namespace GB300Math.NarrowFloatInteger

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

def narrowTarget : String → Option (Nat × Bool × Bool)
  | "s8" => some (8, true, true)
  | "u8" => some (8, false, true)
  | "s16" => some (16, true, true)
  | "u16" => some (16, false, true)
  | "s8_raw" => some (8, true, false)
  | "u8_raw" => some (8, false, false)
  | "s16_raw" => some (16, true, false)
  | "u16_raw" => some (16, false, false)
  | _ => none

def evaluate (src dst mode : String) (modifier input : Nat) : Option Nat := do
  if modifier ≥ 4 || mode ∉ ["rn", "rz", "ru", "rd"] || src ∉ ["f32", "f32_ftz", "f64"] then none else do
    let f : Format := if src = "f64" then ⟨11, 52⟩ else ⟨8, 23⟩
    let (width, signed, ntz) ← narrowTarget dst
    if src = "f64" && !ntz then none else do
      let bits := modify f.width modifier input
      if bits % 2^(f.width-1) > f.infinity then
        return if src = "f64" || !ntz then 2^(width-1) else 0
      let bits := if src = "f32_ftz" && bits % 2^31 < 2^23 then bits / 2^31 * 2^31 else bits
      return toInteger f width signed mode bits % 2^32

theorem signed_finite_saturation : evaluate "f32" "s8" "rn" 0 0xc3800000 = some 0xffffff80 := by decide

theorem signed_nan_raw_encoding : evaluate "f32" "s8_raw" "rn" 0 0x7f800001 = some 0x80 := by decide

theorem fp64_nan_raw_encoding : evaluate "f64" "s16" "rn" 0 0x7ff0000000000001 = some 0x8000 := by decide

theorem flush_changes_ceil : evaluate "f32_ftz" "s8" "ru" 0 1 = some 0 ∧ evaluate "f32" "s8" "ru" 0 1 = some 1 := by decide

#print axioms signed_finite_saturation
#print axioms signed_nan_raw_encoding
#print axioms fp64_nan_raw_encoding
#print axioms flush_changes_ceil

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.NarrowFloatInteger

open GB300Math.NarrowFloatInteger

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, op, modeName, modsText, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    unless modeName ∈ ["rn", "rz", "ru", "rd"] do
      throw (IO.userError "Invalid rounding mode")
    let some mods := modsText.toNat? | throw (IO.userError "Invalid modifiers")
    let some expected := errorsText.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let some model := evaluate precision op modeName mods a |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {precision} {op} {modeName} mods={mods} row={i} a={a} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
