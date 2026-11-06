import Std

namespace GB300Math.Conversions

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

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

theorem nearest_even_error (negative : Bool) (n shift : Nat) :
    2 * n ≤ 2 * (roundMagnitude "rn" negative n shift) * 2 ^ shift + 2 ^ shift ∧
    2 * (roundMagnitude "rn" negative n shift) * 2 ^ shift ≤ 2 * n + 2 ^ shift := by
  have hp : 0 < 2 ^ shift := Nat.two_pow_pos shift
  have hr := Nat.mod_lt n hp
  have hd := Nat.mod_add_div n (2 ^ shift)
  rw [Nat.mul_comm (2 ^ shift) (n / 2 ^ shift)] at hd
  simp only [roundMagnitude]
  split <;> simp_all only [decide_eq_true_eq] <;> simp only [Nat.add_zero, Nat.add_mul, Nat.mul_add, Nat.mul_one, Nat.mul_assoc] <;> omega

theorem nearest_even_ties : roundMagnitude "rn" false 5 1 = 2 ∧
    roundMagnitude "rn" false 7 1 = 4 := by decide

theorem signed_zero_preserved : integral ⟨8, 23⟩ "rn" 0x80000000 = 0x80000000 := by decide

theorem nan_integer_width_difference : toInteger ⟨8, 23⟩ 32 true "rn" 0x7fc00000 = 0 ∧
    toInteger ⟨8, 23⟩ 64 true "rn" 0x7fc00000 = 0x8000000000000000 := by decide

end GB300Math.Conversions

open GB300Math.Conversions

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected tab-separated conversion manifest")
  let cases ← IO.FS.readFile manifest
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [src, dst, mode, input, output] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    unless mode ∈ ["rn", "rz", "ru", "rd", "ra"] do
      throw (IO.userError s!"Invalid rounding: {mode}")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let some model := apply src dst mode a |
        throw (IO.userError s!"Unsupported: {src} {dst}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {src} {dst} {mode} row={i} a={a} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {src} {dst} {mode} cases={xs.size / 32} mismatches={mismatches}"
    failures := failures + mismatches
  return if failures = 0 then 0 else 1
