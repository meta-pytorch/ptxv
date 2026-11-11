import Std

namespace GB300Math.AlternateFloat

def grid : Nat := 512

def unit : Nat := 2^grid

inductive Value where
  | finite (negative : Bool) (magnitude : Nat)
  | infinity (negative : Bool)
  | nan
  deriving DecidableEq, Repr

def negate : Value → Value
  | .finite s n => .finite (!s) n
  | .infinity s => .infinity (!s)
  | .nan => .nan

def signed (negative : Bool) (n : Nat) : Int := if negative then -(n : Int) else n

def add : Value → Value → Value
  | .nan, _ | _, .nan => .nan
  | .infinity sa, .infinity sb => if sa == sb then .infinity sa else .nan
  | .infinity s, _ | _, .infinity s => .infinity s
  | .finite sa a, .finite sb b =>
    let z := signed sa a + signed sb b
    .finite (if z = 0 then sa && sb else z < 0) z.natAbs

def mul : Value → Value → Value
  | .nan, _ | _, .nan => .nan
  | .infinity sa, .infinity sb => .infinity (sa != sb)
  | .infinity sa, .finite sb b => if b = 0 then .nan else .infinity (sa != sb)
  | .finite sa a, .infinity sb => if a = 0 then .nan else .infinity (sa != sb)
  | .finite sa a, .finite sb b => .finite (sa != sb) (a*b/unit)

def fields (format : String) : Option (Nat × Nat × Nat) :=
  match format with
  | "e4m3" => some (8, 3, 7)
  | "e5m2" => some (8, 2, 15)
  | "e3m2" => some (6, 2, 3)
  | "e2m3" => some (6, 3, 1)
  | "e2m1" | "e2m1p4" => some (4, 1, 1)
  | _ => none

def decode (format : String) (bits : Nat) : Option Value := do
  unless bits < 256 do none
  if format = "ue8m0" then
    return if bits = 255 then .nan else .finite false (2^(bits+grid-127))
  let some (width, fraction, bias) := fields format | none
  let raw := bits % 2^width
  let negative := raw ≥ 2^(width-1)
  let magnitude := raw % 2^(width-1)
  if format = "e4m3" && magnitude = 127 then return .nan
  if format = "e5m2" && magnitude ≥ 124 then
    return if magnitude = 124 then .infinity negative else .nan
  let exponent := magnitude / 2^fraction
  let mantissa := magnitude % 2^fraction + if exponent = 0 then 0 else 2^fraction
  let power := max exponent 1 + grid - bias - fraction
  return .finite negative (mantissa * 2^power)

def nearestEven (m d : Nat) : Nat :=
  if d = 0 then m else
    let q := m / 2^d
    let r := m % 2^d
    q + if r > 2^(d-1) ∨ (r = 2^(d-1) ∧ q % 2 = 1) then 1 else 0

def encodeFinite (format : String) (saturate negative : Bool) (m : Nat) : Nat :=
  if m = 0 then (if negative then 128 else 0) else
  let fraction : Nat := if format = "e4m3" then 3 else 2
  let bias : Nat := if format = "e4m3" then 7 else 15
  let maxCode : Nat := if format = "e4m3" then 126 else 123
  let exponent : Int := (m.log2 : Int) - grid + bias
  let shift := (max exponent 1 + grid - bias - fraction).toNat
  let raw := (max (exponent-1) 0).toNat * 2^fraction + nearestEven m shift
  let magnitude := if raw ≤ maxCode then raw
    else if saturate then maxCode else if format = "e4m3" then 127 else 124
  (if negative then 128 else 0) + magnitude

def encodeRaw (format : String) (saturate : Bool) : Value → Nat
  | .nan => 127
  | .infinity s => (if s then 128 else 0) +
    if saturate then (if format = "e4m3" then 126 else 123)
    else if format = "e4m3" then 127 else 124
  | .finite s m => encodeFinite format saturate s m

def encode (format : String) (saturate : Bool) (value : Value) : Nat :=
  let raw := encodeRaw format saturate value
  if format = "e4m3" && raw % 128 = 127 then 127 else raw

def validFormat (format : String) : Bool :=
  format ∈ ["e4m3", "e5m2", "e3m2", "e2m3", "e2m1", "e2m1p4", "ue8m0"]

def legal (op dst a b : String) : Bool :=
  (op ∈ ["add", "sub", "mul", "fma"]) && (dst ∈ ["e4m3", "e5m2"]) &&
  validFormat a && validFormat b && (if op = "add" || op = "sub" then b = dst else true)

def laneBits (format : String) (packed lane : Nat) : Nat :=
  if format = "e2m1" then packed / 2^(4*lane) % 16 else packed / 2^(8*lane) % 256

def evaluate (op dst af bf : String) (saturate : Bool) (a b c : Nat) : Option Nat := do
  unless legal op dst af bf do none
  unless a < 2^(if af = "e2m1" then 16 else 32) ∧
      b < 2^(if bf = "e2m1" then 16 else 32) ∧ c < 2^32 do none
  let mut output := 0
  for lane in [:4] do
    let x ← decode af (laneBits af a lane)
    let y ← decode bf (laneBits bf b lane)
    let z ← decode dst (laneBits dst c lane)
    let value := if op = "add" then add x y
      else if op = "sub" then add x (negate y)
      else if op = "mul" then mul x y else add (mul x y) z
    output := output + encode dst saturate value * 2^(8*lane)
  return output

theorem opposite_infinities : add (.infinity false) (.infinity true) = .nan := by decide
theorem zero_times_infinity : mul (.finite true 0) (.infinity false) = .nan := by decide
theorem preserves_negative_zero : encode "e4m3" true (add (.finite true 0) (.finite true 0)) = 128 := by decide
theorem half_even_down : nearestEven 5 1 = 2 := by decide
theorem half_even_up : nearestEven 7 1 = 4 := by decide
theorem fp8_exact_add : evaluate "add" "e4m3" "e4m3" "e4m3" false 0x38383838 0x38383838 0 = some 0x40404040 := by decide +kernel

#print axioms opposite_infinities
#print axioms zero_times_infinity
#print axioms preserves_negative_zero
#print axioms half_even_down
#print axioms half_even_up
#print axioms fp8_exact_add

def read32 (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do word := word + data[offset+j]!.toNat * 2^(8*j)
  return word
end GB300Math.AlternateFloat
open GB300Math.AlternateFloat

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected alternate-float manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failures := 0
  for row in rows do
    let [op, dst, af, bf, sat, input, output, errors] := row.splitOn "\t" | throw (IO.userError "Malformed manifest row")
    unless sat = "0" ∨ sat = "1" do throw (IO.userError "Invalid saturation flag")
    unless legal op dst af bf do throw (IO.userError "Invalid operation")
    let some expected := errors.toNat? | throw (IO.userError "Invalid expected mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size do throw (IO.userError "Invalid record sizes")
    let mut mismatches := 0
    for i in [:xs.size/16] do
      let a := read32 xs (16*i)
      let b := read32 xs (16*i+4)
      let c := read32 xs (16*i+8)
      unless read32 xs (16*i+12) = 0x93b471ea do throw (IO.userError "Invalid input sentinel")
      let some target := evaluate op dst af bf (sat = "1") a b c | throw (IO.userError "Out-of-domain operands")
      let actual := read32 ys (16*i)
      if actual ≠ target ∨ read32 ys (16*i+4) ≠ a ∨ read32 ys (16*i+8) ≠ b ∨ read32 ys (16*i+12) ≠ c then
        if mismatches < 5 then IO.println s!"witness {op} {dst} {af} {bf} row={i} a={a} b={b} c={c} expected={target} actual={actual}"
        mismatches := mismatches+1
    IO.println s!"result {output} records={xs.size/16} mismatches={mismatches} expected={expected}"
    if mismatches ≠ expected then failures := failures+1
  return if failures = 0 then 0 else 1
