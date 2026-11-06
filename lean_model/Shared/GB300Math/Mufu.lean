import GB300Math.MufuTables
import GB300Math.Fp64Seed
import GB300Math.TanhSmall
import GB300Math.TanhTables

namespace GB300Math.Mufu

def canonicalNaN : Nat := 0x7fffffff
def magnitude (x : Nat) : Nat := x % 2 ^ 31
def sign (x : Nat) : Nat := x &&& 0x80000000
def exponent (x : Nat) : Nat := x / 2 ^ 23 % 256
def fraction (x : Nat) : Nat := x % 2 ^ 23

def interpolate (table : MufuTables.Interpolator) (index : Nat) : Int :=
  let (c0, c1, c2) := table.coefficients[index / 2 ^ table.n]!
  let k := index % 2 ^ table.n
  let quadratic := c2 * (Fp64Seed.squareTruncated k table.col : Int)
  let quadratic := if table.c ≥ 0 then quadratic * 2 ^ table.c.toNat
                   else quadratic / 2 ^ (-table.c).toNat
  (c0 * 2 ^ table.a + table.r + c1 * (k : Int) * 2 ^ table.b + quadratic) / 2 ^ table.sh

def fixedToWord (mag : Nat) (frac : Int) : Nat :=
  let lead := mag.log2
  let mant := if lead ≥ 23 then mag / 2 ^ (lead - 23) else mag * 2 ^ (23 - lead)
  ((lead : Int) - frac + 127).toNat * 2 ^ 23 + mant % 2 ^ 23

def ex2 (x : Nat) : Nat :=
  let e := exponent x
  if magnitude x > 0x7f800000 then canonicalNaN
  else if e ≥ 135 then if sign x ≠ 0 then 0 else 0x7f800000
  else
    let m : Int := (fraction x : Int) + 2 ^ 23
    let q : Int := if e = 0 then 0 else if e ≥ 127 then m * 2 ^ (e - 127)
      else if 127 - e < 24 then m / 2 ^ (127 - e) else 0
    let q := if sign x ≠ 0 then -q - (if q % 2 ^ 23 ≠ 0 then 1 else 0) else q
    let mant := interpolate MufuTables.ex2 (q % 2 ^ 23).toNat
    let be := q / 2 ^ 23 + 127
    if be ≤ 0 then 0 else if be ≥ 255 then 0x7f800000
    else be.toNat * 2 ^ 23 + mant.toNat % 2 ^ 23

def rcp (x : Nat) : Nat :=
  let e := exponent x
  let s := sign x
  if magnitude x > 0x7f800000 then canonicalNaN
  else if e = 0 then s + 0x7f800000
  else if e = 255 then s
  else
    let mant := (interpolate MufuTables.rcp (fraction x)).toNat
    let be : Int := (if mant = 2 ^ 24 then 254 else 253) - (e : Int)
    if be ≤ 0 then s else s + be.toNat * 2 ^ 23 + mant % 2 ^ 23

def rootIndex (x : Nat) : Nat × Int :=
  let e : Int := (exponent x : Int) - 127
  let parity := e % 2
  (parity.toNat * 2 ^ 23 + fraction x, (e - parity) / 2)

def rsq (x : Nat) : Nat :=
  let e := exponent x
  if magnitude x > 0x7f800000 then canonicalNaN
  else if e = 0 then sign x + 0x7f800000
  else if sign x ≠ 0 then canonicalNaN
  else if e = 255 then 0
  else
    let (index, t) := rootIndex x
    if index = 0 then (127 - t).toNat * 2 ^ 23
    else
      let mant := (interpolate MufuTables.rsq index).toNat
      let be := (if mant = 2 ^ 24 then 127 else 126) - t
      be.toNat * 2 ^ 23 + mant % 2 ^ 23

def sqrt (x : Nat) : Nat :=
  let e := exponent x
  if magnitude x > 0x7f800000 then canonicalNaN
  else if e = 0 then sign x
  else if sign x ≠ 0 then canonicalNaN
  else if e = 255 then 0x7f800000
  else
    let (index, t) := rootIndex x
    (127 + t).toNat * 2 ^ 23 + (interpolate MufuTables.sqrt index).toNat % 2 ^ 23

def lg2 (x : Nat) : Nat :=
  let e := exponent x
  if magnitude x > 0x7f800000 then canonicalNaN
  else if e = 0 then 0xff800000
  else if sign x ≠ 0 then canonicalNaN
  else if e = 255 then 0x7f800000
  else if x = 0x3f800000 then 0
  else
    let l := interpolate MufuTables.lg2 (fraction x)
    let ex : Int := (e : Int) - 127
    let v := ex * 2 ^ 36 + l
    if v > 0 then fixedToWord v.toNat 36
    else
      let mag := ((-ex - 1) * 2 ^ 36 + (2 ^ 36 - 1 - l)).toNat
      0x80000000 + if mag = 0 then 0 else fixedToWord mag 36

def sincos (x : Nat) (cosine : Bool) : Nat :=
  let e := exponent x
  if e = 255 then canonicalNaN
  else
    let m := fraction x + 2 ^ 23
    let sh : Int := (e : Int) - 125
    let q := (if e = 0 ∨ sh ≥ 25 then 0 else if sh ≥ 0 then m * 2 ^ sh.toNat
              else if -sh < 32 then m / 2 ^ (-sh).toNat else 0) % 2 ^ 25
    let q := if cosine then (q + 2 ^ 23) % 2 ^ 25 else q
    let negative := if cosine then q / 2 ^ 24 = 1
                    else (sign x ≠ 0) != (q / 2 ^ 24 = 1)
    let u := q % 2 ^ 23
    let u := if q / 2 ^ 23 % 2 = 1 then 0x7fffff - u else u
    let l := (interpolate MufuTables.sin u).toNat
    (if negative then 0x80000000 else 0) + if l = 0 then 0 else fixedToWord l 37

def tanh (x : Nat) : Option Nat :=
  if magnitude x > 0x7f800000 then some canonicalNaN
  else if exponent x < 119 then some x
  else if exponent x ≥ 130 then some (sign x + 0x3f800000)
  else
    let e := exponent x
    let (discard, coefficients) := TanhTables.binades[e - 119]!
    let index := fraction x / 2 ^ discard
    let k := index % 2 ^ 17
    let (a, b, c) := coefficients[index / 2 ^ 17]!
    let value := a * 2 ^ 34 + b * (k : Int) * 2 ^ 17 +
      c * (Fp64Seed.squareTruncated k 19 : Int)
    some (sign x + fixedToWord value.toNat (184 - (min e 126 : Int)))

def tanh16 (input : Nat) : Nat :=
  let h := input % 2 ^ 16
  let mag := h % 2 ^ 15
  if mag > 0x7c00 then 0x7fff
  else if mag < 0x1c00 then h
  else
    let s := h / 2 ^ 15
    let e := mag / 2 ^ 10
    let x := s * 2 ^ 31 + (if e = 31 then 255 else e + 112) * 2 ^ 23 +
      mag % 2 ^ 10 * 2 ^ 13
    let y := (tanh x).getD 0
    s * 2 ^ 15 + (exponent y - 112) * 2 ^ 10 + fraction y / 2 ^ 13

def evaluate32 (operation : String) (input : Nat) : Option Nat :=
  let x := input % 2 ^ 32
  match operation with
  | "ex2" => some (ex2 x)
  | "rcp" => some (rcp x)
  | "rsq" => some (rsq x)
  | "sqrt" => some (sqrt x)
  | "lg2" => some (lg2 x)
  | "sin" => some (sincos x false)
  | "cos" => some (sincos x true)
  | "tanh" => tanh x
  | "tanh16" => some (tanh16 x)
  | "tanh16x2" => some (tanh16 x + 2 ^ 16 * tanh16 (x / 2 ^ 16))
  | _ => none

def widen16 (input : Nat) : Nat :=
  let h := input % 2 ^ 16
  let s := h / 2 ^ 15 * 2 ^ 31
  let e := h / 2 ^ 10 % 32
  let f := h % 2 ^ 10
  if e = 31 then s + 0x7f800000 + f * 2 ^ 13
  else if e ≠ 0 then s + (e + 112) * 2 ^ 23 + f * 2 ^ 13
  else if f = 0 then s
  else s + (f.log2 + 103) * 2 ^ 23 + f * 2 ^ (23 - f.log2) % 2 ^ 23

def narrow16 (x : Nat) : Nat :=
  let s := sign x / 2 ^ 16
  let e := exponent x
  let m := fraction x + if e = 0 then 0 else 2 ^ 23
  if magnitude x > 0x7f800000 then 0x7fff
  else if e ≥ 143 then s + 0x7c00
  else if e ≥ 113 then s + (e - 112) * 2 ^ 10 + fraction x / 2 ^ 13
  else s + m / 2 ^ (126 - e)

def evaluate (operation : String) (input : Nat) : Option Nat := do
  if operation.endsWith ".f16" then
    let y ← evaluate32 (operation.dropEnd 4).toString (widen16 input)
    return narrow16 y
  evaluate32 operation input

theorem ex2_zero : ex2 0 = 0x3f800000 := by decide +kernel
theorem lg2_one : lg2 0x3f800000 = 0 := by decide +kernel
theorem rsq_negative_zero : rsq 0x80000000 = 0xff800000 := by decide +kernel
theorem sqrt_negative_zero : sqrt 0x80000000 = 0x80000000 := by decide +kernel
theorem tanh_downward_step :
    tanh 0x3dbfffff = some 0x3dbf7082 ∧ tanh 0x3dc00000 = some 0x3dbf7060 := by
  decide +kernel
theorem tanh16_truncates : tanh16 0x1c08 = 0x1c07 := by decide +kernel

#print axioms ex2_zero
#print axioms lg2_one
#print axioms rsq_negative_zero
#print axioms sqrt_negative_zero
#print axioms tanh_downward_step
#print axioms tanh16_truncates

end GB300Math.Mufu
