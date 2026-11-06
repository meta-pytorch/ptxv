import GB300Math.Mufu
import GB300Math.MufuSm103

namespace GB300Math.MufuReciprocalBounds

def output (bf16 : Bool) (input : Nat) : Nat :=
  if bf16 then (MufuSm103.evaluateBf16 "rcp" input).getD 65536
  else (Mufu.evaluate "rcp.f16" input).getD 65536

def fractionBits (bf16 : Bool) : Nat := if bf16 then 7 else 10
def bias (bf16 : Bool) : Nat := if bf16 then 127 else 15
def infinity (bf16 : Bool) : Nat := if bf16 then 0x7f80 else 0x7c00

def dyadic (bf16 : Bool) (input : Nat) : Nat × Nat :=
  let word := input % 32768
  let bits := fractionBits bf16
  let exponent := word / 2^bits
  let significand := word % 2^bits + if exponent = 0 then 0 else 2^bits
  let exponent := max exponent 1
  (significand * 2^(exponent - (bias bf16 + bits)),
    2^((bias bf16 + bits) - exponent))

def bound (bf16 : Bool) (result : Nat) : Nat × Nat :=
  if bf16 then (233, 32768)
  else if result % 32768 < 1024 then (985, 262144)
  else (1927, 2097152)

def inDomain (bf16 : Bool) (input result : Nat) : Bool :=
  input % 32768 > 0 && input % 32768 < infinity bf16 &&
  result % 32768 > 0 && result % 32768 < infinity bf16

def valid (bf16 : Bool) (input : Nat) : Bool :=
  let result := output bf16 input
  if !inDomain bf16 input result then true
  else
    let (xn, xd) := dyadic bf16 input
    let (yn, yd) := dyadic bf16 result
    let (bn, bd) := bound bf16 result
    result < 65536 && input / 32768 = result / 32768 &&
    xn * yn ≤ xd * yd &&
    (xd * yd - xn * yn) * bd ≤ bn * xd * yd

def flushDomain (input : Nat) : Bool :=
  let magnitude := input % 32768
  if magnitude = 0 || magnitude ≥ 0x7f80 then true
  else (output true input % 32768 = 0) == (magnitude ≥ 0x7e81)

theorem f16_normal_attained :
    output false 0x07d5 = 0x7015 ∧
    let (xn, xd) := dyadic false 0x07d5
    let (yn, yd) := dyadic false 0x7015
    (xd * yd - xn * yn) * 2097152 = 1927 * xd * yd := by decide +kernel

theorem f16_subnormal_attained :
    output false 0x7bc2 = 0x0107 ∧
    let (xn, xd) := dyadic false 0x7bc2
    let (yn, yd) := dyadic false 0x0107
    (xd * yd - xn * yn) * 262144 = 985 * xd * yd := by decide +kernel

theorem bf16_normal_attained :
    output true 0x00f1 = 0x7e07 ∧
    let (xn, xd) := dyadic true 0x00f1
    let (yn, yd) := dyadic true 0x7e07
    (xd * yd - xn * yn) * 32768 = 233 * xd * yd := by decide +kernel

theorem bf16_flush_boundary :
    output true 0x7e80 = 0x0080 ∧ output true 0x7e81 = 0 := by decide +kernel

#print axioms f16_normal_attained
#print axioms f16_subnormal_attained
#print axioms bf16_normal_attained
#print axioms bf16_flush_boundary

end GB300Math.MufuReciprocalBounds
