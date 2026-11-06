import MufuReciprocalBounds

namespace GB300Math.MufuRootBounds

open MufuReciprocalBounds

def output (bf16 reciprocal : Bool) (input : Nat) : Nat :=
  let op := if reciprocal then "rsq" else "sqrt"
  if bf16 then (MufuSm103.evaluateBf16 op input).getD 65536
  else (Mufu.evaluate (op ++ ".f16") input).getD 65536

def domain (bf16 : Bool) (input result : Nat) : Bool :=
  input > 0 && input < infinity bf16 && result > 0 && result < infinity bf16

def ratio (bf16 reciprocal : Bool) (input result : Nat) : Nat × Nat :=
  let (xn, xd) := dyadic bf16 input
  let (yn, yd) := dyadic bf16 result
  if reciprocal then (xn * yn^2, xd * yd^2)
  else (yn^2 * xd, yd^2 * xn)

def lower (bf16 reciprocal : Bool) : Nat × Nat :=
  if reciprocal then
    let denominator := if bf16 then 128 else 1024
    ((denominator-1)^2, denominator^2)
  else if bf16 then (64, 65) else (512, 513)

def upper (bf16 reciprocal : Bool) : Nat × Nat :=
  if reciprocal then
    let denominator := if bf16 then 128 else 1024
    ((denominator+1)^2, denominator^2)
  else (1, 1)

def valid (bf16 reciprocal : Bool) (input : Nat) : Bool :=
  let result := output bf16 reciprocal input
  if !domain bf16 input result then true
  else
    let (numerator, denominator) := ratio bf16 reciprocal input result
    let (ln, ld) := lower bf16 reciprocal
    let (un, ud) := upper bf16 reciprocal
    denominator > 0 &&
    ln * denominator ≤ numerator * ld && numerator * ud ≤ un * denominator

def flushDomain (reciprocal : Bool) (input : Nat) : Bool :=
  if input = 0 || input ≥ 0x7f80 then true
  else
    let result := output true reciprocal input
    (result = (if reciprocal then 0x7f80 else 0)) == (input < 128)

theorem f16_sqrt_attained :
    output false false 0x0402 = 0x2000 ∧
    let (n, d) := ratio false false 0x0402 0x2000
    n * 513 = d * 512 := by decide +kernel

theorem bf16_sqrt_attained :
    output true false 0x0082 = 0x2000 ∧
    let (n, d) := ratio true false 0x0082 0x2000
    n * 65 = d * 64 := by decide +kernel

theorem bf16_tiny_results :
    output true false 1 = 0 ∧ output true true 1 = 0x7f80 := by decide +kernel

#print axioms f16_sqrt_attained
#print axioms bf16_sqrt_attained
#print axioms bf16_tiny_results

end GB300Math.MufuRootBounds
