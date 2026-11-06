import MufuRootBounds

namespace GB300Math.MufuRsqTightBounds

open MufuRootBounds

def lower (bf16 : Bool) : Nat × Nat :=
  if bf16 then (2067347, 2097152) else (133960757, 134217728)

def upper (bf16 : Bool) : Nat × Nat :=
  if bf16 then (1, 1) else (268435475, 268435456)

def valid (bf16 : Bool) (input : Nat) : Bool :=
  let result := output bf16 true input
  if !domain bf16 input result then true
  else
    let q := ratio bf16 true input result
    decide (0 < q.2) &&
      decide ((lower bf16).1 * q.2 ≤ q.1 * (lower bf16).2) &&
      decide (q.1 * (upper bf16).2 ≤ (upper bf16).1 * q.2)

theorem f16_attained : output false true 0x0bcd = 0x540c ∧
    (ratio false true 0x0bcd 0x540c).1 * 134217728 =
      133960757 * (ratio false true 0x0bcd 0x540c).2 := by
  decide +kernel

theorem bf16_attained : output true true 0x0156 = 0x5e8b ∧
    (ratio true true 0x0156 0x5e8b).1 * 2097152 =
      2067347 * (ratio true true 0x0156 0x5e8b).2 := by
  decide +kernel

#print axioms f16_attained
#print axioms bf16_attained

end GB300Math.MufuRsqTightBounds
