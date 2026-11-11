namespace GB300Math.Stochastic

def roundMantissa (mantissa entropy drop bits : Nat) : Nat :=
  (mantissa + (entropy % 2 ^ bits) * 2 ^ (drop - bits)) / 2 ^ drop

def lane (bf16 : Bool) (input entropy : Nat) : Nat :=
  let x := input % 2 ^ 32
  let fractionBits := if bf16 then 7 else 10
  let bias := if bf16 then 127 else 15
  let limit := (if bf16 then 255 else 31) * 2 ^ fractionBits
  let sign := (x / 2 ^ 31) * 32768
  let e : Nat := x / 2 ^ 23 % 256
  let fraction := x % 2 ^ 23
  if e = 255 then
    if fraction ≠ 0 then 0x7fff else sign + limit - 1
  else
    let mantissa := fraction + if e = 0 then 0 else 2 ^ 23
    let targetExponent : Int := ((max e 1 : Nat) : Int) - 127 + bias
    let drop := 23 - fractionBits + (max 0 (1 - targetExponent)).toNat
    let value := (max 0 (targetExponent - 1)).toNat * 2 ^ fractionBits +
      roundMantissa mantissa entropy drop (23 - fractionBits)
    sign + min value (limit - 1)

def pair (bf16 : Bool) (x y entropy : Nat) : Nat :=
  lane bf16 x entropy + lane bf16 y (entropy / 65536) * 65536

theorem fp16_exact_grid (q entropy : Nat) :
    roundMantissa (q * 8192) entropy 13 13 = q := by
  unfold roundMantissa
  simp only [Nat.sub_self, Nat.pow_zero, Nat.mul_one]
  omega

theorem bf16_exact_grid (q entropy : Nat) :
    roundMantissa (q * 65536) entropy 16 16 = q := by
  unfold roundMantissa
  simp only [Nat.sub_self, Nat.pow_zero, Nat.mul_one]
  omega

theorem normal_rounding_error (m entropy bits : Nat) :
    roundMantissa m entropy bits bits * 2 ^ bits < m + 2 ^ bits ∧
    m < roundMantissa m entropy bits bits * 2 ^ bits + 2 ^ bits := by
  simp only [roundMantissa, Nat.sub_self, Nat.pow_zero, Nat.mul_one]
  have positive : 0 < 2 ^ bits := Nat.two_pow_pos bits
  have hEntropy := Nat.mod_lt entropy positive
  have hRemainder := Nat.mod_lt (m + entropy % 2 ^ bits) positive
  have hDivision := Nat.mod_add_div (m + entropy % 2 ^ bits) (2 ^ bits)
  rw [Nat.mul_comm (2 ^ bits)] at hDivision
  omega

#print axioms fp16_exact_grid
#print axioms bf16_exact_grid
#print axioms normal_rounding_error

end GB300Math.Stochastic
