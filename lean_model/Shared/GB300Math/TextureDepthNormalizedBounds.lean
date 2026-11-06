import GB300Math.TextureDepthNormalized

namespace GB300Math.TextureDepthNormalizedBounds

open TextureDepthNormalized

def units (word : Nat) : Nat :=
  if word / 2^23 = 0 then word % 2^23
  else (2^23 + word % 2^23) * 2^(word/2^23-1)

theorem scaled_division (mantissa exponent : Nat) (h : 0 < exponent ∧ exponent ≤ 129) :
    mantissa * 2^(exponent-1) / 2^129 = mantissa / 2^(130-exponent) := by
  have hp : (2 : Nat)^129 = 2^(130-exponent)*2^(exponent-1) := by
    rw [← Nat.pow_add]
    congr 1
    omega
  rw [hp, Nat.mul_div_mul_right _ _ (Nat.pow_pos (by decide : 0 < (2 : Nat)))]

theorem fixed20_units (word : Nat) (h : word ≤ 0x3f800000) :
    fixed20 word = units word / 2^129 := by
  have hs : word / 2^31 ≠ 1 := by omega
  have hn : TextureDepthCompare.isNaN word = false := by
    simp only [TextureDepthCompare.isNaN, TextureDepthCompare.magnitude]
    simp
    omega
  by_cases he : word = 0x3f800000
  · subst word
    decide +kernel
  · have hw : ¬ word ≥ 0x3f800000 := by omega
    by_cases hz : word / 2^23 = 0
    · have hm : word % 2^23 < 2^129 := by omega
      simp [fixed20, hs, hn, hw, hz, units, Nat.div_eq_of_lt hm]
    · have hscl := scaled_division (2^23 + word % 2^23) (word/2^23) (by omega)
      simpa [fixed20, hs, hn, hw, hz, units] using hscl.symm

theorem decoded_error (word : Nat) (h : word ≤ 0x3f800000) :
    -((2^148 + 65535*2^129 : Nat) : Int) <
      (reference word : Int)*2^149 - (units word : Int)*65535 ∧
    (reference word : Int)*2^149 - (units word : Int)*65535 ≤ (2^148 : Int) := by
  unfold reference
  rw [fixed20_units word h]
  exact reference_error (units word)

theorem greater_than_half_ulp :
    reference 0x370fffff = 0 ∧
    2*(units 0x370fffff)*65535 > 2^149 := by
  decide +kernel

theorem unit_interval (word : Nat) (h : word ≤ 0x3f800000) :
    reference word ≤ 65535 := by
  apply quantize_bounded
  unfold fixed20
  split
  · omega
  · split
    · omega
    · split
      · omega
      · have he : word/2^23 ≤ 126 := by omega
        have hd : 16 ≤ 2^(130-word/2^23) := by
          have hpow := Nat.pow_le_pow_right (by decide : 1 ≤ (2 : Nat)) (show 4 ≤ 130-word/2^23 by omega)
          exact hpow
        have hr : word % 2^23 < 2^23 := Nat.mod_lt _ (by decide)
        have hdiv := Nat.div_le_div_left (a := 2^23+word%2^23) hd (show 0 < 16 by decide)
        exact Nat.le_trans hdiv (by omega)

end GB300Math.TextureDepthNormalizedBounds
