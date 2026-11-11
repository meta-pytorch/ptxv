import GB300Math.StochasticBoundaries
import GB300Math.StochasticSigned

namespace GB300Math.StochasticFloor

open StochasticSm103 StochasticValue StochasticUnderflow StochasticBoundaries

theorem round_zero (m r drop bits : Nat) (hw : bits ≤ drop)
    (hm : m < 2^(drop-bits)) : roundMantissa m r drop bits = 0 := by
  have hp : 2^drop = 2^bits*2^(drop-bits) := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hr : r%2^bits+1 ≤ 2^bits := by have := Nat.mod_lt r (Nat.two_pow_pos bits); omega
  have hu := Nat.mul_le_mul_right (2^(drop-bits)) hr
  have hl := Nat.add_lt_add_right hm ((r%2^bits)*2^(drop-bits))
  unfold roundMantissa
  rw [hp]
  apply Nat.div_eq_of_lt
  nlinarith

theorem floor_is_underflow (f : Format) (x : Nat) (hx : x < floorWord f) :
    x/8388608 ≤ 127-(parameters f).2.2.1 := by
  cases f <;> norm_num [floorWord, parameters] at * <;> omega

theorem floor_mantissa_small (f : Format) (x : Nat) (hx : x < floorWord f) :
    sourceMantissa (x/8388608) (x%8388608) < StochasticUnderflow.quantum f (x/8388608) := by
  by_cases hf : f = .bf16
  · subst f
    have hz : x = 0 := by norm_num [floorWord] at hx; omega
    simp [hz, sourceMantissa, StochasticUnderflow.quantum, drop, entropyBits, fractionBits, parameters]
  · have hd : 24 ≤ drop f (x/8388608)-entropyBits f := by
      cases f <;> norm_num [floorWord, drop, entropyBits, fractionBits, parameters, isPair] at * <;> omega
    have hp := Nat.pow_le_pow_right (by decide : 1 ≤ 2) hd
    have ht : x%8388608 < 8388608 := Nat.mod_lt _ (by decide)
    have hm : sourceMantissa (x/8388608) (x%8388608) < 2^24 := by
      unfold sourceMantissa
      split_ifs <;> omega
    exact lt_of_lt_of_le hm hp

theorem positive_prefix_zero (f : Format) (relu satfinite : Bool) (x r : Nat)
    (hx : x < floorWord f) : lane f relu satfinite x r = 0 := by
  have he := floor_is_underflow f x hx
  have ht : x%8388608 < 8388608 := Nat.mod_lt _ (by decide)
  have hbits := bitvalue f relu satfinite (x/8388608) (x%8388608) r he ht
  have hrebuild : x/8388608*8388608+x%8388608 = x := by omega
  rw [hrebuild] at hbits
  rw [hbits]
  exact round_zero _ _ _ _ (entropy_width f (x/8388608) he) (floor_mantissa_small f x hx)

theorem negative_prefix_zero (f : Format) (satfinite : Bool) (x r : Nat)
    (hx : x < floorWord f) :
    lane f false satfinite (x+2147483648) r = 2^((parameters f).1+(parameters f).2.1) := by
  have hfinite : x < 2139095040 := by cases f <;> norm_num [floorWord] at hx <;> omega
  rw [StochasticSigned.lane_negative f satfinite x r hfinite,
    positive_prefix_zero f false satfinite x r hx, Nat.add_zero]

#print axioms round_zero
#print axioms floor_is_underflow
#print axioms floor_mantissa_small
#print axioms positive_prefix_zero
#print axioms negative_prefix_zero

end GB300Math.StochasticFloor
