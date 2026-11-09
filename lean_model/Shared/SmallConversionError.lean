import Mathlib.Algebra.Order.Group.MinMax
import Mathlib.Tactic
import SmallConversions

namespace GB300Math.SmallConversionError

open GB300Math.SmallConversions

theorem nearest_rational_error (m d : Nat) :
    |(nearestEven m d : ℚ) - (m : ℚ) / 2^d| ≤ 1/2 := by
  have hp : (0 : ℚ) < 2^d := by positivity
  have h := nearest_even_half_quantum m d
  have hlo : (2 : ℚ)*m ≤ 2*((nearestEven m d : ℚ)*2^d)+2^d := by
    exact_mod_cast h.1
  have hhi : (2 : ℚ)*((nearestEven m d : ℚ)*2^d) ≤ 2*m+2^d := by
    exact_mod_cast h.2
  have lower : (nearestEven m d : ℚ)-1/2 ≤ (m : ℚ)/2^d := by
    apply (le_div_iff₀ hp).mpr
    nlinarith
  have upper : (m : ℚ)/2^d ≤ (nearestEven m d : ℚ)+1/2 := by
    apply (div_le_iff₀ hp).mpr
    nlinarith
  rw [abs_le]
  constructor <;> linarith

theorem clipped_rational_error (m d limit : Nat) :
    |(min (nearestEven m d) limit : ℚ) - min ((m : ℚ) / 2^d) limit| ≤ 1/2 := by
  have h := abs_min_sub_min_le_max (nearestEven m d : ℚ) (limit : ℚ)
    ((m : ℚ) / 2^d) (limit : ℚ)
  apply h.trans
  apply max_le (nearest_rational_error m d)
  norm_num

theorem signed_s2f6_error (negative : Bool) (m d : Nat) :
    let limit := if negative then 128 else 127
    let sign : ℚ := if negative then -1 else 1
    |sign * (min (nearestEven m d) limit : ℚ) / 64 -
      sign * min ((m : ℚ) / 2^d) limit / 64| ≤ 1/128 := by
  have h := clipped_rational_error m d (if negative then 128 else 127)
  dsimp
  cases negative <;> simp only [Bool.false_eq_true, ↓reduceIte] at *
  all_goals
    rw [← sub_div, ← mul_sub, abs_div, abs_mul]
    norm_num only [abs_one, abs_neg, one_mul, abs_of_pos (by norm_num : (0 : ℚ) < 64)]
    linarith

theorem positive_scaled_s2f6_error (x scale : Nat)
    (finite : x < 0x7f800000) (validScale : scale < 255)
    (fractional : 0 < 144 - ((max (x / 2^23) 1 : Nat) : Int) + ((scale : Int)-127)) :
    let mantissa : Nat := x % 2^23 + if x / 2^23 = 0 then 0 else 2^23
    let shift : Nat := (144 - ((max (x / 2^23) 1 : Nat) : Int) + ((scale : Int)-127)).toNat
    |(downS2F6 true false true x scale : ℚ)/64 -
      min ((mantissa : ℚ)/2^shift) 127 /64| ≤ 1/128 := by
  have hx : x < 2^31 := by omega
  have hmod : x % 2^31 = x := Nat.mod_eq_of_lt hx
  have hdiv : x / 2^31 = 0 := Nat.div_eq_of_lt hx
  have hf : ¬x > 0x7f800000 := by omega
  have hs : scale ≠ 255 := by omega
  have hi : x ≠ 0x7f800000 := by omega
  have hd := not_le_of_gt fractional
  have h := signed_s2f6_error false
    (x % 2^23 + if x / 2^23 = 0 then 0 else 2^23)
    (144 - ((max (x / 2^23) 1 : Nat) : Int) + ((scale : Int)-127)).toNat
  simpa only [downS2F6, hmod, hdiv, hf, hs, hi, Bool.false_eq_true,
    and_false, false_or, Nat.zero_mod, Nat.zero_ne_one, ↓reduceIte, hd,
    Nat.cast_min, Nat.cast_ofNat, one_mul] using h

theorem s2f6_bound_attained :
    |(downS2F6 true false true 0x3c000000 127 : ℚ)/64 - 1/128| = 1/128 := by
  decide +kernel

#print axioms nearest_rational_error
#print axioms clipped_rational_error
#print axioms signed_s2f6_error
#print axioms positive_scaled_s2f6_error
#print axioms s2f6_bound_attained

end GB300Math.SmallConversionError
