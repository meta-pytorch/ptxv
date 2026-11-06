import AlternateFloatFloatLib
import Mathlib.Tactic

set_option maxRecDepth 100000
set_option exponentiation.threshold 1024

namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem product_grid_divisible (a b : Nat)
    (ha : a % 2^385 = 0) (hb : b % 2^385 = 0) : unit ∣ a*b := by
  obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero ha
  obtain ⟨l, hl⟩ := Nat.dvd_of_mod_eq_zero hb
  refine ⟨k*l*2^258, ?_⟩
  rw [hk, hl]
  norm_num [unit, grid]
  ring

theorem product_integer_division_exact (a b : Nat)
    (ha : a % 2^385 = 0) (hb : b % 2^385 = 0) : a*b/unit*unit = a*b :=
  Nat.div_mul_cancel (product_grid_divisible a b ha hb)

theorem finite_multiply_exact (sa sb : Bool) (a b : Nat)
    (ha : a % 2^385 = 0) (hb : b % 2^385 = 0) :
    rational (some (mul (.finite sa a) (.finite sb b))) =
      some (((if sa then -1 else 1) * (a : ℚ) / unit) *
        ((if sb then -1 else 1) * (b : ℚ) / unit)) := by
  have hu : (unit : ℚ) ≠ 0 := by norm_num [unit, grid]
  have hp : ((a*b/unit : Nat) : ℚ) = (a : ℚ)*(b : ℚ)/(unit : ℚ) := by
    rw [Nat.cast_div (product_grid_divisible a b ha hb) hu, Nat.cast_mul]
  cases sa <;> cases sb <;> dsimp only [mul, rational]
  all_goals rw [hp]; simp; ring

#print axioms product_grid_divisible
#print axioms product_integer_division_exact
#print axioms finite_multiply_exact
end GB300Math.AlternateFloatFloatLib
