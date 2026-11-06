import AlternateFloatPairBounds
import AlternateFloatRationalBounds

set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 1024
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem rational_scaled_bound (z r n t u : ℚ) (hn : 0 < n) (hu : 0 < u)
    (h : n * |r-z| ≤ max |z| (n*t)) :
    |r/u-z/u| ≤ max (|z/u|/n) (t/u) := by
  rw [← sub_div, abs_div, abs_of_pos hu]
  by_cases hz : |z| ≤ n*t
  · rw [max_eq_right hz] at h
    have hb : |r-z| ≤ t := by nlinarith
    exact le_trans (div_le_div_of_nonneg_right hb (le_of_lt hu)) (le_max_right _ _)
  · rw [max_eq_left (le_of_not_ge hz)] at h
    have hb : |r-z| ≤ |z|/n := by apply (le_div_iff₀ hn).2; nlinarith
    have hf : (|z|/n)/u = |z/u|/n := by
      rw [abs_div, abs_of_pos hu]
      ring
    rw [← hf]
    exact le_trans (div_le_div_of_nonneg_right hb (le_of_lt hu)) (le_max_left _ _)

theorem e4m3_grid_to_rational (z r : Int)
    (h : 17*(r-z).natAbs ≤ max z.natAbs (17*2^502)) :
    |(r : ℚ)/unit-(z : ℚ)/unit| ≤ max (|(z : ℚ)/unit|/17) (1/2^10) := by
  have hc : (17 : ℚ)*((r-z).natAbs : ℚ) ≤ max (z.natAbs : ℚ) (17*2^502) := by
    exact_mod_cast h
  rw [Nat.cast_natAbs, Int.cast_abs, Int.cast_sub, Nat.cast_natAbs, Int.cast_abs] at hc
  have hu : (0 : ℚ) < unit := by norm_num [unit, grid]
  have result := rational_scaled_bound (z : ℚ) (r : ℚ) 17 (2^502) unit (by norm_num) hu hc
  convert result using 1
  norm_num [unit, grid]

theorem e5m2_grid_to_rational (z r : Int)
    (h : 9*(r-z).natAbs ≤ max z.natAbs (9*2^495)) :
    |(r : ℚ)/unit-(z : ℚ)/unit| ≤ max (|(z : ℚ)/unit|/9) (1/2^17) := by
  have hc : (9 : ℚ)*((r-z).natAbs : ℚ) ≤ max (z.natAbs : ℚ) (9*2^495) := by
    exact_mod_cast h
  rw [Nat.cast_natAbs, Int.cast_abs, Int.cast_sub, Nat.cast_natAbs, Int.cast_abs] at hc
  have hu : (0 : ℚ) < unit := by norm_num [unit, grid]
  have result := rational_scaled_bound (z : ℚ) (r : ℚ) 9 (2^495) unit (by norm_num) hu hc
  convert result using 1
  norm_num [unit, grid]

theorem pair_bound_extract (op dst af bf : String) (a b : Nat)
    (x y ideal output : Value) (z r : Int)
    (hx : decode af a = some x) (hy : decode bf b = some y)
    (hi : (if op = "add" then add x y else if op = "sub" then add x (negate y) else mul x y) = ideal)
    (hz : gridInteger ideal = some z)
    (ho : decode dst (encode dst false ideal) = some output)
    (hr : gridInteger output = some r)
    (h : pairBound op dst af bf a b = true) :
    (if dst = "e4m3" then 17 else 9) * (r-z).natAbs ≤
      max z.natAbs (if dst = "e4m3" then 17*2^502 else 9*2^495) := by
  simpa [pairBound, hx, hy, hi, hz, ho, hr] using h

#print axioms rational_scaled_bound
#print axioms e4m3_grid_to_rational
#print axioms e5m2_grid_to_rational
#print axioms pair_bound_extract
end GB300Math.AlternateFloatFloatLib
