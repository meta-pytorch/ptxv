import AlternateFloatAlgebra

set_option maxRecDepth 100000
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem signed_natAbs (z : Int) : signed (decide (z < 0)) z.natAbs = z := by
  unfold signed
  split
  next h =>
    have hn : z < 0 := of_decide_eq_true h
    rw [Int.ofNat_natAbs_of_nonpos (by omega)]
    omega
  next h =>
    have hn : 0 ≤ z := by simp only [decide_eq_true_eq] at h; omega
    exact Int.natAbs_of_nonneg hn

theorem signed_cast (s : Bool) (m : Nat) :
    ((signed s m : Int) : ℚ) = (if s then -1 else 1) * (m : ℚ) := by
  cases s <;> simp [signed]

theorem finite_add_exact (sa sb : Bool) (a b : Nat) :
    rational (some (add (.finite sa a) (.finite sb b))) =
      some (((if sa then -1 else 1) * (a : ℚ) / unit) +
        ((if sb then -1 else 1) * (b : ℚ) / unit)) := by
  let z := signed sa a + signed sb b
  have signed_sum : signed (if z = 0 then sa && sb else decide (z < 0)) z.natAbs = z := by
    by_cases hz : z = 0
    · simp [hz, signed]
    · simp only [hz, ite_false]; exact signed_natAbs z
  have hr : ((signed (if z = 0 then sa && sb else decide (z < 0)) z.natAbs : Int) : ℚ) =
      ((signed sa a : Int) : ℚ) + ((signed sb b : Int) : ℚ) := by
    rw [signed_sum]
    simp [z]
  dsimp only [add, rational]
  change some ((if (if z = 0 then sa && sb else decide (z < 0)) then -1 else 1) *
      (z.natAbs : ℚ) / unit) = _
  simp only [signed_cast] at hr
  rw [hr, add_div]

#print axioms signed_natAbs
#print axioms finite_add_exact
end GB300Math.AlternateFloatFloatLib
