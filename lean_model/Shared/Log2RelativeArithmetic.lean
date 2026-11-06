import Log2OneSidedTable
namespace GB300Math.Log2RelativeArithmetic
set_option maxHeartbeats 3000000

theorem positive (n l z y : ℝ) (hn : 1≤ n) (hl : 0≤ l ∧ l≤ 1)
    (hz : -58/1000000000≤ z-l ∧ z-l≤ 177/1000000000)
    (hy : -n/2^23≤ y-(n+z) ∧ y-(n+z)≤ 0) :
    |y-(n+l)|≤|n+l|/2^22 := by
  rw [abs_of_pos (by linarith [hl.1] : 0< n+l)]
  apply abs_le.mpr
  constructor <;> linarith [hz.1,hz.2,hy.1,hy.2,hl.1]

theorem negative_large (m l z y : ℝ) (hm : 3≤ m) (hl : 0≤ l ∧ l≤ 1)
    (hz : -58/1000000000≤ z-l ∧ z-l≤ 177/1000000000)
    (hy : 0≤ y-(-m+z) ∧ y-(-m+z)≤(m-1)/2^23+1/2^36) :
    |y-(-m+l)|≤|-m+l|/2^22 := by
  rw [abs_of_neg (by linarith [hl.2] : -m+l< 0)]
  apply abs_le.mpr
  constructor <;> linarith [hz.1,hz.2,hy.1,hy.2,hl.1,hl.2]

theorem negative_two (l z y : ℝ) (hl : 0≤ l ∧ l≤ 1)
    (hzlo : -58/1000000000≤ z-l)
    (hzhi : z-l+1/2^23+1/2^36≤(2-l)/2^22)
    (hy : 0≤ y-(-2+z) ∧ y-(-2+z)≤ 1/2^23+1/2^36) :
    |y-(-2+l)|≤|-2+l|/2^22 := by
  rw [abs_of_neg (by linarith [hl.2] : -2+l< 0)]
  apply abs_le.mpr
  constructor <;> linarith [hzlo,hzhi,hy.1,hy.2,hl.1,hl.2]

noncomputable def logMantissa (f : Fin (2^23)) : ℝ := Real.log (1+(f.val:ℝ)/2^23)/Real.log 2

theorem log_range (f : Fin (2^23)) : 0≤ logMantissa f ∧ logMantissa f≤ 1 := by
  have hlog:=Log2NearOne.log_two_interval
  have hlogp : 0< Real.log 2 := by linarith
  have hx : 1≤ 1+(f.val:ℝ)/2^23 := by have:=Nat.cast_nonneg (α:=ℝ) f.val;linarith
  have hf : (f.val:ℝ)≤ 2^23 := by exact_mod_cast (Nat.le_of_lt f.isLt)
  have hm:=Real.log_le_log (by linarith : 0< 1+(f.val:ℝ)/2^23) (show 1+(f.val:ℝ)/2^23≤ 2 by linarith)
  dsimp [logMantissa]
  constructor
  · exact div_nonneg (Real.log_nonneg hx) hlogp.le
  · apply (div_le_iff₀ hlogp).mpr
    simpa using hm
#print axioms positive
#print axioms negative_large
#print axioms negative_two
#print axioms log_range
end GB300Math.Log2RelativeArithmetic
