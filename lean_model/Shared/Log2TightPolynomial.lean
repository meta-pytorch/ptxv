import Log2Polynomial
import Log2TightCells
namespace GB300Math.Log2TightPolynomial
open Log2Cells Log2TightCells
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

theorem bernstein_interval (a b c t lo hi : ℝ) (ht : 0≤t ∧ t≤1)
    (h0 : lo≤a ∧ a≤hi) (h1 : lo≤a+b/2 ∧ a+b/2≤hi)
    (h2 : lo≤a+b+c ∧ a+b+c≤hi) :
    lo≤a+b*t+c*t^2 ∧ a+b*t+c*t^2≤hi := by
  have w0 : 0≤(1-t)^2 := sq_nonneg _
  have w1 : 0≤2*t*(1-t) := mul_nonneg (mul_nonneg (by norm_num) ht.1) (sub_nonneg.mpr ht.2)
  have w2 : 0≤t^2 := sq_nonneg _
  have p0:=mul_le_mul_of_nonneg_right h0.2 w0
  have p1:=mul_le_mul_of_nonneg_right h1.2 w1
  have p2:=mul_le_mul_of_nonneg_right h2.2 w2
  have n0:=mul_le_mul_of_nonneg_right h0.1 w0
  have n1:=mul_le_mul_of_nonneg_right h1.1 w1
  have n2:=mul_le_mul_of_nonneg_right h2.1 w2
  constructor <;> nlinarith only [p0,p1,p2,n0,n1,n2]

theorem polynomial_interval (j : Fin 512) (t : ℝ) (ht : 0≤t ∧ t≤1) :
    -173/1000000000+1/2^36≤Log2Polynomial.nativePoly j.val t-Log2Polynomial.localPoly j.val t ∧
    Log2Polynomial.nativePoly j.val t-Log2Polynomial.localPoly j.val t≤
      173/1000000000-(correction j.val:ℝ) := by
  rcases hd : differences j.val with ⟨a,b,c⟩
  have hv := cells_tight j
  simp only [Tight,hd] at hv
  have h0 : -173/1000000000+1/2^36≤(a:ℝ) ∧ (a:ℝ)≤173/1000000000-(correction j.val:ℝ) := by
    constructor
    · simpa using (Rat.cast_le (K:=ℝ)).mpr hv.1
    · simpa using (Rat.cast_le (K:=ℝ)).mpr hv.2.1
  have h1 : -173/1000000000+1/2^36≤(a:ℝ)+(b:ℝ)/2 ∧ (a:ℝ)+(b:ℝ)/2≤173/1000000000-(correction j.val:ℝ) := by
    constructor
    · simpa using (Rat.cast_le (K:=ℝ)).mpr hv.2.2.1
    · simpa using (Rat.cast_le (K:=ℝ)).mpr hv.2.2.2.1
  have h2 : -173/1000000000+1/2^36≤(a:ℝ)+(b:ℝ)+(c:ℝ) ∧ (a:ℝ)+(b:ℝ)+(c:ℝ)≤173/1000000000-(correction j.val:ℝ) := by
    constructor
    · simpa using (Rat.cast_le (K:=ℝ)).mpr hv.2.2.2.2.1
    · simpa using (Rat.cast_le (K:=ℝ)).mpr hv.2.2.2.2.2
  have hid:=Log2Polynomial.difference_identity j.val t
  rw [hd] at hid
  dsimp only at hid
  rw [←hid]
  exact bernstein_interval _ _ _ t _ _ ht h0 h1 h2

#print axioms bernstein_interval
#print axioms polynomial_interval
end GB300Math.Log2TightPolynomial
