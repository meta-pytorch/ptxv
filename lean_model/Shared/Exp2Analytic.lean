import Log2Series
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Tactic

namespace GB300Math.Exp2Analytic
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

def logRat : Rat := 6931471805599453/10^16

def seriesQ (x : Rat) : Rat := ∑ k∈Finset.range 16,x^k/k.factorial
noncomputable def series (x : ℝ) : ℝ := ∑ k∈Finset.range 16,x^k/k.factorial

theorem log_rat_close : |(Log2Series.polynomialQ 2)-logRat| ≤ 1/10^14 := by decide +kernel

theorem log_rat_error : |Real.log 2-(logRat:ℝ)| ≤ 2/10^14 := by
  have h := Log2Series.approximation 2 (by norm_num)
  have hr : |Log2Series.polynomial 2-(logRat:ℝ)| ≤ 1/10^14 := by
    have hq := (Rat.cast_le (K:=ℝ)).mpr log_rat_close
    simpa only [Rat.cast_abs,Rat.cast_sub,Rat.cast_div,Rat.cast_one,Rat.cast_pow,Rat.cast_ofNat,Log2Series.polynomial_cast] using hq
  have ht := abs_sub_le (Real.log 2) (Log2Series.polynomial 2) (logRat:ℝ)
  linarith

theorem series_cast (x : Rat) : (seriesQ x:ℝ)=series (x:ℝ) := by
  unfold seriesQ series
  push_cast
  rfl

theorem series_error (x : ℝ) (hx : |x|≤1) : |Real.exp x-series x|≤1/10^12 := by
  have h:=Real.exp_bound hx (n:=16) (by decide)
  have hp : |x|^16≤1 := by simpa using pow_le_pow_left₀ (abs_nonneg x) hx 16
  change |Real.exp x-series x| ≤ _ at h
  norm_num at h
  nlinarith

theorem exp_le_three (x : ℝ) (hx : x≤1) : Real.exp x≤3 := by
  have h:=Real.exp_bound' (x:=1) (by norm_num) (by norm_num) (n:=3) (by decide)
  norm_num [Finset.sum_range_succ] at h
  exact (Real.exp_le_exp.mpr hx).trans (by linarith)

theorem perturb (a b : ℝ) (ha : a≤1) (hd : |b-a|≤1) :
    |Real.exp b-Real.exp a|≤6*|b-a| := by
  have he : Real.exp a≤3 := exp_le_three a ha
  have h:=Real.abs_exp_sub_one_le hd
  have hi : Real.exp b-Real.exp a=Real.exp a*(Real.exp (b-a)-1) := by
    rw [mul_sub,←Real.exp_add]
    congr 1
    · congr 1;ring
    · ring
  rw [hi,abs_mul,abs_of_pos (Real.exp_pos a)]
  have hh:=mul_le_mul_of_nonneg_left h (le_of_lt (Real.exp_pos a))
  nlinarith [abs_nonneg (b-a)]

theorem local_error (a d v : ℝ) (ha : a≤1) (hd : 0≤d ∧ d≤3/1024)
    (hv : |Real.exp a-v|≤2/10^12) :
    |Real.exp (a+d)-v*(1+d+d^2/2)|≤1/50000000 := by
  have he : Real.exp a≤3 := exp_le_three a ha
  have h:=Real.exp_bound (show |d|≤1 by rw [abs_of_nonneg hd.1];linarith) (n:=3) (by decide)
  norm_num [Finset.sum_range_succ,abs_of_nonneg hd.1] at h
  have hd3 : d^3≤(3/1024:ℝ)^3 := pow_le_pow_left₀ hd.1 hd.2 3
  have hp : 0≤1+d+d^2/2 ∧ 1+d+d^2/2≤2 := by
    have hd2 := pow_le_pow_left₀ hd.1 hd.2 2
    constructor <;> nlinarith
  have h1:=mul_le_mul_of_nonneg_left h (le_of_lt (Real.exp_pos a))
  have h2:=mul_le_mul_of_nonneg_right hv hp.1
  have hi : Real.exp (a+d)-v*(1+d+d^2/2)=
      Real.exp a*(Real.exp d-(1+d+d^2/2))+(Real.exp a-v)*(1+d+d^2/2) := by rw [Real.exp_add];ring
  rw [hi]
  have ht:=abs_add_le (Real.exp a*(Real.exp d-(1+d+d^2/2))) ((Real.exp a-v)*(1+d+d^2/2))
  rw [abs_mul,abs_mul,abs_of_pos (Real.exp_pos a),abs_of_nonneg hp.1] at ht
  nlinarith [Real.exp_pos a]

#print axioms log_rat_close
#print axioms log_rat_error
#print axioms series_cast
#print axioms series_error
#print axioms perturb
#print axioms local_error
end GB300Math.Exp2Analytic
