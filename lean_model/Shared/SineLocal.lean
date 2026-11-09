import SineTaylor
namespace GB300Math.SineLocal
open Set
set_option maxHeartbeats 2000000

theorem within_deriv (a x : ℝ) (hx : a ≠ x) (n : Nat) :
    iteratedDerivWithin n Real.sin (uIcc a x) a = iteratedDeriv n Real.sin a := by
  rw [iteratedDerivWithin_eq_iteratedFDerivWithin,iteratedDeriv_eq_iteratedFDeriv,
    iteratedFDerivWithin_eq_iteratedFDeriv (uniqueDiffOn_uIcc hx) Real.contDiff_sin.contDiffAt left_mem_uIcc]

theorem taylor_two (a x : ℝ) (hx : a ≠ x) :
    taylorWithinEval Real.sin 2 (uIcc a x) a x =
      Real.sin a+Real.cos a*(x-a)-Real.sin a*(x-a)^2/2 := by
  rw [taylor_within_apply]
  simp_rw [within_deriv a x hx]
  norm_num [Finset.sum_range_succ,Real.iteratedDeriv_add_one_sin,
    Real.iteratedDeriv_add_one_cos]
  ring

theorem local_remainder (a d : ℝ) (hd : 0 ≤ d) :
    |Real.sin (a+d)-(Real.sin a+Real.cos a*d-Real.sin a*d^2/2)| ≤ d^3/6 := by
  by_cases hz : d=0
  · simp [hz]
  have hn : a ≠ a+d := by intro he; apply hz; linarith
  obtain ⟨y,hy,h⟩ := taylor_mean_remainder_lagrange_iteratedDeriv (n:=2) hn
    (Real.contDiff_sin.contDiffOn : ContDiffOn ℝ (2+1) Real.sin (uIcc a (a+d)))
  rw [taylor_two a (a+d) hn] at h
  have he : a+d-a=d := by ring
  rw [he] at h
  have hi : iteratedDeriv 3 Real.sin y = -Real.cos y := by
    rw [show 3=2*1+1 by decide,Real.iteratedDeriv_odd_sin]
    norm_num
  rw [h,hi,abs_div,abs_mul,abs_neg,abs_pow,abs_of_nonneg hd]
  have hm := mul_le_mul_of_nonneg_right (Real.abs_cos_le_one y) (pow_nonneg hd 3)
  norm_num at hm ⊢
  linarith

theorem approximate_centers (a d s c : ℝ) (hd : 0 ≤ d ∧ d ≤ 1/128)
    (hs : |Real.sin a-s| ≤ 1/1000000000)
    (hc : |Real.cos a-c| ≤ 1/1000000000) :
    |Real.sin (a+d)-(s+c*d-s*d^2/2)| ≤ 1/10000000 := by
  have ht := local_remainder a d hd.1
  have hp2 : d^2 ≤ (1/128:ℝ)^2 := pow_le_pow_left₀ hd.1 hd.2 2
  have hp3 : d^3 ≤ (1/128:ℝ)^3 := pow_le_pow_left₀ hd.1 hd.2 3
  have hs' := abs_le.mp hs
  have hc' := abs_le.mp hc
  have hcd := mul_le_mul_of_nonneg_right hc'.2 hd.1
  have hcd' := mul_le_mul_of_nonneg_right hc'.1 hd.1
  have hsd := mul_le_mul_of_nonneg_right hs'.2 (sq_nonneg d)
  have hsd' := mul_le_mul_of_nonneg_right hs'.1 (sq_nonneg d)
  have h := abs_le.mp ht
  apply abs_le.mpr
  constructor <;> nlinarith [sq_nonneg d]
#print axioms local_remainder
#print axioms approximate_centers
end GB300Math.SineLocal
