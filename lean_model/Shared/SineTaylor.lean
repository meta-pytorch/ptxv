import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Tactic
set_option maxRecDepth 12000
set_option maxHeartbeats 3000000
namespace GB300Math.SineTaylor
open Set
noncomputable def polynomial (x : ℝ) : ℝ :=
  x-x^3/6+x^5/120-x^7/5040+x^9/362880-x^11/39916800+
    x^13/6227020800-x^15/1307674368000+x^17/355687428096000

theorem within_deriv (x : ℝ) (hx : 0 ≠ x) (n : Nat) :
    iteratedDerivWithin n Real.sin (uIcc 0 x) 0 = iteratedDeriv n Real.sin 0 := by
  rw [iteratedDerivWithin_eq_iteratedFDerivWithin,iteratedDeriv_eq_iteratedFDeriv,
    iteratedFDerivWithin_eq_iteratedFDeriv (uniqueDiffOn_uIcc hx) Real.contDiff_sin.contDiffAt left_mem_uIcc]

theorem taylor_polynomial (x : ℝ) (hx : 0 ≠ x) :
    taylorWithinEval Real.sin 17 (uIcc 0 x) 0 x = polynomial x := by
  rw [taylor_within_apply]
  simp_rw [within_deriv x hx]
  norm_num [Finset.sum_range_succ,Real.iteratedDeriv_add_one_sin,
    Real.iteratedDeriv_add_one_cos,polynomial]
  ring

theorem approximation (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) :
    |Real.sin x-polynomial x| ≤ 1/10000000000 := by
  by_cases hz : x=0
  · simp [hz,polynomial]
  obtain ⟨y,hy,h⟩ := taylor_mean_remainder_lagrange_iteratedDeriv (n:=17) (Ne.symm hz)
    (Real.contDiff_sin.contDiffOn : ContDiffOn ℝ (17+1) Real.sin (uIcc 0 x))
  rw [taylor_polynomial x (Ne.symm hz)] at h
  have hd : iteratedDeriv 18 Real.sin y = -Real.sin y := by
    rw [show 18=2*9 by decide,Real.iteratedDeriv_even_sin]
    norm_num
  rw [h,hd,abs_div,abs_mul,abs_neg,abs_pow,abs_of_nonneg (sub_nonneg.mpr hx.1)]
  have hm := mul_le_mul_of_nonneg_right (Real.abs_sin_le_one y) (pow_nonneg hx.1 18)
  have hp := pow_le_pow_left₀ hx.1 hx.2 18
  norm_num at hm hp ⊢
  nlinarith only [hm,hp]

def piRat : Rat := 3141592653589793/1000000000000000

theorem pi_approx : |Real.pi-(piRat:ℝ)| ≤ 1/1000000000000000 := by
  have lo := Real.pi_gt_d20
  have hi := Real.pi_lt_d20
  norm_num [piRat]
  exact abs_le.mpr ⟨by linarith,by linarith⟩

theorem phase_approximation (u : ℝ) (hu : 0 ≤ u ∧ u ≤ 1) :
    |Real.sin (Real.pi/2*u)-polynomial ((piRat:ℝ)/2*u)| ≤ 1/1000000000 := by
  have hr : 0 ≤ (piRat:ℝ)/2*u ∧ (piRat:ℝ)/2*u ≤ 2 := by
    norm_num [piRat]
    constructor <;> nlinarith [hu.1,hu.2]
  have h := approximation _ hr
  have hp := Real.abs_sin_sub_sin_le (Real.pi/2*u) ((piRat:ℝ)/2*u)
  have hd : Real.pi/2*u-(piRat:ℝ)/2*u=(Real.pi-(piRat:ℝ))*(u/2) := by ring
  rw [hd,abs_mul,abs_of_nonneg (div_nonneg hu.1 (by norm_num) : 0 ≤ u/2)] at hp
  have hc := mul_le_mul_of_nonneg_right pi_approx (div_nonneg hu.1 (by norm_num) : 0 ≤ u/2)
  have ht := abs_sub_le (Real.sin (Real.pi/2*u)) (Real.sin ((piRat:ℝ)/2*u)) (polynomial ((piRat:ℝ)/2*u))
  nlinarith [hu.2]
#print axioms approximation
#print axioms phase_approximation
end GB300Math.SineTaylor
