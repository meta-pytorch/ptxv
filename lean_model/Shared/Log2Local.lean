/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2Series
import SineLocal
namespace GB300Math.Log2Local
open Set
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

theorem log_first : iteratedDeriv 1 Real.log = fun x => x⁻¹ := by
  rw [show 1=0+1 by decide,iteratedDeriv_succ]
  simp

theorem log_second : iteratedDeriv 2 Real.log = fun x => -(x^2)⁻¹ := by
  rw [show 2=1+1 by decide,iteratedDeriv_succ,log_first,deriv_inv']

theorem log_third (x : ℝ) (hx : x≠0) : iteratedDeriv 3 Real.log x = 2/x^3 := by
  rw [show 3=2+1 by decide,iteratedDeriv_succ,log_second]
  have h := (((hasDerivAt_id x).pow 2).inv (pow_ne_zero 2 hx)).neg
  change deriv (-(id ^ 2)⁻¹) x = 2 / x ^ 3
  rw [h.deriv]
  simp only [Pi.pow_apply, id_eq]
  field_simp
  ring

theorem within_deriv (a x : ℝ) (ha : a≠0) (hx : a≠x) (n : Nat) :
    iteratedDerivWithin n Real.log (uIcc a x) a=iteratedDeriv n Real.log a := by
  rw [iteratedDerivWithin_eq_iteratedFDerivWithin,iteratedDeriv_eq_iteratedFDeriv,
    iteratedFDerivWithin_eq_iteratedFDeriv (uniqueDiffOn_uIcc hx) (Real.contDiffAt_log.mpr ha) left_mem_uIcc]

theorem taylor_two (a x : ℝ) (ha : a≠0) (hx : a≠x) :
    taylorWithinEval Real.log 2 (uIcc a x) a x=
      Real.log a+(x-a)/a-(x-a)^2/(2*a^2) := by
  rw [taylor_within_apply]
  simp_rw [within_deriv a x ha hx]
  norm_num [Finset.sum_range_succ,log_first,log_second]
  ring

theorem local_remainder (a d : ℝ) (ha : 1≤a) (hd : 0≤d) :
    |Real.log (a+d)-(Real.log a+d/a-d^2/(2*a^2))|≤d^3/3 := by
  by_cases hz : d=0
  · simp [hz]
  have hdp : 0<d := lt_of_le_of_ne hd (Ne.symm hz)
  have hn : a≠a+d := by intro h;apply hz;linarith
  have hc : ContDiffOn ℝ (2+1) Real.log (uIcc a (a+d)) := by
    intro x hx
    have hx' : a≤x := by rw [uIcc_of_le (by linarith)] at hx;exact hx.1
    exact (Real.contDiffAt_log.mpr (by linarith : x≠0)).contDiffWithinAt
  obtain ⟨y,hy,h⟩ := taylor_mean_remainder_lagrange_iteratedDeriv (n:=2) hn hc
  have hy' : a≤y := by
    rw [uIoo_of_lt (by linarith)] at hy
    exact hy.1.le
  rw [taylor_two a (a+d) (by linarith) hn,show a+d-a=d by ring,log_third y (by linarith)] at h
  have hden : 1≤y^3 := one_le_pow₀ (by linarith)
  have hb : |2/y^3|≤2 := by
    rw [abs_of_nonneg (by positivity)]
    apply (div_le_iff₀ (by positivity : 0<y^3)).mpr
    linarith
  rw [h,abs_div,abs_mul,abs_pow,abs_of_nonneg hd]
  have hm := mul_le_mul_of_nonneg_right hb (pow_nonneg hd 3)
  norm_num at hm ⊢
  linarith

#print axioms log_third
#print axioms local_remainder
end GB300Math.Log2Local
