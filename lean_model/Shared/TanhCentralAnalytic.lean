/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Exp2Analytic
import Mathlib.Analysis.Complex.Trigonometric
namespace GB300Math.TanhCentralAnalytic
open Exp2Analytic
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

theorem power_error (x y : ℝ) (hx : 0≤x ∧ x≤1) (hy : 0≤y ∧ y≤1) (n : Nat) :
    |x^n-y^n|≤n*|x-y| := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hy1 : y^n≤1 := by simpa using pow_le_pow_left₀ hy.1 hy.2 n
    have he : x^(n+1)-y^(n+1)=x*(x^n-y^n)+(x-y)*y^n := by ring
    rw [he]
    have ht:=abs_add_le (x*(x^n-y^n)) ((x-y)*y^n)
    rw [abs_mul,abs_mul,abs_of_nonneg hx.1,abs_of_nonneg (pow_nonneg hy.1 _)] at ht
    have h1:=mul_le_mul_of_nonneg_right hx.2 (abs_nonneg (x^n-y^n))
    have h2:=mul_le_mul_of_nonneg_left hy1 (abs_nonneg (x-y))
    push_cast
    nlinarith

theorem center_error (a s c : ℝ) (ha : 0≤a ∧ a≤8)
    (hs : s=series (-a/8)) (hr : 0≤s ∧ s≤1)
    (hc : |s^16-c|≤1/10^15) : |Real.exp (-2*a)-c|≤17/10^12 := by
  have he:=series_error (-a/8) (by rw [abs_of_nonpos (by linarith)];linarith)
  have hp:=power_error (Real.exp (-a/8)) s
    ⟨le_of_lt (Real.exp_pos _),by simpa using Real.exp_le_exp.mpr (show -a/8≤0 by linarith)⟩ hr 16
  have hn : Real.exp (-2*a)=(Real.exp (-a/8))^16 := by
    have h:=Real.exp_nat_mul (-a/8) 16
    convert h using 1
    congr 1
    ring
  rw [hs] at hp
  rw [hn]
  have ht:=abs_sub_le ((Real.exp (-a/8))^16) ((series (-a/8))^16) c
  rw [hs] at hc
  norm_num at hp
  linarith

theorem local_exp_error (a c d : ℝ) (ha : 0≤a) (hc : |Real.exp (-2*a)-c|≤17/10^12)
    (hd : 0≤d ∧ d≤1/128) :
    |Real.exp (-2*(a+d))-c*(1-2*d+2*d^2)|≤1/1000000 := by
  have he : Real.exp (-2*a)≤1 := by simpa using Real.exp_le_exp.mpr (show -2*a≤0 by linarith)
  have ht:=Real.exp_bound (x := -2*d) (by rw [abs_of_nonpos (by linarith)];linarith) (n:=3) (by decide)
  norm_num [Finset.sum_range_succ,abs_of_nonpos (show -2*d≤0 by linarith)] at ht
  rw [abs_of_nonneg hd.1] at ht
  have hte : |Real.exp (-2*d)-(1-2*d+2*d^2)|≤(16/9)*d^3 := by
    convert ht using 1 <;> ring_nf
  have hd3:=pow_le_pow_left₀ hd.1 hd.2 3
  have hpoly : 0≤1-2*d+2*d^2 ∧ 1-2*d+2*d^2≤1 := by
    have h:=mul_nonneg hd.1 (show 0≤1-d by linarith)
    constructor <;> nlinarith [sq_nonneg d]
  have hi : Real.exp (-2*(a+d))-c*(1-2*d+2*d^2)=
    Real.exp (-2*a)*(Real.exp (-2*d)-(1-2*d+2*d^2))+
    (Real.exp (-2*a)-c)*(1-2*d+2*d^2) := by
    rw [show -2*(a+d)=(-2*a)+(-2*d) by ring,Real.exp_add];ring
  rw [hi]
  have hh:=abs_add_le (Real.exp (-2*a)*(Real.exp (-2*d)-(1-2*d+2*d^2)))
    ((Real.exp (-2*a)-c)*(1-2*d+2*d^2))
  rw [abs_mul,abs_mul,abs_of_pos (Real.exp_pos _),abs_of_nonneg hpoly.1] at hh
  have h1:=mul_le_mul_of_nonneg_left hte (le_of_lt (Real.exp_pos (-2*a)))
  have h2:=mul_le_mul_of_nonneg_right hc hpoly.1
  nlinarith [Real.exp_pos (-2*a)]

theorem tanh_negative_exp (x : ℝ) : Real.tanh x=(1-Real.exp (-2*x))/(1+Real.exp (-2*x)) := by
  rw [Real.tanh_eq,Real.exp_neg,show -2*x=-(x+x) by ring,Real.exp_neg,Real.exp_add]
  have he:=Real.exp_pos x
  field_simp

theorem rational_error (x y v : ℝ) (hv : 0≤v)
    (hr : |y*(1+v)-(1-v)|≤1/100000)
    (he : |Real.exp (-2*x)-v|≤1/1000000) : |y-Real.tanh x|≤12/1000000 := by
  let e:=Real.exp (-2*x)
  have hepos : 0<e := Real.exp_pos _
  have hd : 0<1+v := by linarith
  have hid : y-(1-v)/(1+v)=(y*(1+v)-(1-v))/(1+v) := by field_simp
  have hy : |y-(1-v)/(1+v)|≤1/100000 := by
    rw [hid,abs_div,abs_of_pos hd]
    apply (div_le_iff₀ hd).mpr
    nlinarith
  have hrid : (1-v)/(1+v)-(1-e)/(1+e)=2*(e-v)/((1+v)*(1+e)) := by
    field_simp
    ring
  have hve : |(1-v)/(1+v)-(1-e)/(1+e)|≤2/1000000 := by
    rw [hrid,abs_div,abs_mul,abs_of_pos (show (0:ℝ)<2 by norm_num),abs_of_pos (show (0:ℝ)<(1+v)*(1+e) by positivity)]
    apply (div_le_iff₀ (by positivity : (0:ℝ)<(1+v)*(1+e))).mpr
    change |e-v|≤_ at he
    nlinarith [mul_nonneg hv hepos.le]
  rw [tanh_negative_exp]
  have ht:=abs_sub_le y ((1-v)/(1+v)) ((1-e)/(1+e))
  dsimp [e] at ht hve
  linarith

#print axioms power_error
#print axioms center_error
#print axioms local_exp_error
#print axioms tanh_negative_exp
#print axioms rational_error
end GB300Math.TanhCentralAnalytic
