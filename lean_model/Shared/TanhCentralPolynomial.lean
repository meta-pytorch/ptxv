/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TanhCentralData
namespace GB300Math.TanhCentralPolynomial
open TanhCentralCells TanhCentralAnalytic Exp2Analytic
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

theorem quartic_bound (a b c d e t : ℝ) (ht : 0≤t ∧ t≤1) :
    |a+b*t+c*t^2+d*t^3+e*t^4|≤|a|+|b|+|c|+|d|+|e| := by
  have term (x:ℝ) (n:Nat) : |x*t^n|≤|x| := by
    rw [abs_mul,abs_of_nonneg (pow_nonneg ht.1 _)]
    have h : t^n≤1 := by simpa using pow_le_pow_left₀ ht.1 ht.2 n
    simpa using mul_le_mul_of_nonneg_left h (abs_nonneg x)
  have h1:=term b 1
  have h2:=term c 2
  have h3:=term d 3
  have h4:=term e 4
  simp only [pow_one] at h1
  have ha:=abs_add_le a (b*t)
  have hb:=abs_add_le (a+b*t) (c*t^2)
  have hc:=abs_add_le (a+b*t+c*t^2) (d*t^3)
  have hd:=abs_add_le (a+b*t+c*t^2+d*t^3) (e*t^4)
  linarith

theorem bernstein4 (a b c d e t bound:ℝ) (ht:0≤t ∧ t≤1)
    (h0:|a|≤bound) (h1:|a+b/4|≤bound) (h2:|a+b/2+c/6|≤bound)
    (h3:|a+3*b/4+c/2+d/4|≤bound) (h4:|a+b+c+d+e|≤bound) :
    |a+b*t+c*t^2+d*t^3+e*t^4|≤bound := by
  have hleft:0≤1-t := sub_nonneg.mpr ht.2
  have htpos:=ht.1
  have w0:0≤(1-t)^4 := by positivity
  have w1:0≤4*t*(1-t)^3 := by positivity
  have w2:0≤6*t^2*(1-t)^2 := by positivity
  have w3:0≤4*t^3*(1-t) := by positivity
  have w4:0≤t^4 := by positivity
  have l0:=mul_le_mul_of_nonneg_right (abs_le.mp h0).1 w0
  have l1:=mul_le_mul_of_nonneg_right (abs_le.mp h1).1 w1
  have l2:=mul_le_mul_of_nonneg_right (abs_le.mp h2).1 w2
  have l3:=mul_le_mul_of_nonneg_right (abs_le.mp h3).1 w3
  have l4:=mul_le_mul_of_nonneg_right (abs_le.mp h4).1 w4
  have u0:=mul_le_mul_of_nonneg_right (abs_le.mp h0).2 w0
  have u1:=mul_le_mul_of_nonneg_right (abs_le.mp h1).2 w1
  have u2:=mul_le_mul_of_nonneg_right (abs_le.mp h2).2 w2
  have u3:=mul_le_mul_of_nonneg_right (abs_le.mp h3).2 w3
  have u4:=mul_le_mul_of_nonneg_right (abs_le.mp h4).2 w4
  apply abs_le.mpr
  constructor <;> nlinarith only [l0,l1,l2,l3,l4,u0,u1,u2,u3,u4]

noncomputable def nativePoly (e u : Nat) (t:ℝ) : ℝ :=
  let (a,b,c):=nativeCoefficients e u
  (a:ℝ)+(b:ℝ)*t+(c:ℝ)*t^2

noncomputable def approxExp (e u : Nat) (t:ℝ) : ℝ :=
  (center e u:ℝ)*(1-2*(step e:ℝ)*t+2*((step e:ℝ)*t)^2)

theorem residual_bound (e u : Nat) (t:ℝ) (ht:0≤t ∧ t≤1)
    (hv:Valid e u) :
    |nativePoly e u t*(1+approxExp e u t)-(1-approxExp e u t)|≤1/100000 := by
  let r:=TanhCentralCells.residual e u
  have hh:=hv.2.2.2.2.2.2.2.2.2
  change BernsteinValid e u at hh
  rcases hh with ⟨h0,h1,h2,h3,h4⟩
  have h0r : |(r[0]!:ℝ)|≤1/100000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr h0
  have h1r : |(r[0]!:ℝ)+(r[1]!:ℝ)/4|≤1/100000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr h1
  have h2r : |(r[0]!:ℝ)+(r[1]!:ℝ)/2+(r[2]!:ℝ)/6|≤1/100000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr h2
  have h3r : |(r[0]!:ℝ)+3*(r[1]!:ℝ)/4+(r[2]!:ℝ)/2+(r[3]!:ℝ)/4|≤1/100000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr h3
  have h4r : |(r[0]!:ℝ)+(r[1]!:ℝ)+(r[2]!:ℝ)+(r[3]!:ℝ)+(r[4]!:ℝ)|≤1/100000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr h4
  have hb:=bernstein4 (r[0]!:ℝ) (r[1]!:ℝ) (r[2]!:ℝ) (r[3]!:ℝ) (r[4]!:ℝ) t _ ht h0r h1r h2r h3r h4r
  convert hb using 1
  unfold nativePoly approxExp
  dsimp [r,TanhCentralCells.residual]
  rcases nativeCoefficients e u with ⟨a,b,c⟩
  dsimp only
  push_cast
  congr 1
  ring

theorem ideal_error (e:Fin 11) (u:Fin 512) (hv:Valid e.val u.val)
    (t:ℝ) (ht:0≤t ∧ t≤1) :
    |nativePoly e.val u.val t-Real.tanh ((start e.val u.val:ℝ)+(step e.val:ℝ)*t)|≤12/1000000 := by
  rcases hv with ⟨ha,hA,hs,hS,hc,hC,hr,hR,hclose,hres⟩
  have har : 0≤(start e.val u.val:ℝ) ∧ (start e.val u.val:ℝ)≤8 := by exact_mod_cast And.intro ha hA
  have hsr : 0≤(step e.val:ℝ) ∧ (step e.val:ℝ)≤1/128 := by
    constructor
    · exact (Rat.cast_nonneg (K:=ℝ)).mpr hs.le
    · simpa using (Rat.cast_le (K:=ℝ)).mpr hS
  have hcr : 0≤(center e.val u.val:ℝ) := by exact_mod_cast hc
  have hseries : 0≤series (-(start e.val u.val:ℝ)/8) ∧ series (-(start e.val u.val:ℝ)/8)≤1 := by
    have h:=(show 0≤(seriesQ (-start e.val u.val/8):ℝ) ∧ (seriesQ (-start e.val u.val/8):ℝ)≤1 by exact_mod_cast And.intro hr hR)
    simpa only [series_cast,Rat.cast_div,Rat.cast_neg,Rat.cast_ofNat] using h
  have hcenter : |Real.exp (-2*(start e.val u.val:ℝ))-(center e.val u.val:ℝ)|≤17/10^12 := by
    apply center_error _ _ _ har rfl hseries
    have h:=(Rat.cast_le (K:=ℝ)).mpr hclose
    simpa only [Rat.cast_abs,Rat.cast_sub,Rat.cast_pow,series_cast,Rat.cast_div,Rat.cast_neg,Rat.cast_ofNat,Rat.cast_one] using h
  have hd : 0≤(step e.val:ℝ)*t ∧ (step e.val:ℝ)*t≤1/128 := by
    constructor
    · exact mul_nonneg hsr.1 ht.1
    · nlinarith
  have he:=local_exp_error (start e.val u.val:ℝ) (center e.val u.val:ℝ) ((step e.val:ℝ)*t) har.1 hcenter hd
  refine rational_error _ _ (approxExp e.val u.val t) ?_ (residual_bound _ _ _ ht ⟨ha,hA,hs,hS,hc,hC,hr,hR,hclose,hres⟩) ?_
  · unfold approxExp
    apply mul_nonneg hcr
    nlinarith [sq_nonneg ((step e.val:ℝ)*t)]
  · simpa [approxExp,mul_assoc] using he

#print axioms bernstein4
#print axioms quartic_bound
#print axioms residual_bound
#print axioms ideal_error
end GB300Math.TanhCentralPolynomial
