/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2RelativeCells
import Log2OneSidedPolynomial
namespace GB300Math.Log2RelativePolynomial
open Log2Cells Log2TightCells Log2RelativeCells
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

theorem right_bound (j : Fin 512) (t : ℝ) (ht : 0≤t ∧ t≤1) :
    Real.log (1+(j.val+t)/512)/Real.log 2≤(rightIdeal j.val:ℝ) := by
  have hlog:=Log2NearOne.log_two_interval
  have hlogp : 0<Real.log 2 := by linarith
  have hx : 0<1+(j.val+t)/512 := by have:=Nat.cast_nonneg (α:=ℝ) j.val;linarith [ht.1]
  have hmono := (div_le_div_iff_of_pos_right hlogp).mpr
    (Real.log_le_log hx (show 1+(j.val+t)/512≤1+(j.val+1)/512 by linarith [ht.2]))
  unfold rightIdeal
  by_cases hj : j.val<511
  · rw [ite_eq_left hj]
    let next : Fin 512 := ⟨j.val+1,by omega⟩
    have hn:=Log2Centers.center_error next
    have he : (x next.val:ℝ)=1+(j.val+1)/512 := by dsimp [x,next];push_cast;ring
    rw [he] at hn
    have hb:=abs_le.mp hn
    push_cast
    have hv : center next.val=center (j.val+1) := rfl
    rw [hv] at hb
    linarith [hb.2]
  · rw [ite_eq_right hj]
    have he : j.val=511 := by have:=j.isLt;omega
    rw [he] at hmono ⊢
    norm_num at hmono ⊢
    simpa [div_self (ne_of_gt hlogp)] using hmono

theorem polynomial_upper (j : Fin 512) (t : ℝ) (ht : 0≤t ∧ t≤1) :
    Log2Polynomial.nativePoly j.val t-Log2Polynomial.localPoly j.val t≤(upper j.val:ℝ) := by
  rcases hd : differences j.val with ⟨a,b,c⟩
  have hrel:=cells_relative j
  simp only [RelativeCell,hd] at hrel
  have h0 : (a:ℝ)≤(upper j.val:ℝ) := by exact (Rat.cast_le (K:=ℝ)).mpr hrel.1
  have h1 : (a:ℝ)+(b:ℝ)/2≤(upper j.val:ℝ) := by simpa using (Rat.cast_le (K:=ℝ)).mpr hrel.2.1
  have h2 : (a:ℝ)+(b:ℝ)+(c:ℝ)≤(upper j.val:ℝ) := by simpa using (Rat.cast_le (K:=ℝ)).mpr hrel.2.2
  have w0 : 0≤(1-t)^2 := sq_nonneg _
  have w1 : 0≤2*t*(1-t) := mul_nonneg (mul_nonneg (by norm_num) ht.1) (sub_nonneg.mpr ht.2)
  have w2 : 0≤t^2 := sq_nonneg _
  have p0:=mul_le_mul_of_nonneg_right h0 w0
  have p1:=mul_le_mul_of_nonneg_right h1 w1
  have p2:=mul_le_mul_of_nonneg_right h2 w2
  have hid:=Log2Polynomial.difference_identity j.val t
  rw [hd] at hid
  dsimp only at hid
  rw [←hid]
  nlinarith only [p0,p1,p2]

#print axioms right_bound
#print axioms polynomial_upper
end GB300Math.Log2RelativePolynomial
