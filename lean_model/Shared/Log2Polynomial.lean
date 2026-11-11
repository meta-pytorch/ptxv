/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2Centers
import SinePolynomialBounds
namespace GB300Math.Log2Polynomial
open Log2Cells
set_option maxRecDepth 12000
set_option maxHeartbeats 5000000
noncomputable def nativePoly (j : Nat) (t : ℝ) : ℝ :=
  let (a,b,c) := nativeCoefficients j
  (a:ℝ)+(b:ℝ)*t+(c:ℝ)*t^2
noncomputable def localPoly (j : Nat) (t : ℝ) : ℝ :=
  (center j:ℝ)+(linear j:ℝ)*t+(quadratic j:ℝ)*t^2

theorem difference_identity (j : Nat) (t : ℝ) :
    let (a,b,c) := differences j
    (a:ℝ)+(b:ℝ)*t+(c:ℝ)*t^2=nativePoly j t-localPoly j t := by
  unfold differences nativePoly localPoly
  rcases nativeCoefficients j with ⟨a,b,c⟩
  dsimp only
  push_cast
  ring

theorem polynomial_error (j : Fin 512) (t : ℝ) (ht : 0≤t ∧ t≤1) :
    |nativePoly j.val t-localPoly j.val t|≤14/100000000 := by
  rcases hd : differences j.val with ⟨a,b,c⟩
  have hv := cells_valid j
  simp only [CellValid,hd] at hv
  have h0 : |(a:ℝ)|≤14/100000000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hv.2.2.2.1
  have h1 : |(a:ℝ)+(b:ℝ)/2|≤14/100000000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hv.2.2.2.2.1
  have h2 : |(a:ℝ)+(b:ℝ)+(c:ℝ)|≤14/100000000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hv.2.2.2.2.2
  have hi := difference_identity j.val t
  rw [hd] at hi
  dsimp only at hi
  rw [←hi]
  exact SinePolynomialBounds.bernstein _ _ _ t _ ht h0 h1 h2

theorem local_error (j : Fin 512) (t : ℝ) (ht : 0≤t ∧ t≤1) :
    |Real.log ((x j.val:ℝ)+t/512)/Real.log 2-localPoly j.val t|≤4/1000000000 := by
  have hx := Log2Centers.x_range j
  have hx' : (1:ℝ)≤(x j.val:ℝ) := by exact_mod_cast hx.1
  have hd : 0≤t/512 := div_nonneg ht.1 (by norm_num)
  have hrem := Log2Local.local_remainder (x j.val:ℝ) (t/512) hx' hd
  have hlog := Log2NearOne.log_two_interval
  have hlogp : 0<Real.log 2 := by linarith
  have hdiv := (div_le_div_iff_of_pos_right hlogp).mpr hrem
  have hc := abs_le.mp (Log2Centers.center_error j)
  have hl := abs_le.mp (Log2Centers.linear_error j)
  have hq := abs_le.mp (Log2Centers.quadratic_error j)
  have hl0 := mul_le_mul_of_nonneg_right hl.1 ht.1
  have hl1 := mul_le_mul_of_nonneg_right hl.2 ht.1
  have hq0 := mul_le_mul_of_nonneg_right hq.1 (sq_nonneg t)
  have hq1 := mul_le_mul_of_nonneg_right hq.2 (sq_nonneg t)
  let ideal := Real.log (x j.val:ℝ)/Real.log 2+
    1/(512*(x j.val:ℝ)*Real.log 2)*t+
    (-1)/(2*512^2*(x j.val:ℝ)^2*Real.log 2)*t^2
  have hp : |ideal-localPoly j.val t|≤6/1000000000000 := by
    dsimp [ideal,localPoly]
    apply abs_le.mpr
    have ht2 : t^2≤1 := by nlinarith [sq_nonneg (1-t)]
    constructor <;> nlinarith only [hc.1,hc.2,hl0,hl1,hq0,hq1,ht.1,ht.2,ht2]
  have hid : (Real.log ((x j.val:ℝ)+t/512)-
    (Real.log (x j.val:ℝ)+(t/512)/(x j.val:ℝ)-(t/512)^2/(2*(x j.val:ℝ)^2)))/Real.log 2=
      Real.log ((x j.val:ℝ)+t/512)/Real.log 2-ideal := by dsimp [ideal];ring
  have hr : |Real.log ((x j.val:ℝ)+t/512)/Real.log 2-ideal|≤(t/512)^3/(3*Real.log 2) := by
    rw [←hid,abs_div,abs_of_pos hlogp]
    convert hdiv using 1; ring
  have ht3 : (t/512)^3≤(1/512:ℝ)^3 := pow_le_pow_left₀ hd (by linarith) 3
  have hr' : (t/512)^3/(3*Real.log 2)≤36/10000000000 := by
    apply (div_le_iff₀ (by positivity : 0<3*Real.log 2)).mpr
    nlinarith only [ht3,hlog.1]
  have htriangle := abs_sub_le (Real.log ((x j.val:ℝ)+t/512)/Real.log 2) ideal (localPoly j.val t)
  linarith

theorem cell_error (j : Fin 512) (t : ℝ) (ht : 0≤t ∧ t≤1) :
    |nativePoly j.val t-Real.log (1+(j.val+t)/512)/Real.log 2|≤144/1000000000 := by
  have hp := polynomial_error j t ht
  have hl := local_error j t ht
  have hx : (x j.val:ℝ)+t/512=1+(j.val+t)/512 := by dsimp [x];push_cast;ring
  rw [hx] at hl
  have h := abs_sub_le (nativePoly j.val t) (localPoly j.val t)
    (Real.log (1+(j.val+t)/512)/Real.log 2)
  rw [abs_sub_comm] at hl
  linarith
#print axioms polynomial_error
#print axioms local_error
#print axioms cell_error
end GB300Math.Log2Polynomial
