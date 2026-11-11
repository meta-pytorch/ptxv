/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SineCellsProof
namespace GB300Math.SinePolynomialBounds
open SineCells
set_option maxRecDepth 12000
set_option maxHeartbeats 4000000

theorem bernstein (a b c t bound : ℝ) (ht : 0 ≤ t ∧ t ≤ 1)
    (h0 : |a| ≤ bound) (h1 : |a+b/2| ≤ bound) (h2 : |a+b+c| ≤ bound) :
    |a+b*t+c*t^2| ≤ bound := by
  have w0 : 0 ≤ (1-t)^2 := sq_nonneg _
  have w1 : 0 ≤ 2*t*(1-t) := mul_nonneg (mul_nonneg (by norm_num) ht.1) (sub_nonneg.mpr ht.2)
  have w2 : 0 ≤ t^2 := sq_nonneg _
  have p0 := mul_le_mul_of_nonneg_right (abs_le.mp h0).2 w0
  have p1 := mul_le_mul_of_nonneg_right (abs_le.mp h1).2 w1
  have p2 := mul_le_mul_of_nonneg_right (abs_le.mp h2).2 w2
  have n0 := mul_le_mul_of_nonneg_right (abs_le.mp h0).1 w0
  have n1 := mul_le_mul_of_nonneg_right (abs_le.mp h1).1 w1
  have n2 := mul_le_mul_of_nonneg_right (abs_le.mp h2).1 w2
  apply abs_le.mpr
  constructor <;> nlinarith only [p0,p1,p2,n0,n1,n2]

noncomputable def nativePoly (j : Nat) (h : ℝ) : ℝ :=
  let (a,b,c) := nativeCoefficients j
  (a:ℝ)+(b:ℝ)*h+(c:ℝ)*h^2

noncomputable def localTaylor (j : Nat) (h : ℝ) : ℝ :=
  (centerSin j:ℝ)+(centerCos j:ℝ)*((SineTaylor.piRat:ℝ)/512*h)-
    (centerSin j:ℝ)*((SineTaylor.piRat:ℝ)/512*h)^2/2

theorem difference_identity (j : Nat) (h : ℝ) :
    let (a,b,c) := differences j
    (a:ℝ)+(b:ℝ)*h+(c:ℝ)*h^2 = nativePoly j h-localTaylor j h := by
  unfold differences nativePoly localTaylor
  rcases nativeCoefficients j with ⟨a,b,c⟩
  dsimp only
  push_cast
  ring

theorem polynomial_error (j : Fin 256) (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) :
    |nativePoly j.val t-localTaylor j.val t| ≤ 1/5000000 := by
  rcases hd : differences j.val with ⟨a,b,c⟩
  have hv := cells_valid j
  simp only [CellValid,hd] at hv
  have h0 : |(a:ℝ)| ≤ 1/5000000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr (abs_le.mpr ⟨hv.2.2.2.2.1,hv.2.2.2.2.2.1⟩ : |a| ≤ 1/5000000)
  have h1 : |(a:ℝ)+(b:ℝ)/2| ≤ 1/5000000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr (abs_le.mpr ⟨hv.2.2.2.2.2.2.1,hv.2.2.2.2.2.2.2.1⟩ : |a+b/2| ≤ 1/5000000)
  have h2 : |(a:ℝ)+(b:ℝ)+(c:ℝ)| ≤ 1/5000000 := by simpa using (Rat.cast_le (K:=ℝ)).mpr (abs_le.mpr ⟨hv.2.2.2.2.2.2.2.2.1,hv.2.2.2.2.2.2.2.2.2⟩ : |a+b+c| ≤ 1/5000000)
  have hi := difference_identity j.val t
  rw [hd] at hi
  dsimp only at hi
  rw [←hi]
  exact bernstein _ _ _ t _ ht h0 h1 h2

theorem cell_ideal_error (j : Fin 256) (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) :
    |nativePoly j.val t-Real.sin (Real.pi*(j.val+t)/512)| ≤ 301/1000000000 := by
  have hj := j.isLt
  have hjr : (j.val:ℝ) ≤ 255 := by exact_mod_cast (show j.val ≤ 255 by omega)
  have hd : 0 ≤ (SineTaylor.piRat:ℝ)/512*t ∧ (SineTaylor.piRat:ℝ)/512*t ≤ 1/128 := by
    norm_num [SineTaylor.piRat]
    constructor <;> nlinarith [ht.1,ht.2]
  have h := SineLocal.approximate_centers ((SineTaylor.piRat:ℝ)*j.val/512)
    ((SineTaylor.piRat:ℝ)/512*t) (centerSin j.val:ℝ) (centerCos j.val:ℝ)
    hd (center_sin j) (center_cos j)
  have he : (SineTaylor.piRat:ℝ)*j.val/512+(SineTaylor.piRat:ℝ)/512*t = (SineTaylor.piRat:ℝ)*(j.val+t)/512 := by ring
  rw [he] at h
  change |Real.sin ((SineTaylor.piRat:ℝ)*(j.val+t)/512)-localTaylor j.val t| ≤ 1/10000000 at h
  have hp := polynomial_error j t ht
  have hl := Real.abs_sin_sub_sin_le (Real.pi*(j.val+t)/512) ((SineTaylor.piRat:ℝ)*(j.val+t)/512)
  have he' : Real.pi*(j.val+t)/512-(SineTaylor.piRat:ℝ)*(j.val+t)/512 = (Real.pi-(SineTaylor.piRat:ℝ))*((j.val+t)/512) := by ring
  rw [he',abs_mul,abs_of_nonneg (div_nonneg (add_nonneg (Nat.cast_nonneg _) ht.1) (by norm_num) : 0 ≤ (j.val+t)/512)] at hl
  have hpi := mul_le_mul_of_nonneg_right SineTaylor.pi_approx (div_nonneg (add_nonneg (Nat.cast_nonneg _) ht.1) (by norm_num) : 0 ≤ (j.val+t)/512)
  have htriangle := abs_sub_le (nativePoly j.val t) (localTaylor j.val t) (Real.sin ((SineTaylor.piRat:ℝ)*(j.val+t)/512))
  have htriangle2 := abs_sub_le (nativePoly j.val t) (Real.sin ((SineTaylor.piRat:ℝ)*(j.val+t)/512)) (Real.sin (Real.pi*(j.val+t)/512))
  rw [abs_sub_comm (localTaylor j.val t)] at htriangle
  rw [abs_sub_comm (Real.sin ((SineTaylor.piRat:ℝ)*(j.val+t)/512))] at htriangle2
  nlinarith [ht.2]
#print axioms bernstein
#print axioms cell_ideal_error
end GB300Math.SinePolynomialBounds
