/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RsqPtxCells

set_option maxRecDepth 1000000
set_option maxHeartbeats 10000000

namespace GB300Math.RsqPtxBounds
open RsqBounds

theorem cell_error (p : Parameters) (left k dropped : Int)
    (hcell : CellBound p left) (hl : 0 ≤ left ∧ left+2047 ≤ 131071)
    (hw : left ≤ k ∧ k ≤ left+2047)
    (hd : 0 ≤ dropped ∧ dropped ≤ 4718593) :
    let z := (p.a+p.b*k+p.c*(k*k-dropped))/2^34;
    -residualBound ≤ (p.m+p.step*k)*z^2-2^71 ∧
    (p.m+p.step*k)*z^2-2^71 ≤ residualBound := by
  rcases ht : taylor p left with ⟨q0,q1,q2,q3,q4,q5⟩
  simp only [CellBound,ht] at hcell
  rcases hcell with ⟨hc,hb,hm,hs,hnlo,hnhi,hlo,hhi⟩
  have hbox := polynomial_box q0 q1 q2 q3 q4 q5 (k-left) 2047 (by omega) (by omega)
  have heq : q0+q1*(k-left)+q2*(k-left)^2+q3*(k-left)^3+q4*(k-left)^4+q5*(k-left)^5 =
      (p.m+p.step*k)*(numerator p k)^2-2^139 := by
    have vals := ht
    simp only [taylor,Prod.mk.injEq] at vals
    rcases vals with ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩
    dsimp [numerator]
    ring
  rw [heq] at hbox
  let q := numerator p k
  let n := p.a+p.b*k+p.c*(k*k-dropped)
  let z := n/2^34
  let delta := p.c*dropped+n%2^34
  let m := p.m+p.step*k
  have hqlo := numerator_decreasing p k 131071 hc hb (by omega) (by omega) (by omega)
  have hqhi := numerator_decreasing p 0 k hc hb (by omega) (by omega) (by omega)
  have hqleft := numerator_decreasing p left k hc hb hl.1 hw.1 (by omega)
  have hqright := numerator_decreasing p k (left+2047) hc hb (by omega) hw.2 hl.2
  have hd0 := mul_nonneg hc hd.1
  have hd1 := mul_le_mul_of_nonneg_left hd.2 hc
  have hr : 0 ≤ n%2^34 ∧ n%2^34 ≤ 2^34-1 := by omega
  have hdelta : 0 ≤ delta ∧ delta ≤ loss p := by dsimp [delta,loss];omega
  have hn : z*2^34 = q-delta := by
    have hv : n = z*2^34+n%2^34 := by dsimp [z];omega
    dsimp [q,n,delta,numerator] at *
    nlinarith only [hv]
  have hzb : 2^23 ≤ z ∧ z ≤ 2^24 := by
    dsimp [numerator,q] at *
    constructor <;> omega
  have hmlo : 0 < p.m+p.step*left ∧ p.m+p.step*left ≤ m := by
    have ha := mul_nonneg hs hl.1
    have hb := mul_le_mul_of_nonneg_left hw.1 hs
    dsimp [m];omega
  have hmhi : m ≤ p.m+p.step*(left+2047) := by
    dsimp [m]
    have h := mul_le_mul_of_nonneg_left hw.2 hs
    omega
  have hq0 : 0 ≤ q := by omega
  have hqleft0 : 0 ≤ numerator p left := by omega
  have hloss0 : 0 ≤ q^2-(q-delta)^2 := by
    have hp := mul_nonneg hdelta.1 (show 0 ≤ 2*q-delta by omega)
    nlinarith only [hp]
  have hloss1 : q^2-(q-delta)^2 ≤ 2*q*loss p := by
    have hp := mul_nonneg (show 0 ≤ 2*q by omega) (show 0 ≤ loss p-delta by omega)
    nlinarith [sq_nonneg delta]
  have hscale0 := mul_nonneg (show 0 ≤ m by omega) hloss0
  have hscale1 := mul_le_mul_of_nonneg_left hloss1 (show 0 ≤ m by omega)
  have hproduct := mul_le_mul hmhi hqleft hq0 (show 0 ≤ p.m+p.step*(left+2047) by omega)
  have hproduct2 := mul_le_mul_of_nonneg_right hproduct (show 0 ≤ 2*loss p by omega)
  have hres : -residualBound ≤ m*z^2-2^71 ∧
      m*z^2-2^71 ≤ residualBound := by
    have hid : (m*z^2-2^71)*2^68 = (m*q^2-2^139)-m*(q^2-(q-delta)^2) := by
      rw [show (2 : Int)^68 = (2^34)^2 by norm_num]
      calc
        _ = m*(z*2^34)^2-2^139 := by ring
        _ = _ := by rw [hn];ring
    dsimp [m,q] at hid hscale0 hscale1 hproduct2 ⊢
    constructor <;> nlinarith only [hid,hbox.1,hbox.2,hlo,hhi,hscale0,hscale1,hproduct2]
  exact hres

theorem interpolate_residual (index : Nat) (hi : index < 2^24) :
    let z := Mufu.interpolate MufuTables.rsq index
    let m : Int := (2^23+index%2^23)*2^(index/2^23);
    -residualBound ≤ m*z^2-2^71 ∧ m*z^2-2^71 ≤ residualBound := by
  let segment : Fin 128 := ⟨index/131072,by omega⟩
  let k := index%131072
  let cell : Fin 64 := ⟨k/2048,by dsimp [k];omega⟩
  let drop := k*k-Fp64Seed.squareTruncated k 19
  have hs : Fp64Seed.squareTruncated k 19 ≤ k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hd : 0 ≤ (drop : Int) ∧ (drop : Int) ≤ 4718593 := by
    constructor
    · exact Int.natCast_nonneg _
    · exact_mod_cast SqrtBounds.square_drop_bound k
  have hl : 0 ≤ (cell.val : Int)*2048 ∧ (cell.val : Int)*2048+2047 ≤ 131071 := by
    have := cell.isLt
    omega
  have hw : (cell.val : Int)*2048 ≤ (k : Int) ∧ (k : Int) ≤ (cell.val : Int)*2048+2047 := by
    dsimp [cell];omega
  have h : let z := ((parameters segment.val).a+(parameters segment.val).b*(k : Int)+
      (parameters segment.val).c*((k : Int)*(k : Int)-(drop : Int)))/2^34;
      -residualBound ≤ ((parameters segment.val).m+(parameters segment.val).step*(k : Int))*z^2-2^71 ∧
      ((parameters segment.val).m+(parameters segment.val).step*(k : Int))*z^2-2^71 ≤ residualBound := by
    rcases cells_valid segment cell with hcell | hexact
    · exact cell_error (parameters segment.val) ((cell.val : Int)*2048) k drop hcell hl hw hd
    · let offset : Fin 2048 := ⟨k%2048,by omega⟩
      have heq : cell.val*2048+offset.val = k := by dsimp [cell,offset];omega
      have hex := hexact offset
      dsimp only at hex
      rw [heq] at hex
      have hc : (k : Int)*(k : Int)-(drop : Int) = Fp64Seed.squareTruncated k 19 := by
        have hh := Nat.sub_add_cancel hs
        have hj : (drop : Int)+(Fp64Seed.squareTruncated k 19 : Int) = (k : Int)*(k : Int) := by exact_mod_cast hh
        omega
      rw [hc]
      exact hex
  have hsquare : (k : Int)*(k : Int)-(drop : Int) = Fp64Seed.squareTruncated k 19 := by
    have hh := Nat.sub_add_cancel hs
    have hc : (drop : Int)+(Fp64Seed.squareTruncated k 19 : Int) = (k : Int)*(k : Int) := by
      exact_mod_cast hh
    omega
  have hpar : segment.val/64 = index/2^23 := by dsimp [segment];omega
  have hrem : (segment.val%64 : Int)*131072+(k : Int) = index%2^23 := by
    dsimp [segment,k];omega
  have hm : (parameters segment.val).m+(parameters segment.val).step*(k : Int) =
      (2^23+(index%2^23 : Nat))*2^(index/2^23) := by
    dsimp [parameters]
    rw [hpar]
    push_cast
    calc
      _ = (8388608+((segment.val%64 : Int)*131072+(k : Int)))*2^(index/2^23) := by ring
      _ = _ := by rw [hrem];norm_num
  dsimp only at h ⊢
  rw [hm,hsquare] at h
  convert h using 1 <;>
    dsimp [Mufu.interpolate,MufuTables.rsq,parameters,segment,k] <;> norm_num <;> ring_nf


theorem relative_from_squared (x y e : ℝ) (hx : 0 < x) (hy : 0 ≤ y)
    (he : 0 ≤ e ∧ e ≤ 1)
    (hlo : (1-e)^2 ≤ x*y^2) (hhi : x*y^2 ≤ (1+e)^2) :
    |y-1/Real.sqrt x| ≤ e/Real.sqrt x := by
  have hs : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  have hsq : (y*Real.sqrt x)^2 = x*y^2 := by rw [mul_pow,Real.sq_sqrt hx.le];ring
  have hprod : 0 ≤ y*Real.sqrt x := mul_nonneg hy hs.le
  have hl : 1-e ≤ y*Real.sqrt x := by nlinarith only [hlo,hsq,hprod,he.2]
  have hh : y*Real.sqrt x ≤ 1+e := by nlinarith only [hhi,hsq,hprod,he.1]
  have hb : |y*Real.sqrt x-1| ≤ e := abs_le.mpr ⟨by linarith,by linarith⟩
  have hid : y-1/Real.sqrt x = (y*Real.sqrt x-1)/Real.sqrt x := by field_simp
  rw [hid,abs_div,abs_of_pos hs]
  exact div_le_div_of_nonneg_right hb hs.le

theorem normalized_relative_error (index : Nat) (hi : index < 2^24) :
    |normalizedOutput index-1/Real.sqrt (SqrtBounds.normalizedInput index)| ≤
      ((17 : ℝ)/2^27)/Real.sqrt (SqrtBounds.normalizedInput index) := by
  by_cases hz : index = 0
  · norm_num [normalizedOutput,hz,SqrtBounds.normalizedInput]
  have h := interpolate_residual index hi
  dsimp only at h
  have hlo := (Int.cast_le (R := ℝ)).mpr h.1
  have hhi := (Int.cast_le (R := ℝ)).mpr h.2
  dsimp [residualBound] at hlo hhi
  push_cast at hlo hhi
  have hm : (index : Int)%8388608 = ((index%8388608 : Nat) : Int) := by omega
  rw [hm] at hlo hhi
  simp only [Int.cast_natCast] at hlo hhi
  have hb := RsqBounds.interpolate_error index hi
  have hy : 0 ≤ normalizedOutput index := by
    have hr : 0 ≤ (Mufu.interpolate MufuTables.rsq index : ℝ) := by exact_mod_cast (show (0 : Int) ≤ Mufu.interpolate MufuTables.rsq index by omega)
    simp only [normalizedOutput,ite_eq_right hz]
    positivity
  apply relative_from_squared _ _ _ (by dsimp [SqrtBounds.normalizedInput];positivity) hy (by norm_num)
  · dsimp [SqrtBounds.normalizedInput,normalizedOutput]
    rw [ite_eq_right hz]
    norm_num at hlo ⊢
    nlinarith only [hlo]
  · dsimp [SqrtBounds.normalizedInput,normalizedOutput]
    rw [ite_eq_right hz]
    norm_num at hhi ⊢
    nlinarith only [hhi]

open SqrtBounds (positiveNormalValue scaleExponent mantissaIndex outputScale)

theorem rsq_bits_relative_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rsq (exponent*2^23+fraction))-
      1/Real.sqrt (positiveNormalValue (exponent*2^23+fraction))| ≤
      ((17 : ℝ)/2^27)/Real.sqrt (positiveNormalValue (exponent*2^23+fraction)) := by
  have hp := SqrtBounds.index_properties exponent fraction hf
  have hs : 0 < outputScale exponent := by unfold outputScale;positivity
  have ho := output_decode_scaled exponent fraction he hf
  have hxo : positiveNormalValue (Mufu.rsq (exponent*2^23+fraction)) =
      normalizedOutput (mantissaIndex exponent fraction)/outputScale exponent := by
    apply (eq_div_iff hs.ne').mpr
    exact ho
  rw [hxo,SqrtBounds.input_decode_scaled exponent fraction (by omega) hf,
    Real.sqrt_mul (show 0 ≤ SqrtBounds.normalizedInput (mantissaIndex exponent fraction) by
      unfold SqrtBounds.normalizedInput;positivity),Real.sqrt_sq hs.le]
  rw [show 1/(Real.sqrt (SqrtBounds.normalizedInput (mantissaIndex exponent fraction))*outputScale exponent) =
    (1/Real.sqrt (SqrtBounds.normalizedInput (mantissaIndex exponent fraction)))/outputScale exponent by ring,
    ← sub_div,abs_div,abs_of_pos hs]
  simpa only [div_div,mul_assoc] using div_le_div_of_nonneg_right (normalized_relative_error _ hp.1) hs.le

theorem rational_bound_stronger_than_ptx : (17 : ℝ)/2^27 < (2 : ℝ)^(-(229 : ℝ)/10) := by
  have hp : ((2 : ℝ)^(-(229 : ℝ)/10))^10 = (2 : ℝ)^(-229 : Int) := by
    rw [← Real.rpow_natCast,← Real.rpow_mul (by norm_num : (0 : ℝ) ≤ 2)]
    norm_num
  by_contra h
  have he : (2 : ℝ)^(-(229 : ℝ)/10) ≤ (17 : ℝ)/2^27 := le_of_not_gt h
  have hpow := pow_le_pow_left₀ (Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 2) _).le he 10
  rw [hp] at hpow
  norm_num at hpow

theorem rsq_bits_ptx_relative_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rsq (exponent*2^23+fraction))-
      1/Real.sqrt (positiveNormalValue (exponent*2^23+fraction))| ≤
      ((2 : ℝ)^(-(229 : ℝ)/10))/Real.sqrt (positiveNormalValue (exponent*2^23+fraction)) := by
  apply (rsq_bits_relative_error exponent fraction he hf).trans
  exact div_le_div_of_nonneg_right rational_bound_stronger_than_ptx.le (Real.sqrt_nonneg _)

#print axioms cells_valid
#print axioms normalized_relative_error
#print axioms rsq_bits_relative_error
#print axioms rational_bound_stronger_than_ptx
#print axioms rsq_bits_ptx_relative_error
end GB300Math.RsqPtxBounds
