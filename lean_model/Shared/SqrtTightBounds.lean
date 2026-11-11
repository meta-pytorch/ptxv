/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SqrtBounds
set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace GB300Math.SqrtTightBounds
open GB300Math.SqrtBounds

abbrev CellBound (p : Parameters) (left : Int) : Prop :=
  let (q0,q1,q2,q3,q4) := taylor p left
  let e := -p.c*4718593
  let bound := (2*(p.a/2^36)-1)*2^72
  p.c ≤ 0 ∧ 0 ≤ p.b+262142*p.c ∧ 2^59 ≤ p.a ∧
  qmax p+e < 2^60 ∧
  -bound+2*qmax p*2^36 ≤ lower q0 q1 q2 q3 q4 8191 ∧
  upper q0 q1 q2 q3 q4 8191+2*qmax p*e+e^2 ≤ bound

theorem cells_valid : ∀ segment : Fin 128, ∀ cell : Fin 16,
    CellBound (parameters segment.val) ((cell.val : Int)*8192) := by decide +kernel

theorem cell_error (p : Parameters) (left k dropped : Int)
    (hcell : CellBound p left) (hk : 0 ≤ k ∧ k ≤ 131071)
    (hwindow : 0 ≤ k-left ∧ k-left ≤ 8191)
    (hd : 0 ≤ dropped ∧ dropped ≤ 4718593) :
    let z := (p.a+p.b*k+p.c*(k*k-dropped))/2^36
    2^23 ≤ z ∧ z < 2^24 ∧
    -(2*z-1) ≤ z*z-(p.m+p.step*k) ∧ z*z-(p.m+p.step*k) ≤ 2*z-1 := by
  rcases ht : taylor p left with ⟨q0,q1,q2,q3,q4⟩
  simp only [CellBound,ht] at hcell
  rcases hcell with ⟨hc,hb,ha,hmax,hlo,hhi⟩
  have hq := numerator_range p k hc hb hk
  have hbox := polynomial_box q0 q1 q2 q3 q4 (k-left) 8191 hwindow.1 hwindow.2
  have heq : q0+q1*(k-left)+q2*(k-left)^2+q3*(k-left)^3+q4*(k-left)^4 =
      (p.a+p.b*k+p.c*k*k)^2-(p.m+p.step*k)*2^72 := by
    have vals := ht
    simp only [taylor,Prod.mk.injEq] at vals
    rcases vals with ⟨rfl,rfl,rfl,rfl,rfl⟩
    ring
  rw [heq] at hbox
  let q := p.a+p.b*k+p.c*k*k
  let n := p.a+p.b*k+p.c*(k*k-dropped)
  let z := n/2^36
  let delta := -p.c*dropped-n%2^36
  let e := -p.c*4718593
  have hr : 0 ≤ n%2^36 ∧ n%2^36 ≤ 2^36-1 := by omega
  have hd0 := mul_nonneg (show 0 ≤ -p.c by omega) hd.1
  have hd1 := mul_le_mul_of_nonneg_left hd.2 (show 0 ≤ -p.c by omega)
  have he0 : 0 ≤ e := by dsimp [e]; omega
  have hdelta : -2^36 ≤ delta ∧ delta ≤ e := by dsimp [delta,e]; omega
  have hn : z*2^36 = q+delta := by
    have hv : n = z*2^36+n%2^36 := by dsimp [z]; omega
    dsimp [q,delta,n] at *
    nlinarith only [hv]
  have hzn : 2^23 ≤ z ∧ z < 2^24 := by
    dsimp [q] at hn
    constructor <;> omega
  have hnmin : p.a ≤ n := by dsimp [n,q] at *; nlinarith only [hq.1,hd0]
  have hmin : p.a/2^36 ≤ z := by dsimp [z]; omega
  have hloss := mul_nonneg (show 0 ≤ 2*q by dsimp [q]; omega)
    (show 0 ≤ delta+2^36 by omega)
  have hqmax := mul_nonneg (show 0 ≤ 2*(qmax p-q) by dsimp [q]; omega)
    (show (0 : Int) ≤ 2^36 by norm_num)
  have hlower : -2*qmax p*2^36 ≤ (q+delta)^2-q^2 := by nlinarith [sq_nonneg delta]
  have hstep := mul_nonneg (show 0 ≤ e-delta by omega)
    (show 0 ≤ 2*q+e+delta by dsimp [q]; omega)
  have hu := mul_nonneg (show 0 ≤ 2*(qmax p-q) by dsimp [q]; omega) he0
  have hupper : (q+delta)^2-q^2 ≤ 2*qmax p*e+e^2 := by nlinarith
  have hid : (z*z-(p.m+p.step*k))*2^72 =
      (q^2-(p.m+p.step*k)*2^72)+((q+delta)^2-q^2) := by
    rw [← hn]
    ring
  change 2^23 ≤ z ∧ z < 2^24 ∧ _
  refine ⟨hzn.1,hzn.2,?_,?_⟩
  · dsimp [q,e] at hid hlower hupper
    nlinarith only [hid,hlower,hbox.1,hlo,hmin]
  · dsimp [q,e] at hid hlower hupper
    nlinarith only [hid,hupper,hbox.2,hhi,hmin]

theorem interpolate_error (index : Nat) (hi : index < 2^24) :
    let z := Mufu.interpolate MufuTables.sqrt index
    let target : Int := (2^23+index%2^23)*2^(23+index/2^23)
    2^23 ≤ z ∧ z < 2^24 ∧ -(2*z-1) ≤ z*z-target ∧ z*z-target ≤ 2*z-1 := by
  let segment : Fin 128 := ⟨index/131072,by omega⟩
  let k := index%131072
  let cell : Fin 16 := ⟨k/8192,by dsimp [k];omega⟩
  let drop := k*k-Fp64Seed.squareTruncated k 19
  have hs : Fp64Seed.squareTruncated k 19 ≤ k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hd : 0 ≤ (drop : Int) ∧ (drop : Int) ≤ 4718593 := by
    constructor
    · exact Int.natCast_nonneg _
    · exact_mod_cast square_drop_bound k
  have hk : 0 ≤ (k : Int) ∧ (k : Int) ≤ 131071 := by dsimp [k];omega
  have hw : 0 ≤ (k : Int)-(cell.val : Int)*8192 ∧
      (k : Int)-(cell.val : Int)*8192 ≤ 8191 := by dsimp [cell];omega
  have h := cell_error (parameters segment.val) ((cell.val : Int)*8192) k drop
    (cells_valid segment cell) hk hw hd
  have hsquare : (k : Int)*(k : Int)-(drop : Int) = Fp64Seed.squareTruncated k 19 := by
    have hh := Nat.sub_add_cancel hs
    have hc : (drop : Int)+(Fp64Seed.squareTruncated k 19 : Int) = (k : Int)*(k : Int) := by
      exact_mod_cast hh
    omega
  have hpar : segment.val/64 = index/2^23 := by dsimp [segment];omega
  have hrem : (segment.val%64 : Int)*131072+(k : Int) = index%2^23 := by
    dsimp [segment,k];omega
  have hm : (parameters segment.val).m+(parameters segment.val).step*(k : Int) =
      (2^23+(index%2^23 : Nat))*2^(23+index/2^23) := by
    dsimp [parameters]
    rw [hpar]
    push_cast
    calc
      _ = (8388608+((segment.val%64 : Int)*131072+(k : Int)))*2^(23+index/2^23) := by ring
      _ = _ := by rw [hrem]; norm_num
  dsimp only at h ⊢
  rw [hm,hsquare] at h
  convert h using 1 <;>
    dsimp [Mufu.interpolate,MufuTables.sqrt,parameters,segment,k] <;> norm_num <;> ring_nf


theorem sqrt_absolute_error (index : Nat) (hi : index < 2^24) :
    |normalizedOutput index - Real.sqrt (normalizedInput index)| ≤ 1/2^23 := by
  have h := interpolate_error index hi
  dsimp only at h
  have hy : (1 : ℝ) ≤ normalizedOutput index := by
    dsimp [normalizedOutput]
    have hz : (2 : ℝ)^23 ≤ (Mufu.interpolate MufuTables.sqrt index : ℝ) := by exact_mod_cast h.1
    norm_num at hz ⊢
    linarith
  have hx : (0 : ℝ) ≤ normalizedInput index := by unfold normalizedInput; positivity
  have hlo := (Int.cast_le (R := ℝ)).mpr h.2.2.1
  have hhi := (Int.cast_le (R := ℝ)).mpr h.2.2.2
  push_cast at hlo hhi
  have hm : (index : Int)%8388608 = ((index%8388608 : Nat) : Int) := by omega
  rw [hm] at hlo hhi
  simp only [Int.cast_natCast] at hlo hhi
  have he : normalizedOutput index ^ 2 - normalizedInput index =
      ((Mufu.interpolate MufuTables.sqrt index : ℝ)^2-
      ((2 : ℝ)^23+((index%2^23 : Nat) : ℝ))*2^(23+index/2^23))/2^46 := by
    dsimp [normalizedInput,normalizedOutput]
    rw [pow_add]
    ring
  have hz : (Mufu.interpolate MufuTables.sqrt index : ℝ) = normalizedOutput index*2^23 := by
    dsimp [normalizedOutput]
    ring
  rw [hz] at hlo hhi he
  have hs := Real.sq_sqrt hx
  have hs0 := Real.sqrt_nonneg (normalizedInput index)
  have he1 : (normalizedOutput index - 1/2^23)^2 ≤ normalizedInput index := by
    norm_num at hlo hhi he ⊢
    nlinarith only [hhi,he]
  have he2 : normalizedInput index ≤ (normalizedOutput index + 1/2^23)^2 := by
    norm_num at hlo hhi he ⊢
    nlinarith only [hlo,he]
  apply abs_le.mpr
  constructor
  · by_contra hc
    have hl : normalizedOutput index + 1/2^23 < Real.sqrt (normalizedInput index) := by linarith
    have hp := mul_pos (sub_pos.mpr hl)
      (show 0 < Real.sqrt (normalizedInput index)+(normalizedOutput index+1/2^23) by linarith)
    nlinarith only [hp,he2,hs]
  · by_contra hc
    have hl : Real.sqrt (normalizedInput index) < normalizedOutput index - 1/2^23 := by linarith
    have hp := mul_pos (sub_pos.mpr hl)
      (show 0 < normalizedOutput index-1/2^23+Real.sqrt (normalizedInput index) by linarith)
    nlinarith only [hp,he1,hs]

theorem sqrt_bits_absolute_error (index : Nat) (hi : index < 2^24) :
    |positiveNormalValue (Mufu.sqrt (0x3f800000+index))-
      Real.sqrt (positiveNormalValue (0x3f800000+index))| ≤ 1/2^23 := by
  rw [output_decode index hi,input_decode index hi]
  exact sqrt_absolute_error index hi
theorem sqrt_bits_scaled_absolute_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.sqrt (exponent*2^23+fraction))-
      Real.sqrt (positiveNormalValue (exponent*2^23+fraction))| ≤ outputScale exponent/2^23 := by
  have hp := index_properties exponent fraction hf
  have hs : 0 < outputScale exponent := by unfold outputScale; positivity
  rw [input_decode_scaled exponent fraction (by omega) hf,output_decode_scaled exponent fraction he hf,
    Real.sqrt_mul (show 0 ≤ normalizedInput (mantissaIndex exponent fraction) by unfold normalizedInput;positivity),
    Real.sqrt_sq (le_of_lt hs)]
  rw [← sub_mul,abs_mul,abs_of_pos hs]
  have h := mul_le_mul_of_nonneg_right (sqrt_absolute_error (mantissaIndex exponent fraction) hp.1) (le_of_lt hs)
  linarith


#print axioms cells_valid
#print axioms sqrt_bits_absolute_error
#print axioms sqrt_bits_scaled_absolute_error
end GB300Math.SqrtTightBounds
