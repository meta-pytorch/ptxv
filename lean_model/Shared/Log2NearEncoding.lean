/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2NearFixed
import Log2Encoding
import Root32FloatLib
namespace GB300Math.Log2NearEncoding
open Log2Encoding
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

theorem decoded_error (v : Int) (hv : -(2^36: Int)<v ∧ v<2^36) :
    ∃ y : Rat, Root32FloatLib.value (encoded v)=some y ∧
      |(y:ℝ)-(v:ℝ)/2^36|≤1/2^24+1/2^36 := by
  by_cases hp : v>0
  · rw [encoded,ite_eq_left hp]
    obtain ⟨y,hy,hlo,hhi⟩ := Log2NearFixed.fixed_error v.toNat (by omega)
    refine ⟨y,hy,?_⟩
    have hvr : (v.toNat:ℝ)=(v:ℝ) := by exact_mod_cast Int.toNat_of_nonneg hp.le
    have hl : (0:ℝ)≤(v.toNat:ℝ)/2^36-(y:ℝ) := by simpa using (Rat.cast_nonneg (K:=ℝ)).mpr hlo
    have hh : (v.toNat:ℝ)/2^36-(y:ℝ)≤1/2^24 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hhi
    rw [hvr] at hl hh
    apply abs_le.mpr;constructor <;> linarith
  · rw [encoded,ite_eq_right hp]
    by_cases hz : (-v-1).toNat=0
    · rw [ite_eq_left hz]
      refine ⟨0,by decide +kernel,?_⟩
      have hvr : -1≤(v:ℝ) ∧ (v:ℝ)≤0 := by
        constructor
        · exact_mod_cast (by omega : -1≤v)
        · exact_mod_cast (by omega : v≤0)
      norm_num
      apply abs_le.mpr;constructor <;> linarith
    · rw [ite_eq_right hz]
      let mag := (-v-1).toNat
      obtain ⟨y,hy,hlo,hhi⟩ := Log2NearFixed.fixed_error mag (by dsimp [mag];omega)
      have hneg := RcpSigned.signed_decode (Mufu.fixedToWord mag 36) y hy
      refine ⟨-y,?_,?_⟩
      · simpa only [Nat.add_comm,Root32FloatLib.value,mag] using hneg
      have hmag : (mag:ℝ)=-(v:ℝ)-1 := by
        have h : (mag:Int)=-v-1 := by dsimp [mag];exact Int.toNat_of_nonneg (by omega)
        exact_mod_cast h
      have hl : (0:ℝ)≤(mag:ℝ)/2^36-(y:ℝ) := by simpa using (Rat.cast_nonneg (K:=ℝ)).mpr hlo
      have hh : (mag:ℝ)/2^36-(y:ℝ)≤1/2^24 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hhi
      rw [hmag] at hl hh
      push_cast
      apply abs_le.mpr;constructor <;> linarith

#print axioms decoded_error
end GB300Math.Log2NearEncoding
