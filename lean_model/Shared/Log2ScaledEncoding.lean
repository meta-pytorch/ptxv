/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2ScaledFixed
import Log2Encoding
namespace GB300Math.Log2ScaledEncoding
open Log2Encoding
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

theorem positive (v : Int) (scale : Nat) (hs : 1≤ scale ∧ scale≤ 127)
    (hv : 0< v ∧ v<(scale+1:Nat)*2^36) :
    ∃ y : Rat,Root32FloatLib.value (encoded v)=some y ∧
      -(scale:ℝ)/2^23≤(y:ℝ)-(v:ℝ)/2^36 ∧ (y:ℝ)-(v:ℝ)/2^36≤ 0 := by
  rw [encoded,ite_eq_left hv.1]
  obtain ⟨y,hy,hlo,hhi⟩:=Log2ScaledFixed.fixed_error v.toNat scale hs (by omega)
  refine ⟨y,hy,?_⟩
  have hvr : (v.toNat:ℝ)=(v:ℝ) := by exact_mod_cast Int.toNat_of_nonneg hv.1.le
  have hl : (0:ℝ)≤(v.toNat:ℝ)/2^36-(y:ℝ) := by simpa using (Rat.cast_nonneg (K:=ℝ)).mpr hlo
  have hh : (v.toNat:ℝ)/2^36-(y:ℝ)≤(scale:ℝ)/2^23 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hhi
  rw [hvr] at hl hh
  constructor <;> linarith

theorem negative (v : Int) (scale : Nat) (hs : 1≤ scale ∧ scale≤ 127)
    (hv : v< 0 ∧ -v≤(scale+1:Nat)*2^36) :
    ∃ y : Rat,Root32FloatLib.value (encoded v)=some y ∧
      0≤(y:ℝ)-(v:ℝ)/2^36 ∧ (y:ℝ)-(v:ℝ)/2^36≤(scale:ℝ)/2^23+1/2^36 := by
  rw [encoded,ite_eq_right (by omega : ¬v> 0)]
  by_cases hz : (-v-1).toNat=0
  · rw [ite_eq_left hz]
    refine ⟨0,by decide +kernel,?_⟩
    have he : v = -1 := by omega
    rw [he]
    have hsr : (0:ℝ)≤ scale := Nat.cast_nonneg _
    norm_num
    positivity
  · rw [ite_eq_right hz]
    let mag := (-v-1).toNat
    obtain ⟨y,hy,hlo,hhi⟩:=Log2ScaledFixed.fixed_error mag scale hs (by dsimp [mag];omega)
    have hneg:=RcpSigned.signed_decode (Mufu.fixedToWord mag 36) y hy
    refine ⟨-y,?_,?_⟩
    · simpa only [Nat.add_comm,Root32FloatLib.value,mag] using hneg
    have hmag : (mag:ℝ)=-(v:ℝ)-1 := by
      have h : (mag:Int)=-v-1 := by dsimp [mag];exact Int.toNat_of_nonneg (by omega)
      exact_mod_cast h
    have hl : (0:ℝ)≤(mag:ℝ)/2^36-(y:ℝ) := by simpa using (Rat.cast_nonneg (K:=ℝ)).mpr hlo
    have hh : (mag:ℝ)/2^36-(y:ℝ)≤(scale:ℝ)/2^23 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hhi
    rw [hmag] at hl hh
    push_cast
    constructor <;> linarith
#print axioms positive
#print axioms negative
end GB300Math.Log2ScaledEncoding
