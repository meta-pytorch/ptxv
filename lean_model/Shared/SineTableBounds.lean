/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SinePolynomialBounds
import SineInterpolationInteger
import SineFixedDecode
namespace GB300Math.SineTableBounds
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 12000
set_option maxHeartbeats 6000000

theorem table_error (index : Nat) (hi : index<2^23) :
    |(Mufu.interpolate MufuTables.sin index:ℝ)/2^37-Real.sin (Real.pi*index/2^24)| ≤ 401/1000000000 := by
  let j : Fin 256 := ⟨index/32768,by omega⟩
  let t : ℝ := (index%32768:Nat)/32768
  have ht : 0 ≤ t ∧ t ≤ 1 := by
    have hr : ((index%32768:Nat):ℝ) ≤ 32768 := by exact_mod_cast (show index%32768≤32768 by omega)
    dsimp [t]
    constructor
    · positivity
    · linarith
  have hp := SinePolynomialBounds.cell_ideal_error j t ht
  have hg : j.val/4=index/131072 := by dsimp [j];omega
  have hk : (j.val%4)*32768+index%32768=index%131072 := by dsimp [j];omega
  have hindex : j.val*32768+index%32768=index := by
    dsimp only [j]
    simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div index 32768
  have hindexr : (j.val:ℝ)*32768+((index%32768:Nat):ℝ)=index := by exact_mod_cast hindex
  have hangle : Real.pi*(j.val+t)/512=Real.pi*index/2^24 := by
    dsimp [t]
    have hmul := congrArg (fun x:ℝ => Real.pi*x) hindexr
    nlinarith only [hmul]
  rw [hangle] at hp
  rcases hcoeff : MufuTables.sin.coefficients[index/131072]! with ⟨a,b,c⟩
  let k : Int := (index%131072:Nat)
  let n : Int := a*2^31+b*k*2^18+c*k*k
  have hpoly : SinePolynomialBounds.nativePoly j.val t=(n:ℝ)/2^56 := by
    unfold SinePolynomialBounds.nativePoly SineCells.nativeCoefficients
    rw [hg,hcoeff]
    dsimp only
    push_cast
    have hkr : ((j.val%4:Nat):ℝ)*32768+((index%32768:Nat):ℝ)=(index%131072:Nat) := by exact_mod_cast hk
    dsimp only [t,n,k]
    simp only [Int.cast_add,Int.cast_mul,Int.cast_pow,Int.cast_ofNat,Int.cast_natCast]
    rw [←hkr]
    ring
  rw [hpoly] at hp
  have h := SineInterpolationInteger.interpolation_facts index hi
  rw [hcoeff] at h
  dsimp only at h
  have hlo : -(2^19:ℝ) ≤ (Mufu.interpolate MufuTables.sin index:ℝ)*2^19-(n:ℝ) := by exact_mod_cast h.2.2.1
  have hhi : (Mufu.interpolate MufuTables.sin index:ℝ)*2^19-(n:ℝ) ≤ (1265*4718593:ℝ) := by exact_mod_cast h.2.2.2
  have herror : |(Mufu.interpolate MufuTables.sin index:ℝ)/2^37-(n:ℝ)/2^56| ≤ 1/10000000 := by
    apply abs_le.mpr
    constructor <;> nlinarith only [hlo,hhi]
  have ht := abs_sub_le ((Mufu.interpolate MufuTables.sin index:ℝ)/2^37) ((n:ℝ)/2^56) (Real.sin (Real.pi*index/2^24))
  linarith

def magnitudeWord (index : Nat) : Nat :=
  let l := (Mufu.interpolate MufuTables.sin index).toNat
  if l=0 then 0 else Mufu.fixedToWord l 37

theorem decoded_table (index : Nat) (hi : index<2^23) :
    ∃ y : Rat,
      (Model.ofNatBits (magnitudeWord index) : Model FloatFormat.binary32).toRat?=some y ∧
      |(y:ℝ)-Real.sin (Real.pi*index/2^24)| ≤ 401/1000000000+1/2^23 := by
  have h := SineInterpolationInteger.interpolation_facts index hi
  dsimp only at h
  have hz : ((Mufu.interpolate MufuTables.sin index).toNat:Int)=Mufu.interpolate MufuTables.sin index := Int.toNat_of_nonneg h.1
  have hzr : ((Mufu.interpolate MufuTables.sin index).toNat:ℝ)=(Mufu.interpolate MufuTables.sin index:ℝ) := by exact_mod_cast hz
  have he := table_error index hi
  rw [←hzr] at he
  unfold magnitudeWord
  by_cases zero : (Mufu.interpolate MufuTables.sin index).toNat=0
  · rw [ite_eq_left zero]
    refine ⟨0,by decide +kernel,?_⟩
    rw [zero] at he
    norm_num at he ⊢
    linarith
  · rw [ite_eq_right zero]
    obtain ⟨y,hy,hylo,hyhi⟩ := SineFixedDecode.fixed_error (Mufu.interpolate MufuTables.sin index).toNat (by omega)
    refine ⟨y,hy,?_⟩
    have hylo' : 0 ≤ ((Mufu.interpolate MufuTables.sin index).toNat:ℝ)/2^37-(y:ℝ) := by simpa using (Rat.cast_nonneg (K:=ℝ)).mpr hylo
    have hyhi' : ((Mufu.interpolate MufuTables.sin index).toNat:ℝ)/2^37-(y:ℝ) ≤ 1/2^23 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hyhi
    have hb := abs_le.mp he
    apply abs_le.mpr
    constructor <;> linarith
#print axioms table_error
#print axioms decoded_table
end GB300Math.SineTableBounds
