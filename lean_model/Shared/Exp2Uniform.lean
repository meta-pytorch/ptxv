/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Exp2Argument
import Exp2Relative
namespace GB300Math.Exp2Uniform
open FloatLib.Floats.Formats.BinaryInterchange
open Exp2Boundary
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

theorem split_argument (q : Int) :
    (q:ℝ)/2^23=((q/2^23:Int):ℝ)+(((q%2^23).toNat:Nat):ℝ)/2^23 := by
  have hrem : 0≤q%2^23 := by omega
  have hi : q%2^23+2^23*(q/2^23)=q := by omega
  have hr : ((q%2^23:Int):ℝ)+2^23*((q/2^23:Int):ℝ)=(q:ℝ) := by exact_mod_cast hi
  have hc : (((q%2^23).toNat:Nat):ℝ)=((q%2^23:Int):ℝ) := by exact_mod_cast Int.toNat_of_nonneg hrem
  rw [hc]
  linarith

theorem ideal_factor (q : Int) (x : ℝ) :
    (2:ℝ)^x=(2:ℝ)^(q/2^23)*
      Real.exp (Real.log 2*((((q%2^23).toNat:Nat):ℝ)/2^23))*
      Real.exp (Real.log 2*(x-(q:ℝ)/2^23)) := by
  have hq:=split_argument q
  have he : Real.log 2*x=Real.log 2*((q/2^23:Int):ℝ)+
    Real.log 2*((((q%2^23).toNat:Nat):ℝ)/2^23)+Real.log 2*(x-(q:ℝ)/2^23) := by rw [hq];ring
  have hp : Real.exp (Real.log 2*((q/2^23:Int):ℝ))=(2:ℝ)^(q/2^23) := by
    have h:=Real.rpow_def_of_pos (by norm_num : (0:ℝ)<2) (((q/2^23:Int):ℝ))
    rw [Real.rpow_intCast] at h
    exact h.symm
  rw [Real.rpow_def_of_pos (by norm_num),he,Real.exp_add,Real.exp_add,hp]

theorem finite_output_error (word : Nat) (hw : word<2^32)
    (hf : Mufu.magnitude word<0x7f800000)
    (ho : 0<Mufu.ex2 word ∧ Mufu.ex2 word<0x7f800000) :
    ∃ x y:Rat,(Model.ofNatBits word:Model FloatFormat.binary32).toRat?=some x ∧
      (Model.ofNatBits (Mufu.ex2 word):Model FloatFormat.binary32).toRat?=some y ∧
      |(y:ℝ)-(2:ℝ)^(x:ℝ)|/((2:ℝ)^(x:ℝ))≤1/2^22 := by
  have he : Mufu.exponent word≤134 := by
    by_contra h
    have hh : 135≤Mufu.exponent word := by omega
    simp only [Mufu.ex2,not_lt.mpr hf.le,ite_false,hh,ite_true] at ho
    split_ifs at ho <;> omega
  let q:=signedFixed word
  let index:Nat:=(q%2^23).toNat
  have hi : index<2^23 := by dsimp [index];omega
  have hz:=Exp2Interpolation.interpolation_facts index hi
  have hm : 2^23≤(Mufu.interpolate MufuTables.ex2 index).toNat ∧
      (Mufu.interpolate MufuTables.ex2 index).toNat<2^24 := by
    exact ⟨by omega,by omega⟩
  have hpack:=ex2_pack word he hf
  have hbe : 1≤outputExponent word ∧ outputExponent word≤254 := by
    have hzero : ¬outputExponent word≤0 := by
      intro h
      have hz':= (pack_zero (outputExponent word) (Mufu.interpolate MufuTables.ex2 index).toNat).mpr h
      rw [←hpack] at hz'
      omega
    have hinf : ¬255≤outputExponent word := by
      intro h
      have hi':=(pack_infinite (outputExponent word) (Mufu.interpolate MufuTables.ex2 index).toNat).mpr h
      rw [←hpack] at hi'
      omega
    omega
  obtain ⟨x,hx,hxv⟩:=Exp2Decode.finite word hw hf
  obtain ⟨y,hy,hyv⟩:=Exp2Decode.packed (outputExponent word)
    (Mufu.interpolate MufuTables.ex2 index).toNat hbe hm
  have hcast : ((Mufu.interpolate MufuTables.ex2 index).toNat:ℝ)=
    (Mufu.interpolate MufuTables.ex2 index:ℝ) := by
      exact_mod_cast Int.toNat_of_nonneg (show 0≤Mufu.interpolate MufuTables.ex2 index by omega)
  have heq : outputExponent word-127=q/2^23 := by dsimp [outputExponent,q];omega
  rw [hcast,heq] at hyv
  have hout : (Model.ofNatBits (Mufu.ex2 word):Model FloatFormat.binary32).toRat?=some y := by rw [hpack];exact hy
  refine ⟨x,y,hx,hout,?_⟩
  have hd : |(x:ℝ)-(q:ℝ)/2^23|≤1/2^23 := by
    rw [hxv,abs_sub_comm]
    exact Exp2Argument.argument_error word hw
  have hdir : 0≤(x:ℝ)-(q:ℝ)/2^23 ∨ index=0 := by
    by_cases hq : q=0
    · right;simp [index,hq]
    · left
      rw [hxv]
      exact sub_nonneg.mpr (Exp2Argument.nonzero_floor word hw hq)
  have hn:=Exp2Relative.normalized_exp_tight index hi ((x:ℝ)-(q:ℝ)/2^23) hd hdir
  have hs : 0<(2:ℝ)^(q/2^23) := zpow_pos (by norm_num) _
  have hscaled:=mul_le_mul_of_nonneg_left hn hs.le
  have hideal:=ideal_factor q (x:ℝ)
  change (2:ℝ)^(x:ℝ)=(2:ℝ)^(q/2^23)*Real.exp (Real.log 2*((index:ℝ)/2^23))*Real.exp (Real.log 2*((x:ℝ)-(q:ℝ)/2^23)) at hideal
  have hiabs : |(y:ℝ)-(2:ℝ)^(x:ℝ)|=(2:ℝ)^(q/2^23)*
      |(Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
       Real.exp (Real.log 2*((index:ℝ)/2^23))*Real.exp (Real.log 2*((x:ℝ)-(q:ℝ)/2^23))| := by
    rw [hyv,hideal]
    rw [show (Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23*(2:ℝ)^(q/2^23)-
      (2:ℝ)^(q/2^23)*Real.exp (Real.log 2*((index:ℝ)/2^23))*Real.exp (Real.log 2*((x:ℝ)-(q:ℝ)/2^23))=
      (2:ℝ)^(q/2^23)*((Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
       Real.exp (Real.log 2*((index:ℝ)/2^23))*Real.exp (Real.log 2*((x:ℝ)-(q:ℝ)/2^23))) by ring]
    rw [abs_mul,abs_of_pos hs]
  apply (div_le_iff₀ (Real.rpow_pos_of_pos (by norm_num) _)).mpr
  rw [hiabs,hideal]
  nlinarith only [hscaled]

#print axioms split_argument
#print axioms ideal_factor
#print axioms finite_output_error
end GB300Math.Exp2Uniform
