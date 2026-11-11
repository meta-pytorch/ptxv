/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SineModelBridge
import SineArgumentError
namespace GB300Math.SineUniform
open FloatLib.Floats.Formats.BinaryInterchange
open SineModelBridge
set_option maxRecDepth 12000
set_option maxHeartbeats 6000000

theorem total_error (y a b : ℝ) (hya : |y-a| ≤ 71/100000000)
    (hab : |a-b| ≤ Real.pi/2^24) : |y-b| ≤ 1/2^20 := by
  have h := abs_sub_le y a b
  have hp := Real.pi_lt_d4
  nlinarith only [h,hya,hab,hp]

theorem finite_parameter_bound (negative : Bool) (e : Fin 255) (f : Fin (2^23)) :
    ∃ x ys yc : Rat,
      (Model.ofNatBits (word negative e f) : Model FloatFormat.binary32).toRat?=some x ∧
      (Model.ofNatBits (Mufu.sincos (word negative e f) false) : Model FloatFormat.binary32).toRat?=some ys ∧
      (Model.ofNatBits (Mufu.sincos (word negative e f) true) : Model FloatFormat.binary32).toRat?=some yc ∧
      |(ys:ℝ)-Real.sin (2*Real.pi*(x:ℝ))| ≤ 1/2^20 ∧
      |(yc:ℝ)-Real.cos (2*Real.pi*(x:ℝ))| ≤ 1/2^20 := by
  obtain ⟨x,r,n,hx,hrelation,hrlo,hrhi⟩ := positive_reduction e f
  have hq : phase e f<2^25 := by unfold phase SineArgument.rawPhase;exact Nat.mod_lt _ (by decide)
  have hqc : (phase e f+2^23)%2^25<2^25 := Nat.mod_lt _ (by decide)
  obtain ⟨ys,hys,hserror⟩ := SinePhaseResult.decoded_phase (phase e f) hq negative
  obtain ⟨yc,hyc,hcerror⟩ := SinePhaseResult.decoded_phase ((phase e f+2^23)%2^25) hqc false
  simp only [Bool.false_eq_true,ite_false] at hcerror
  rw [SineArgumentError.cosine_phase (phase e f) hq] at hcerror
  have hrelationr : (x:ℝ)=(n:ℝ)+(phase e f:ℝ)/2^25+(r:ℝ) := by
    have h := congrArg (fun z : Rat => (z : ℝ)) hrelation
    simpa using h
  have hrr : 0≤(r:ℝ) ∧ (r:ℝ)≤1/2^25 := by
    constructor
    · exact (Rat.cast_nonneg (K:=ℝ)).mpr hrlo
    · simpa using (Rat.cast_le (K:=ℝ)).mpr hrhi
  have he := SineArgumentError.reduced_error (x:ℝ) r n (phase e f) hrelationr hrr
  have shape := kernel_shape negative e f
  have houtS : (Model.ofNatBits (Mufu.sincos (word negative e f) false) : Model FloatFormat.binary32).toRat?=some ys := by rw [shape.1];exact hys
  have houtC : (Model.ofNatBits (Mufu.sincos (word negative e f) true) : Model FloatFormat.binary32).toRat?=some yc := by rw [shape.2];exact hyc
  cases negative
  · refine ⟨x,ys,yc,hx,houtS,houtC,?_,?_⟩
    · simp only [Bool.false_eq_true,ite_false] at hserror
      exact total_error _ _ _ hserror (by simpa only [abs_sub_comm] using he.1)
    · exact total_error _ _ _ hcerror (by simpa only [abs_sub_comm] using he.2)
  · have hin := RcpSigned.signed_decode (word false e f) x hx
    have hin' : (Model.ofNatBits (word true e f) : Model FloatFormat.binary32).toRat?=some (-x) := by
      simpa [word,Nat.add_comm] using hin
    refine ⟨-x,ys,yc,hin',houtS,houtC,?_,?_⟩
    · simp only [ite_true] at hserror
      have hneg : |(-Real.sin (Real.pi*phase e f/2^24))-Real.sin (2*Real.pi*(-(x:ℝ)))| ≤ Real.pi/2^24 := by
        rw [show 2*Real.pi*(-(x:ℝ))=-(2*Real.pi*(x:ℝ)) by ring,Real.sin_neg,neg_sub_neg]
        exact he.1
      simpa using total_error _ _ _ hserror hneg
    · have hneg : |Real.cos (Real.pi*phase e f/2^24)-Real.cos (2*Real.pi*(-(x:ℝ)))| ≤ Real.pi/2^24 := by
        rw [show 2*Real.pi*(-(x:ℝ))=-(2*Real.pi*(x:ℝ)) by ring,Real.cos_neg,abs_sub_comm]
        exact he.2
      simpa using total_error _ _ _ hcerror hneg

theorem raw_finite_bound (w : Nat) (hw : w<2^32) (he : Mufu.exponent w<255) :
    ∃ x ys yc : Rat,
      (Model.ofNatBits w : Model FloatFormat.binary32).toRat?=some x ∧
      (Model.ofNatBits (Mufu.sincos w false) : Model FloatFormat.binary32).toRat?=some ys ∧
      (Model.ofNatBits (Mufu.sincos w true) : Model FloatFormat.binary32).toRat?=some yc ∧
      |(ys:ℝ)-Real.sin (2*Real.pi*(x:ℝ))| ≤ 1/2^20 ∧
      |(yc:ℝ)-Real.cos (2*Real.pi*(x:ℝ))| ≤ 1/2^20 := by
  let negative := decide (2^31≤w)
  let e : Fin 255 := ⟨Mufu.exponent w,he⟩
  let f : Fin (2^23) := ⟨Mufu.fraction w,by unfold Mufu.fraction;omega⟩
  have hsum : (if 2^31≤w then 2^31 else 0)+((w/2^23%256)*2^23+w%2^23)=w := by
    norm_num only [Nat.reducePow]
    split_ifs <;> omega
  have hword : word negative e f=w := by
    simpa [word,negative,e,f,Mufu.exponent,Mufu.fraction] using hsum
  have h := finite_parameter_bound negative e f
  rwa [hword] at h

theorem nonfinite_bits (w : Nat) (he : Mufu.exponent w=255) :
    Mufu.sincos w false=0x7fffffff ∧ Mufu.sincos w true=0x7fffffff := by
  simp [Mufu.sincos,he,Mufu.canonicalNaN]

#print axioms raw_finite_bound
#print axioms nonfinite_bits
theorem finite_count : 2*255*2^23=4278190080 := by decide +kernel
#print axioms total_error
#print axioms finite_parameter_bound
#print axioms finite_count
end GB300Math.SineUniform
