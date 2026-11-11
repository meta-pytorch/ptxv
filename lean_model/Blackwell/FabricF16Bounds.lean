/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF16
import FloatLib.Floats.Formats.Flocq.Theory.Error.Relative
import Mathlib.Tactic

namespace FabricF16Bounds
open FloatLib.Floats.Formats.BinaryInterchange FloatLib.Floats.Formats.Flocq

theorem normalize_zero_value (value : Model FloatFormat.binary16) :
    (Model.ofNatBits (if value.toNatBits%32768=0 then 0 else value.toNatBits) :
      Model FloatFormat.binary16).toReal=value.toReal := by
  by_cases h : value.toNatBits%32768=0
  · have hb : value.toNatBits<65536 := Model.toNatBits_lt_two_pow value
    have hz : value.toNatBits=0 ∨ value.toNatBits=32768 := by omega
    have hv : value=(Model.ofNatBits value.toNatBits : Model FloatFormat.binary16) := (Model.ofNatBits_toNatBits value).symm
    rw [ite_eq_left h,hv]
    rcases hz with hz|hz
    · rw [hz]
    · rw [hz]
      change (Model.zero FloatFormat.binary16 false).toReal=(Model.zero FloatFormat.binary16 true).toReal
      simp
  · simp [h]

theorem finite_predict (words : List Nat) (hf : ∀w∈words,w%32768<31744) :
    FabricF16.predict words =
      let result:=Model.sum FloatFormat.binary16 (words.toArray.map fun w=>(Model.ofNatBits w : Model FloatFormat.binary16)) .nearestEven
      if result.toNatBits%32768=0 then 0 else result.toNatBits := by
  have hn : words.filter (fun w=>decide (w%32768>31744))=[] := by
    simp only [List.filter_eq_nil_iff,Bool.not_eq_true,decide_eq_false_iff_not]
    intro w hw;have:=hf w hw;omega
  have hi : words.filter (fun w=>decide (w%32768=31744))=[] := by
    simp only [List.filter_eq_nil_iff,Bool.not_eq_true,decide_eq_false_iff_not]
    intro w hw;have:=hf w hw;omega
  simp [FabricF16.predict,hn,hi]

theorem predict_value (words : List Nat) (hf : ∀w∈words,w%32768<31744) :
    (Model.ofNatBits (FabricF16.predict words) : Model FloatFormat.binary16).toReal=
      (Model.sum FloatFormat.binary16 (words.toArray.map fun w=>(Model.ofNatBits w : Model FloatFormat.binary16)) .nearestEven).toReal := by
  rw [finite_predict words hf]
  exact normalize_zero_value _

theorem normalize_zero_finite (value : Model FloatFormat.binary16) :
    (Model.ofNatBits (if value.toNatBits%32768=0 then 0 else value.toNatBits) :
      Model FloatFormat.binary16).isFinite=value.isFinite := by
  by_cases h : value.toNatBits%32768=0
  · have hb : value.toNatBits<65536 := Model.toNatBits_lt_two_pow value
    have hz : value.toNatBits=0 ∨ value.toNatBits=32768 := by omega
    have hv : value=(Model.ofNatBits value.toNatBits : Model FloatFormat.binary16) := (Model.ofNatBits_toNatBits value).symm
    rw [ite_eq_left h,hv]
    rcases hz with hz|hz <;> rw [hz]
    decide +kernel
  · simp [h]

theorem exp_word (word : Nat) :
    (Model.ofNatBits word : Model FloatFormat.binary16).expField=word%32768/1024 := by
  rw [Model.expField_eq_expFieldImpl_apply]
  change ((word%65536)>>>10)&&&(2^5-1)=word%32768/1024
  rw [Nat.and_two_pow_sub_one_eq_mod,Nat.shiftRight_eq_div_pow]
  omega

theorem finite_raw (word : Nat) (hf : word%32768<31744) :
    (Model.ofNatBits word : Model FloatFormat.binary16).isFinite=true := by
  change ((Model.ofNatBits word : Model FloatFormat.binary16).expField != 31)=true
  rw [exp_word]
  simp only [bne_iff_ne]
  omega

theorem exact_decoded_sum (values : List (Model FloatFormat.binary16)) :
    (Model.Reduction.Semantics.finiteSum values:ℝ)=(values.map Model.toReal).sum := by
  have hc : ∀v:Model FloatFormat.binary16,
      (Model.Reduction.Semantics.finiteContribution v:ℝ)=v.toReal := by
    intro v
    rw [Model.toReal_eq]
    unfold Model.Reduction.Semantics.finiteContribution
    cases v.toDyadic? <;> simp [FloatLib.Numerics.Dyadic.cast_toRat]
  induction values with
  | nil => simp [Model.Reduction.Semantics.finiteSum]
  | cons v vs ih =>
      simpa [Model.Reduction.Semantics.finiteSum,Rat.cast_add,hc] using congrArg (fun x=>v.toReal+x) ih

theorem round_normal_error (x:ℝ) (hn : (2:ℝ)^(-14:Int)≤|x|) :
    |Model.roundAt FloatFormat.binary16 x-x|≤|x|/2048 := by
  have hx : x≠0 := by intro h;norm_num [h] at hn
  have h:=relative_error_round_FLT_normal (β:=FloatLib.Numerics.binaryRadix)
    (-24) 11 (by norm_num) nearestEven x hx (by
      convert hn using 1
      norm_num [bpow]
      change ((2:ℝ)^14)⁻¹=1/16384
      norm_num)
  have hp : bpow FloatLib.Numerics.binaryRadix (1-11)=(1:ℝ)/1024 := by
    change (2:ℝ)^(-10:Int)=1/1024
    norm_num
  rw [hp] at h
  change |_-x|/|x|≤1/1024/2 at h
  have he:=(div_le_iff₀ (abs_pos.mpr hx)).mp h
  convert he using 1 <;> first | rfl | ring

theorem exact_sum_normal_error (values : Array (Model FloatFormat.binary16))
    (hf : ∀v∈values.toList,Model.isFinite v=true)
    (hout : Model.isFinite (Model.sum FloatFormat.binary16 values .nearestEven)=true)
    (hn : (2:ℝ)^(-14:Int)≤|(Model.Reduction.Semantics.finiteSum values.toList:ℝ)|) :
    |(Model.sum FloatFormat.binary16 values .nearestEven).toReal-
      (Model.Reduction.Semantics.finiteSum values.toList:ℝ)|≤
      |(Model.Reduction.Semantics.finiteSum values.toList:ℝ)|/2048 := by
  have hz : Model.Reduction.Semantics.finiteSum values.toList≠0 := by
    intro h;norm_num [h] at hn
  have h:=Model.Reduction.sumWithStatus_value_toReal_eq_roundAt_of_finite_nonzero
    FloatFormat.binary16 values (by decide) hf hz hout
  change (Model.sum FloatFormat.binary16 values .nearestEven).toReal=_ at h
  rw [h]
  exact round_normal_error _ hn

theorem decoded_normal_error (words : List Nat)
    (hf : ∀w∈words,w%32768<31744)
    (hout : (Model.ofNatBits (FabricF16.predict words) : Model FloatFormat.binary16).isFinite=true)
    (hn : (2:ℝ)^(-14:Int)≤|(words.map fun w=>(Model.ofNatBits w : Model FloatFormat.binary16).toReal).sum|) :
    |(Model.ofNatBits (FabricF16.predict words) : Model FloatFormat.binary16).toReal-
      (words.map fun w=>(Model.ofNatBits w : Model FloatFormat.binary16).toReal).sum|≤
      |(words.map fun w=>(Model.ofNatBits w : Model FloatFormat.binary16).toReal).sum|/2048 := by
  let values:=words.toArray.map fun w=>(Model.ofNatBits w : Model FloatFormat.binary16)
  have hv : ∀v∈values.toList,v.isFinite=true := by
    intro v hm
    obtain ⟨w,hw,rfl⟩ := List.mem_map.mp (by simpa [values] using hm)
    exact finite_raw w (hf w hw)
  have he : (Model.Reduction.Semantics.finiteSum values.toList:ℝ)=
      (words.map fun w=>(Model.ofNatBits w : Model FloatFormat.binary16).toReal).sum := by
    rw [exact_decoded_sum]
    simp [values,List.map_map,Function.comp_def]
  have ho : (Model.sum FloatFormat.binary16 values .nearestEven).isFinite=true := by
    rw [finite_predict words hf] at hout
    exact (normalize_zero_finite _).symm.trans hout
  rw [predict_value words hf]
  have h:=exact_sum_normal_error values hv ho (by simpa [he] using hn)
  rw [he] at h
  exact h


#print axioms normalize_zero_value
#print axioms finite_predict
#print axioms predict_value
#print axioms round_normal_error
#print axioms exact_sum_normal_error
#print axioms normalize_zero_finite
#print axioms exp_word
#print axioms finite_raw
#print axioms exact_decoded_sum
#print axioms decoded_normal_error
end FabricF16Bounds
