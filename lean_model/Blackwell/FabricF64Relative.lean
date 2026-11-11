/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF64Bounds
import FloatLib.Floats.Formats.Flocq.Theory.Error.Relative
import FloatLib.Floats.Formats.Flocq.Theory.Format.Theorems
namespace FabricF64Relative
open FabricF64 FabricF64Bounds FloatLib.Floats.Formats.BinaryInterchange FloatLib.Floats.Formats.Flocq
set_option maxHeartbeats 2000000
set_option maxRecDepth 100000
set_option exponentiation.threshold 2048

theorem round_small (value:Int) (hsmall:value.natAbs<2^53)
    (finite:(Model.ofNatBits (FabricF64.round value):Model FloatFormat.binary64).isFinite=true) :
    (Model.ofNatBits (FabricF64.round value):Model FloatFormat.binary64).toReal=(value:ℝ)/2^1074 := by
  have hformat : FLTFormat (β:=FloatLib.Numerics.binaryRadix) (-1074) 53 ((value:ℝ)/2^1074) := by
    refine ⟨by norm_num,⟨value,-1074⟩,?_,?_,by norm_num⟩
    · change (value:ℝ)/2^1074=(value:ℝ)*(2:ℝ)^(-1074:Int)
      norm_num [zpow_neg,div_eq_mul_inv]
    · exact hsmall
  have hg:=generic_of_fltFormat (β:=FloatLib.Numerics.binaryRadix) (-1074) 53 (by norm_num) _ hformat
  rw [round_word] at finite ⊢
  rw [Model.toReal_roundDyadic_eq_roundAt _ (by decide) _ finite,dyadic_value]
  exact round_preserves_generic nearestEven _ hg

theorem round_relative (value:Int)
    (finite:(Model.ofNatBits (FabricF64.round value):Model FloatFormat.binary64).isFinite=true) :
    |(Model.ofNatBits (FabricF64.round value):Model FloatFormat.binary64).toReal-(value:ℝ)/2^1074| ≤
      |(value:ℝ)/2^1074|/2^53 := by
  by_cases hs:value.natAbs<2^53
  · rw [round_small value hs finite,sub_self,abs_zero]
    positivity
  · have hn : (2:ℝ)^(-1022:Int) ≤ |(value:ℝ)/2^1074| := by
      have hv:(2^53:ℝ) ≤ |(value:ℝ)| := by
        have h: (2^53:ℝ) ≤ (value.natAbs:ℝ) := by exact_mod_cast (show 2^53 ≤ value.natAbs by omega)
        simpa using h
      rw [abs_div,abs_of_pos (by positivity:(0:ℝ)<2^1074)]
      norm_num [zpow_neg] at *
      linarith
    have hx:(value:ℝ)/2^1074≠0 := by
      have hv:value≠0 := by intro h;simp [h] at hs
      exact div_ne_zero (by exact_mod_cast hv) (by positivity)
    have hr:=relative_error_round_FLT_normal (β:=FloatLib.Numerics.binaryRadix) (-1074) 53 (by norm_num)
      nearestEven ((value:ℝ)/2^1074) hx hn
    rw [round_word] at finite ⊢
    rw [Model.toReal_roundDyadic_eq_roundAt _ (by decide) _ finite,dyadic_value]
    have hb:bpow FloatLib.Numerics.binaryRadix (1-53)=(1:ℝ)/2^52 := by
      change (2:ℝ)^(-52:Int)=1/2^52
      norm_num
    rw [hb] at hr
    change |_-_|/|(value:ℝ)/2^1074| ≤ 1/2^52/2 at hr
    have he:=(div_le_iff₀ (abs_pos.mpr hx)).mp hr
    convert he using 1 <;> first | rfl | ring

def alignmentLoss (words:List Nat) := words.length*(2^fineShift words-1)

theorem coarse_dominates (words:List Nat) (hn:words.length ≤ 4) :
    128*alignmentLoss words<2^coarseShift words := by
  by_cases hz:fineShift words=0
  · simp [alignmentLoss,hz]
  · have hs:coarseShift words=fineShift words+9 := by
      dsimp [coarseShift,fineShift] at *
      omega
    rw [hs,Nat.pow_add]
    have hp:=Nat.two_pow_pos (fineShift words)
    dsimp [alignmentLoss]
    have hsub:2^fineShift words-1+1=2^fineShift words := by omega
    nlinarith

theorem unflushed_relative (words:List Nat) (hn:words.length ≤ 4) (hz:accumulator words≠0) :
    128*(accumulator words-(words.map fun w=>(units w).getD 0).sum).natAbs ≤ (accumulator words).natAbs := by
  have hc:=coarse_dominates words hn
  have hl:=sum_loss (fineShift words) (words.map fun w=>(units w).getD 0)
  have hnf:¬(finiteSum words).natAbs<2^coarseShift words := by
    intro h
    exact hz (by simp [accumulator,flush,h])
  have he:accumulator words=finiteSum words := by simp [accumulator,flush,hnf]
  rw [he]
  have hb:(finiteSum words-(words.map fun w=>(units w).getD 0).sum).natAbs ≤ alignmentLoss words := by
    simpa [finiteSum,List.map_map,Function.comp_def,alignmentLoss] using hl
  omega

theorem unflushed_real (words:List Nat) (hn:words.length ≤ 4) (hz:accumulator words≠0) :
    128*|(accumulator words:ℝ)/2^1074-(((words.map fun w=>(units w).getD 0).sum:Int):ℝ)/2^1074| ≤
      |(accumulator words:ℝ)/2^1074| := by
  have h:=unflushed_relative words hn hz
  have hr:(128:ℝ)*|((accumulator words-(words.map fun w=>(units w).getD 0).sum:Int):ℝ)| ≤ |(accumulator words:ℝ)| := by
    have hh:(128:ℝ)*(accumulator words-(words.map fun w=>(units w).getD 0).sum).natAbs ≤ ((accumulator words).natAbs:ℝ) := by exact_mod_cast h
    simpa using hh
  rw [Int.cast_sub] at hr
  rw [←sub_div,abs_div,abs_div,abs_of_pos (by positivity:(0:ℝ)<2^1074)]
  rw [←mul_div_assoc,div_le_div_iff_of_pos_right (by positivity)]
  exact hr

theorem decoded_relative (words:List Nat)
    (valid:∀w∈words,w<2^64 ∧ w%2^63<2047*2^52) (hn:words.length ≤ 4)
    (finite:(Model.ofNatBits (predict words):Model FloatFormat.binary64).isFinite=true) :
    let y:=(Model.ofNatBits (predict words):Model FloatFormat.binary64).toReal
    let s:=(words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary64).toReal).sum
    |y-s| ≤ |s| ∧ (y≠0 → (127*2^46:ℝ)*|y-s| ≤ (2^46+1)*|s|) := by
  let a:ℝ:=(accumulator words:ℝ)/2^1074
  let s:ℝ:=(((words.map fun w=>(units w).getD 0).sum:Int):ℝ)/2^1074
  let y:=(Model.ofNatBits (predict words):Model FloatFormat.binary64).toReal
  have hu:∀w∈words,∃v,units w=some v := by
    intro w hw
    exact FabricF64Decode.units_defined w (valid w hw).1 (finite_raw w (valid w hw).2)
  dsimp only
  rw [FabricF64Decode.sum_units_value words hu]
  change |y-s| ≤ |s| ∧ (y≠0 → (127*2^46:ℝ)*|y-s| ≤ (2^46+1)*|s|)
  have ho:=predict_finite words (fun w hw=>(valid w hw).2)
  have hfinite:(Model.ofNatBits (FabricF64.round (accumulator words)):Model FloatFormat.binary64).isFinite=true := by
    simpa only [ho] using finite
  have hr:|y-a| ≤ |a|/2^53 := by
    dsimp [y,a]
    rw [ho]
    exact round_relative (accumulator words) hfinite
  by_cases hz:accumulator words=0
  · have ha:a=0 := by simp [a,hz]
    have hy:y=0 := by rw [ha] at hr;simpa using hr
    simp [hy]
  · have ha:128*|a-s| ≤ |a| := unflushed_real words hn hz
    have habs:=abs_sub_le a s 0
    have ht:=abs_sub_le y a s
    have hnonneg:=abs_nonneg s
    simp only [sub_zero] at habs
    norm_num at hr ⊢
    constructor
    · nlinarith
    · intro _
      nlinarith
#print axioms round_small
#print axioms round_relative
#print axioms coarse_dominates
#print axioms unflushed_relative
#print axioms unflushed_real
#print axioms decoded_relative
end FabricF64Relative
