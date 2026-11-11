/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF16StochasticBounds
import FloatLib.Floats.Formats.Flocq.Theory.Error.Bounds
namespace FabricF16V2Bounds
open FabricF16V2
open FloatLib.Floats.Formats.BinaryInterchange FloatLib.Floats.Formats.Flocq

def asDyadic (value : Int) : FloatLib.Numerics.Dyadic :=
  {negative := value < 0, significand := value.natAbs, exponent := -24}

theorem dyadic_value (value : Int) : (asDyadic value).toReal = (value : ℝ)/(2^24 : ℝ) := by
  cases value <;>
    simp [asDyadic, FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand,
      zpow_neg, div_eq_mul_inv]

theorem round_word (value : Int) :
    (Model.ofNatBits (FabricF16V2.round value) : Model FloatFormat.binary16) =
      Model.roundDyadic FloatFormat.binary16 (asDyadic value) := by
  unfold FabricF16V2.round
  exact Model.ofNatBits_toNatBits _

theorem round_error (value : Int)
    (finite : Model.isFinite (Model.ofNatBits (FabricF16V2.round value) : Model FloatFormat.binary16) = true) :
    |Model.toReal (Model.ofNatBits (FabricF16V2.round value) : Model FloatFormat.binary16) -
        (value : ℝ)/(2^24 : ℝ)| ≤
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.binary16)
        ((value : ℝ)/(2^24 : ℝ))/2 := by
  rw [round_word] at finite ⊢
  rw [Model.toReal_roundDyadic_eq_roundAt _ (by decide) _ finite, dyadic_value]
  exact error_bound_ulp (rnd := nearestEven) _

theorem accumulate_finite (words : List Nat)
    (finite_inputs : ∀ w ∈ words, w % 32768 < 31744) :
    FabricF16V2.accumulate words = .finite
      ((words.map fun w => (FabricF16V2.units w).getD 0).sum) := by
  have hn : ∀ w ∈ words, ¬ 31744 < w % 32768 := by
    intro w hw
    have := finite_inputs w hw
    omega
  have hi : ∀ w ∈ words, ¬ w % 32768 = 31744 := by
    intro w hw
    have := finite_inputs w hw
    omega
  have hnf : words.filter (fun w => decide (w % 32768 > 31744)) = [] := by
    simp only [List.filter_eq_nil_iff, Bool.not_eq_true, decide_eq_false_iff_not]
    exact hn
  have hif : words.filter (fun w => decide (w % 32768 = 31744)) = [] := by
    simp only [List.filter_eq_nil_iff, Bool.not_eq_true, decide_eq_false_iff_not]
    exact hi
  simp [FabricF16V2.accumulate, hnf, hif]

theorem round_relative (value:Int)
    (hf:(Model.ofNatBits (FabricF16V2.round value):Model FloatFormat.binary16).isFinite=true) :
    |(Model.ofNatBits (FabricF16V2.round value):Model FloatFormat.binary16).toReal-(value:ℝ)/2^24|≤
      |(value:ℝ)/2^24|/2048 := by
  rw [round_word] at hf ⊢
  rw [Model.toReal_roundDyadic_eq_roundAt _ (by decide) _ hf,dyadic_value]
  by_cases hn : (2:ℝ)^(-14:Int)≤|(value:ℝ)/2^24|
  · exact FabricF16Bounds.round_normal_error _ hn
  · have hs : value.natAbs<1024 := by
      rw [abs_div,abs_of_pos (by positivity : (0:ℝ)<2^24)] at hn
      have hv : |(value:ℝ)|<(1024:ℝ) := by norm_num at hn;linarith
      have hv' : (value.natAbs:ℝ)<1024 := by simpa using hv
      exact_mod_cast hv'
    rw [FabricF16ExactBounds.small_round_exact value hs]
    simp only [sub_self,abs_zero]
    positivity

theorem round_zero : FabricF16V2.round 0=0 := by decide +kernel

theorem decoded_zero : (Model.ofNatBits 0:Model FloatFormat.binary16).toReal=0 := by
  change (Model.zero FloatFormat.binary16 false).toReal=0
  simp

theorem output_round_error (states:Array Accum) (lane:Nat) (value:Int) (sr:Bool)
    (hv:states[lane]! =.finite value)
    (hf:(Model.ofNatBits (output states lane sr):Model FloatFormat.binary16).isFinite=true) :
    |(Model.ofNatBits (output states lane sr):Model FloatFormat.binary16).toReal-(value:ℝ)/2^24|≤
      |(value:ℝ)/2^24|/(if sr then 1024 else 2048) := by
  cases sr
  · simp only [output,hv,Bool.not_false,↓reduceIte] at hf ⊢
    exact round_relative value hf
  · exact FabricF16StochasticBounds.decoded_round_error states lane value hv hf

theorem output_piecewise_error (states:Array Accum) (lane shift:Nat) (value:Int) (sr:Bool)
    (hv:states[lane]! =.finite value)
    (hf:(Model.ofNatBits (finalOutput states lane shift sr):Model FloatFormat.binary16).isFinite=true) :
    |(Model.ofNatBits (finalOutput states lane shift sr):Model FloatFormat.binary16).toReal-(value:ℝ)/2^24|≤
      if value.natAbs<2^shift then |(value:ℝ)/2^24|
      else |(value:ℝ)/2^24|/(if sr then 1024 else 2048) := by
  by_cases h:value.natAbs<2^shift
  · simp [finalOutput,hv,h,decoded_zero]
  · simp only [finalOutput,hv,h,↓reduceIte] at hf ⊢
    exact output_round_error states lane value sr hv hf

theorem output_global_error (states:Array Accum) (lane shift:Nat) (value:Int) (sr:Bool)
    (hv:states[lane]! =.finite value)
    (hf:(Model.ofNatBits (finalOutput states lane shift sr):Model FloatFormat.binary16).isFinite=true) :
    |(Model.ofNatBits (finalOutput states lane shift sr):Model FloatFormat.binary16).toReal-(value:ℝ)/2^24|≤
      |(value:ℝ)/2^24| := by
  have h:=output_piecewise_error states lane shift value sr hv hf
  split at h
  · exact h
  · cases sr <;> simp only [Bool.false_eq_true,↓reduceIte] at h <;> linarith [abs_nonneg ((value:ℝ)/2^24)]

theorem coarse_flush_loss (words:List Nat) (value:Int) :
    (flush (coarseShift words) value-value).natAbs≤3 := by
  simp only [flush,coarseShift]
  split <;> split <;> simp_all
  all_goals omega

theorem decoded_model_error (words:List Nat) (states:Array Accum) (lane:Nat) (sr:Bool)
    (valid:∀w∈words,w<65536 ∧ w%32768<31744)
    (hs:states[lane]! =accumulate words)
    (hf:(Model.ofNatBits (finalOutput states lane (coarseShift words) sr):Model FloatFormat.binary16).isFinite=true) :
    |(Model.ofNatBits (finalOutput states lane (coarseShift words) sr):Model FloatFormat.binary16).toReal-
      (words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16).toReal).sum|≤
      |(words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16).toReal).sum| := by
  let value:=(words.map fun w=>(units w).getD 0).sum
  have hv:states[lane]! =.finite value := by rw [hs];exact accumulate_finite words (fun w hw=>(valid w hw).2)
  have hu : ∀w∈words,∃v,FabricF16ExactBounds.units w=some v := by
    intro w hw
    exact FabricF16ExactBounds.units_defined w (valid w hw).1 (FabricF16Bounds.finite_raw w (valid w hw).2)
  rw [FabricF16ExactBounds.sum_units_value words hu]
  exact output_global_error states lane (coarseShift words) value sr hv hf

theorem output_nonzero_error (states:Array Accum) (lane shift:Nat) (value:Int) (sr:Bool)
    (hv:states[lane]! =.finite value)
    (hf:(Model.ofNatBits (finalOutput states lane shift sr):Model FloatFormat.binary16).isFinite=true)
    (hz:(Model.ofNatBits (finalOutput states lane shift sr):Model FloatFormat.binary16).toReal≠0) :
    |(Model.ofNatBits (finalOutput states lane shift sr):Model FloatFormat.binary16).toReal-(value:ℝ)/2^24|≤
      |(value:ℝ)/2^24|/(if sr then 1024 else 2048) := by
  have h:=output_piecewise_error states lane shift value sr hv hf
  split at h
  · rename_i hsmall
    simp [finalOutput,hv,hsmall,decoded_zero] at hz
  · exact h

theorem output_absolute_error (words:List Nat) (states:Array Accum) (lane:Nat) (value:Int) (sr:Bool)
    (hv:states[lane]! =.finite value)
    (hf:(Model.ofNatBits (finalOutput states lane (coarseShift words) sr):Model FloatFormat.binary16).isFinite=true) :
    |(Model.ofNatBits (finalOutput states lane (coarseShift words) sr):Model FloatFormat.binary16).toReal-(value:ℝ)/2^24|≤
      3/2^24+|(value:ℝ)/2^24|/(if sr then 1024 else 2048) := by
  have h:=output_piecewise_error states lane (coarseShift words) value sr hv hf
  split at h
  · rename_i hsmall
    have hn:value.natAbs≤3 := by
      simp only [coarseShift] at hsmall
      split at hsmall <;> norm_num at hsmall <;> omega
    have hr:|(value:ℝ)|≤3 := by
      have hn':(value.natAbs:ℝ)≤3 := by exact_mod_cast hn
      simpa using hn'
    rw [abs_div,abs_of_pos (by positivity : (0:ℝ)<2^24)] at h ⊢
    have hdiv : |(value:ℝ)|/2^24≤(3:ℝ)/2^24 := div_le_div_of_nonneg_right hr (by positivity)
    cases sr <;> simp only [Bool.false_eq_true,↓reduceIte] <;> linarith [abs_nonneg (value:ℝ)]
  · linarith

theorem decoded_nonzero_error (words:List Nat) (states:Array Accum) (lane:Nat) (sr:Bool)
    (valid:∀w∈words,w<65536 ∧ w%32768<31744)
    (hs:states[lane]! =accumulate words)
    (hf:(Model.ofNatBits (finalOutput states lane (coarseShift words) sr):Model FloatFormat.binary16).isFinite=true)
    (hz:(Model.ofNatBits (finalOutput states lane (coarseShift words) sr):Model FloatFormat.binary16).toReal≠0) :
    |(Model.ofNatBits (finalOutput states lane (coarseShift words) sr):Model FloatFormat.binary16).toReal-
      (words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16).toReal).sum|≤
      |(words.map fun w=>(Model.ofNatBits w:Model FloatFormat.binary16).toReal).sum|/(if sr then 1024 else 2048) := by
  let value:=(words.map fun w=>(units w).getD 0).sum
  have hv:states[lane]! =.finite value := by rw [hs];exact accumulate_finite words (fun w hw=>(valid w hw).2)
  have hu : ∀w∈words,∃v,FabricF16ExactBounds.units w=some v := by
    intro w hw
    exact FabricF16ExactBounds.units_defined w (valid w hw).1 (FabricF16Bounds.finite_raw w (valid w hw).2)
  rw [FabricF16ExactBounds.sum_units_value words hu]
  exact output_nonzero_error states lane (coarseShift words) value sr hv hf hz

#print axioms output_nonzero_error
#print axioms output_absolute_error
#print axioms decoded_nonzero_error
#print axioms dyadic_value
#print axioms round_word
#print axioms round_error
#print axioms accumulate_finite
#print axioms round_relative
#print axioms round_zero
#print axioms decoded_zero
#print axioms output_round_error
#print axioms output_piecewise_error
#print axioms output_global_error
#print axioms coarse_flush_loss
#print axioms decoded_model_error
end FabricF16V2Bounds
