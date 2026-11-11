/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF32Bounds
import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Proof
import FloatLib.Floats.Formats.Flocq.Theory.Error.Bounds
namespace FabricF32RoundBounds
open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.Flocq
open GB300Math.TensorAlignmentLoss

def asDyadic (value : Int) : FloatLib.Numerics.Dyadic :=
  {negative := value < 0, significand := value.natAbs, exponent := -149}

theorem dyadic_value (value : Int) : (asDyadic value).toReal = (value : ℝ)/(2^149 : ℝ) := by
  cases value <;>
    simp [asDyadic, FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand,
      zpow_neg, div_eq_mul_inv]

theorem round_word (value : Int) :
    (Model.ofNatBits (FabricF32.round value) : Model (.ieee 8 23)) =
      Model.roundDyadic (.ieee 8 23) (asDyadic value) := by
  unfold FabricF32.round
  exact Model.ofNatBits_toNatBits _

theorem round_error (value : Int)
    (finite : Model.isFinite (Model.ofNatBits (FabricF32.round value) : Model (.ieee 8 23)) = true) :
    |Model.toReal (Model.ofNatBits (FabricF32.round value) : Model (.ieee 8 23)) -
        (value : ℝ)/(2^149 : ℝ)| ≤
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf (.ieee 8 23))
        ((value : ℝ)/(2^149 : ℝ))/2 := by
  rw [round_word] at finite ⊢
  rw [Model.toReal_roundDyadic_eq_roundAt _ (by decide) _ finite, dyadic_value]
  exact error_bound_ulp (rnd := nearestEven) _

def acc (words : List Nat) (values : List Int) : Int :=
  FabricF32.flush (FabricF32.coarseShift words) (FabricF32.aligned words values)

def loss (words : List Nat) (values : List Int) : Nat :=
  values.length*(2^FabricF32.fineShift words-1)+(2^FabricF32.coarseShift words-1)

theorem accumulator_real_error (words : List Nat) (values : List Int) :
    |(acc words values : ℝ)/(2^149 : ℝ) - (exactSum values : ℝ)/(2^149 : ℝ)| ≤
      (loss words values : ℝ)/(2^149 : ℝ) := by
  have h := FabricF32Bounds.accumulator_loss words values
  have hr : |((acc words values - exactSum values : Int) : ℝ)| ≤ (loss words values : ℝ) := by
    have hn : (((acc words values-exactSum values).natAbs : Nat) : ℝ) ≤ (loss words values : ℝ) := by
      exact_mod_cast h
    simpa using hn
  rw [Int.cast_sub] at hr
  rw [← sub_div, abs_div, abs_of_pos (by positivity : (0:ℝ)<2^149)]
  exact div_le_div_of_nonneg_right hr (by positivity)

theorem final_error (words : List Nat) (values : List Int)
    (finite : Model.isFinite (Model.ofNatBits (FabricF32.round (acc words values)) : Model (.ieee 8 23)) = true) :
    |Model.toReal (Model.ofNatBits (FabricF32.round (acc words values)) : Model (.ieee 8 23)) -
        (exactSum values : ℝ)/(2^149 : ℝ)| ≤
      (loss words values : ℝ)/(2^149 : ℝ) +
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf (.ieee 8 23))
        ((acc words values : ℝ)/(2^149 : ℝ))/2 := by
  have h1 := round_error (acc words values) finite
  have h2 := accumulator_real_error words values
  calc
    _ ≤ |Model.toReal (Model.ofNatBits (FabricF32.round (acc words values)) : Model (.ieee 8 23)) -
      (acc words values : ℝ)/(2^149 : ℝ)| +
      |(acc words values : ℝ)/(2^149 : ℝ) - (exactSum values : ℝ)/(2^149 : ℝ)| := abs_sub_le _ _ _
    _ ≤ _ := by linarith

theorem predict_finite (words : List Nat)
    (finite_inputs : ∀ w ∈ words, w % 2^31 < 0x7f800000) :
    FabricF32.predict words = FabricF32.round (acc words (words.map fun w => (FabricF32.units w).getD 0)) := by
  have hn : ¬ ∃ w ∈ words, 0x7f800000 < w % 2147483648 := by
    rintro ⟨w, hw, h⟩
    have := finite_inputs w hw
    omega
  have hi : ¬ ∃ w ∈ words, w % 2147483648 = 0x7f800000 := by
    rintro ⟨w, hw, h⟩
    have := finite_inputs w hw
    omega
  simp [FabricF32.predict, hn, hi, acc]

theorem finite_model_error (words : List Nat)
    (finite_inputs : ∀ w ∈ words, w % 2^31 < 0x7f800000)
    (finite_output : Model.isFinite (Model.ofNatBits (FabricF32.predict words) : Model (.ieee 8 23)) = true) :
    let values := words.map fun w => (FabricF32.units w).getD 0
    |Model.toReal (Model.ofNatBits (FabricF32.predict words) : Model (.ieee 8 23)) -
        (exactSum values : ℝ)/(2^149 : ℝ)| ≤
      (loss words values : ℝ)/(2^149 : ℝ) +
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf (.ieee 8 23))
        ((acc words values : ℝ)/(2^149 : ℝ))/2 := by
  rw [predict_finite words finite_inputs] at finite_output ⊢
  exact final_error words _ finite_output

theorem midpoint_rounding_bits : FabricF32.round 16777217 = 16777216 := by decide +kernel

theorem midpoint_rounding_error :
    (Model.ofNatBits (FabricF32.round 16777217) : Model (.ieee 8 23)).toRat? =
      some ((16777217:Rat)/2^149 - 1/2^149) := by decide +kernel

#print axioms predict_finite
#print axioms finite_model_error
#print axioms midpoint_rounding_bits
#print axioms midpoint_rounding_error
#print axioms accumulator_real_error
#print axioms final_error
#print axioms dyadic_value
#print axioms round_word
#print axioms round_error
end FabricF32RoundBounds
