/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RcpPtxWrapper
import FloatLib.Numerics.Quantization.Deterministic.Rational
set_option maxRecDepth 10000
set_option maxHeartbeats 400000
namespace GB300Math.RcpPtxBounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem low_pack_word (negative : Bool) (r : Nat) (hr : 0 < r ∧ r ≤ 2^23) :
    Approx.asWord (if r = 2^23 then Model.ofFields FloatFormat.binary32 negative 1 0
      else Model.ofFields FloatFormat.binary32 negative 0 r) =
      (if negative then 2^31 else 0)+r := by
  by_cases h : r = 2^23
  · rw [h]
    simpa using fields_word negative 1 0 (by omega) (by omega)
  · rw [ite_eq_right h]
    have hn := Model.isNaN_eq_false_of_isFinite_eq_true _
      (Model.isFinite_ofFields_ieee FloatFormat.binary32 (by decide) negative 0 r (by decide))
    simp only [Approx.asWord,hn,Bool.false_eq_true,↓reduceIte]
    simpa using fields_bits negative 0 r (by omega) (by omega)

theorem quarter_round_range (f shift : Nat) (hf : f < 2^23)
    (hs : 1 ≤ shift ∧ shift ≤ 2) :
    let r := FloatLib.Numerics.roundShiftRightEven (2^23+f) shift
    0 < r ∧ r ≤ 2^23 := by
  have hl := FloatLib.Numerics.shiftRight_le_roundShiftRightEven (2^23+f) shift
  have hh := FloatLib.Numerics.roundShiftRightEven_le_shiftRight_add1 (2^23+f) shift
  have hs0 := hs.1
  have hs1 := hs.2
  interval_cases shift <;> norm_num at hl hh ⊢ <;> omega

theorem mul_quarter_low_word (negative : Bool) (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 2) (hf : f < 2^23) :
    Approx.mul ((if negative then 2^31 else 0)+e*2^23+f) 0x3e800000 =
      (if negative then 2^31 else 0)+
        FloatLib.Numerics.roundShiftRightEven (2^23+f) (3-e) := by
  have hright : Approx.asFloat 0x3e800000 = Model.ofFields FloatFormat.binary32 false 125 0 := by
    simpa using (fields_float false 125 0 (by omega) (by norm_num)).symm
  change Approx.asWord (Model.mul _ _) = _
  rw [← fields_float negative e f (by omega) hf,hright,Model.Proof.mul_eq_spec]
  unfold Model.Spec.mul
  rw [decode_normal negative e f ⟨he.1,by omega⟩ hf,
      decode_normal false 125 0 (by omega) (by norm_num)]
  simp
  rw [Model.roundDyadic_eq_roundDyadicImpl]
  have hlog : ((8388608+f)*8388608).log2 = 46 := by
    apply (Nat.log2_eq_iff (by omega)).mpr
    constructor <;> omega
  have hround1 := FloatLib.Numerics.roundShiftRightEven_mul_two_pow (2^23+f) 23 1
  have hround2 := FloatLib.Numerics.roundShiftRightEven_mul_two_pow (2^23+f) 23 2
  norm_num at hround1 hround2
  have hfmt : FloatFormat.binary32.isIEEE = true := by decide
  have hmin : FloatFormat.ieeeMinNormalExponent FloatFormat.binary32 = -126 := by rfl
  have halign : FloatFormat.ieeeSubnormalAlignExp FloatFormat.binary32 = 149 := by rfl
  have hwidth : FloatFormat.binary32.fracWidth = 23 := by rfl
  have hr1 := quarter_round_range f 1 hf (by omega)
  have hr2 := quarter_round_range f 2 hf (by omega)
  dsimp only at hr1 hr2
  have hr1n : FloatLib.Numerics.roundShiftRightEven (2^23+f) 1 ≠ 0 := by omega
  have hr2n : FloatLib.Numerics.roundShiftRightEven (2^23+f) 2 ≠ 0 := by omega
  norm_num at hr1n hr2n
  have he0 := he.1
  have he1 := he.2
  have hprod : (8388608+f)*8388608 ≠ 0 := by omega
  interval_cases e <;>
    norm_num [Model.roundDyadicImpl,Model.ieeeRoundDyadicImpl,
      hfmt,hmin,halign,hwidth,hlog,hround1,hround2,hr1n,hr2n,hprod,Model.pow2, Nat.shiftLeft_eq]
  all_goals
    simp only [show (-25 : Int) = Int.negSucc 24 from rfl,
      show (-24 : Int) = Int.negSucc 23 from rfl]
    norm_num [hround1,hround2,hr1n,hr2n]
  · exact low_pack_word negative _ hr2
  · exact low_pack_word negative _ hr1

theorem mul_quarter_low_word_left (negative : Bool) (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 2) (hf : f < 2^23) :
    Approx.mul 0x3e800000 ((if negative then 2^31 else 0)+e*2^23+f) =
      (if negative then 2^31 else 0)+
        FloatLib.Numerics.roundShiftRightEven (2^23+f) (3-e) := by
  have hleft : Approx.asFloat 0x3e800000 = Model.ofFields FloatFormat.binary32 false 125 0 := by
    simpa using (fields_float false 125 0 (by omega) (by norm_num)).symm
  have hc : Approx.mul 0x3e800000 ((if negative then 2^31 else 0)+e*2^23+f) =
      Approx.mul ((if negative then 2^31 else 0)+e*2^23+f) 0x3e800000 := by
    change Approx.asWord (Model.mul _ _) = Approx.asWord (Model.mul _ _)
    rw [← fields_float negative e f (by omega) hf,hleft]
    rw [mul_normal_comm false negative 125 0 e f (by omega) (by norm_num) ⟨he.1,by omega⟩ hf]
  rw [hc]
  exact mul_quarter_low_word negative e f he hf

#print axioms mul_quarter_low_word
#print axioms mul_quarter_low_word_left
end GB300Math.RcpPtxBounds
