/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured
import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.SignedSemantics.Core
import MufuRcp32Scaling
import Mathlib.Tactic
set_option maxRecDepth 8000
namespace GB300Math.RcpSigned
open FloatLib.Floats.Formats.BinaryInterchange

theorem toggle_word (w : Nat) :
    Model.toggleSign (Model.ofNatBits w : Model FloatFormat.binary32) =
      Model.ofNatBits (w+2^31) := by
  apply congrArg Model.ofBits
  change (BitVec.ofNat 32 w) ^^^ BitVec.ofNat 32 (2^31) = BitVec.ofNat 32 (w+2^31)
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_ofNat, hi, decide_true, Bool.true_and]
  rw [show (2147483648 : Nat).testBit i = decide (31=i) from @Nat.testBit_two_pow 31 i]
  by_cases he : i = 31
  · subst i
    simpa only [decide_true, Bool.xor_true, Nat.add_comm] using (Nat.testBit_two_pow_add_eq w 31).symm
  · have hl : i < 31 := by omega
    rw [Nat.add_comm, Nat.testBit_two_pow_add_gt hl]
    simp [Ne.symm he]

theorem neg_rat (x : Model FloatFormat.binary32) (v : Rat)
    (h : x.toRat? = some v) : (Model.neg x).toRat? = some (-v) := by
  unfold Model.toRat? at h ⊢
  cases hd : x.toDyadic? with
  | none => simp [hd] at h
  | some d =>
    simp only [hd, Option.map_some, Option.some.injEq] at h
    rw [Model.toDyadic?_neg_of_toDyadic?_some x hd]
    simp only [Option.map_some]
    rw [← h]
    apply congrArg some
    cases hn : d.negative <;>
      simp [Model.negDyadic, FloatFormat.binary32, FloatFormat.supportsSignedZero,
        FloatLib.Numerics.Dyadic.toRat, FloatLib.Numerics.Dyadic.signedSignificand, hn]

theorem signed_decode (w : Nat) (v : Rat)
    (h : (Model.ofNatBits w : Model FloatFormat.binary32).toRat? = some v) :
    (Model.ofNatBits (w+2^31) : Model FloatFormat.binary32).toRat? = some (-v) := by
  rw [← toggle_word]
  rw [← Model.neg_eq_toggleSign_of_supportsSignedZero _ (by decide)]
  exact neg_rat _ v h

#print axioms toggle_word
#print axioms neg_rat
#print axioms signed_decode
end GB300Math.RcpSigned

namespace GB300Math.RcpSigned

theorem sign_zero (w : Nat) (hw : w < 2^31) : Mufu.sign w = 0 := by
  unfold Mufu.sign
  change w &&& 2^31 = 0
  rw [Nat.and_two_pow, Nat.testBit_lt_two_pow hw]
  exact Nat.zero_mul _

theorem sign_set (w : Nat) (hw : w < 2^31) : Mufu.sign (2^31+w) = 2^31 := by
  unfold Mufu.sign
  change (2^31+w) &&& 2^31 = 2^31
  rw [Nat.and_two_pow]
  have hb : (2^31+w).testBit 31 = true := by
    rw [Nat.testBit_two_pow_add_eq, Nat.testBit_lt_two_pow hw]
    rfl
  rw [hb]
  exact Nat.one_mul _

theorem exponent_sign (w : Nat) : Mufu.exponent (2^31+w) = Mufu.exponent w := by
  unfold Mufu.exponent
  omega

theorem fraction_sign (w : Nat) : Mufu.fraction (2^31+w) = Mufu.fraction w := by
  unfold Mufu.fraction
  omega

theorem magnitude_sign (w : Nat) : Mufu.magnitude (2^31+w) = Mufu.magnitude w := by
  unfold Mufu.magnitude
  omega

#print axioms sign_zero
#print axioms sign_set
#print axioms exponent_sign
#print axioms fraction_sign
#print axioms magnitude_sign
end GB300Math.RcpSigned

namespace GB300Math.RcpSigned

theorem rcp_sign (w : Nat) (finite_or_inf : w ≤ 0x7f800000) :
    Mufu.rcp (2^31+w) = 2^31+Mufu.rcp w := by
  have hw : w < 2^31 := by omega
  have hm : Mufu.magnitude w = w := Nat.mod_eq_of_lt hw
  unfold Mufu.rcp
  rw [magnitude_sign, exponent_sign, fraction_sign, sign_zero w hw, sign_set w hw, hm]
  have hn : ¬ w > 0x7f800000 := by omega
  simp only [hn, ite_false]
  split_ifs <;> simp only [Nat.zero_add, Nat.add_zero, Nat.add_assoc]

#print axioms rcp_sign
end GB300Math.RcpSigned


namespace GB300Math.RcpSigned
open FloatLib.Floats.Formats.BinaryInterchange

theorem negative_normal_floatlib_error (e : Fin 252) (f : Fin (2^23)) :
    let w := (1+e.val)*2^23+f.val
    ∃ x y : Rat,
      Model.toRat? (Model.ofNatBits (w+2^31) : Model FloatFormat.binary32) = some x ∧
      Model.toRat? (Model.ofNatBits (Mufu.rcp (w+2^31)) : Model FloatFormat.binary32) = some y ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 1/(2^23 : ℝ) := by
  dsimp only
  obtain ⟨x,y,hx,hy,hbound⟩ := MufuRcp32Bounds.positive_normal_floatlib_error e f
  refine ⟨-x,-y,signed_decode _ _ hx,?_,?_⟩
  · have hw : (1+e.val)*2^23+f.val ≤ 0x7f800000 := by
      have he := e.isLt
      have hf := f.isLt
      omega
    have hs := rcp_sign ((1+e.val)*2^23+f.val) hw
    rw [Nat.add_comm (2^31) _, Nat.add_comm (2^31) _] at hs
    rw [hs]
    exact signed_decode _ _ hy
  · simpa only [Rat.cast_neg, one_div, inv_neg, neg_sub_neg, abs_sub_comm, abs_neg] using hbound

#print axioms negative_normal_floatlib_error
end GB300Math.RcpSigned
