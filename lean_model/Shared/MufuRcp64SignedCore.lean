/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured
import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.SignedSemantics.Core
import MufuRcp64ScaleAccuracy
import Mathlib.Tactic
set_option maxRecDepth 8000
namespace GB300Math.Rcp64Signed
open FloatLib.Floats.Formats.BinaryInterchange

theorem toggle_word (w : Nat) :
    Model.toggleSign (Model.ofNatBits w : Model FloatFormat.binary64) =
      Model.ofNatBits (w+2^63) := by
  apply congrArg Model.ofBits
  change (BitVec.ofNat 64 w) ^^^ BitVec.ofNat 64 (2^63) = BitVec.ofNat 64 (w+2^63)
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_ofNat, hi, decide_true, Bool.true_and]
  rw [show (9223372036854775808 : Nat).testBit i = decide (63=i) from @Nat.testBit_two_pow 63 i]
  by_cases he : i = 63
  · subst i
    simpa only [decide_true, Bool.xor_true, Nat.add_comm] using (Nat.testBit_two_pow_add_eq w 63).symm
  · have hl : i < 63 := by omega
    rw [Nat.add_comm, Nat.testBit_two_pow_add_gt hl]
    simp [Ne.symm he]

theorem neg_rat (x : Model FloatFormat.binary64) (v : Rat)
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
      simp [Model.negDyadic, FloatFormat.binary64, FloatFormat.supportsSignedZero,
        FloatLib.Numerics.Dyadic.toRat, FloatLib.Numerics.Dyadic.signedSignificand, hn]

theorem signed_decode (w : Nat) (v : Rat)
    (h : (Model.ofNatBits w : Model FloatFormat.binary64).toRat? = some v) :
    (Model.ofNatBits (w+2^63) : Model FloatFormat.binary64).toRat? = some (-v) := by
  rw [← toggle_word]
  rw [← Model.neg_eq_toggleSign_of_supportsSignedZero _ (by decide)]
  exact neg_rat _ v h

#print axioms toggle_word
#print axioms neg_rat
#print axioms signed_decode
end GB300Math.Rcp64Signed
