/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuRcp32Classification
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange
theorem full_normal_fields (e : Fin 2046) (f : Fin (2^52)) :
    let x : Model FloatFormat.binary64 := Model.ofNatBits ((1+e.val)*2^52+f.val)
    x.expField = 1+e.val ∧ x.fracField = f.val ∧ x.signBit = false := by
  have he := e.isLt
  have hf := f.isLt
  have hw : (1+e.val)*2^52+f.val < 2^64 := by omega
  have hn : (1+e.val)*2^52+f.val < 2^63 := by omega
  constructor
  · change (((BitVec.ofNat 64 ((1+e.val)*2^52+f.val) >>> 52) &&& (2047 : BitVec 64)).toNat) = _
    simp only [BitVec.toNat_and, BitVec.toNat_ushiftRight, BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt hw, Nat.shiftRight_eq_div_pow]
    change ((((1+e.val)*2^52+f.val) / 2^52) &&& (2^11-1)) = _
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  · constructor
    · change ((BitVec.ofNat 64 ((1+e.val)*2^52+f.val) &&& (4503599627370495 : BitVec 64)).toNat) = _
      simp only [BitVec.toNat_and, BitVec.toNat_ofNat]
      rw [Nat.mod_eq_of_lt hw]
      change (((1+e.val)*2^52+f.val) &&& (2^52-1)) = _
      rw [Nat.and_two_pow_sub_one_eq_mod]
      omega
    · rw [Model.signBit_eq_msb]
      change (BitVec.ofNat 64 ((1+e.val)*2^52+f.val)).msb = false
      rw [BitVec.msb_eq_decide]
      simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hw]
      simpa using hn

theorem full_decode_normal_fields (x : Model FloatFormat.binary64) (e : Fin 2046) (f : Fin (2^52))
    (he : x.expField = 1+e.val) (hf : x.fracField = f.val) (hs : x.signBit = false) :
    Model.toRat? x = some (((2^52+f.val : Nat) : Rat) * (2 : Rat)^((e.val : Int)-1074)) := by
  have hieee : FloatFormat.binary64.isIEEE = true := by decide
  have hen : (1+e.val == FloatFormat.binary64.expAllOnesNat) = false := by
    change (1+e.val == 2047) = false
    simp only [beq_eq_false_iff_ne]
    have h := e.isLt
    omega
  have hez : (1+e.val == 0) = false := by simp
  simp only [Model.toRat?, Model.toDyadic?, hieee, ite_true,
    Model.ieeeToDyadic?, Model.IEEE.isNaN, Model.IEEE.isInf, he, hf, hs, hen, hez]
  norm_num [FloatLib.Numerics.Dyadic.toRat, FloatLib.Numerics.Dyadic.signedSignificand,
    Model.pow2, FloatFormat.binary64, FloatFormat.ieeeNormalMantissaExpOffset, FloatFormat.bias]
  ring_nf
  simp
theorem full_decode_normal (e : Fin 2046) (f : Fin (2^52)) :
    Model.toRat? (Model.ofNatBits ((1+e.val)*2^52+f.val) : Model FloatFormat.binary64) =
      some (((2^52+f.val : Nat) : Rat) * (2 : Rat)^((e.val : Int)-1074)) := by
  have h := full_normal_fields e f
  exact full_decode_normal_fields _ e f h.1 h.2.1 h.2.2

theorem decode_full_normalized (f : Fin (2^20)) (low : Fin (2^32)) :
    (Model.ofNatBits ((1023*2^20+f.val)*2^32+low.val) : Model FloatFormat.binary64).toRat? =
      some ((((2^20+f.val)*2^32+low.val : Nat) : Rat)/2^52) := by
  let frac : Fin (2^52) := ⟨f.val*2^32+low.val, by have hf := f.isLt; have hl := low.isLt; omega⟩
  have d := full_decode_normal (1022 : Fin 2046) frac
  have hw : (1023*2^20+f.val)*2^32+low.val = (1+(1022 : Fin 2046).val)*2^52+frac.val := by
    dsimp [frac]
    ring
  rw [hw,d]
  dsimp [frac]
  norm_num
  ring_nf

#print axioms full_normal_fields
#print axioms full_decode_normal_fields
#print axioms full_decode_normal
#print axioms decode_full_normalized
end GB300Math.Rcp64Bounds
