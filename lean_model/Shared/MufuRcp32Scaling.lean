/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured
import MufuRcp32Accuracy
namespace GB300Math.MufuRcp32Bounds
open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

theorem normal_fields (e : Fin 254) (f : Fin (2^23)) :
    let x : Model FloatFormat.binary32 := Model.ofNatBits ((1+e.val)*2^23+f.val)
    x.expField = 1+e.val ∧ x.fracField = f.val ∧ x.signBit = false := by
  have he := e.isLt
  have hf := f.isLt
  have hw : (1+e.val)*2^23+f.val < 2^32 := by omega
  have hn : (1+e.val)*2^23+f.val < 2^31 := by omega
  constructor
  · change (((BitVec.ofNat 32 ((1+e.val)*2^23+f.val) >>> 23) &&& (255 : BitVec 32)).toNat) = _
    simp only [BitVec.toNat_and, BitVec.toNat_ushiftRight, BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt hw, Nat.shiftRight_eq_div_pow]
    change ((((1+e.val)*2^23+f.val) / 2^23) &&& (2^8-1)) = _
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  · constructor
    · change ((BitVec.ofNat 32 ((1+e.val)*2^23+f.val) &&& (8388607 : BitVec 32)).toNat) = _
      simp only [BitVec.toNat_and, BitVec.toNat_ofNat]
      rw [Nat.mod_eq_of_lt hw]
      change (((1+e.val)*2^23+f.val) &&& (2^23-1)) = _
      rw [Nat.and_two_pow_sub_one_eq_mod]
      omega
    · rw [Model.signBit_eq_msb]
      change (BitVec.ofNat 32 ((1+e.val)*2^23+f.val)).msb = false
      rw [BitVec.msb_eq_decide]
      simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hw]
      simpa using hn

theorem decode_normal_fields (x : Model FloatFormat.binary32) (e : Fin 254) (f : Fin (2^23))
    (he : x.expField = 1+e.val) (hf : x.fracField = f.val) (hs : x.signBit = false) :
    Model.toRat? x = some (((2^23+f.val : Nat) : Rat) * (2 : Rat)^((e.val : Int)-149)) := by
  have hieee : FloatFormat.binary32.isIEEE = true := by decide
  have hen : (1+e.val == FloatFormat.binary32.expAllOnesNat) = false := by
    change (1+e.val == 255) = false
    simp only [beq_eq_false_iff_ne]
    have h := e.isLt
    omega
  have hez : (1+e.val == 0) = false := by simp
  simp only [Model.toRat?, Model.toDyadic?, hieee, ite_true,
    Model.ieeeToDyadic?, Model.IEEE.isNaN, Model.IEEE.isInf, he, hf, hs, hen, hez]
  norm_num [FloatLib.Numerics.Dyadic.toRat, FloatLib.Numerics.Dyadic.signedSignificand,
    Model.pow2, FloatFormat.binary32, FloatFormat.ieeeNormalMantissaExpOffset, FloatFormat.bias]
  ring_nf
  simp
theorem rcp_fields (x r e : Nat) (hex : Mufu.exponent x = 1+e)
    (hmag : Mufu.magnitude x ≤ 0x7f800000) (hs : Mufu.sign x = 0)
    (hr : (Mufu.interpolate MufuTables.rcp (Mufu.fraction x)).toNat = r)
    (he : e < 252) :
    Mufu.rcp x = ((if r = 2^24 then 253 else 252)-e)*2^23+r%2^23 := by
  have hz : ¬1+e = 0 := by omega
  have hi : ¬1+e = 255 := by omega
  have hm : ¬Mufu.magnitude x > 0x7f800000 := by omega
  unfold Mufu.rcp
  rw [hex, hs, hr]
  simp only [hm, hz, hi, ite_false]
  by_cases h : r = 2^24
  · simp only [h, ite_true]
    have hb : ¬(254 : Int)-((1+e : Nat) : Int) ≤ 0 := by omega
    rw [ite_eq_right hb]
    have hbe : ((254 : Int)-((1+e : Nat) : Int)).toNat = 253-e := by omega
    simp only [Nat.zero_add, hbe]
  · simp only [h, ite_false]
    have hb : ¬(253 : Int)-((1+e : Nat) : Int) ≤ 0 := by omega
    rw [ite_eq_right hb]
    have hbe : ((253 : Int)-((1+e : Nat) : Int)).toNat = 252-e := by omega
    simp only [Nat.zero_add, hbe]
theorem decode_normal (e : Fin 254) (f : Fin (2^23)) :
    Model.toRat? (Model.ofNatBits ((1+e.val)*2^23+f.val) : Model FloatFormat.binary32) =
      some (((2^23+f.val : Nat) : Rat) * (2 : Rat)^((e.val : Int)-149)) := by
  have h := normal_fields e f
  exact decode_normal_fields _ e f h.1 h.2.1 h.2.2

theorem reciprocal_max_bits
    (D : Nat → Option Rat)
    (decode : ∀ (e : Fin 254) (f : Fin (2^23)),
      D ((1+e.val)*2^23+f.val) =
        some (((2^23+f.val : Nat) : Rat) * (2 : Rat)^((e.val : Int)-149)))
    (e : Fin 252) :
    D ((253-e.val)*2^23) = some ((2 : Rat)^24 * (2 : Rat)^(102-(e.val : Int))) := by
  have he := e.isLt
  let eo : Fin 254 := ⟨252-e.val, by omega⟩
  have d := decode eo (0 : Fin (2^23))
  simp only [Fin.val_zero, Nat.add_zero, Nat.cast_pow, Nat.cast_ofNat] at d
  have hword : (1+eo.val)*2^23 = (253-e.val)*2^23 := by dsimp [eo]; omega
  rw [hword] at d
  apply d.trans
  apply congrArg some
  have hp : (eo.val : Int)-149 = (102-(e.val : Int))+1 := by dsimp [eo]; omega
  rw [hp, zpow_add₀ (by norm_num : (2 : Rat) ≠ 0), zpow_one]
  ring

#print axioms reciprocal_max_bits

theorem reciprocal_other_bits
    (D : Nat → Option Rat)
    (decode : ∀ (e : Fin 254) (f : Fin (2^23)),
      D ((1+e.val)*2^23+f.val) =
        some (((2^23+f.val : Nat) : Rat) * (2 : Rat)^((e.val : Int)-149)))
    (e : Fin 252) (r : Nat) (rlo : 2^23 ≤ r) (rhi : r < 2^24) :
    D ((252-e.val)*2^23+r%2^23) = some ((r : Rat)*(2 : Rat)^(102-(e.val : Int))) := by
  have he := e.isLt
  let eo : Fin 254 := ⟨251-e.val, by omega⟩
  let fo : Fin (2^23) := ⟨r%2^23, Nat.mod_lt _ (by norm_num)⟩
  have d := decode eo fo
  have hword : (1+eo.val)*2^23+fo.val = (252-e.val)*2^23+r%2^23 := by dsimp [eo,fo]; omega
  rw [hword] at d
  apply d.trans
  apply congrArg some
  have hp : (eo.val : Int)-149 = 102-(e.val : Int) := by dsimp [eo]; omega
  have hm : 2^23+fo.val = r := by dsimp [fo]; omega
  rw [hp, hm]

#print axioms reciprocal_other_bits

theorem reciprocal_bits_value_generic
    (D : Nat → Option Rat)
    (decode : ∀ (e : Fin 254) (f : Fin (2^23)),
      D ((1+e.val)*2^23+f.val) =
        some (((2^23+f.val : Nat) : Rat) * (2 : Rat)^((e.val : Int)-149)))
    (e : Fin 252) (r : Nat) (rlo : 2^23 ≤ r) (rhi : r ≤ 2^24) :
    D (((if r = 2^24 then 253 else 252)-e.val)*2^23+r%2^23) =
      some ((r : Rat)*(2 : Rat)^(102-(e.val : Int))) := by
  by_cases h : r = 2^24
  · have hm : r%2^23=0 := by omega
    rw [ite_eq_left h, hm, Nat.add_zero]
    have d := reciprocal_max_bits D decode e
    have hc : (r : Rat) = (2 : Rat)^24 := by rw [h]; norm_cast
    rw [hc]
    exact d
  · rw [ite_eq_right h]
    exact reciprocal_other_bits D decode e r rlo (by omega)

theorem reciprocal_bits_value (e : Fin 252) (r : Nat) (rlo : 2^23 ≤ r) (rhi : r ≤ 2^24) :
    Model.toRat? (Model.ofNatBits
      (((if r = 2^24 then 253 else 252)-e.val)*2^23+r%2^23) : Model FloatFormat.binary32) =
      some ((r : Rat)*(2 : Rat)^(102-(e.val : Int))) :=
  reciprocal_bits_value_generic
    (fun n => Model.toRat? (Model.ofNatBits n : Model FloatFormat.binary32))
    decode_normal e r rlo rhi

theorem sign_zero_of_lt (w : Nat) (h : w < 2^31) : Mufu.sign w = 0 := by
  have bit : w.testBit 31 = false := Nat.testBit_lt_two_pow h
  calc
    Mufu.sign w = (w.testBit 31).toNat * 2^31 := Nat.and_two_pow w 31
    _ = 0 * 2^31 := congrArg (fun b : Bool => b.toNat * 2^31) bit
    _ = 0 := Nat.zero_mul _

theorem encoded_normal_rcp (e : Fin 252) (f : Fin (2^23)) :
    Mufu.rcp ((1+e.val)*2^23+f.val) =
      let r := (Mufu.interpolate MufuTables.rcp f.val).toNat
      ((if r = 2^24 then 253 else 252)-e.val)*2^23+r%2^23 := by
  have he := e.isLt
  have hf := f.isLt
  have hx : Mufu.exponent ((1+e.val)*2^23+f.val) = 1+e.val := by unfold Mufu.exponent; omega
  have hm : Mufu.magnitude ((1+e.val)*2^23+f.val) ≤ 0x7f800000 := by
    unfold Mufu.magnitude
    rw [Nat.mod_eq_of_lt (show (1+e.val)*2^23+f.val < 2^31 by omega)]
    omega
  have hs : Mufu.sign ((1+e.val)*2^23+f.val) = 0 :=
    sign_zero_of_lt _ (by omega)
  have hfrac : Mufu.fraction ((1+e.val)*2^23+f.val) = f.val := by unfold Mufu.fraction; omega
  have rr : (Mufu.interpolate MufuTables.rcp (Mufu.fraction ((1+e.val)*2^23+f.val))).toNat = (Mufu.interpolate MufuTables.rcp f.val).toNat := by
    rw [hfrac]
  exact rcp_fields _ _ e.val hx hm hs rr he

theorem decode_rcp_normal_generic
    (D : Nat → Option Rat)
    (decode : ∀ (e : Fin 252) (r : Nat), 2^23 ≤ r → r ≤ 2^24 →
      D (((if r = 2^24 then 253 else 252)-e.val)*2^23+r%2^23) =
        some ((r : Rat)*(2 : Rat)^(102-(e.val : Int))))
    (e : Fin 252) (f : Fin (2^23)) :
    D (Mufu.rcp ((1+e.val)*2^23+f.val)) =
      some ((Mufu.interpolate MufuTables.rcp f.val : Rat)*(2 : Rat)^(102-(e.val : Int))) := by
  have he := e.isLt
  have hf := f.isLt
  have hr := mantissa_range f
  let r := (Mufu.interpolate MufuTables.rcp f.val).toNat
  have hn : 0 ≤ Mufu.interpolate MufuTables.rcp f.val := le_trans (by norm_num) hr.1
  have hcast : (r : Int) = Mufu.interpolate MufuTables.rcp f.val := Int.toNat_of_nonneg hn
  have rlo : 2^23 ≤ r := Int.toNat_le_toNat hr.1
  have rhi : r ≤ 2^24 := Int.toNat_le_toNat hr.2
  rw [encoded_normal_rcp]
  have d := decode e r rlo rhi
  have hc : (r : Rat) = (Mufu.interpolate MufuTables.rcp f.val : Rat) := by exact_mod_cast hcast
  simpa only [hc] using d

theorem decode_rcp_normal (e : Fin 252) (f : Fin (2^23)) :
    Model.toRat? (Model.ofNatBits
      (Mufu.rcp ((1+e.val)*2^23+f.val)) : Model FloatFormat.binary32) =
      some ((Mufu.interpolate MufuTables.rcp f.val : Rat)*(2 : Rat)^(102-(e.val : Int))) :=
  decode_rcp_normal_generic
    (fun n => Model.toRat? (Model.ofNatBits n : Model FloatFormat.binary32))
    reciprocal_bits_value e f

theorem scaled_relative_error (e : Int) (f : Fin (2^23)) :
    let x : ℝ := (2^23+(f.val : ℝ))*2^(e-149)
    let y : ℝ := (Mufu.interpolate MufuTables.rcp f.val : ℝ)*2^(102-e)
    |y-1/x|/|1/x| ≤ 1/(2^23 : ℝ) := by
  dsimp only
  have hx : (2^23+(f.val : ℝ))*(2 : ℝ)^(e-149) ≠ 0 := by positivity
  rw [relative_eq_product _ _ hx]
  have product : ((2^23+(f.val : ℝ))*2^(e-149)) *
      ((Mufu.interpolate MufuTables.rcp f.val : ℝ)*2^(102-e)) =
      (2^23+(f.val : ℝ))*(Mufu.interpolate MufuTables.rcp f.val : ℝ)/2^47 := by
    calc
      _ = (2^23+(f.val : ℝ))*(Mufu.interpolate MufuTables.rcp f.val : ℝ)*
          ((2 : ℝ)^(e-149)*2^(102-e)) := by ring
      _ = _ := by
        rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
        have hp : e-149+(102-e) = -47 := by ring
        rw [hp]
        norm_num
        ring
  rw [product]
  have h := normalized_relative_error f
  rw [relative_eq_product _ _ (by dsimp [normalizedInput]; positivity)] at h
  have identity : normalizedInput f.val * normalizedOutput f.val =
      (2^23+(f.val : ℝ))*(Mufu.interpolate MufuTables.rcp f.val : ℝ)/2^47 := by
    dsimp [normalizedInput, normalizedOutput]
    ring
  rw [identity] at h
  exact h

theorem positive_normal_floatlib_error (e : Fin 252) (f : Fin (2^23)) :
    ∃ x y : Rat,
      Model.toRat? (Model.ofNatBits ((1+e.val)*2^23+f.val) : Model FloatFormat.binary32) = some x ∧
      Model.toRat? (Model.ofNatBits (Mufu.rcp ((1+e.val)*2^23+f.val)) : Model FloatFormat.binary32) = some y ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 1/(2^23 : ℝ) := by
  refine ⟨((2^23+f.val : Nat) : Rat)*(2 : Rat)^((e.val : Int)-149),
    (Mufu.interpolate MufuTables.rcp f.val : Rat)*(2 : Rat)^(102-(e.val : Int)),
    ?_, decode_rcp_normal e f, ?_⟩
  · exact decode_normal ⟨e.val, by have h := e.isLt; omega⟩ f
  · have h := scaled_relative_error e.val f
    norm_num at h ⊢
    exact h

theorem highest_normal_exponent_flushes (w : Nat)
    (he : Mufu.exponent w = 254) (hm : Mufu.magnitude w ≤ 0x7f800000) :
    Mufu.rcp w = Mufu.sign w := by
  have hmag : ¬Mufu.magnitude w > 0x7f800000 := by omega
  unfold Mufu.rcp
  rw [he]
  simp only [hmag, ite_false]
  split_ifs <;> simp_all

theorem flushed_relative_error (x : ℝ) (hx : x ≠ 0) :
    |0-1/x|/|1/x| = 1 := by
  rw [zero_sub, abs_neg]
  exact div_self (abs_ne_zero.mpr (one_div_ne_zero hx))

theorem normal_tail_examples :
    Mufu.rcp 0x7e800000 = 0x00800000 ∧
    Mufu.rcp 0x7e800001 = 0 ∧
    Mufu.rcp 0x7f000000 = 0 := by
  decide +kernel

#print axioms sign_zero_of_lt
#print axioms encoded_normal_rcp
#print axioms highest_normal_exponent_flushes
#print axioms flushed_relative_error
#print axioms normal_tail_examples
#print axioms normal_fields
#print axioms decode_normal_fields
#print axioms rcp_fields
#print axioms decode_normal
#print axioms reciprocal_bits_value
#print axioms decode_rcp_normal
#print axioms scaled_relative_error
#print axioms positive_normal_floatlib_error
end GB300Math.MufuRcp32Bounds
