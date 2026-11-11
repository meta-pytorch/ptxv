/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu
import MufuRcp32Signed
import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.PowerProof
import FloatLib.Floats.Formats.BinaryInterchange.Configured
import TanhTail
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Tactic
namespace GB300Math.TanhPlateau
open FloatLib.Floats.Formats.BinaryInterchange

theorem tail_error (x : ℝ) (hx : 8 ≤ x) : |1-Real.tanh x| < 1/(2^22:ℝ) := by
  exact lt_trans (TanhTail.positive_tail x hx) (by norm_num)

theorem boundary_lower_error : 1/(2^23:ℝ) < |1-Real.tanh 8| := by
  exact lt_trans (by norm_num) TanhTail.tail_error_at_eight

theorem native_positive_bits (word : Nat) (hm : Mufu.magnitude word ≤ 0x7f800000)
    (he : 130 ≤ Mufu.exponent word) : Mufu.tanh word = some (Mufu.sign word+0x3f800000) := by
  simp [Mufu.tanh, Nat.not_lt.mpr hm, Nat.not_lt.mpr (show 119 ≤ Mufu.exponent word by omega), he]

theorem boundary_bits : Mufu.tanh 0x41000000 = some 0x3f800000 := by decide +kernel

theorem boundary_decode :
    (Model.ofNatBits 0x41000000 : Model FloatFormat.binary32).toRat? = some 8 ∧
    (Model.ofNatBits 0x3f800000 : Model FloatFormat.binary32).toRat? = some 1 := by decide +kernel


def tailWord (e : Fin 125) (f : Fin (2^23)) : Nat := (130+e.val)*2^23+f.val

theorem tail_rat (e : Fin 125) (f : Fin (2^23)) :
    (Model.ofNatBits (tailWord e f) : Model FloatFormat.binary32).toRat? =
      some (((2^23+f.val : Nat) : Rat)*(2:Rat)^((e.val:Int)-20)) := by
  let ei : Fin 254 := ⟨129+e.val, by have := e.isLt; omega⟩
  have h := MufuRcp32Bounds.decode_normal ei f
  have he : (ei.val:Int)-149 = (e.val:Int)-20 := by dsimp [ei]; omega
  have hw : (1+ei.val)*2^23+f.val = tailWord e f := by simp [ei, tailWord]; omega
  rwa [hw, he] at h

theorem tail_decode (e : Fin 125) (f : Fin (2^23)) :
    Model.toReal (Model.ofNatBits (tailWord e f) : Model FloatFormat.binary32) =
      ((2^23+f.val : Nat) : ℝ)*(2:ℝ)^((e.val:Int)-20) := by
  simpa using Model.Algebraic.toReal_of_toRat?_eq_some (tail_rat e f)

theorem tail_value_lower (e : Fin 125) (f : Fin (2^23)) :
    8 ≤ Model.toReal (Model.ofNatBits (tailWord e f) : Model FloatFormat.binary32) := by
  rw [tail_decode]
  have hexp : (e.val:Int)-20 = (e.val:Int)+(-20) := by omega
  rw [hexp, zpow_add₀ (by norm_num : (2:ℝ) ≠ 0), zpow_natCast]
  have hp : (1:ℝ) ≤ 2^e.val := one_le_pow₀ (by norm_num)
  have hf : (2^23:ℝ) ≤ ((2^23+f.val:Nat):ℝ) := by exact_mod_cast (Nat.le_add_right (2^23) f.val)
  norm_num at hf ⊢
  nlinarith

theorem tail_bits (e : Fin 125) (f : Fin (2^23)) :
    Mufu.tanh (tailWord e f) = some 0x3f800000 := by
  have he := e.isLt
  have hf := f.isLt
  have hw : tailWord e f < 2^31 := by unfold tailWord; omega
  have hs := RcpSigned.sign_zero (tailWord e f) hw
  have hm : Mufu.magnitude (tailWord e f) ≤ 0x7f800000 := by unfold Mufu.magnitude tailWord; omega
  have hx : 130 ≤ Mufu.exponent (tailWord e f) := by unfold Mufu.exponent tailWord; omega
  simpa [hs] using native_positive_bits (tailWord e f) hm hx

theorem decoded_tail_error (e : Fin 125) (f : Fin (2^23)) :
    |Model.toReal (Model.ofNatBits 0x3f800000 : Model FloatFormat.binary32) -
      Real.tanh (Model.toReal (Model.ofNatBits (tailWord e f) : Model FloatFormat.binary32))| < 1/(2^22:ℝ) := by
  have ho := Model.Algebraic.toReal_of_toRat?_eq_some boundary_decode.2
  norm_num at ho
  rw [ho]
  exact tail_error _ (tail_value_lower e f)

theorem negative_tail_error (x : ℝ) (hx : 8 ≤ x) :
    |-1-Real.tanh (-x)| < 1/(2^22:ℝ) := by
  rw [Real.tanh_neg, show (-1:ℝ)- -Real.tanh x = -(1-Real.tanh x) by ring, abs_neg]
  exact tail_error x hx


theorem negative_tail_bits (e : Fin 125) (f : Fin (2^23)) :
    Mufu.tanh (2^31+tailWord e f) = some 0xbf800000 := by
  have he := e.isLt
  have hf := f.isLt
  have hw : tailWord e f < 2^31 := by unfold tailWord; omega
  have hs := RcpSigned.sign_set (tailWord e f) hw
  have hm : Mufu.magnitude (2^31+tailWord e f) ≤ 0x7f800000 := by unfold Mufu.magnitude tailWord; omega
  have hx : 130 ≤ Mufu.exponent (2^31+tailWord e f) := by unfold Mufu.exponent tailWord; omega
  have h := native_positive_bits (2^31+tailWord e f) hm hx
  rw [hs] at h
  exact h

theorem sign_decode (w : Nat)
    (finite : (Model.ofNatBits w : Model FloatFormat.binary32).isFinite = true) :
    Model.toReal (Model.ofNatBits (2^31+w) : Model FloatFormat.binary32) =
      -Model.toReal (Model.ofNatBits w : Model FloatFormat.binary32) := by
  rw [Nat.add_comm, ← RcpSigned.toggle_word]
  rw [← Model.neg_eq_toggleSign_of_supportsSignedZero _ (by decide)]
  exact Model.toReal_neg _ finite

theorem decoded_negative_tail_error (e : Fin 125) (f : Fin (2^23)) :
    |Model.toReal (Model.ofNatBits 0xbf800000 : Model FloatFormat.binary32) -
      Real.tanh (Model.toReal (Model.ofNatBits (2^31+tailWord e f) : Model FloatFormat.binary32))| < 1/(2^22:ℝ) := by
  have hd : (Model.ofNatBits 0xbf800000 : Model FloatFormat.binary32).toRat? = some (-1) := by decide +kernel
  have ho := Model.Algebraic.toReal_of_toRat?_eq_some hd
  norm_num at ho
  rw [ho, sign_decode _ (Model.Algebraic.isFinite_of_toRat?_eq_some (tail_rat e f))]
  exact negative_tail_error _ (tail_value_lower e f)

theorem finite_tail_coverage (word : Nat) (hw : 0x41000000 ≤ word) (hf : word < 0x7f800000) :
    ∃ e : Fin 125, ∃ f : Fin (2^23), word = tailWord e f := by
  refine ⟨⟨word/2^23-130, by omega⟩, ⟨word%2^23, Nat.mod_lt _ (by norm_num)⟩, ?_⟩
  simp only [tailWord]
  omega


theorem positive_native_bound (word : Nat) (hw : 0x41000000 ≤ word) (hf : word < 0x7f800000) :
    Mufu.tanh word = some 0x3f800000 ∧
      |Model.toReal (Model.ofNatBits 0x3f800000 : Model FloatFormat.binary32) -
        Real.tanh (Model.toReal (Model.ofNatBits word : Model FloatFormat.binary32))| < 1/(2^22:ℝ) := by
  obtain ⟨e, f, rfl⟩ := finite_tail_coverage word hw hf
  exact ⟨tail_bits e f, decoded_tail_error e f⟩

theorem negative_native_bound (word : Nat) (hw : 0x41000000 ≤ word) (hf : word < 0x7f800000) :
    Mufu.tanh (2^31+word) = some 0xbf800000 ∧
      |Model.toReal (Model.ofNatBits 0xbf800000 : Model FloatFormat.binary32) -
        Real.tanh (Model.toReal (Model.ofNatBits (2^31+word) : Model FloatFormat.binary32))| < 1/(2^22:ℝ) := by
  obtain ⟨e, f, rfl⟩ := finite_tail_coverage word hw hf
  exact ⟨negative_tail_bits e f, decoded_negative_tail_error e f⟩

theorem decoded_boundary_error :
    1/(2^23:ℝ) < |Model.toReal (Model.ofNatBits 0x3f800000 : Model FloatFormat.binary32) -
      Real.tanh (Model.toReal (Model.ofNatBits 0x41000000 : Model FloatFormat.binary32))| := by
  have hi := Model.Algebraic.toReal_of_toRat?_eq_some boundary_decode.1
  have ho := Model.Algebraic.toReal_of_toRat?_eq_some boundary_decode.2
  norm_num at hi ho
  rw [hi, ho]
  exact boundary_lower_error

theorem decoded_tail_tight_error (e : Fin 125) (f : Fin (2^23)) :
    |Model.toReal (Model.ofNatBits 0x3f800000 : Model FloatFormat.binary32) -
      Real.tanh (Model.toReal (Model.ofNatBits (tailWord e f) : Model FloatFormat.binary32))| < 2251/10000000000 := by
  have ho := Model.Algebraic.toReal_of_toRat?_eq_some boundary_decode.2
  norm_num at ho
  rw [ho]
  exact TanhTail.positive_tail _ (tail_value_lower e f)

theorem decoded_negative_tail_tight_error (e : Fin 125) (f : Fin (2^23)) :
    |Model.toReal (Model.ofNatBits 0xbf800000 : Model FloatFormat.binary32) -
      Real.tanh (Model.toReal (Model.ofNatBits (2^31+tailWord e f) : Model FloatFormat.binary32))| < 2251/10000000000 := by
  have hd : (Model.ofNatBits 0xbf800000 : Model FloatFormat.binary32).toRat? = some (-1) := by decide +kernel
  have ho := Model.Algebraic.toReal_of_toRat?_eq_some hd
  norm_num at ho
  rw [ho, sign_decode _ (Model.Algebraic.isFinite_of_toRat?_eq_some (tail_rat e f))]
  exact TanhTail.negative_tail _ (by have h := tail_value_lower e f; linarith)

theorem decoded_boundary_tight_error :
    225/1000000000 < |Model.toReal (Model.ofNatBits 0x3f800000 : Model FloatFormat.binary32) -
      Real.tanh (Model.toReal (Model.ofNatBits 0x41000000 : Model FloatFormat.binary32))| := by
  have hi := Model.Algebraic.toReal_of_toRat?_eq_some boundary_decode.1
  have ho := Model.Algebraic.toReal_of_toRat?_eq_some boundary_decode.2
  norm_num at hi ho
  rw [hi, ho]
  exact TanhTail.tail_error_at_eight

theorem positive_relative_error (x : ℝ) (hx : 8 ≤ x) :
    |1-Real.tanh x| / |Real.tanh x| < 2251/9999997749 := by
  have ha := TanhTail.positive_tail x hx
  have hl := le_abs_self (1-Real.tanh x)
  have hp : 0 < Real.tanh x := by linarith
  rw [abs_of_pos hp]
  apply (div_lt_iff₀ hp).mpr
  nlinarith

theorem decoded_tail_relative_error (e : Fin 125) (f : Fin (2^23)) :
    let x := Model.toReal (Model.ofNatBits (tailWord e f) : Model FloatFormat.binary32)
    |Model.toReal (Model.ofNatBits 0x3f800000 : Model FloatFormat.binary32) - Real.tanh x| /
      |Real.tanh x| < 2251/9999997749 := by
  have ho := Model.Algebraic.toReal_of_toRat?_eq_some boundary_decode.2
  norm_num at ho
  dsimp only
  rw [ho]
  exact positive_relative_error _ (tail_value_lower e f)

theorem negative_relative_error (x : ℝ) (hx : 8 ≤ x) :
    |-1-Real.tanh (-x)| / |Real.tanh (-x)| < 2251/9999997749 := by
  rw [Real.tanh_neg, show (-1:ℝ)- -Real.tanh x = -(1-Real.tanh x) by ring, abs_neg, abs_neg]
  exact positive_relative_error x hx

theorem decoded_negative_tail_relative_error (e : Fin 125) (f : Fin (2^23)) :
    let x := Model.toReal (Model.ofNatBits (2^31+tailWord e f) : Model FloatFormat.binary32)
    |Model.toReal (Model.ofNatBits 0xbf800000 : Model FloatFormat.binary32) - Real.tanh x| /
      |Real.tanh x| < 2251/9999997749 := by
  have hd : (Model.ofNatBits 0xbf800000 : Model FloatFormat.binary32).toRat? = some (-1) := by decide +kernel
  have ho := Model.Algebraic.toReal_of_toRat?_eq_some hd
  norm_num at ho
  dsimp only
  rw [ho, sign_decode _ (Model.Algebraic.isFinite_of_toRat?_eq_some (tail_rat e f))]
  exact negative_relative_error _ (tail_value_lower e f)

theorem relative_below_ptx_bound : (2251:ℝ)/9999997749 < 1/(2^11:ℝ) := by norm_num

#print axioms positive_relative_error
#print axioms decoded_tail_relative_error
#print axioms negative_relative_error
#print axioms decoded_negative_tail_relative_error
#print axioms relative_below_ptx_bound
#print axioms decoded_tail_tight_error
#print axioms decoded_negative_tail_tight_error
#print axioms decoded_boundary_tight_error
#print axioms positive_native_bound
#print axioms negative_native_bound
#print axioms decoded_boundary_error
#print axioms negative_tail_bits
#print axioms sign_decode
#print axioms decoded_negative_tail_error
#print axioms finite_tail_coverage
#print axioms tail_rat
#print axioms tail_decode
#print axioms tail_value_lower
#print axioms tail_bits
#print axioms decoded_tail_error
#print axioms negative_tail_error
#print axioms tail_error
#print axioms boundary_lower_error
#print axioms native_positive_bits
#print axioms boundary_bits
#print axioms boundary_decode
end GB300Math.TanhPlateau
