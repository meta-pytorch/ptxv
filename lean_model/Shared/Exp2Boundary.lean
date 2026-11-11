/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Exp2Bounds

namespace GB300Math.Exp2Boundary
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

def positiveFixed (word : Nat) : Int :=
  let e := Mufu.exponent word
  let m : Int := (Mufu.fraction word : Int)+2^23
  if e = 0 then 0 else if e ≥ 127 then m*2^(e-127)
    else if 127-e < 24 then m/2^(127-e) else 0

def signedFixed (word : Nat) : Int :=
  let q := positiveFixed word
  if Mufu.sign word ≠ 0 then -q-(if q%2^23 ≠ 0 then 1 else 0) else q

def outputExponent (word : Nat) : Int := signedFixed word/2^23+127

theorem fixed_nonnegative (word : Nat) : 0 ≤ positiveFixed word := by
  dsimp [positiveFixed]
  split_ifs <;> positivity

theorem positive_fixed_high (word : Nat) (he : 127 ≤ Mufu.exponent word) :
    positiveFixed word = (Mufu.fraction word+2^23 : Nat)*2^(Mufu.exponent word-127) := by
  have hn : Mufu.exponent word ≠ 0 := by omega
  simp [positiveFixed,hn,he,Nat.cast_add]

theorem positive_fixed_low (word : Nat) (he : Mufu.exponent word ≤ 126) :
    positiveFixed word < 2^23 := by
  have hf : Mufu.fraction word < 2^23 := by dsimp [Mufu.fraction];omega
  dsimp [positiveFixed]
  split_ifs with hz hhigh hsmall
  · norm_num
  · omega
  · have hp : (2 : Int) ≤ 2^(127-Mufu.exponent word) := by
      have h := pow_le_pow_right₀ (by norm_num : (1 : Int) ≤ 2) (show 1 ≤ 127-Mufu.exponent word by omega)
      simpa using h
    have hm : (Mufu.fraction word : Int)+2^23 < 2^24 := by omega
    exact (Int.ediv_lt_iff_lt_mul (by omega)).mpr (by nlinarith)
  · norm_num

theorem magnitude_fields (word : Nat) :
    Mufu.magnitude word = Mufu.exponent word*2^23+Mufu.fraction word := by
  dsimp [Mufu.magnitude,Mufu.exponent,Mufu.fraction]
  omega

theorem fixed_below64 (word : Nat) (he : Mufu.exponent word ≤ 132) :
    positiveFixed word < 64*2^23 := by
  by_cases hlow : Mufu.exponent word ≤ 126
  · have h := positive_fixed_low word hlow
    omega
  · rw [positive_fixed_high word (by omega)]
    have hp : (2 : Int)^(Mufu.exponent word-127) ≤ 32 := by
      have h := pow_le_pow_right₀ (by norm_num : (1 : Int) ≤ 2) (show Mufu.exponent word-127 ≤ 5 by omega)
      simpa using h
    have hm : ((Mufu.fraction word+2^23 : Nat) : Int) < 2^24 := by
      have hf : Mufu.fraction word < 2^23 := by dsimp [Mufu.fraction];omega
      omega
    have hh := mul_le_mul_of_nonneg_left hp (Int.natCast_nonneg (Mufu.fraction word+2^23))
    nlinarith

theorem fixed_threshold126 (word : Nat) (he : Mufu.exponent word ≤ 134) :
    (126*2^23 < positiveFixed word ↔ 0x42fc0000 < Mufu.magnitude word) := by
  rw [magnitude_fields]
  have hf : Mufu.fraction word < 2^23 := by dsimp [Mufu.fraction];omega
  by_cases hlo : Mufu.exponent word ≤ 132
  · have h := fixed_below64 word hlo
    omega
  · have hh : Mufu.exponent word = 133 ∨ Mufu.exponent word = 134 := by omega
    rw [positive_fixed_high word (by omega)]
    rcases hh with hh|hh <;> rw [hh] <;> norm_num <;> omega

theorem fixed_threshold128 (word : Nat) (he : Mufu.exponent word ≤ 134) :
    (128*2^23 ≤ positiveFixed word ↔ 0x43000000 ≤ Mufu.magnitude word) := by
  rw [magnitude_fields]
  have hf : Mufu.fraction word < 2^23 := by dsimp [Mufu.fraction];omega
  by_cases hlo : Mufu.exponent word ≤ 132
  · have h := fixed_below64 word hlo
    omega
  · have hh : Mufu.exponent word = 133 ∨ Mufu.exponent word = 134 := by omega
    rw [positive_fixed_high word (by omega)]
    rcases hh with hh|hh <;> rw [hh] <;> norm_num <;> omega

theorem exponent_underflows (word : Nat) (he : Mufu.exponent word ≤ 134) :
    (outputExponent word ≤ 0 ↔ Mufu.sign word ≠ 0 ∧ 0x42fc0000 < Mufu.magnitude word) := by
  have hp := fixed_nonnegative word
  have ht := fixed_threshold126 word he
  dsimp [outputExponent,signedFixed]
  split_ifs <;> omega

theorem exponent_overflows (word : Nat) (he : Mufu.exponent word ≤ 134) :
    (255 ≤ outputExponent word ↔ Mufu.sign word = 0 ∧ 0x43000000 ≤ Mufu.magnitude word) := by
  have hp := fixed_nonnegative word
  have ht := fixed_threshold128 word he
  dsimp [outputExponent,signedFixed]
  split_ifs <;> omega

def pack (be : Int) (mantissa : Nat) : Nat :=
  if be ≤ 0 then 0 else if be ≥ 255 then 0x7f800000
  else be.toNat*2^23+mantissa%2^23

theorem pack_zero (be : Int) (mantissa : Nat) : pack be mantissa = 0 ↔ be ≤ 0 := by
  have hm : mantissa%2^23 < 2^23 := Nat.mod_lt _ (by decide)
  unfold pack
  split_ifs <;> simp_all

theorem pack_infinite (be : Int) (mantissa : Nat) : pack be mantissa = 0x7f800000 ↔ 255 ≤ be := by
  have hm : mantissa%2^23 < 2^23 := Nat.mod_lt _ (by decide)
  unfold pack
  split_ifs <;> simp_all <;> omega

theorem ex2_pack (word : Nat) (he : Mufu.exponent word ≤ 134)
    (hf : Mufu.magnitude word < 0x7f800000) :
    Mufu.ex2 word = pack (outputExponent word)
      (Mufu.interpolate MufuTables.ex2 ((signedFixed word)%2^23).toNat).toNat := by
  have hhi : ¬135 ≤ Mufu.exponent word := by omega
  simp only [Mufu.ex2,not_lt.mpr hf.le,ite_false,hhi]
  rfl

theorem zero_iff (word : Nat) (hf : Mufu.magnitude word < 0x7f800000) :
    (Mufu.ex2 word = 0 ↔ Mufu.sign word ≠ 0 ∧ 0x42fc0000 < Mufu.magnitude word) := by
  by_cases he : Mufu.exponent word ≤ 134
  · rw [ex2_pack word he hf,pack_zero,exponent_underflows word he]
  · have hm : 0x42fc0000 < Mufu.magnitude word := by rw [magnitude_fields];omega
    have hh : 135 ≤ Mufu.exponent word := by omega
    simp [Mufu.ex2,not_lt.mpr hf.le,hh,hm]

theorem infinite_iff (word : Nat) (hf : Mufu.magnitude word < 0x7f800000) :
    (Mufu.ex2 word = 0x7f800000 ↔ Mufu.sign word = 0 ∧ 0x43000000 ≤ Mufu.magnitude word) := by
  by_cases he : Mufu.exponent word ≤ 134
  · rw [ex2_pack word he hf,pack_infinite,exponent_overflows word he]
  · have hm : 0x43000000 ≤ Mufu.magnitude word := by rw [magnitude_fields];omega
    have hh : 135 ≤ Mufu.exponent word := by omega
    simp [Mufu.ex2,not_lt.mpr hf.le,hh,hm]

theorem sign_zero_iff (word : Nat) (hw : word < 2^32) :
    (Mufu.sign word = 0 ↔ word < 2^31) := by
  have hs : word &&& 2^31 ≤ 2^31 := Nat.and_le_right
  have hd := Nat.and_div_two_pow (a := word) (b := 2^31) (n := 31)
  have hm := Nat.and_mod_two_pow (a := word) (b := 2^31) (n := 31)
  have hq : word/2^31 = 0 ∨ word/2^31 = 1 := by omega
  dsimp [Mufu.sign]
  rcases hq with hq|hq <;> rw [hq] at hd <;> norm_num at hd hm <;> omega

theorem zero_bit_range (word : Nat) (hw : word < 2^32)
    (hf : Mufu.magnitude word < 0x7f800000) :
    (Mufu.ex2 word = 0 ↔ 0xc2fc0000 < word) := by
  rw [zero_iff word hf]
  have hs := sign_zero_iff word hw
  dsimp [Mufu.magnitude] at *
  omega

theorem infinite_bit_range (word : Nat) (hw : word < 2^32)
    (hf : Mufu.magnitude word < 0x7f800000) :
    (Mufu.ex2 word = 0x7f800000 ↔ 0x43000000 ≤ word ∧ word < 2^31) := by
  rw [infinite_iff word hf,sign_zero_iff word hw]
  dsimp [Mufu.magnitude]
  omega

theorem pack_class (be : Int) (mantissa : Nat) :
    pack be mantissa = 0 ∨ (0x00800000 ≤ pack be mantissa ∧ pack be mantissa < 0x7f800000) ∨ pack be mantissa = 0x7f800000 := by
  have hm : mantissa%2^23 < 2^23 := Nat.mod_lt _ (by decide)
  unfold pack
  split_ifs <;> simp_all
  omega

theorem no_subnormal_output (word : Nat) (hf : Mufu.magnitude word < 0x7f800000) :
    Mufu.ex2 word = 0 ∨ (0x00800000 ≤ Mufu.ex2 word ∧ Mufu.ex2 word < 0x7f800000) ∨ Mufu.ex2 word = 0x7f800000 := by
  by_cases he : Mufu.exponent word ≤ 134
  · rw [ex2_pack word he hf]
    exact pack_class _ _
  · have hh : 135 ≤ Mufu.exponent word := by omega
    by_cases hs : Mufu.sign word = 0 <;> simp [Mufu.ex2,not_lt.mpr hf.le,hh,hs]

theorem subnormal_inputs_return_one (word : Nat) (hw : word < 2^32)
    (he : Mufu.exponent word = 0) : Mufu.ex2 word = 0x3f800000 :=
  Exp2Bounds.ex2_small_bits word hw (by omega)

theorem zero_range_relative_error_one (word : Nat) (hw : word < 2^32)
    (hf : Mufu.magnitude word < 0x7f800000) (hz : 0xc2fc0000 < word) :
    |Exp2Bounds.finiteValue (Mufu.ex2 word)-(2 : ℝ)^(Exp2Bounds.finiteValue word)| /
      |(2 : ℝ)^(Exp2Bounds.finiteValue word)| = 1 := by
  rw [(zero_bit_range word hw hf).mpr hz]
  have hv : Exp2Bounds.finiteValue 0 = 0 := by norm_num [Exp2Bounds.finiteValue,Mufu.exponent,Mufu.fraction]
  rw [hv,zero_sub,abs_neg]
  exact div_self (abs_ne_zero.mpr (ne_of_gt (Real.rpow_pos_of_pos (by norm_num) _)))

theorem next_negative_flush : Mufu.ex2 0xc2fc0001 = 0 := by decide +kernel
theorem negative_boundary_normal : Mufu.ex2 0xc2fc0000 = 0x00800000 := by decide +kernel
theorem previous_negative_nonzero : Mufu.ex2 0xc2fbffff ≠ 0 := by decide +kernel
theorem positive_boundary_infinite : Mufu.ex2 0x43000000 = 0x7f800000 := by decide +kernel
theorem previous_positive_finite : Mufu.ex2 0x42ffffff < 0x7f800000 := by decide +kernel

#print axioms zero_iff
#print axioms infinite_iff
#print axioms zero_bit_range
#print axioms infinite_bit_range
#print axioms no_subnormal_output
#print axioms subnormal_inputs_return_one
#print axioms zero_range_relative_error_one
#print axioms next_negative_flush
#print axioms negative_boundary_normal
#print axioms previous_negative_nonzero
#print axioms positive_boundary_infinite
#print axioms previous_positive_finite
end GB300Math.Exp2Boundary
