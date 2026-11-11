/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.SmallConversionNative
import GB300Math.SmallConversionNativeSm103

namespace GB300Math.SmallConversionNativeSm103
open SmallConversions

theorem nearest_unchanged (fmt : Format) (relu sat : Bool) (x : Nat) :
    downNative fmt "rn" relu sat x = SmallConversionNative.downNative fmt "rn" relu sat x := by
  simp [downNative, SmallConversionNative.downNative, roundNative, SmallConversionNative.roundNative]

theorem nonnegative_unchanged (fmt : Format) (mode : String) (relu sat : Bool) (x : Nat)
    (h : x / 2^31 % 2 ≠ 1) :
    downNative fmt mode relu sat x = SmallConversionNative.downNative fmt mode relu sat x := by
  simp [downNative, SmallConversionNative.downNative, roundNative, SmallConversionNative.roundNative, h]

theorem saturated_unchanged (fmt : Format) (mode : String) (relu : Bool) (x : Nat) :
    downNative fmt mode relu true x = SmallConversionNative.downNative fmt mode relu true x := by
  simp [downNative, SmallConversionNative.downNative, roundNative, SmallConversionNative.roundNative]

def finiteCode (fmt : Format) (mode : String) (negative : Bool) (mag : Nat) : Nat :=
  let e := mag / 2^23
  let m := mag % 2^23 + if e = 0 then 0 else 2^23
  let target : Int := (max e 1 : Nat) - 127 + (fmt.bias : Int)
  let shift := 23 - (fmt.fractionBits : Int) + max (1-target) 0
  (max (target-1) 0).toNat * 2^fmt.fractionBits + roundNative mode negative m shift

theorem negative_finite_overflow (fmt : Format) (sat : Bool) (x : Nat)
    (hfinite : x % 2^31 < 0x7f800000) (hnegative : x / 2^31 % 2 = 1)
    (hoverflow : finiteCode fmt "rp" true (x % 2^31) > fmt.limit) :
    downNative fmt "rp" false sat x = 2^(fmt.exponentBits + fmt.fractionBits) + fmt.limit := by
  have hn : ¬x % 2^31 > 0x7f800000 := by omega
  have hi : x % 2^31 ≠ 0x7f800000 := by omega
  simp [finiteCode] at hoverflow
  simp [downNative, hn, hi, hnegative]
  omega

theorem positive_finite_overflow_e4 (x : Nat)
    (hfinite : x % 2^31 < 0x7f800000) (hpositive : x / 2^31 % 2 ≠ 1)
    (hoverflow : finiteCode ⟨4,3,7,126⟩ "rp" false (x % 2^31) > 126) :
    downNative ⟨4,3,7,126⟩ "rp" false false x = 127 := by
  have hn : ¬x % 2^31 > 0x7f800000 := by omega
  have hi : x % 2^31 ≠ 0x7f800000 := by omega
  simp [finiteCode] at hoverflow
  simp [downNative, hn, hi, hpositive]
  omega

theorem positive_finite_overflow_e5 (x : Nat)
    (hfinite : x % 2^31 < 0x7f800000) (hpositive : x / 2^31 % 2 ≠ 1)
    (hoverflow : finiteCode ⟨5,2,15,123⟩ "rp" false (x % 2^31) > 123) :
    downNative ⟨5,2,15,123⟩ "rp" false false x = 124 := by
  have hn : ¬x % 2^31 > 0x7f800000 := by omega
  have hi : x % 2^31 ≠ 0x7f800000 := by omega
  simp [finiteCode] at hoverflow
  simp [downNative, hn, hi, hpositive]
  omega

theorem source_nan_e4 (mode : String) (relu sat : Bool) (x : Nat)
    (h : x % 2^31 > 0x7f800000) : downNative ⟨4,3,7,126⟩ mode relu sat x = 127 := by
  simp [downNative, h]

theorem source_nan_e5 (mode : String) (relu sat : Bool) (x : Nat)
    (h : x % 2^31 > 0x7f800000) : downNative ⟨5,2,15,123⟩ mode relu sat x = 127 := by
  simp [downNative, h]

theorem source_infinity_e4 (mode : String) (x : Nat)
    (h : x % 2^31 = 0x7f800000) : downNative ⟨4,3,7,126⟩ mode false false x = 127 := by
  simp [downNative, h]

theorem source_negative_infinity_e5 (mode : String) (x : Nat)
    (h : x % 2^31 = 0x7f800000) (hs : x / 2^31 % 2 = 1) :
    downNative ⟨5,2,15,123⟩ mode false false x = 252 := by
  simp [downNative, h, hs]

def LeastUpwardFinite (representable : Int → Prop) (x result : Int) : Prop :=
  representable result ∧ x ≤ result ∧ ∀ y, representable y → x ≤ y → result ≤ y

theorem negative_finite_clamp (fmt : Format) (sat : Bool) (x : Nat)
    (hfinite : x % 2^31 < 0x7f800000) (hnegative : x / 2^31 % 2 = 1) :
    downNative fmt "rp" false sat x = 2^(fmt.exponentBits + fmt.fractionBits) +
      min (finiteCode fmt "rp" true (x % 2^31)) fmt.limit := by
  have hn : ¬x % 2^31 > 0x7f800000 := by omega
  have hi : x % 2^31 ≠ 0x7f800000 := by omega
  by_cases hv : fmt.limit < finiteCode fmt "rp" true (x % 2^31)
  · have hm := Nat.min_eq_right (Nat.le_of_lt hv)
    simp [finiteCode] at hv hm
    simp [downNative, finiteCode, hn, hi, hnegative, hv, hm]
  · have hm := Nat.min_eq_left (Nat.le_of_not_gt hv)
    simp [finiteCode] at hv hm
    simp [downNative, finiteCode, hn, hi, hnegative, hm]
    omega

theorem round_up_negative (m : Nat) (shift : Int) (hd : 0 < shift) :
    roundNative "rp" true m shift = m / 2^shift.toNat := by
  have hd' : ¬shift ≤ 0 := by omega
  have hmode : ("rp" : String) ≠ "rn" := by decide
  simp only [roundNative, hd', hmode, ↓reduceIte]
  simp only [Bool.not_true, Bool.false_eq_true, and_false, false_and, ↓reduceIte,
    Nat.add_zero]

theorem e4_negative_at_or_beyond_max (x : Nat) (sat : Bool)
    (hfinite : x % 2^31 < 0x7f800000) (hnegative : x / 2^31 % 2 = 1)
    (hlarge : 0x43e00000 ≤ x % 2^31) : downNative ⟨4,3,7,126⟩ "rp" false sat x = 254 := by
  rw [negative_finite_clamp _ _ _ hfinite hnegative]
  clear hfinite hnegative
  generalize hmag : x % 2^31 = mag at hlarge ⊢
  clear hmag x
  have he : 135 ≤ mag / 2^23 := by omega
  have ht : 1 ≤ ((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7 := by omega
  have hezero : mag / 2^23 ≠ 0 := by omega
  have hmode : ("rp" : String) ≠ "rn" := by decide
  have htarget : max (1 - (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7)) 0 = 0 := by omega
  have hvalue : 126 ≤ finiteCode ⟨4,3,7,126⟩ "rp" true (mag) := by
    dsimp only [finiteCode]
    simp only [ite_eq_right hezero]
    change 126 ≤ (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7 - 1) 0).toNat * 8 + roundNative "rp" true (mag % 2^23 + 8388608) (20 + max (1 - (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7)) 0)
    rw [htarget]
    rw [round_up_negative _ (20+0) (by decide)]
    change 126 ≤ (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7 - 1) 0).toNat * 8 + (mag % 2^23 + 8388608) / 1048576
    by_cases heq : mag / 2^23 = 135
    · have hf : 6291456 ≤ mag % 2^23 := by omega
      simp only [heq]
      omega
    · have hh : 15 ≤ (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7 - 1) 0).toNat := by omega
      omega
  simp only [Nat.min_eq_right hvalue]

theorem e5_negative_at_or_beyond_max (x : Nat) (sat : Bool)
    (hfinite : x % 2^31 < 0x7f800000) (hnegative : x / 2^31 % 2 = 1)
    (hlarge : 0x47600000 ≤ x % 2^31) : downNative ⟨5,2,15,123⟩ "rp" false sat x = 251 := by
  rw [negative_finite_clamp _ _ _ hfinite hnegative]
  clear hfinite hnegative
  generalize hmag : x % 2^31 = mag at hlarge ⊢
  clear hmag x
  have he : 142 ≤ mag / 2^23 := by omega
  have ht : 1 ≤ ((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15 := by omega
  have hezero : mag / 2^23 ≠ 0 := by omega
  have hmode : ("rp" : String) ≠ "rn" := by decide
  have htarget : max (1 - (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15)) 0 = 0 := by omega
  have hvalue : 123 ≤ finiteCode ⟨5,2,15,123⟩ "rp" true (mag) := by
    dsimp only [finiteCode]
    simp only [ite_eq_right hezero]
    change 123 ≤ (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15 - 1) 0).toNat * 4 + roundNative "rp" true (mag % 2^23 + 8388608) (21 + max (1 - (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15)) 0)
    rw [htarget]
    rw [round_up_negative _ (21+0) (by decide)]
    change 123 ≤ (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15 - 1) 0).toNat * 4 + (mag % 2^23 + 8388608) / 2097152
    by_cases heq : mag / 2^23 = 142
    · have hf : 6291456 ≤ mag % 2^23 := by omega
      simp only [heq]
      omega
    · have hh : 30 ≤ (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15 - 1) 0).toNat := by omega
      omega
  simp only [Nat.min_eq_right hvalue]

theorem round_up_positive (m : Nat) (shift : Int) (hd : 0 < shift) :
    roundNative "rp" false m shift = m / 2^shift.toNat +
      if m % 2^shift.toNat ≠ 0 then 1 else 0 := by
  have hd' : ¬shift ≤ 0 := by omega
  have hmode : ("rp" : String) ≠ "rn" := by decide
  simp only [roundNative, hd', hmode, ↓reduceIte]
  simp only [Bool.not_false, true_and]

theorem e4_positive_beyond_max (x : Nat)
    (hfinite : x % 2^31 < 0x7f800000) (hpositive : x / 2^31 % 2 ≠ 1)
    (hlarge : 0x43e00000 < x % 2^31) : downNative ⟨4,3,7,126⟩ "rp" false false x = 127 := by
  apply positive_finite_overflow_e4 x hfinite hpositive
  clear hfinite hpositive
  generalize hmag : x % 2^31 = mag at hlarge ⊢
  clear hmag x
  have he : 135 ≤ mag / 2^23 := by omega
  have hezero : mag / 2^23 ≠ 0 := by omega
  have htarget : max (1 - (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7)) 0 = 0 := by omega
  dsimp only [finiteCode]
  simp only [ite_eq_right hezero]
  change 126 < (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7 - 1) 0).toNat * 8 + roundNative "rp" false (mag % 2^23 + 8388608) (20 + max (1 - (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7)) 0)
  rw [htarget, round_up_positive _ (20+0) (by decide)]
  change 126 < (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7 - 1) 0).toNat * 8 + ((mag % 2^23 + 8388608) / 1048576 + (if (mag % 2^23 + 8388608) % 1048576 ≠ 0 then 1 else 0))
  by_cases heq : mag / 2^23 = 135
  · have hf : 6291456 < mag % 2^23 := by omega
    simp only [heq]
    split <;> omega
  · have hh : 15 ≤ (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 7 - 1) 0).toNat := by omega
    split <;> omega

theorem e5_positive_beyond_max (x : Nat)
    (hfinite : x % 2^31 < 0x7f800000) (hpositive : x / 2^31 % 2 ≠ 1)
    (hlarge : 0x47600000 < x % 2^31) : downNative ⟨5,2,15,123⟩ "rp" false false x = 124 := by
  apply positive_finite_overflow_e5 x hfinite hpositive
  clear hfinite hpositive
  generalize hmag : x % 2^31 = mag at hlarge ⊢
  clear hmag x
  have he : 142 ≤ mag / 2^23 := by omega
  have hezero : mag / 2^23 ≠ 0 := by omega
  have htarget : max (1 - (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15)) 0 = 0 := by omega
  dsimp only [finiteCode]
  simp only [ite_eq_right hezero]
  change 123 < (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15 - 1) 0).toNat * 4 + roundNative "rp" false (mag % 2^23 + 8388608) (21 + max (1 - (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15)) 0)
  rw [htarget, round_up_positive _ (21+0) (by decide)]
  change 123 < (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15 - 1) 0).toNat * 4 + ((mag % 2^23 + 8388608) / 2097152 + (if (mag % 2^23 + 8388608) % 2097152 ≠ 0 then 1 else 0))
  by_cases heq : mag / 2^23 = 142
  · have hf : 6291456 < mag % 2^23 := by omega
    simp only [heq]
    split <;> omega
  · have hh : 30 ≤ (max (((max (mag / 2^23) 1 : Nat) : Int) - 127 + 15 - 1) 0).toNat := by omega
    split <;> omega

theorem negative_overflow_least_upward (representable : Int → Prop) (limit x : Int)
    (hlimit : representable (-limit)) (hbound : ∀ y, representable y → -limit ≤ y)
    (hoverflow : x < -limit) : LeastUpwardFinite representable x (-limit) := by
  exact ⟨hlimit, by omega, fun y hy _ => hbound y hy⟩

theorem positive_overflow_no_finite_upward (representable : Int → Prop) (limit x : Int)
    (hbound : ∀ y, representable y → y ≤ limit) (hoverflow : limit < x) :
    ¬∃ y, representable y ∧ x ≤ y := by
  rintro ⟨y, hy, hxy⟩
  have := hbound y hy
  omega

theorem negative_rp_quantum_bound (m d : Nat) (hd : 0 < d) :
    let q := roundNative "rp" true m (d : Int)
    q * 2^d ≤ m ∧ m < (q + 1) * 2^d := by
  have hp := Nat.two_pow_pos d
  have hr := Nat.mod_lt m hp
  have hm := Nat.mod_add_div m (2^d)
  have hd' : ¬(d : Int) ≤ 0 := by omega
  have hmode : ("rp" : String) ≠ "rn" := by decide
  simp only [roundNative, hd', hmode, ↓reduceIte, Int.toNat_natCast]
  simp only [Bool.not_true, Bool.false_eq_true, and_false, false_and, ↓reduceIte,
    Nat.add_zero, Nat.add_mul, Nat.one_mul]
  rw [Nat.mul_comm (2^d) (m / 2^d)] at hm
  omega

#print axioms nearest_unchanged
#print axioms nonnegative_unchanged
#print axioms saturated_unchanged
#print axioms negative_finite_overflow
#print axioms positive_finite_overflow_e4
#print axioms positive_finite_overflow_e5
#print axioms source_nan_e4
#print axioms source_nan_e5
#print axioms source_infinity_e4
#print axioms source_negative_infinity_e5
#print axioms negative_overflow_least_upward
#print axioms positive_overflow_no_finite_upward
#print axioms negative_rp_quantum_bound
#print axioms negative_finite_clamp
#print axioms e4_negative_at_or_beyond_max
#print axioms e5_negative_at_or_beyond_max
#print axioms round_up_negative
#print axioms round_up_positive
#print axioms e4_positive_beyond_max
#print axioms e5_positive_beyond_max
end GB300Math.SmallConversionNativeSm103
