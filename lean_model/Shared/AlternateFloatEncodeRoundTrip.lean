/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AlternateFloatEncodeBound
import AlternateFloatPackRoundTrip
set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 4096
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem nearestEven_le_of_lt (m d p : Nat) (hm : m < p*2^d) :
    nearestEven m d ≤ p := by
  have hq : m / 2^d < p := (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos d)).mpr hm
  unfold nearestEven
  split
  · rename_i h; subst d; simp at hm; omega
  · dsimp only; split <;> omega

theorem e4m3_encode_normal_roundtrip (m : Nat) (s : Bool)
    (hm : m ≠ 0) (hk : 506 ≤ m.log2)
    (hr : (m.log2-506)*8 + nearestEven m (m.log2-3) ≤ 126) :
    decode "e4m3" (encodeFinite "e4m3" false s m) =
      some (.finite s (nearestEven m (m.log2-3)*2^(m.log2-3))) := by
  obtain ⟨hqlo,hqhi⟩ := rounded_significand_range m 3 hm (by omega)
  norm_num at hqlo hqhi
  have helo : 1 ≤ m.log2-505 := by omega
  have hehi : m.log2-505 < 16 := by omega
  have he : (m.log2 : Int)-grid+7 = ((m.log2-505 : Nat) : Int) := by simp [grid]; omega
  have hmax : max ((m.log2 : Int)-grid+7) 1 = ((m.log2-505 : Nat) : Int) := by rw [he]; omega
  have hmax0 : max (((m.log2 : Int)-grid+7)-1) 0 = ((m.log2-506 : Nat) : Int) := by rw [he]; omega
  have hshift : (((m.log2-505 : Nat) : Int)+grid-7-3).toNat = m.log2-3 := by simp [grid]; omega
  have hraw : (m.log2-505-1)*8 + nearestEven m (m.log2-3) ≤ 126 := by omega
  have h := e4m3_normal_pack_roundtrip s ⟨m.log2-505, hehi⟩ ⟨nearestEven m (m.log2-3), by omega⟩ helo hqlo hraw
  have hp : m.log2-505+502 = m.log2-3 := by omega
  simp only [hp] at h
  simpa [encodeFinite, hm, hmax, hmax0, hshift, hr, Nat.sub_sub,
    Nat.add_assoc] using h

theorem e5m2_encode_normal_roundtrip (m : Nat) (s : Bool)
    (hm : m ≠ 0) (hk : 498 ≤ m.log2)
    (hr : (m.log2-498)*4 + nearestEven m (m.log2-2) ≤ 123) :
    decode "e5m2" (encodeFinite "e5m2" false s m) =
      some (.finite s (nearestEven m (m.log2-2)*2^(m.log2-2))) := by
  obtain ⟨hqlo,hqhi⟩ := rounded_significand_range m 2 hm (by omega)
  norm_num at hqlo hqhi
  have helo : 1 ≤ m.log2-497 := by omega
  have hehi : m.log2-497 < 31 := by omega
  have he : (m.log2 : Int)-grid+15 = ((m.log2-497 : Nat) : Int) := by simp [grid]; omega
  have hmax : max ((m.log2 : Int)-grid+15) 1 = ((m.log2-497 : Nat) : Int) := by rw [he]; omega
  have hmax0 : max (((m.log2 : Int)-grid+15)-1) 0 = ((m.log2-498 : Nat) : Int) := by rw [he]; omega
  have hshift : (((m.log2-497 : Nat) : Int)+grid-15-2).toNat = m.log2-2 := by simp [grid]; omega
  have hraw : (m.log2-497-1)*4 + nearestEven m (m.log2-2) ≤ 123 := by omega
  have h := e5m2_normal_pack_roundtrip s ⟨m.log2-497, hehi⟩ ⟨nearestEven m (m.log2-2), by omega⟩ helo hqlo hraw
  have hp : m.log2-497+495 = m.log2-2 := by omega
  simp only [hp] at h
  simpa [encodeFinite, hm, hmax, hmax0, hshift, hr, Nat.sub_sub,
    Nat.add_assoc] using h


theorem e4m3_encode_subnormal_roundtrip (m : Nat) (s : Bool)
    (hk : m.log2 ≤ 505) :
    decode "e4m3" (encodeFinite "e4m3" false s m) =
      some (.finite s (nearestEven m 503 * 2^503)) := by
  by_cases hm : m = 0
  · subst m; cases s <;> decide +kernel
  have hmhi : m < 2^506 := (Nat.log2_lt hm).mp (by omega)
  have hqhi : nearestEven m 503 ≤ 8 := nearestEven_le_of_lt m 503 8 (by norm_num at hmhi ⊢; omega)
  have hmax : max ((m.log2 : Int)-512+7) 1 = 1 := by omega
  have hmax0 : max (((m.log2 : Int)-512+7)-1) 0 = 0 := by omega
  have hr : nearestEven m 503 ≤ 126 := by omega
  have h := e4m3_subnormal_pack_roundtrip s ⟨nearestEven m 503, by omega⟩
  simpa [encodeFinite, hm, grid, hmax, hmax0, hr] using h

theorem e5m2_encode_subnormal_roundtrip (m : Nat) (s : Bool)
    (hk : m.log2 ≤ 497) :
    decode "e5m2" (encodeFinite "e5m2" false s m) =
      some (.finite s (nearestEven m 496 * 2^496)) := by
  by_cases hm : m = 0
  · subst m; cases s <;> decide +kernel
  have hmhi : m < 2^498 := (Nat.log2_lt hm).mp (by omega)
  have hqhi : nearestEven m 496 ≤ 4 := nearestEven_le_of_lt m 496 4 (by norm_num at hmhi ⊢; omega)
  have hmax : max ((m.log2 : Int)-512+15) 1 = 1 := by omega
  have hmax0 : max (((m.log2 : Int)-512+15)-1) 0 = 0 := by omega
  have hr : nearestEven m 496 ≤ 123 := by omega
  have h := e5m2_subnormal_pack_roundtrip s ⟨nearestEven m 496, by omega⟩
  simpa [encodeFinite, hm, grid, hmax, hmax0, hr] using h

#print axioms nearestEven_le_of_lt
#print axioms e4m3_encode_normal_roundtrip
#print axioms e5m2_encode_normal_roundtrip
#print axioms e4m3_encode_subnormal_roundtrip
#print axioms e5m2_encode_subnormal_roundtrip
end GB300Math.AlternateFloatFloatLib
