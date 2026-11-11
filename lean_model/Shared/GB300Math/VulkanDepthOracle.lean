/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureDepth24

set_option maxRecDepth 10000

namespace GB300Math.VulkanDepthOracle

open TextureDepthCompare

def cuda16 (word : Nat) : Nat :=
  if word / 2^31 = 1 || isNaN word then 0
  else if word ≥ 0x3f800000 then 65535
  else if word / 2^23 < 107 then 0
  else ((2^23 + word % 2^23) / 2^(130-word/2^23) * 131070 + 1048575) / 2^21

theorem cuda16_reference (word : Nat) :
    cuda16 word = TextureDepthNormalized.reference word := by
  unfold cuda16 TextureDepthNormalized.reference TextureDepthNormalized.fixed20
  split
  · simp [TextureDepthNormalized.quantize]
  · split
    · decide +kernel
    · rename_i hsign hlarge
      by_cases he : word / 2^23 < 107
      · have hm := Nat.mod_lt word (by decide : 0 < 2^23)
        have hp : 2^24 ≤ 2^(130-word/2^23) :=
          Nat.pow_le_pow_right (by decide) (by omega)
        have hz : (2^23 + word % 2^23) / 2^(130-word/2^23) = 0 :=
          Nat.div_eq_of_lt (by omega)
        simp only [he, ↓reduceIte]
        split <;> simp [TextureDepthNormalized.quantize, hz]
      · have hn : word / 2^23 ≠ 0 := by omega
        simp only [he, hn, ↓reduceIte, TextureDepthNormalized.quantize]
        have hk (k : Nat) : k*131070+1048575 = 2*k*65535+2^20-1 := by omega
        exact congrArg (fun n => n / 2^21) (hk _)

theorem cuda16_product_fits (fixed : Nat) (h : fixed ≤ 2^20) :
    fixed * 131070 + 1048575 < 2^64 := by omega

theorem halfUp_shift (m s : Nat) (hs : 0 < s) :
    (2*m + 2^s)/(2*2^s) = (m+2^(s-1))/2^s := by
  have hp : 2^s = 2 * 2^(s-1) := by
    calc
      2^s = 2^((s-1)+1) := by congr 1; omega
      _ = 2^(s-1)*2 := Nat.pow_succ ..
      _ = 2*2^(s-1) := Nat.mul_comm ..
  rw [hp]
  have hn : 2*m+2*2^(s-1) = 2*(m+2^(s-1)) := by omega
  rw [hn]
  exact Nat.mul_div_mul_left _ _ (by decide : 0<2)

theorem halfUp_scale (m s : Nat) :
    (2*m*2^24 + 2^(24+s))/(2*2^(24+s)) = (2*m+2^s)/(2*2^s) := by
  rw [Nat.pow_add]
  have hn : 2*m*2^24+2^24*2^s = 2^24*(2*m+2^s) := by omega
  have hd : 2*(2^24*2^s) = 2^24*(2*2^s) := by omega
  rw [hn,hd]
  exact Nat.mul_div_mul_left _ _ (by decide : 0<2^24)

def cuda24 (word : Nat) : Nat :=
  if word / 2^31 = 1 || TextureDepthCompare.isNaN word then 0
  else if word ≥ 0x3f800000 then 0xffffff
  else if word / 2^23 < 102 then 0
  else
    let m := 2^23 + word % 2^23
    if word < 0x3d800000 then
      (m+2^(126-word/2^23-1))/2^(126-word/2^23)
    else
      let n := m * 0xffffff
      let s := 150-word/2^23
      n / 2^s + if n % 2^s > 2^(s-1) ||
        (n % 2^s == 2^(s-1) && (n / 2^s) % 2 == 1) then 1 else 0

theorem nearestEven_halfway (n s : Nat) (hs : 0<s) :
    TextureDepth24.nearestEven n (2^s) =
      n/2^s + if n%2^s > 2^(s-1) ||
        (n%2^s == 2^(s-1) && (n/2^s)%2 == 1) then 1 else 0 := by
  have hp : 2^s = 2*2^(s-1) := by
    calc
      2^s = 2^((s-1)+1) := by congr 1; omega
      _ = 2^(s-1)*2 := Nat.pow_succ ..
      _ = 2*2^(s-1) := Nat.mul_comm ..
  unfold TextureDepth24.nearestEven
  dsimp only
  split <;> split <;>
    simp_all only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  all_goals omega

theorem cuda24_reference (word : Nat) :
    cuda24 word = TextureDepth24.reference word := by
  unfold cuda24 TextureDepth24.reference
  split
  · rfl
  · split
    · rfl
    · rename_i hsign hlarge
      have he : word / 2^23 < 127 := by omega
      have hm : word % 2^23 < 2^23 := Nat.mod_lt _ (by decide)
      by_cases hz : word / 2^23 = 0
      · simp [hz]
      · simp only [hz, ↓reduceIte]
        by_cases hl : word < 0x3d800000
        · have hel : word / 2^23 < 123 := by omega
          have hp : 150-word/2^23 = 24+(126-word/2^23) := by omega
          simp only [hl, ↓reduceIte, hp]
          rw [halfUp_scale, halfUp_shift _ _ (by omega)]
          split
          · rename_i ht
            have hs : 25 ≤ 126-word/2^23 := by omega
            have hp2 : 2^24 ≤ 2^(126-word/2^23-1) :=
              Nat.pow_le_pow_right (by decide) (by omega)
            have hp3 : 2^(126-word/2^23) = 2*2^(126-word/2^23-1) := by
              calc
                _ = 2^((126-word/2^23-1)+1) := by congr 1; omega
                _ = _ := by rw [Nat.pow_succ, Nat.mul_comm]
            symm
            exact Nat.div_eq_of_lt (by omega)
          · rfl
        · have ht : ¬ word/2^23 < 102 := by omega
          simp only [hl, ht, ↓reduceIte]
          exact (nearestEven_halfway _ _ (by omega)).symm

theorem cuda24_product_fits (m : Nat) (h : m < 2^24) :
    m*0xffffff < 2^48 := by omega


theorem cuda16_shift_safe (word : Nat) (h : word < 0x3f800000)
    (he : 107 ≤ word/2^23) : 4 ≤ 130-word/2^23 ∧ 130-word/2^23 ≤ 23 := by omega

theorem cuda24_low_shift_safe (word : Nat) (h : word < 0x3d800000)
    (he : 102 ≤ word/2^23) : 4 ≤ 126-word/2^23 ∧ 126-word/2^23 ≤ 24 := by omega

theorem cuda24_high_shift_safe (word : Nat) (hl : 0x3d800000 ≤ word)
    (hh : word < 0x3f800000) : 24 ≤ 150-word/2^23 ∧ 150-word/2^23 ≤ 27 := by omega

theorem d16_half_source : cuda16 0x3f000000 = 32767 := by decide +kernel

theorem d24_first_tie_source : cuda24 0x33000000 = 1 := by decide +kernel

end GB300Math.VulkanDepthOracle
