/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureDepth24
import GB300Math.TextureDepthNormalizedBounds

namespace GB300Math.TextureDepth24Bounds

open TextureDepth24 TextureDepthNormalizedBounds

def halfUp (n d : Nat) : Nat := (2*n+d)/(2*d)

theorem halfUp_scaled (n d g : Nat) (hg : 0 < g) :
    halfUp (n*g) (d*g) = halfUp n d := by
  unfold halfUp
  have hn : 2*(n*g)+d*g = (2*n+d)*g := by simp [Nat.add_mul, Nat.mul_assoc]
  rw [hn, ← Nat.mul_assoc]
  exact Nat.mul_div_mul_right _ _ hg

theorem nearestEven_scaled (n d g : Nat) (hg : 0 < g) :
    nearestEven (n*g) (d*g) = nearestEven n d := by
  simp only [nearestEven, Nat.mul_div_mul_right _ _ hg, Nat.mul_mod_mul_right]
  have hgt : 2*(n%d*g) > d*g ↔ 2*(n%d) > d := by
    simpa only [Nat.mul_assoc] using (Nat.mul_lt_mul_right hg : d*g < (2*(n%d))*g ↔ d < 2*(n%d))
  have heq : 2*(n%d*g) = d*g ↔ 2*(n%d) = d := by
    rw [← Nat.mul_assoc]
    exact ⟨Nat.eq_of_mul_eq_mul_right hg, fun h => congrArg (· * g) h⟩
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  simp only [hgt, heq]

def referenceUnits (word : Nat) : Nat :=
  if word < 0x3d800000 then halfUp (units word * 2^24) (2^149)
  else nearestEven (units word * 0xffffff) (2^149)

theorem binding (word : Nat) (hw : word ≤ 0x3f800000) :
    reference word = referenceUnits word := by
  have hs : word / 2^31 ≠ 1 := by omega
  have hn : TextureDepthCompare.isNaN word = false := by
    simp only [TextureDepthCompare.isNaN, TextureDepthCompare.magnitude]
    simp
    omega
  by_cases he : word = 0x3f800000
  · subst word
    decide +kernel
  · have hw1 : ¬ word ≥ 0x3f800000 := by omega
    by_cases hz : word / 2^23 = 0
    · have hc : word < 0x3d800000 := by omega
      have hm : 2*((word%2^23)*2^24)+2^149 < 2*2^149 := by omega
      simp [reference, hs, hn, hw1, hz, referenceUnits, units, hc, halfUp,
        Nat.div_eq_of_lt hm]
    · let e := word/2^23
      let m := 2^23 + word%2^23
      let g := 2^(e-1)
      let d := 2^(150-e)
      have he : 0 < e ∧ e ≤ 126 := by dsimp [e]; omega
      have hg : 0 < g := Nat.pow_pos (by decide : 0 < (2 : Nat))
      have hp : d*g = 2^149 := by
        dsimp [d,g]
        rw [← Nat.pow_add]
        have hsum : 150-e+(e-1) = 149 := by omega
        rw [hsum]
      have hu : units word = m*g := by simp [units, hz, m, g, e]
      by_cases hc : word < 0x3d800000
      · have ht := halfUp_scaled (m*2^24) d g hg
        rw [hp] at ht
        simpa [reference, hs, hn, hw1, hz, referenceUnits, hu, hc, halfUp,
          m,d,e,Nat.mul_assoc,Nat.mul_comm,Nat.mul_left_comm] using ht.symm
      · have ht := nearestEven_scaled (m*0xffffff) d g hg
        rw [hp] at ht
        simpa [reference, hs, hn, hw1, hz, referenceUnits, hu, hc,
          m,d,e,Nat.mul_assoc,Nat.mul_comm,Nat.mul_left_comm] using ht.symm

theorem low_units (word : Nat) (hw : word < 0x3d800000) : units word < 2^145 := by
  unfold units
  split
  · omega
  · have he : word/2^23-1 ≤ 121 := by omega
    have hp := Nat.pow_le_pow_right (by decide : 1 ≤ (2 : Nat)) he
    have hm : 2^23+word%2^23 ≤ 2^24-1 := by omega
    have h := Nat.mul_le_mul hm hp
    omega

theorem decoded_error (word : Nat) (hw : word ≤ 0x3f800000) :
    -(2^148 : Int) ≤ (reference word : Int)*2^149 - (units word : Int)*0xffffff ∧
    (reference word : Int)*2^149 - (units word : Int)*0xffffff < (9*2^145 : Int) := by
  rw [binding word hw]
  unfold referenceUnits
  split
  · rename_i hc
    have hu := low_units word hc
    have he := halfUp_error (units word*2^24) (2^149) (by decide)
    simp only [halfUp, Int.natCast_mul] at *
    omega
  · have he := nearestEven_error (units word*0xffffff) (2^149) (by decide)
    simp only [Int.natCast_mul] at he
    omega

theorem exceeds_half_ulp :
    reference 0x3d7fffe8 = 0xfffff ∧
    2*((reference 0x3d7fffe8)*2^149 - units 0x3d7fffe8*0xffffff) > 2^149 := by
  decide +kernel

end GB300Math.TextureDepth24Bounds
