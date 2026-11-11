/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyArea
import GB300Math.TextureGradient2D

namespace GB300Math.TextureAnisotropyGeneral
open TextureGradient2D

def multiply (a b : TextureGradient2D.Dyadic) : TextureGradient2D.Dyadic :=
  if a.significand=0 ∨ b.significand=0 then ⟨0,0⟩ else
    let x := a.significand.natAbs
    let y := b.significand.natAbs
    let p := x*y-TextureAnisotropyArea.omitted x y+448
    normalized ⟨signed (a.significand*b.significand<0) p,a.scale+b.scale⟩

def cap (major minor : Int) (anisotropy : Nat) : Int :=
  max (min major minor) (major-(anisotropy.log2:Int)*256)

def position (ax ay bx byy anisotropy : Nat) : Nat :=
  let ax := fromBits ax
  let ay := fromBits ay
  let bx := fromBits bx
  let byy := fromBits byy
  let major := norm ax ay bx byy
  if major.significand=0 then 0 else
    let lm := TextureAnisotropyArea.logarithm major.significand.natAbs major.scale
    let area := guardedAdd (multiply ax byy) (negate (multiply ay bx))
    let lower := lm-(anisotropy.log2:Int)*256
    let minor := if area.significand=0 then lower else
      TextureAnisotropyArea.logarithm area.significand.natAbs area.scale-lm
    min 2048 (2048+cap lm minor anisotropy).toNat

def sample (ax ay bx byy anisotropy : Nat) : Nat :=
  TextureMipmap.encodeQ8 (256+position ax ay bx byy anisotropy)

theorem cap_one (major minor : Int) : cap major minor 1=major := by
  have h : (1:Nat).log2=0 := by decide
  simp only [cap, h]
  omega

theorem cap_bounds (major minor : Int) (anisotropy : Nat) :
    major-(anisotropy.log2:Int)*256 ≤ cap major minor anisotropy ∧
    cap major minor anisotropy ≤ major := by
  unfold cap
  omega

theorem multiply_zero_left (b : TextureGradient2D.Dyadic) : multiply ⟨0,0⟩ b=⟨0,0⟩ := by
  simp [multiply]

theorem multiply_zero_right (a : TextureGradient2D.Dyadic) : multiply a ⟨0,0⟩=⟨0,0⟩ := by
  simp [multiply]

theorem position_bounded (ax ay bx byy anisotropy : Nat) :
    position ax ay bx byy anisotropy ≤ 2048 := by
  unfold position
  dsimp only
  split
  · omega
  · exact Nat.min_le_left _ _

theorem subnormal_input (v : Nat) (h : v / 2^23 % 256 = 0) :
    TextureGradient2D.fromBits v = ⟨0, 0⟩ := by
  simp [TextureGradient2D.fromBits, h]

theorem nan_input (v : Nat) (he : v / 2^23 % 256 = 255)
    (hm : v % 2^23 ≠ 0) : TextureGradient2D.fromBits v = ⟨0, 0⟩ := by
  simp [TextureGradient2D.fromBits, he, hm]

#print axioms subnormal_input
#print axioms nan_input
#print axioms cap_one
#print axioms cap_bounds
#print axioms multiply_zero_left
#print axioms multiply_zero_right
#print axioms position_bounded
end GB300Math.TextureAnisotropyGeneral
