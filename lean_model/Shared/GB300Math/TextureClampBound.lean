/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureClampOrder

namespace GB300Math.TextureClampBound

theorem rounded_merge (a b y : Nat) (hy : y≤256) :
    let late := (a*(256-y)+128)/256+b-(b*y+128)/256
    let early := ((a+b)*(256-y)+128)/256
    late≤early+1 ∧ early≤late+1 := by
  have hby : b*y≤b*256 := Nat.mul_le_mul_left b hy
  have hid : a*(256-y)+b*256=(a+b)*(256-y)+b*y := by
    calc
      a*(256-y)+b*256=a*(256-y)+b*((256-y)+y) := by rw [Nat.sub_add_cancel hy]
      _=(a+b)*(256-y)+b*y := by simp [Nat.mul_add,Nat.add_mul,Nat.add_assoc]
  dsimp only
  omega

theorem edge_mass_error (w x y : Nat) (hx : x≤256) (hy : y≤256) :
    let late := TextureAnisotropyOriented.plane w x y
    let early := TextureAnisotropyOriented.plane w 0 y
    late.1+late.2.1≤early.1+1 ∧ early.1≤late.1+late.2.1+1 := by
  have h := TextureAnisotropyTaps.upperWeight_bounded w x false hx
  simp only [TextureAnisotropyTaps.upperWeight,Bool.false_eq_true,↓reduceIte] at h
  have hby : ((w*x+128)/256)*y≤((w*x+128)/256)*256 := Nat.mul_le_mul_left _ hy
  have hg := rounded_merge (w-(w*x+128)/256) ((w*x+128)/256) y hy
  rw [Nat.sub_add_cancel h] at hg
  simp only [TextureAnisotropyOriented.plane,Nat.mul_zero,Nat.zero_add,Nat.reduceDiv,
    Nat.sub_zero,Nat.zero_mul] at ⊢
  omega

theorem bound_attained :
    let late := TextureAnisotropyOriented.plane 101 37 62
    let early := TextureAnisotropyOriented.plane 101 0 62
    early.1-(late.1+late.2.1)=1 := by decide +kernel

#print axioms rounded_merge
#print axioms edge_mass_error
#print axioms bound_attained

end GB300Math.TextureClampBound
