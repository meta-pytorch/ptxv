/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureClampBound
import Mathlib

namespace GB300Math.TextureClampReal

theorem conserved_weights_error (a b c d : Nat) (hm : a+b=c+d)
    (hac : a≤c+1 ∧ c≤a+1) (v₀ v₁ : ℝ) :
    |((a:ℝ)*v₀+b*v₁)/256-((c:ℝ)*v₀+d*v₁)/256|≤|v₀-v₁|/256 := by
  have hm' : (a:ℝ)+b=c+d := by exact_mod_cast hm
  have h₀ : (a:ℝ)≤c+1 := by exact_mod_cast hac.1
  have h₁ : (c:ℝ)≤a+1 := by exact_mod_cast hac.2
  have hd : |(a:ℝ)-c|≤1 := abs_le.mpr ⟨by linarith,by linarith⟩
  have hi : ((a:ℝ)*v₀+b*v₁)/256-((c:ℝ)*v₀+d*v₁)/256=
      ((a:ℝ)-c)*(v₀-v₁)/256 := by
    have hb : (b:ℝ)=c+d-a := by linarith
    rw [hb]
    ring
  rw [hi,abs_div,abs_mul]
  rw [abs_of_pos (by norm_num : (0:ℝ)<256)]
  exact div_le_div_of_nonneg_right
    (by simpa using mul_le_mul_of_nonneg_right hd (abs_nonneg (v₀-v₁))) (by norm_num)

theorem edge_error (w x y : Nat) (hx : x≤256) (hy : y≤256) (v₀ v₁ : ℝ) :
    let late := TextureAnisotropyOriented.plane w x y
    let early := TextureAnisotropyOriented.plane w 0 y
    |(((late.1+late.2.1:Nat):ℝ)*v₀+(late.2.2.1+late.2.2.2)*v₁)/256-
      ((early.1:ℝ)*v₀+early.2.2.1*v₁)/256|≤|v₀-v₁|/256 := by
  let p := TextureAnisotropyOriented.plane w x y
  let q := TextureAnisotropyOriented.plane w 0 y
  have hp := TextureAnisotropyOriented.plane_sum w x y hx hy
  have hq := TextureAnisotropyOriented.plane_sum w 0 y (by omega) hy
  have hzeros : q.2.1=0 ∧ q.2.2.2=0 := by
    simp [q,TextureAnisotropyOriented.plane]
  have hm : (p.1+p.2.1)+(p.2.2.1+p.2.2.2)=q.1+q.2.2.1 := by
    change p.1+p.2.1+p.2.2.1+p.2.2.2=w at hp
    change q.1+q.2.1+q.2.2.1+q.2.2.2=w at hq
    omega
  simpa only [p,q,Nat.cast_add] using conserved_weights_error _ _ _ _ hm
    (TextureClampBound.edge_mass_error w x y hx hy) v₀ v₁

theorem edge_bound_attained :
    let late := TextureAnisotropyOriented.plane 101 37 62
    let early := TextureAnisotropyOriented.plane 101 0 62
    |(((late.1+late.2.1:Nat):ℝ)*0+(late.2.2.1+late.2.2.2)*256)/256-
      ((early.1:ℝ)*0+early.2.2.1*256)/256|=1 := by
  norm_num [TextureAnisotropyOriented.plane]

#print axioms conserved_weights_error
#print axioms edge_error
#print axioms edge_bound_attained

end GB300Math.TextureClampReal
