/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Mathlib
import GB300Math.TextureAnisotropyMipmapStep

namespace GB300Math.TextureAnisotropyMipmapRealBounds
set_option maxHeartbeats 0
set_option maxRecDepth 100000

open TextureAnisotropyMipmapStep

theorem root_error (f : Fin 257) (r : ℝ) (hr : 0≤r)
    (hroot : r^256=(2:ℝ)^f.val) :
    -(3:ℝ)/512 < (step f.val:ℝ)/128-r/2 ∧
      (step f.val:ℝ)/128-r/2 ≤ (1:ℝ)/512 := by
  have hn := exact_root_brackets f
  have hs := (step_bounds f).1
  have hsub : 1≤4*step f.val := by omega
  have hlo : (((4*step f.val-1:Nat):ℝ))^256 ≤ (256:ℝ)^256*2^f.val := by
    exact_mod_cast hn.1
  have hhi : (256:ℝ)^256*2^f.val < (((4*step f.val+3:Nat):ℝ))^256 := by
    exact_mod_cast hn.2
  rw [Nat.cast_sub hsub] at hlo
  norm_num only [Nat.cast_mul,Nat.cast_add,Nat.cast_ofNat] at hlo hhi
  have lower : 4*(step f.val:ℝ)-1 ≤ 256*r := by
    by_contra h
    have hp := pow_lt_pow_left₀ (lt_of_not_ge h) (by positivity : (0:ℝ)≤256*r)
      (by decide : (256:Nat)≠0)
    rw [mul_pow,hroot] at hp
    norm_num only at hp
    exact (not_lt_of_ge hlo) hp
  have upper : 256*r < 4*(step f.val:ℝ)+3 := by
    by_contra h
    have hp := pow_le_pow_left₀ (by positivity : (0:ℝ)≤4*(step f.val:ℝ)+3)
      (le_of_not_gt h) 256
    rw [mul_pow,hroot] at hp
    norm_num only at hp
    exact (not_lt_of_ge hp) hhi
  constructor <;> linarith

theorem exp_root (f : Nat) :
    (Real.exp (Real.log 2*(f:ℝ)/256))^256 = (2:ℝ)^f := by
  rw [← Real.exp_nat_mul]
  norm_num only [Nat.cast_ofNat]
  have he : (256:ℝ)*(Real.log 2*(f:ℝ)/256)=(f:ℝ)*Real.log 2 := by ring
  rw [he,Real.exp_nat_mul,Real.exp_log (by norm_num : (0:ℝ)<2)]

theorem half_step_ideal_error (f : Fin 257) :
    -(3:ℝ)/512 < (step f.val:ℝ)/128-Real.exp (Real.log 2*(f.val:ℝ)/256)/2 ∧
      (step f.val:ℝ)/128-Real.exp (Real.log 2*(f.val:ℝ)/256)/2 ≤ (1:ℝ)/512 := by
  exact root_error f _ (Real.exp_pos _).le (exp_root f.val)

#print axioms root_error
#print axioms exp_root
#print axioms half_step_ideal_error
end GB300Math.TextureAnisotropyMipmapRealBounds
