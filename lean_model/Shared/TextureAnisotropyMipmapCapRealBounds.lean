/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TextureAnisotropyMipmapRealBounds
import GB300Math.TextureAnisotropyMipmapCaps

namespace GB300Math.TextureAnisotropyMipmapCapRealBounds
open TextureAnisotropyMipmap TextureAnisotropyMipmapRealBounds
set_option maxHeartbeats 0

noncomputable def effective (a : Nat) : ℝ := Real.exp (Real.log 2*(cap a:ℝ)/256)

theorem effective_bounds (a : Fin 16) :
    2*((a.val:ℝ)+1)/3 ≤ effective (a.val+1) ∧
      effective (a.val+1) ≤ (a.val:ℝ)+1 := by
  have hn := TextureAnisotropyMipmapCaps.effective_cap_brackets a
  have hr : 0≤effective (a.val+1) := (Real.exp_pos _).le
  have he : (effective (a.val+1))^256=(2:ℝ)^(cap (a.val+1)) := exp_root _
  have hlo : (2*((a.val:ℝ)+1))^256 ≤ (3:ℝ)^256*2^(cap (a.val+1)) := by
    exact_mod_cast hn.1
  have hhi : (2:ℝ)^(cap (a.val+1)) ≤ ((a.val:ℝ)+1)^256 := by
    exact_mod_cast hn.2
  have lo : 2*((a.val:ℝ)+1)≤3*effective (a.val+1) := by
    by_contra h
    have hp := pow_lt_pow_left₀ (lt_of_not_ge h) (by positivity : (0:ℝ)≤3*effective (a.val+1))
      (by decide : (256:Nat)≠0)
    rw [mul_pow,he] at hp
    exact (not_lt_of_ge hlo) hp
  have hi : effective (a.val+1)≤(a.val:ℝ)+1 := by
    by_contra h
    have hp := pow_lt_pow_left₀ (lt_of_not_ge h) (by positivity : (0:ℝ)≤(a.val:ℝ)+1)
      (by decide : (256:Nat)≠0)
    rw [he] at hp
    exact (not_lt_of_ge hhi) hp
  constructor <;> linarith

theorem descriptor3_effective : effective 3=2 := by
  unfold effective
  norm_num [cap]
  convert Real.exp_log (by norm_num : (0:ℝ)<2) using 1

#print axioms effective_bounds
#print axioms descriptor3_effective
end GB300Math.TextureAnisotropyMipmapCapRealBounds
