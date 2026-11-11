/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureGradient3D
import GB300Math.StochasticValue
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

namespace GB300Math.TextureGradient3DBounds

noncomputable def spatial (x y z : ℝ) : ℝ := z+11/32*y+1/4*x

theorem spatial_squared_upper (x y z : ℝ) :
    1024*(spatial x y z)^2 ≤ 1209*(x^2+y^2+z^2) := by
  unfold spatial
  nlinarith [sq_nonneg (32*x-8*z), sq_nonneg (11*x-8*y), sq_nonneg (32*y-11*z)]

theorem spatial_squared_lower (x y z : ℝ) (hx : 0 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) :
    2601*(x^2+y^2+z^2) ≤ 3072*(spatial x y z)^2 := by
  have ha : 0 ≤ z-y := sub_nonneg.mpr hyz
  have hb : 0 ≤ y-x := sub_nonneg.mpr hxy
  have hab := mul_nonneg ha hb
  have hac := mul_nonneg ha hx
  have hbc := mul_nonneg hb hx
  unfold spatial
  nlinarith [sq_nonneg (z-y),sq_nonneg (y-x)]

theorem lower_bound_attained :
    3072*(spatial 1 1 1)^2 = 2601*((1:ℝ)^2+1^2+1^2) := by
  norm_num [spatial]

theorem upper_bound_attained :
    1024*(spatial 8 11 32)^2 = 1209*((8:ℝ)^2+11^2+32^2) := by
  norm_num [spatial]

theorem diagonal_position :
    TextureGradient3D.position 0x3f000000 0x3f000000 0x3f000000
      0x3d800000 0x3d800000 0x3d800000 0 0 0 = 429 := by decide +kernel

theorem diagonal_output :
    StochasticValue.rationalValue 8 23 127 0x7f7fffff
      (TextureGradient3D.sample 0x3f000000 0x3f000000 0x3f000000
        0x3d800000 0x3d800000 0x3d800000 0 0 0) = some (1+429/256) := by decide +kernel

noncomputable def diagonalEuclideanLod : ℝ := 1+Real.log 3/(2*Real.log 2)

theorem diagonal_lod_gap : (29:ℝ)/256 < diagonalEuclideanLod-429/256 := by
  have hl2 : 0 < Real.log (2:ℝ) := Real.log_pos (by norm_num)
  have hp : (2:ℝ)^202 < (3:ℝ)^128 := by
    exact_mod_cast (show (2:Nat)^202 < 3^128 by decide +kernel)
  have hl := Real.log_lt_log (by positivity : (0:ℝ) < 2^202) hp
  rw [Real.log_pow, Real.log_pow] at hl
  have hh : ((101:ℝ)/128)*(2*Real.log 2) < Real.log 3 := by norm_num at hl; nlinarith
  have hd : (101:ℝ)/128 < Real.log 3/(2*Real.log 2) :=
    (lt_div_iff₀ (by positivity : (0:ℝ) < 2*Real.log 2)).2 hh
  unfold diagonalEuclideanLod
  linarith

#print axioms spatial_squared_upper
#print axioms spatial_squared_lower
#print axioms lower_bound_attained
#print axioms upper_bound_attained
#print axioms diagonal_position
#print axioms diagonal_output
#print axioms diagonal_lod_gap
end GB300Math.TextureGradient3DBounds
