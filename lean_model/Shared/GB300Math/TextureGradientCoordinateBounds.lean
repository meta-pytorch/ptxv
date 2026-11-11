/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureGradient3D
set_option maxRecDepth 40000
set_option maxHeartbeats 20000000
namespace GB300Math.TextureGradientCoordinateBounds
open GB300Math.TextureGradient3D

theorem origin_unit_lod : position 0 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 = 1280 := by decide
theorem large_unit_lod : position 0x4b800000 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 = 0 := by decide
theorem origin_unit_sample : sample 0 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 = 0x40c00000 := by decide
theorem large_unit_sample : sample 0x4b800000 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 = 0x3f800000 := by decide
theorem unit_derivative_erased : (adjusted 0x4b800000 0x3f800000).significand = 0 := by decide
theorem nan_negative_coordinate : (adjusted 0x3f000000 0x7fc00000).significand = -512 := by decide

#print axioms origin_unit_lod
#print axioms large_unit_lod
#print axioms origin_unit_sample
#print axioms large_unit_sample
#print axioms unit_derivative_erased
#print axioms nan_negative_coordinate

theorem maximum_lod_gap :
    position 0 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 -
    position 0x4b800000 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 = 1280 := by
  rw [origin_unit_lod, large_unit_lod]

theorem lod_gap_bounded (cx cy cz ax ay az bx byy bz other : Nat) :
    position cx cy cz ax ay az bx byy bz - other ≤ 1280 := by
  exact Nat.le_trans (Nat.sub_le _ _) (position_bounded cx cy cz ax ay az bx byy bz)

#print axioms maximum_lod_gap
#print axioms lod_gap_bounded
end GB300Math.TextureGradientCoordinateBounds
