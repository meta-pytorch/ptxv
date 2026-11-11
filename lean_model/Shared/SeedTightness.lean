/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Fp64SeedBounds
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
namespace GB300Math.SeedTightness

theorem output_bits : Fp64Seed.evaluate false 0x3ff04884ffffffff = 0x3fef717e00000000 := by decide +kernel

theorem relative_error :
    Fp64SeedBounds.normalValue (Fp64Seed.evaluate false 0x3ff04884ffffffff)*
      Fp64SeedBounds.normalValue 0x3ff04884ffffffff-1 = (9401331221565058 : ℚ)/2^73 := by
  rw [output_bits]
  norm_num [Fp64SeedBounds.normalValue,Fp64SeedBounds.positiveNormalValue]

theorem relative_error_interval :
    let e := |Fp64SeedBounds.normalValue (Fp64Seed.evaluate false 0x3ff04884ffffffff)*
      Fp64SeedBounds.normalValue 0x3ff04884ffffffff-1|
    (995 : ℚ)/10^9 < e ∧ e < (996 : ℚ)/10^9 := by
  dsimp only
  rw [relative_error]
  norm_num

#print axioms output_bits
#print axioms relative_error
#print axioms relative_error_interval
end GB300Math.SeedTightness
