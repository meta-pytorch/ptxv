/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Fp64Seed
import FloatLib.Floats.Formats.BinaryInterchange.Configured
import Mathlib.Analysis.SpecialFunctions.Sqrt

set_option maxRecDepth 100000
set_option maxHeartbeats 20000000

namespace GB300Math.Fp64SeedAccuracy

open FloatLib.Floats

def decoded (word : Nat) : ℚ :=
  (ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits word : ExecFloat.Binary 11 52)).getD 0

theorem three_input_decode : decoded 0x4008000000000000 = 3 := by decide +kernel
theorem two_input_decode : decoded 0x4000000000000000 = 2 := by decide +kernel

theorem rcp_three_bits : Fp64Seed.evaluate false 0x4008000000000000 = 0x3fd5555500000000 := by
  decide +kernel

theorem rsq_two_bits : Fp64Seed.evaluate true 0x4000000000000000 = 0x3fe6a09e00000000 := by
  decide +kernel

theorem rcp_three_relative_error :
    |(decoded (Fp64Seed.evaluate false 0x4008000000000000) : ℝ) - 1 / 3| / (1 / 3) =
      (1 : ℝ) / 4194304 := by
  rw [rcp_three_bits]
  have hd : decoded 0x3fd5555500000000 = 1398101 / 4194304 := by decide +kernel
  rw [hd]
  norm_num

theorem rsq_two_absolute_error :
    |(decoded (Fp64Seed.evaluate true 0x4000000000000000) : ℝ) - 1 / Real.sqrt 2| <
      (1 : ℝ) / 4194304 := by
  rw [rsq_two_bits]
  have hd : decoded 0x3fe6a09e00000000 = 741455 / 1048576 := by decide +kernel
  rw [hd]
  have hs : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have he : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hlo : (1414213562 : ℝ) / 1000000000 < Real.sqrt 2 := by nlinarith
  have hhi : Real.sqrt 2 < (1414213563 : ℝ) / 1000000000 := by nlinarith
  apply abs_lt.mpr
  constructor
  · have h : 1 / Real.sqrt 2 < (741455 : ℝ) / 1048576 + 1 / 4194304 := by
      apply (div_lt_iff₀ hs).2
      nlinarith
    push_cast
    linarith
  · have h : (741455 : ℝ) / 1048576 - 1 / 4194304 < 1 / Real.sqrt 2 := by
      apply (lt_div_iff₀ hs).2
      nlinarith
    push_cast
    linarith

#print axioms rcp_three_bits
#print axioms rsq_two_bits
#print axioms rcp_three_relative_error
#print axioms rsq_two_absolute_error

end GB300Math.Fp64SeedAccuracy
