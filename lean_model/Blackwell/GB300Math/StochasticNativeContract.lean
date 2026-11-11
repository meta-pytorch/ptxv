/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticNativeSm103

namespace GB300Math.StochasticNativeContract
open GB300Math.StochasticSm103
open GB300Math.StochasticNativeSm103

theorem saturated_lane_agrees (f : Format) (relu : Bool) (x r : Nat) :
    GB300Math.StochasticNativeSm103.lane f relu true x r =
      GB300Math.StochasticSm103.lane f relu true x r := by
  cases f <;> rfl

theorem entropy_bound (f : Format) (c : Nat) : entropy f c < 65536 := by
  cases f <;> simp [entropy] <;> omega

theorem fp4_entropy_low_byte (c : Nat) :
    entropy .e2m1 c % 256 = c / 256 % 256 := by
  simp [entropy, Nat.add_mod]

theorem fp8_entropy_ignores_merge (r m : Nat) (hm : m < 65536) :
    entropy .e4m3 (r*65536+m) = r % 65536 := by
  simp [entropy]
  omega

theorem fp4_fixed_merge_entropy (c d : Nat) (h : c % 2^24 = d % 2^24) :
    entropy .e2m1 c % 256 = entropy .e2m1 d % 256 := by
  rw [fp4_entropy_low_byte, fp4_entropy_low_byte]
  omega

theorem native_rounding_error (f : Format) (c mantissa drop : Nat) (h : 16 ≤ drop) :
    roundMantissa mantissa (entropy f c) drop 16 * 2^drop < mantissa + 2^drop ∧
    mantissa < roundMantissa mantissa (entropy f c) drop 16 * 2^drop + 2^drop :=
  strict_one_step_error mantissa (entropy f c) drop 16 h

#print axioms saturated_lane_agrees
#print axioms entropy_bound
#print axioms fp4_entropy_low_byte
#print axioms fp8_entropy_ignores_merge
#print axioms fp4_fixed_merge_entropy
#print axioms native_rounding_error
end GB300Math.StochasticNativeContract
