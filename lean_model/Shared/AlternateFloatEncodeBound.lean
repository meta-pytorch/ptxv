/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AlternateFloatNormalRelative
set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 4096
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem rounded_significand_range (m p : Nat) (hm : m ≠ 0) (hp : p ≤ m.log2) :
    2^p ≤ nearestEven m (m.log2-p) ∧
    nearestEven m (m.log2-p) ≤ 2^(p+1) := by
  have hlo := Nat.log2_self_le hm
  have hhi : m < 2^(m.log2+1) := (Nat.log2_lt hm).mp (by omega)
  have hpow : 2^p * 2^(m.log2-p) = 2^m.log2 := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hpowsucc : 2^(p+1) * 2^(m.log2-p) = 2^(m.log2+1) := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hpos := Nat.two_pow_pos (m.log2-p)
  have hquotlo : 2^p ≤ m / 2^(m.log2-p) :=
    (Nat.le_div_iff_mul_le hpos).mpr (by rwa [hpow])
  have hquothi : m / 2^(m.log2-p) < 2^(p+1) :=
    (Nat.div_lt_iff_lt_mul hpos).mpr (by rwa [hpowsucc])
  unfold nearestEven
  split
  · rename_i hz
    have hmp : m.log2 = p := by omega
    rw [hmp] at hlo hhi
    omega
  · dsimp only
    split <;> omega

#print axioms rounded_significand_range
end GB300Math.AlternateFloatFloatLib
