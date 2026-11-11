/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AlternateFloatSpec

namespace GB300Math.AlternateFloat

theorem nearestEven_half_quantum (m d : Nat) (hd : 0 < d) :
    nearestEven m d * 2^d ≤ m + 2^(d-1) ∧
    m ≤ nearestEven m d * 2^d + 2^(d-1) := by
  have hp : 2^d = 2^(d-1)*2 := by
    calc
      2^d = 2^((d-1)+1) := congrArg (fun n : Nat => 2^n) (by omega)
      _ = 2^(d-1)*2 := Nat.pow_succ 2 (d-1)
  have hpos : 0 < 2^d := Nat.two_pow_pos d
  have hrem := Nat.mod_lt m hpos
  have hdiv := Nat.div_add_mod m (2^d)
  rw [Nat.mul_comm (2^d) (m / 2^d)] at hdiv
  simp only [nearestEven, ite_eq_right (by omega : d ≠ 0)]
  split <;> simp only [Nat.add_mul, Nat.one_mul, Nat.add_zero]
  · omega
  · omega

#print axioms nearestEven_half_quantum
end GB300Math.AlternateFloat
