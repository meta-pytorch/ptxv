/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AlternateFloatRationalBounds
import Mathlib.Tactic

set_option maxHeartbeats 0
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem normal_relative_bound (p q : Nat) (m h : ℚ)
    (hp : 0 < p) (hh : 0 < h)
    (hm : 2 * (p : ℚ) * h ≤ m)
    (hl : 2 * (q : ℚ) * h ≤ m + h)
    (hu : m ≤ 2 * (q : ℚ) * h + h) :
    (2 * (p : ℚ) + 1) * |2 * (q : ℚ) * h - m| ≤ m := by
  have hp0 : (0 : ℚ) ≤ p := by positivity
  have hnear : |2 * (q : ℚ) * h - m| ≤ h := by
    apply abs_le.mpr
    constructor <;> linarith
  by_cases hqp : q ≤ p
  · have heq : q = p := by
      by_contra hne
      have hstep : q + 1 ≤ p := by omega
      have hc : (q : ℚ) + 1 ≤ p := by exact_mod_cast hstep
      have hprod : 2 * ((q : ℚ) + 1) * h ≤ 2 * (p : ℚ) * h := by gcongr
      nlinarith
    subst q
    rw [abs_of_nonpos (by linarith : 2 * (p : ℚ) * h - m ≤ 0)]
    have hprod := mul_nonpos_of_nonneg_of_nonpos hp0 (by linarith : m - 2 * (p : ℚ) * h - h ≤ 0)
    nlinarith
  · have hstep : p + 1 ≤ q := by omega
    have hc : (p : ℚ) + 1 ≤ q := by exact_mod_cast hstep
    have hprod : 2 * ((p : ℚ) + 1) * h ≤ 2 * (q : ℚ) * h := by gcongr
    calc
      (2 * (p : ℚ) + 1) * |2 * (q : ℚ) * h - m| ≤ (2 * (p : ℚ) + 1) * h := by gcongr
      _ ≤ m := by nlinarith

theorem nearest_normal_relative_bound (p m d : Nat) (hp : 0 < p) (hd : 0 < d)
    (hm : p * 2^d ≤ m) :
    (2 * (p : ℚ) + 1) * |((nearestEven m d * 2^d : Nat) : ℚ) - m| ≤ m := by
  have hpow : 2^d = 2 * 2^(d-1) := by
    calc
      2^d = 2^((d-1)+1) := congrArg (fun n : Nat => 2^n) (by omega)
      _ = 2 * 2^(d-1) := by rw [Nat.pow_succ, Nat.mul_comm]
  obtain ⟨hl, hu⟩ := GB300Math.AlternateFloat.nearestEven_half_quantum m d hd
  have hlq : ((nearestEven m d * 2^d : Nat) : ℚ) ≤ (m : ℚ) + 2^(d-1) := by exact_mod_cast hl
  have huq : (m : ℚ) ≤ ((nearestEven m d * 2^d : Nat) : ℚ) + 2^(d-1) := by exact_mod_cast hu
  have hmq : (p : ℚ) * 2^d ≤ (m : ℚ) := by exact_mod_cast hm
  have hpq : (2 : ℚ)^d = 2 * 2^(d-1) := by exact_mod_cast hpow
  simp only [Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat] at hlq huq ⊢
  rw [hpq] at hlq huq hmq ⊢
  have r := normal_relative_bound p (nearestEven m d) m (2^(d-1)) hp
    (by positivity) (by nlinarith [hmq]) (by nlinarith [hlq]) (by nlinarith [huq])
  simpa only [mul_assoc, mul_comm, mul_left_comm] using r

#print axioms nearest_normal_relative_bound
#print axioms normal_relative_bound
end GB300Math.AlternateFloatFloatLib
