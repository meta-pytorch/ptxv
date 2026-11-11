/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF32Decode
namespace FabricF32Properties
open FabricF32 GB300Math.TensorAlignmentLoss

theorem maxExponent_perm {xs ys : List Nat} (hp : xs.Perm ys) :
    maxExponent xs = maxExponent ys := by
  unfold maxExponent
  exact hp.foldr_eq' (by intros; omega) 0

theorem fineShift_perm {xs ys : List Nat} (hp : xs.Perm ys) :
    fineShift xs = fineShift ys := by simp [fineShift, maxExponent_perm hp]

theorem coarseShift_perm {xs ys : List Nat} (hp : xs.Perm ys) :
    coarseShift xs = coarseShift ys := by simp [coarseShift, maxExponent_perm hp]

theorem truncatedSum_perm {xs ys : List Int} (hp : xs.Perm ys) (step : Nat) :
    truncatedSum xs step = truncatedSum ys step := by
  unfold truncatedSum
  exact hp.foldr_eq' (by intros; omega) 0

theorem finite_predict_perm {xs ys : List Nat} (hp : xs.Perm ys)
    (hfinite : ∀ w ∈ xs, w % 2^31 < 0x7f800000) : predict xs = predict ys := by
  have hy : ∀ w ∈ ys, w % 2^31 < 0x7f800000 := by
    intro w hw
    exact hfinite w (hp.mem_iff.mpr hw)
  rw [FabricF32RoundBounds.predict_finite xs hfinite, FabricF32RoundBounds.predict_finite ys hy]
  unfold FabricF32RoundBounds.acc aligned
  rw [fineShift_perm hp, coarseShift_perm hp,
    truncatedSum_perm (hp.map (fun w => (units w).getD 0))]

theorem truncateGrid_scale (n : Int) (step k : Nat) (hk : 0 < k) :
    truncateGrid ((k:Int)*n) (k*step) = (k:Int)*truncateGrid n step := by
  have hsign : (k:Int)*n < 0 ↔ n < 0 := by
    have hpos : (0:Int)<k := by exact_mod_cast hk
    constructor <;> intro h <;> nlinarith
  simp only [truncateGrid, Int.natAbs_mul, Int.natAbs_natCast, hsign,
    Nat.mul_div_mul_left _ _ hk, Nat.cast_mul]
  split <;> ring

theorem truncatedSum_scale (values : List Int) (step k : Nat) (hk : 0 < k) :
    truncatedSum (values.map ((k:Int)*·)) (k*step) = (k:Int)*truncatedSum values step := by
  induction values with
  | nil => simp [truncatedSum]
  | cons v vs ih =>
    simp only [List.map_cons, truncatedSum, List.foldr_cons] at *
    rw [truncateGrid_scale v step k hk, ih]
    ring

theorem flush_scale (s q : Nat) (v : Int) :
    flush (s+q) ((2^q:Int)*v) = (2^q:Int)*flush s v := by
  have hcmp : ((2^q:Int)*v).natAbs < 2^(s+q) ↔ v.natAbs < 2^s := by
    rw [Int.natAbs_mul, Int.natAbs_pow, Nat.pow_add]
    norm_num
    rw [Nat.mul_comm (2^s), Nat.mul_lt_mul_left (Nat.two_pow_pos q)]
  simp only [flush, hcmp]
  split <;> simp

theorem accumulator_scale (xs ys : List Nat) (values : List Int) (q : Nat)
    (hf : fineShift ys = fineShift xs + q)
    (hc : coarseShift ys = coarseShift xs + q) :
    FabricF32RoundBounds.acc ys (values.map ((2^q:Int)*·)) =
      (2^q:Int)*FabricF32RoundBounds.acc xs values := by
  unfold FabricF32RoundBounds.acc aligned
  rw [hf, hc, Nat.pow_add, Nat.mul_comm (2^fineShift xs)]
  have h := truncatedSum_scale values (2^fineShift xs) (2^q) (Nat.two_pow_pos q)
  simp only [Nat.cast_pow, Nat.cast_ofNat] at h
  rw [h, flush_scale]

theorem sixteen_thresholds (xs ys : List Nat)
    (he : maxExponent ys = maxExponent xs+16) (hmin : 64 ≤ maxExponent xs) :
    fineShift ys = fineShift xs+16 ∧ coarseShift ys = coarseShift xs+16 := by
  have hx : 4 ≤ maxExponent xs/16 := by omega
  simp only [fineShift, coarseShift, he]
  omega

#print axioms accumulator_scale
#print axioms sixteen_thresholds
#print axioms truncateGrid_scale
#print axioms truncatedSum_scale
#print axioms flush_scale
#print axioms maxExponent_perm
#print axioms fineShift_perm
#print axioms coarseShift_perm
#print axioms truncatedSum_perm
#print axioms finite_predict_perm
end FabricF32Properties
