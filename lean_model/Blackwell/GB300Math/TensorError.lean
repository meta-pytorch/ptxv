/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorBlackwell
namespace GB300Math.TensorError
open GB300Math.Tensor

theorem nonnegative_residual (n s : Nat) :
    ((n:Int) - (n/2^s:Nat)*(2^s:Int)).natAbs < 2^s := by
  have hr := Nat.mod_lt n (Nat.two_pow_pos s)
  have he := congrArg (fun x : Nat => (x:Int)) (Nat.mod_add_div n (2^s))
  simp only [Int.natCast_add, Int.natCast_mul] at he
  have hp : ((2^s:Nat):Int) = (2^s:Int) := by simp
  rw [hp] at he
  have heq : (n:Int) - (n/2^s:Nat)*(2^s:Int) = (n%2^s:Nat) := by
    rw [Int.mul_comm] at he
    omega
  rw [heq]
  exact hr

theorem signed_alignment_residual (n : Int) (s : Nat) :
    (n-shiftTruncate n (-(s:Int))*(2^s:Int)).natAbs < 2^s := by
  cases n with
  | ofNat n =>
    by_cases hs : s = 0
    · subst s; simp [shiftTruncate, show ¬(n:Int)<0 by omega]
    · simp only [shiftTruncate, show ¬ -(s:Int) ≥ 0 by omega, ite_false,
        Int.neg_neg, Int.toNat_natCast]
      exact nonnegative_residual n s
  | negSucc n =>
    by_cases hs : s = 0
    · subst s
      simp [shiftTruncate, Int.negSucc_eq]
      rw [Int.natAbs_of_nonneg (by omega)]
      omega
    · simp only [shiftTruncate, show ¬ -(s:Int) ≥ 0 by omega, ite_false,
        Int.neg_neg, Int.toNat_natCast, show Int.negSucc n < 0 by omega, ite_true]
      have h := nonnegative_residual (n+1) s
      have he : Int.negSucc n - -(↑((Int.negSucc n).natAbs / 2^s)) * (2^s:Int) =
          -((n+1:Int) - ((n+1)/2^s:Nat)*(2^s:Int)) := by
        simp only [Int.negSucc_eq, show (↑n+1:Int) = ((n+1:Nat):Int) by omega, Int.natAbs_neg, Int.natAbs_natCast, Int.neg_mul]
        omega
      rw [he, Int.natAbs_neg]
      exact h


theorem sum_alignment_residual (xs : List Int) (s : Nat) :
    (xs.sum - (xs.map (fun x => shiftTruncate x (-(s:Int)))).sum*(2^s:Int)).natAbs
      ≤ xs.length*(2^s-1) := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have hx := signed_alignment_residual x s
    have ht := Int.natAbs_add_le
      (x-shiftTruncate x (-(s:Int))*(2^s:Int))
      (xs.sum-(xs.map (fun x => shiftTruncate x (-(s:Int)))).sum*(2^s:Int))
    have he : (x::xs).sum - ((x::xs).map (fun x => shiftTruncate x (-(s:Int)))).sum*(2^s:Int) =
        (x-shiftTruncate x (-(s:Int))*(2^s:Int)) +
        (xs.sum-(xs.map (fun x => shiftTruncate x (-(s:Int)))).sum*(2^s:Int)) := by
      simp only [List.sum_cons, List.map_cons, Int.add_mul]
      omega
    rw [he]
    simp only [List.length_cons, Nat.add_mul, Nat.one_mul]
    omega

theorem two_stage_alignment_residual (xs : List Int) (s t : Nat) :
    let aligned := (xs.map (fun x => shiftTruncate x (-(s:Int)))).sum
    (xs.sum - shiftTruncate aligned (-(t:Int))*(2^(s+t):Int)).natAbs ≤
      xs.length*(2^s-1) + (2^t-1)*2^s := by
  dsimp only
  let aligned := (xs.map (fun x => shiftTruncate x (-(s:Int)))).sum
  let rounded := shiftTruncate aligned (-(t:Int))
  have h1 := sum_alignment_residual xs s
  have h2 := signed_alignment_residual aligned t
  change (aligned-rounded*(2^t:Int)).natAbs < 2^t at h2
  have he : xs.sum-rounded*(2^(s+t):Int) =
      (xs.sum-aligned*(2^s:Int)) + (aligned-rounded*(2^t:Int))*(2^s:Int) := by
    rw [Int.pow_add, Int.sub_mul, Int.mul_assoc]
    rw [Int.mul_comm (2^t:Int) (2^s:Int)]
    omega
  change (xs.sum-rounded*(2^(s+t):Int)).natAbs ≤ _
  rw [he]
  have ht := Int.natAbs_add_le (xs.sum-aligned*(2^s:Int))
    ((aligned-rounded*(2^t:Int))*(2^s:Int))
  have hb : (aligned-rounded*(2^t:Int)).natAbs ≤ 2^t-1 := by omega
  have hm := Nat.mul_le_mul_right (2^s) hb
  simp only [Int.natAbs_mul, Int.natAbs_pow, show (2:Int).natAbs = 2 from rfl] at ht
  change (xs.sum-aligned*(2^s:Int)).natAbs ≤ _ at h1
  omega

#print axioms two_stage_alignment_residual
#print axioms signed_alignment_residual
#print axioms sum_alignment_residual
theorem permutation_preserves_quantized_sum {xs ys : List Int}
    (h : xs.Perm ys) (s : Int) :
    (xs.map (fun x => shiftTruncate x s)).sum =
      (ys.map (fun x => shiftTruncate x s)).sum := by
  induction h with
  | nil => rfl
  | cons x h ih => simpa only [List.map_cons, List.sum_cons] using congrArg (fun z => shiftTruncate x s + z) ih
  | swap x y xs => simp [List.map_cons, List.sum_cons, Int.add_left_comm]
  | trans h₁ h₂ ih₁ ih₂ => exact ih₁.trans ih₂

#print axioms permutation_preserves_quantized_sum

theorem permutation_preserves_maximum {xs ys : List Int}
    (h : xs.Perm ys) (z : Int) : xs.foldr max z = ys.foldr max z := by
  induction h with
  | nil => rfl
  | cons x h ih => simpa only [List.foldr_cons] using congrArg (max x) ih
  | swap x y xs => simp only [List.foldr_cons]; omega
  | trans h₁ h₂ ih₁ ih₂ => exact ih₁.trans ih₂
#print axioms permutation_preserves_maximum

end GB300Math.TensorError
