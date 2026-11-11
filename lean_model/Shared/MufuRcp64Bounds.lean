/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuRcp32Classification
set_option maxRecDepth 8000
namespace GB300Math.Rcp64Bounds
open MufuRcp32Bounds

theorem same_interpolation (index : Nat) :
    Fp64Seed.interpolate false index = (Mufu.interpolate MufuTables.rcp index).toNat := by
  rfl

theorem truncate_product (f n : Int) (hf : 0 ≤ f) (hfmax : f ≤ 2^20)
    (hlo : -(2^24 : Int) ≤ (2^23+f*8)*n-2^47)
    (hhi : (2^23+f*8)*n-2^47 ≤ 2^24) :
    -(2^21 : Int) ≤ (2^20+f)*(n/8)-2^41 ∧
    (2^20+f)*(n/8)-2^41 ≤ 2^18 := by
  have ht0 : 0 ≤ n%8 := Int.emod_nonneg _ (by norm_num)
  have ht7 : n%8 ≤ 7 := by have h := Int.emod_lt_of_pos n (by norm_num : (0 : Int) < 8); omega
  have hr : n = 8*(n/8)+n%8 := by omega
  have hd : (2^20+f)*(n%8) ≤ 7*2^21 := by nlinarith
  have hd0 : 0 ≤ (2^20+f)*(n%8) := by positivity
  have hid := congrArg (fun x : Int => (2^23+f*8)*x) hr
  ring_nf at hid
  constructor <;> nlinarith

theorem seed_product_bound (f : Fin (2^20)) :
    -(2^21 : Int) ≤ ((2^20+(f.val : Int))*
      ((Fp64Seed.interpolate false (f.val*8) / 8 : Nat) : Int)-2^41) ∧
    ((2^20+(f.val : Int))*
      ((Fp64Seed.interpolate false (f.val*8) / 8 : Nat) : Int)-2^41) ≤ 2^18 := by
  let i : Fin (2^23) := ⟨f.val*8, by have h := f.isLt; omega⟩
  let n := Fp64Seed.interpolate false (f.val*8)
  have hpos : 0 ≤ Mufu.interpolate MufuTables.rcp i.val :=
    le_trans (by norm_num) (mantissa_range i).1
  have hn : (n : Int) = Mufu.interpolate MufuTables.rcp i.val := by
    dsimp [n,i]
    rw [same_interpolation]
    exact Int.toNat_of_nonneg hpos
  have h := normalized_product_bound i
  rw [← hn] at h
  simp only [i,Nat.cast_mul,Nat.cast_ofNat] at h
  have hi : (f.val : Int) ≤ 2^20 := by have h := f.isLt; omega
  have result := truncate_product f.val n (by positivity) hi h.1 h.2
  simpa only [Int.natCast_ediv,Nat.cast_ofNat] using result

noncomputable def input (f : Nat) : ℝ := (2^20+(f : ℝ))/2^20
noncomputable def output (f : Nat) : ℝ :=
    ((Fp64Seed.interpolate false (f*8)/8 : Nat) : ℝ)/2^21

theorem normalized_error (f : Fin (2^20)) :
    |output f.val-1/input f.val|/|1/input f.val| ≤ 1/(2^20 : ℝ) := by
  rw [relative_eq_product _ _ (by dsimp [input]; positivity)]
  have h := seed_product_bound f
  have lo : -(2^21 : ℝ) ≤ (2^20+(f.val : ℝ))*
      ((Fp64Seed.interpolate false (f.val*8)/8 : Nat) : ℝ)-2^41 := by exact_mod_cast h.1
  have hi : (2^20+(f.val : ℝ))*
      ((Fp64Seed.interpolate false (f.val*8)/8 : Nat) : ℝ)-2^41 ≤ 2^18 := by exact_mod_cast h.2
  have identity : input f.val*output f.val-1 =
      ((2^20+(f.val : ℝ))*((Fp64Seed.interpolate false (f.val*8)/8 : Nat) : ℝ)-2^41)/2^41 := by
    dsimp [input,output]
    ring
  rw [identity,abs_div]
  have hh : |(2^20+(f.val : ℝ))*((Fp64Seed.interpolate false (f.val*8)/8 : Nat) : ℝ)-2^41| ≤ 2^21 :=
    abs_le.mpr ⟨lo, by linarith⟩
  norm_num at hh ⊢
  linarith

#print axioms truncate_product
#print axioms same_interpolation
#print axioms seed_product_bound
#print axioms normalized_error
end GB300Math.Rcp64Bounds
