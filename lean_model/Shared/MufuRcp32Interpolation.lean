/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuRcp32Bounds

namespace GB300Math.MufuRcp32Bounds

set_option maxRecDepth 10000
set_option maxHeartbeats 0

def rawSum (j k : Nat) : Int :=
  let (c0,c1,c2) := MufuTables.rcp.coefficients[j]!
  c0*2^32 + 1060634624 + c1*(k : Int)*2^19 +
    c2*(Fp64Seed.squareTruncated k 17 : Int)*4

theorem rounded_product_bound (m s : Int) (hm : 0 ≤ m)
    (lower : 0 ≤ m*s - 2^81 + (2^24-m)*2^34)
    (upper : m*s ≤ 2^81 + 2^58) :
    -(2^24) ≤ m*(s/2^34)-2^47 ∧ m*(s/2^34)-2^47 ≤ 2^24 := by
  have remainder_nonnegative := Int.emod_nonneg s (by norm_num : (2^34 : Int) ≠ 0)
  have remainder_small := Int.emod_lt_of_pos s (by norm_num : (0 : Int) < 2^34)
  have identity := Int.ediv_mul_add_emod s (2^34)
  have floored : (s/2^34)*2^34 ≤ s := by omega
  have ceiling : s ≤ (s/2^34)*2^34 + 2^34 := by omega
  have first := mul_le_mul_of_nonneg_left floored hm
  have second := mul_le_mul_of_nonneg_left ceiling hm
  constructor <;> nlinarith

theorem segment_product_bound (j : Fin 128) (k : Fin 65536)
    (square : squareValid k.val = true) :
    -(2^24) ≤ (2^23+(j.val : Int)*2^16+k.val)*(rawSum j.val k.val/2^34)-2^47 ∧
    (2^23+(j.val : Int)*2^16+k.val)*(rawSum j.val k.val/2^34)-2^47 ≤ 2^24 := by
  let row := MufuTables.rcp.coefficients[j.val]!
  let c0 := row.1
  let c1 := row.2.1
  let c2 := row.2.2
  let m : Int := 2^23+(j.val : Int)*2^16+k.val
  let hi : Int := c0*2^32 + 1060634624 + c1*(k.val : Int)*2^19 + c2*(k.val : Int)^2*4
  let lo := hi-c2*4*1048577
  have hc : 0 ≤ c2 := (coefficient_certificates j).1
  have hm : 0 ≤ m := by dsimp [m]; positivity
  have sq := square
  simp only [squareValid, Bool.and_eq_true, decide_eq_true_eq] at sq
  have sqlo : (k.val : Int)^2 ≤ (Fp64Seed.squareTruncated k.val 17 : Int)+1048577 := by
    norm_num only [pow_two]
    exact_mod_cast sq.1
  have sqhi : (Fp64Seed.squareTruncated k.val 17 : Int) ≤ (k.val : Int)^2 := by
    norm_num only [pow_two]
    exact_mod_cast sq.2
  have sqlo_mul := mul_le_mul_of_nonneg_left sqlo hc
  have sqhi_mul := mul_le_mul_of_nonneg_left sqhi hc
  have sumlo : lo ≤ rawSum j.val k.val := by
    change hi-c2*4*1048577 ≤ c0*2^32+1060634624+c1*(k.val : Int)*2^19+
      c2*(Fp64Seed.squareTruncated k.val 17 : Int)*4
    dsimp [hi]
    nlinarith
  have sumhi : rawSum j.val k.val ≤ hi := by
    change c0*2^32+1060634624+c1*(k.val : Int)*2^19+
      c2*(Fp64Seed.squareTruncated k.val 17 : Int)*4 ≤ hi
    dsimp [hi]
    nlinarith
  have low := lower_nonnegative j k
  have high := upper_nonnegative j k
  have low_identity : (lowerPoly j.val).eval k.val =
      m*lo-2^81+(2^24-m)*2^34 := by
    simp only [lowerPoly, Cubic.eval]
    dsimp [m, lo, hi, c0, c1, c2, row]
    ring
  have high_identity : (upperPoly j.val).eval k.val =
      2^81+2^58-m*hi := by
    simp only [upperPoly, Cubic.eval]
    dsimp [m, hi, c0, c1, c2, row]
    ring
  rw [low_identity] at low
  rw [high_identity] at high
  have lowmul := mul_le_mul_of_nonneg_left sumlo hm
  have highmul := mul_le_mul_of_nonneg_left sumhi hm
  exact rounded_product_bound m (rawSum j.val k.val) hm (by nlinarith) (by nlinarith)

#print axioms rounded_product_bound
#print axioms segment_product_bound

end GB300Math.MufuRcp32Bounds
