/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import All_e4m3_add
import All_e4m3_sub
import All_e4m3_mul
import All_e5m2_add
import All_e5m2_sub
import All_e5m2_mul
import AlternateFloatPairRational
import AlternateFloatAddExact
set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 1024
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem e4m3_add_rational_error_bound (a b : Fin 256) (x y ideal output : Value) (z r : Int)
    (hx : decode "e4m3" a.val = some x) (hy : decode "e4m3" b.val = some y)
    (hi : add x y = ideal) (hz : gridInteger ideal = some z)
    (ho : decode "e4m3" (encode "e4m3" false ideal) = some output)
    (hr : gridInteger output = some r) :
    |(r : ℚ)/unit-(z : ℚ)/unit| ≤ max (|(z : ℚ)/unit|/17) (1/2^10) := by
  apply e4m3_grid_to_rational
  have hb := pair_bound_extract "add" "e4m3" "e4m3" "e4m3" a.val b.val
    x y ideal output z r hx hy (by simpa using hi) hz ho hr
    (e4m3_add_relative_or_subnormal_bound a b)
  simpa using hb
#print axioms e4m3_add_rational_error_bound

theorem e4m3_sub_rational_error_bound (a b : Fin 256) (x y ideal output : Value) (z r : Int)
    (hx : decode "e4m3" a.val = some x) (hy : decode "e4m3" b.val = some y)
    (hi : add x (negate y) = ideal) (hz : gridInteger ideal = some z)
    (ho : decode "e4m3" (encode "e4m3" false ideal) = some output)
    (hr : gridInteger output = some r) :
    |(r : ℚ)/unit-(z : ℚ)/unit| ≤ max (|(z : ℚ)/unit|/17) (1/2^10) := by
  apply e4m3_grid_to_rational
  have hb := pair_bound_extract "sub" "e4m3" "e4m3" "e4m3" a.val b.val
    x y ideal output z r hx hy (by simpa using hi) hz ho hr
    (e4m3_sub_relative_or_subnormal_bound a b)
  simpa using hb
#print axioms e4m3_sub_rational_error_bound

theorem e4m3_mul_rational_error_bound (a b : Fin 256) (x y ideal output : Value) (z r : Int)
    (hx : decode "e4m3" a.val = some x) (hy : decode "e4m3" b.val = some y)
    (hi : mul x y = ideal) (hz : gridInteger ideal = some z)
    (ho : decode "e4m3" (encode "e4m3" false ideal) = some output)
    (hr : gridInteger output = some r) :
    |(r : ℚ)/unit-(z : ℚ)/unit| ≤ max (|(z : ℚ)/unit|/17) (1/2^10) := by
  apply e4m3_grid_to_rational
  have hb := pair_bound_extract "mul" "e4m3" "e4m3" "e4m3" a.val b.val
    x y ideal output z r hx hy (by simpa using hi) hz ho hr
    (e4m3_mul_relative_or_subnormal_bound a b)
  simpa using hb
#print axioms e4m3_mul_rational_error_bound

theorem e5m2_add_rational_error_bound (a b : Fin 256) (x y ideal output : Value) (z r : Int)
    (hx : decode "e5m2" a.val = some x) (hy : decode "e5m2" b.val = some y)
    (hi : add x y = ideal) (hz : gridInteger ideal = some z)
    (ho : decode "e5m2" (encode "e5m2" false ideal) = some output)
    (hr : gridInteger output = some r) :
    |(r : ℚ)/unit-(z : ℚ)/unit| ≤ max (|(z : ℚ)/unit|/9) (1/2^17) := by
  apply e5m2_grid_to_rational
  have hb := pair_bound_extract "add" "e5m2" "e5m2" "e5m2" a.val b.val
    x y ideal output z r hx hy (by simpa using hi) hz ho hr
    (e5m2_add_relative_or_subnormal_bound a b)
  simpa using hb
#print axioms e5m2_add_rational_error_bound

theorem e5m2_sub_rational_error_bound (a b : Fin 256) (x y ideal output : Value) (z r : Int)
    (hx : decode "e5m2" a.val = some x) (hy : decode "e5m2" b.val = some y)
    (hi : add x (negate y) = ideal) (hz : gridInteger ideal = some z)
    (ho : decode "e5m2" (encode "e5m2" false ideal) = some output)
    (hr : gridInteger output = some r) :
    |(r : ℚ)/unit-(z : ℚ)/unit| ≤ max (|(z : ℚ)/unit|/9) (1/2^17) := by
  apply e5m2_grid_to_rational
  have hb := pair_bound_extract "sub" "e5m2" "e5m2" "e5m2" a.val b.val
    x y ideal output z r hx hy (by simpa using hi) hz ho hr
    (e5m2_sub_relative_or_subnormal_bound a b)
  simpa using hb
#print axioms e5m2_sub_rational_error_bound

theorem e5m2_mul_rational_error_bound (a b : Fin 256) (x y ideal output : Value) (z r : Int)
    (hx : decode "e5m2" a.val = some x) (hy : decode "e5m2" b.val = some y)
    (hi : mul x y = ideal) (hz : gridInteger ideal = some z)
    (ho : decode "e5m2" (encode "e5m2" false ideal) = some output)
    (hr : gridInteger output = some r) :
    |(r : ℚ)/unit-(z : ℚ)/unit| ≤ max (|(z : ℚ)/unit|/9) (1/2^17) := by
  apply e5m2_grid_to_rational
  have hb := pair_bound_extract "mul" "e5m2" "e5m2" "e5m2" a.val b.val
    x y ideal output z r hx hy (by simpa using hi) hz ho hr
    (e5m2_mul_relative_or_subnormal_bound a b)
  simpa using hb
#print axioms e5m2_mul_rational_error_bound

def pairValues (op fmt : String) (a b : Nat) : Option (ℚ × ℚ) := do
  let x ← decode fmt a
  let y ← decode fmt b
  let ideal := if op = "add" then add x y else if op = "sub" then add x (negate y) else mul x y
  let exact ← rational (some ideal)
  let rounded ← rational (decode fmt (encode fmt false ideal))
  return (exact, rounded)

theorem e4m3_relative_tight : pairValues "add" "e4m3" 56 24 = some (17/16, 1) := by decide +kernel
theorem e5m2_relative_tight : pairValues "add" "e5m2" 60 48 = some (9/8, 1) := by decide +kernel
theorem e4m3_subnormal_tight : pairValues "mul" "e4m3" 1 48 = some (1/2^10, 0) := by decide +kernel
theorem e5m2_subnormal_tight : pairValues "mul" "e5m2" 1 56 = some (1/2^17, 0) := by decide +kernel
#print axioms e4m3_relative_tight
#print axioms e5m2_relative_tight
#print axioms e4m3_subnormal_tight
#print axioms e5m2_subnormal_tight
end GB300Math.AlternateFloatFloatLib
