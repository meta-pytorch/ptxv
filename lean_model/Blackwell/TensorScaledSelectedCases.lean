/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorScaledSelected

namespace GB300Math.TensorScaledSelectedCases
open GB300Math.Tensor GB300Math.TensorScaledIdeal

def a : Array Nat := (Array.range 96).map fun j =>
  if j = 32 ∨ j = 48 then 1 else if j = 33 ∨ j = 49 then 2
  else if j = 64 ∨ j = 65 then 10 else 0

def b : Array Nat := (Array.range 96).map fun j =>
  if j = 32 ∨ j = 48 ∨ j = 64 then 1 else if j = 33 ∨ j = 49 then 3
  else if j = 65 then 5 else 0

def ratio (x : GB300Math.Tensor.Dyadic) : Rat :=
  if x.exponent ≥ 0 then (x.significand : Rat) * 2^x.exponent.toNat
  else (x.significand : Rat) / 2^(-x.exponent).toNat

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

theorem ideal_cancels :
    (exactDot false a b #[254,254,254] #[254,219,219] 0).map ratio = some 0 := by
  decide +kernel

theorem empty_scale_does_not_create_overflow :
    TensorScaledSelected.evaluate false a b #[254,254,254] #[254,219,219] 0
      = some 0 := by
  decide +kernel

theorem removing_unused_large_scale_restores_zero :
    TensorScaledSelected.evaluate false a b #[219,254,254] #[219,219,219] 0
      = some 0 := by
  decide +kernel

theorem fp4_zero_iff : ∀ x : Fin 16, fp4Twice x.val = 0 ↔ x.val % 8 = 0 := by
  decide +kernel

theorem product_zero_iff : ∀ x y : Fin 16,
    fp4Twice x.val * fp4Twice y.val = 0 ↔ x.val % 8 = 0 ∨ y.val % 8 = 0 := by
  decide +kernel

#print axioms fp4_zero_iff
#print axioms product_zero_iff
#print axioms ideal_cancels
#print axioms empty_scale_does_not_create_overflow
#print axioms removing_unused_large_scale_restores_zero
end GB300Math.TensorScaledSelectedCases
