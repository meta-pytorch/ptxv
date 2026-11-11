/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorScaledBlackwell

namespace GB300Math.TensorScaledWitness
open GB300Math.Tensor GB300Math.TensorScaledIdeal

def operand : Array Nat :=
  (Array.range 64).map fun i => if i = 0 then 2 else if i = 32 then 1 else 0

def asRat (x : GB300Math.Tensor.Dyadic) : Rat :=
  if x.exponent ≥ 0 then (x.significand : Rat) * 2^x.exponent.toNat
  else (x.significand : Rat) / 2^(-x.exponent).toNat

def ideal (scaleCode : Nat) : Option Rat :=
  (exactDot false operand operand #[127,scaleCode] #[127,127] 0xbf800000).map asRat

def predicted (scaleCode : Nat) : Option Nat :=
  TensorScaledBlackwell.evaluate false operand operand #[127,scaleCode] #[127,127]
    0xbf800000 (-10000)

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem lower_boundary_preserved :
    ideal 94 = some ((1:Rat)/2^35) ∧ predicted 94 = some 0x2e000000 := by decide +kernel

theorem upper_boundary_lost :
    ideal 93 = some ((1:Rat)/2^36) ∧ predicted 93 = some 0 := by decide +kernel

theorem finite_absolute_and_relative_error :
    let exact : Rat := (1:Rat)/2^36
    let observed := asRat (decode 0 23 8)
    exact - observed = (1:Rat)/2^36 ∧ (exact-observed)/exact = 1 := by decide +kernel

theorem lost_result_is_normal_f32 :
    asRat (decode 0x2d800000 23 8) = (1:Rat)/2^36 ∧
    ¬ isZero 0x2d800000 23 8 ∧ ¬ isInf 0x2d800000 23 8 ∧
    ¬ isNaN 0x2d800000 23 8 := by decide +kernel

#print axioms lower_boundary_preserved
#print axioms upper_boundary_lost
#print axioms finite_absolute_and_relative_error
#print axioms lost_result_is_normal_f32
end GB300Math.TensorScaledWitness
