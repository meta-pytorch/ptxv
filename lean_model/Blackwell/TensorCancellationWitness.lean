/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorBlackwell

namespace GB300Math.TensorCancellationWitness
open GB300Math.Tensor

def a : Array Nat := #[0x3c00,0xbc00] ++ Array.replicate 14 0x0800
def b : Array Nat := #[0x3c00,0x3c00] ++ Array.replicate 14 0x0800

def exactUnits : Int := (Array.range 16).foldl (fun total j =>
  let x := decode a[j]! 10 5
  let y := decode b[j]! 10 5
  total + shiftTruncate (x.significand*y.significand) (x.exponent+y.exponent+26)) 0

def outputUnits (word : Nat) : Int :=
  let x := decode word 23 8
  shiftTruncate x.significand (x.exponent+26)

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem all_inputs_finite :
    (a++b).all (fun x => !(isNaN x 10 5 || isInf x 10 5)) = true := by decide +kernel

theorem exact_dyadic_sum : exactUnits = 14 := by decide +kernel

theorem blackwell_prediction :
    TensorBlackwell.alignedDot a b 0 10 5 true = 0 := by decide +kernel

theorem blackwell_absolute_error :
    (exactUnits-outputUnits 0).natAbs = 14 := by decide +kernel

#print axioms all_inputs_finite
#print axioms exact_dyadic_sum
#print axioms blackwell_prediction
#print axioms blackwell_absolute_error
end GB300Math.TensorCancellationWitness
