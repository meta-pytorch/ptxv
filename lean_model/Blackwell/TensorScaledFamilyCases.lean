/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorScaledSelected
import GB300Math.TensorScaledSmall

namespace GB300Math.TensorScaledFamilyCases
open GB300Math.Tensor GB300Math.TensorScaledIdeal

def a (k : Nat) := #[9] ++ Array.replicate (k-1) 0
def b (k : Nat) := #[1] ++ Array.replicate (k-1) 0

def ratio (x : GB300Math.Tensor.Dyadic) : Rat :=
  if x.exponent ≥ 0 then (x.significand : Rat) * 2^x.exponent.toNat
  else (x.significand : Rat) / 2^(-x.exponent).toNat

set_option maxRecDepth 100000
set_option maxHeartbeats 3000000

theorem exact_dot :
    (exactDot false (a 64) (b 64) #[113,127] #[113,127] 0x3f800000).map ratio =
      some ((1073741823 : Rat)/1073741824) := by decide +kernel

theorem fp4_family_prediction :
    TensorScaledSelected.evaluate false (a 64) (b 64) #[113,127] #[113,127] 0x3f800000 =
      some 0x3f7fffff := by decide +kernel

theorem small_family_prediction :
    TensorScaledSmall.dot "e2m1" "e2m1" (a 32) (b 32) 0x3f800000 113 113 (some 25) =
      some 0x3f800000 := by decide +kernel

theorem fp4_absolute_error :
    ((1073741823 : Rat)/1073741824) - ratio (decode 0x3f7fffff 23 8) =
      (63 : Rat)/1073741824 := by decide +kernel

theorem small_absolute_error :
    ratio (decode 0x3f800000 23 8) - ((1073741823 : Rat)/1073741824) =
      (1 : Rat)/1073741824 := by decide +kernel

#print axioms exact_dot
#print axioms fp4_family_prediction
#print axioms small_family_prediction
#print axioms fp4_absolute_error
#print axioms small_absolute_error
end GB300Math.TensorScaledFamilyCases
