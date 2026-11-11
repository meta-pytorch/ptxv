/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TensorPermutationNormalCases

namespace GB300Math.TensorPartitionCases
open GB300Math.Tensor GB300Math.TensorPermutationCases
open GB300Math.TensorPermutationNormalCases

def order : Array Nat := #[1, 3, 5, 7, 9, 11, 13, 14, 0, 2, 4, 6, 8, 10, 12, 15]
def negativeFirstA := order.map fun i => normalA[i]!
def negativeFirstB := order.map fun i => normalB[i]!

set_option maxRecDepth 100000
set_option maxHeartbeats 3000000

theorem every_index_once :
    (List.range 16).all (fun i => (order.toList.filter (· == i)).length == 1) = true := by
  decide +kernel

theorem exact_result_positive :
    exactDot negativeFirstA negativeFirstB normalC = (1 : Rat)/2^29 := by decide +kernel

theorem blackwell_zero_prediction :
    TensorLegacyBlackwell.dot "f16" "f32" negativeFirstA negativeFirstB normalC = 0 := by
  decide +kernel

#print axioms every_index_once
#print axioms exact_result_positive
#print axioms blackwell_zero_prediction
end GB300Math.TensorPartitionCases
