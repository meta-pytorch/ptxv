/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MultimemReduction
set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 4096
namespace GB300Math.MultimemReduction

def witnessInputs : List Nat := [2319920140,1732455079,1730440352,3213447358]

def binaryWitnessResults : Option (List Nat) := do
  let results ← orders.mapM fun order => parenthesizations "add" "f32"
    witnessInputs[order[0]!]! witnessInputs[order[1]!]!
    witnessInputs[order[2]!]! witnessInputs[order[3]!]!
  return uniqueSorted results.flatten

theorem all_binary_orders_return_same_word :
    binaryWitnessResults = some [1739836324] := by decide +kernel

theorem one_final_round_returns_adjacent_word :
    exactFiniteSum "f32" witnessInputs = some 1739836323 := by decide +kernel

theorem captured_word_outside_binary_orders :
    (binaryWitnessResults.map fun words => words.contains 1739836323) = some false := by
  rw [all_binary_orders_return_same_word]
  decide

#print axioms all_binary_orders_return_same_word
#print axioms one_final_round_returns_adjacent_word
#print axioms captured_word_outside_binary_orders
end GB300Math.MultimemReduction
