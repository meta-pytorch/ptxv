/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricBF16
import GB300Math.TensorCancellationBounds

namespace FabricBF16Block
open GB300Math.TensorAlignmentLoss GB300Math.TensorCancellationBounds

theorem signed_accumulation_bound (words : List Nat) (values : List Int) :
    -negativeBudget values (2^(latticeShift words)) ≤ exactSum values-aligned words values ∧
    exactSum values-aligned words values ≤ positiveBudget values (2^(latticeShift words)) := by
  exact cancellation_aware_loss values (2^(latticeShift words)) (Nat.two_pow_pos _)

theorem positive_alignment_guard (words : List Nat) (values : List Int)
    (h : positiveBudget values (2^(latticeShift words)) < exactSum values) :
    0 < aligned words values := by
  exact positive_ideal_retains_sign values (2^(latticeShift words)) (Nat.two_pow_pos _) h

theorem negative_alignment_guard (words : List Nat) (values : List Int)
    (h : exactSum values < -negativeBudget values (2^(latticeShift words))) :
    aligned words values < 0 := by
  exact negative_ideal_retains_sign values (2^(latticeShift words)) (Nat.two_pow_pos _) h

theorem fixed_exponent_pair_cancels (words : List Nat) (n : Int) (values : List Int) :
    aligned words (n :: -n :: values) = aligned words values := by
  exact fixed_lattice_pair_cancels n values (2^(latticeShift words))

#print axioms signed_accumulation_bound
#print axioms positive_alignment_guard
#print axioms negative_alignment_guard
#print axioms fixed_exponent_pair_cancels
end FabricBF16Block
