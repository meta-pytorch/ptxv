/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockTick

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.SparseClock

def foldOwns : List Entry → List Entry → List Entry
  | base, [] => base
  | base, own :: rest => foldOwns (overlay base own) rest

theorem fold_owns_preserves_sortedness (base owns : List Entry) (sorted : Sorted base) :
    Sorted (foldOwns base owns) := by
  induction owns generalizing base with
  | nil => exact sorted
  | cons own rest ih => exact ih _ (overlay_sorted base own sorted)

theorem fold_owns_accumulates_pointwise_maximum (base owns : List Entry) :
    entriesValue (foldOwns base owns) =
      BarrierReduction.join (entriesValue base) (entriesValue owns) := by
  induction owns generalizing base with
  | nil => funext actor; simp [foldOwns, BarrierReduction.join, entriesValue]
  | cons own rest ih =>
    rw [foldOwns, ih, overlay_value]
    funext actor
    simp [BarrierReduction.join, value, entriesValue, Nat.max_assoc]

theorem any_sequence_of_own_insertions_has_sorted_base (state : State) (owns : List Entry)
    (formed : state.WellFormed) :
    (State.mk (foldOwns state.base owns) (0, 0)).WellFormed :=
  fold_owns_preserves_sortedness state.base owns formed

end MemoryWrite.SparseClock
