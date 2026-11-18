/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.EffectsComposition

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.ActionEffects

theorem invalidated_batch_writes (batch : barrier.ArrivalBatch) :
    (do let invalid ← barrier.invalidate_arrival_batch batch
        barrier.arrival_batch_writes invalid) = .ok true := by
  simp [barrier.invalidate_arrival_batch, barrier.arrival_batch_writes]

theorem record_arrival_invalidation_commutes (state : effects.FoldState)
    (key : effects.PhaseKey) (arrival : barrier.ArrivalStep) :
    effects.record_arrival { state with invalidated := true } key arrival = (do
      let next ← effects.record_arrival state key arrival
      .ok { next with invalidated := true }) := by
  unfold effects.record_arrival
  simp only [Std.bind_assoc]
  congr 1
  funext found
  cases found with
  | none =>
    simp only [Std.bind_assoc, bind_ok, Bool.true_or]
    congr 1
    funext batch
    congr 1
    funext entries
    congr 1
    funext index
    change (do
      let pair ← alloc.vec.Vec.index_mut
        (core.slice.index.SliceIndexUsizeSlice effects.ArrivalEntry) entries index
      let batch ← barrier.record_arrival_batch pair.1.batch arrival
      .ok { state with arrivals := pair.2 { pair.1 with batch := batch }, invalidated := true }) =
      (do
        let next ← (do
          let pair ← alloc.vec.Vec.index_mut
            (core.slice.index.SliceIndexUsizeSlice effects.ArrivalEntry) entries index
          let batch ← barrier.record_arrival_batch pair.1.batch arrival
          .ok { state with arrivals := pair.2 { pair.1 with batch := batch }, invalidated := state.invalidated || (alloc.vec.Vec.len state.arrivals != 0#usize) })
        .ok { next with invalidated := true })
    simp only [Std.bind_assoc, bind_ok]
    rfl
  | some index =>
    simp only [bind_ok]
    change (do
      let pair ← alloc.vec.Vec.index_mut
        (core.slice.index.SliceIndexUsizeSlice effects.ArrivalEntry) state.arrivals index
      let batch ← barrier.record_arrival_batch pair.1.batch arrival
      .ok { state with arrivals := pair.2 { pair.1 with batch := batch }, invalidated := true }) =
      (do
        let next ← (do
          let pair ← alloc.vec.Vec.index_mut
            (core.slice.index.SliceIndexUsizeSlice effects.ArrivalEntry) state.arrivals index
          let batch ← barrier.record_arrival_batch pair.1.batch arrival
          .ok { state with arrivals := pair.2 { pair.1 with batch := batch } })
        .ok { next with invalidated := true })
    simp only [Std.bind_assoc, bind_ok]
    rfl

end MemoryWrite.ActionEffects
