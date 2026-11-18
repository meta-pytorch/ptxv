/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReachTotal

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.BarrierReach

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.GraphSource

def BuildPost (successors : alloc.vec.Vec (alloc.vec.Vec Usize))
    (seeds : alloc.vec.Vec U32) (graph : cache_graph.MayWriteGraph) : Prop :=
  graph.succ = successors ∧ graph.writes.val = [] ∧ graph.register_count = 0#usize ∧
  barrier_reach.propagate (alloc.vec.Vec.deref successors) seeds = .ok graph.reach ∧
  PropagationPost successors.val seeds.val graph.reach.val

@[step]
theorem rust_build (successors : alloc.vec.Vec (alloc.vec.Vec Usize))
    (seeds : alloc.vec.Vec U32) (bounded : EdgesBounded successors.val seeds.val.length) :
    graph_source.build successors seeds ⦃ graph => BuildPost successors seeds graph ⦄ := by
  obtain ⟨reach, propagated, meaning⟩ := MemoryWrite.spec_returns _ _
    (rust_propagate (alloc.vec.Vec.deref successors) seeds
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using bounded))
  unfold graph_source.build
  dsimp only
  rw [propagated]
  simp only [bind_ok, WP.spec_ok]
  refine ⟨rfl, rfl, rfl, propagated, ?_⟩
  simpa only [alloc.vec.Vec.deref, Slice.from_val] using meaning

end MemoryWrite.GraphSource
