/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexBuildTotal

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

def PublicationPost (before : cache_graph.MayWriteGraph)
    (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (result : register_index.RegisterIndex × cache_graph.MayWriteGraph) : Prop :=
  Valid result.1 ∧ GraphRepresents result.1 destinations.val result.2.writes.val ∧
  result.2.reach = before.reach ∧ result.2.succ = before.succ ∧
  result.2.register_count = result.1.count ∧
  register_index.build_writes destinations = .ok (result.1, result.2.writes)

@[step]
theorem rust_publish_graph (before : cache_graph.MayWriteGraph)
    (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey)) (fits : BuildFits destinations.val) :
    register_index.publish_graph before destinations ⦃ result => PublicationPost before destinations result ⦄ := by
  obtain ⟨⟨index, writes⟩, built, valid, represents, countBound, pagesBound⟩ := MemoryWrite.spec_returns _ _
    (rust_build_writes_total destinations fits)
  unfold register_index.publish_graph
  rw [built]
  simp only [bind_ok, register_index.RegisterIndex.len]
  change WP.spec (.ok (index, { before with writes := writes, register_count := index.count }))
    (PublicationPost before destinations)
  rw [WP.spec_ok]
  exact ⟨valid, represents, rfl, rfl, rfl, built⟩

end MemoryWrite.RegisterIndex
