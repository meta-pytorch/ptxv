/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.EffectsNamespaces

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 2000000

namespace MemoryWrite.ActionEffects

def sequence : List effects.Effect → effects.FoldState → Result effects.FoldState
  | [], state => .ok state
  | event :: rest, state => do
    let next ← effects.apply state event
    sequence rest next

theorem sequence_append (first second : List effects.Effect) (state : effects.FoldState) :
    sequence (first ++ second) state = (do
      let middle ← sequence first state
      sequence second middle) := by
  induction first generalizing state with
  | nil => simp [sequence]
  | cons event rest ih => simp [sequence, ih, Std.bind_assoc]

theorem source_finding (state : effects.FoldState) :
    effects.apply state .Finding = .ok { state with invalidated := true } := rfl

theorem source_fault (state : effects.FoldState) :
    effects.apply state .Fault = .ok {
      state with summary := { state.summary with all := true }, invalidated := true
    } := rfl

theorem fault_preserves_observations (state : effects.FoldState) :
    effects.apply state .Fault ⦃ next =>
      next.summary.reads = state.summary.reads ∧
      next.summary.writes = state.summary.writes ∧
      next.summary.releases = state.summary.releases ∧
      next.summary.acquires = state.summary.acquires ∧
      next.arrivals = state.arrivals ∧ next.invalidated = true ∧ next.summary.all = true ⦄ := by
  simp [source_fault]

theorem finding_preserves_observations (state : effects.FoldState) :
    effects.apply state .Finding ⦃ next =>
      next.summary = state.summary ∧ next.arrivals = state.arrivals ∧
      next.invalidated = true ⦄ := by
  simp [source_finding]

end MemoryWrite.ActionEffects
