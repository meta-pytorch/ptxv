/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowGraphRefinement
import Instruction.BarrierSeedRows
import Instruction.PartialSuccess

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

def GraphSeedPost (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool))
    (returns : Slice (alloc.vec.Vec Usize)) (source : control_flow.GraphSource) : Prop :=
  control_flow.graph_successors code cut fold returns = .ok source.successors ∧
  control_flow.barrier_seeds code = .ok source.seeds ∧
  source.successors.val.map (fun row => row.val.map UScalar.val) =
    graphRows code.val cut fold.val returns.val ∧ BarrierSeed.Rows code.val source.seeds.val

@[step]
theorem rust_graph_and_seeds (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool))
    (returns : Slice (alloc.vec.Vec Usize))
    (space : ∀ pc < code.val.length,
      (graphRows code.val cut fold.val returns.val)[pc]!.length ≤ Usize.max) :
    control_flow.graph_and_seeds code cut fold returns ⦃ source =>
      GraphSeedPost code cut fold returns source ⦄ := by
  obtain ⟨successors, successorsOk, successorsMeaning⟩ := Instruction.Partial.spec_returns _ _
    (rust_graph_successors code cut fold returns space)
  obtain ⟨seeds, seedsOk, seedMeaning⟩ := Instruction.Partial.spec_returns _ _
    (BarrierSeed.rust_barrier_seeds code)
  unfold control_flow.graph_and_seeds
  rw [successorsOk]
  simp only [bind_ok]
  rw [seedsOk]
  simp only [bind_ok, WP.spec_ok]
  exact ⟨successorsOk, seedsOk, successorsMeaning, seedMeaning⟩

end Instruction.ControlFlow
