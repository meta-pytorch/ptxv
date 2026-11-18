/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.BarrierMemoryPaths
import MemoryWrite.RegisterGraphPublication

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex MemoryWrite.BarrierReach

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

structure ProgramSourceInputs (before : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize)) where
  codeStorage : code.val.length < Usize.max
  targetStorage : ∀ instruction ∈ code.val, TargetStorage instruction.op
  returnStorage : ∀ row ∈ returns.val, row.val.length < Usize.max
  cfgProduced : control_flow.graph_successors code cut fold returns = .ok before.succ
  targetValidity : ∀ (pc : Nat) (instruction : Inst), code.val[pc]? = some instruction →
    TargetsBounded instruction.op code.val.length
  returnValidity : ∀ (pc : Nat) (row : alloc.vec.Vec Usize), returns.val[pc]? = some row →
    ∀ destination ∈ row.val, destination.val < code.val.length
  reachLength : before.reach.val.length = code.val.length
  reachProduced : reachFromSource code (alloc.vec.Vec.deref before.succ) = .ok before.reach
  destinations : Slice (alloc.vec.Vec ProgramKey)
  destinationLength : destinations.val.length = code.val.length
  destinationSource : SourceMemoryRows (code.val.map Inst.op) destinations.val
  registerSpans : ∀ operation ∈ code.val.map Inst.op, optionBounded (memoryWrites operation)
  buildFits : BuildFits destinations.val

def ProgramSourceInputs.publication
    {before : PtxvMemoryWrite.cache_graph.MayWriteGraph}
    {code : Slice Inst} {cut : U8} {fold : Slice (Option Bool)} {returns : Slice (alloc.vec.Vec Usize)}
    (source : ProgramSourceInputs before code cut fold returns)
    (index : PtxvMemoryWrite.register_index.RegisterIndex) (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (published : PublicationPost before source.destinations (index, graph)) :
    ProgramPublication graph code cut fold returns := {
  codeStorage := source.codeStorage
  targetStorage := source.targetStorage
  returnStorage := source.returnStorage
  cfgProduced := by rw [published.2.2.2.1]; exact source.cfgProduced
  targetValidity := source.targetValidity
  returnValidity := source.returnValidity
  reachLength := by rw [published.2.2.1]; exact source.reachLength
  destinations := source.destinations
  destinationLength := source.destinationLength
  destinationSource := source.destinationSource
  registerSpans := source.registerSpans
  index := index
  writes := graph.writes
  built := published.2.2.2.2.2
  graphWrites := rfl
  graphCount := published.2.2.2.2.1
}

theorem published_source_analysis_and_queries_return_true
    (before : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (source : ProgramSourceInputs before code cut fold returns) :
    ∃ index graph, PtxvMemoryWrite.register_index.publish_graph before source.destinations = .ok (index, graph) ∧
      ∀ (mask : U32) (bit : Bit), HasBit mask bit → ∀ (start : Usize), start.val < code.val.length →
        AnalysisFits graph → ∃ sets,
          PtxvMemoryWrite.cache_graph.may_write_sets graph start mask = .ok sets ∧
          ∀ (pc : Usize) (register : Reg),
            RawMemoryWrittenBefore graph code.val cut fold.val returns.val mask start.val pc.val register →
            ToBarrier graph code.val cut fold.val returns.val mask bit pc.val →
            Std.bind (programKeyFromSource register) (fun identity =>
              Std.bind (PtxvMemoryWrite.register_index.RegisterIndex.get index identity)
                (fun dense => PtxvMemoryWrite.cache_graph.may_at sets pc dense)) = .ok true := by
  obtain ⟨⟨index, graph⟩, published, specification⟩ := MemoryWrite.spec_returns _ _
    (rust_publish_graph before source.destinations source.buildFits)
  refine ⟨index, graph, published, ?_⟩
  intro mask bit requested start inside fits
  let publication := source.publication index graph specification
  have reachProduced : reachFromSource code (alloc.vec.Vec.deref graph.succ) = .ok graph.reach := by
    rw [specification.2.2.2.1, specification.2.2.1]
    exact source.reachProduced
  simpa only [publication, ProgramSourceInputs.publication] using
    actual_barrier_memory_analysis_and_queries_return_true graph code cut fold returns publication
      reachProduced mask bit requested start inside fits

end Instruction.ControlFlow
