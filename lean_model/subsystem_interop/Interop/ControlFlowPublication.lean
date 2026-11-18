/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.ControlFlowBounds
import Instruction.ControlFlowStorage

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

structure ProgramPublication (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize)) where
  codeStorage : code.val.length < Usize.max
  targetStorage : ∀ instruction ∈ code.val, TargetStorage instruction.op
  returnStorage : ∀ row ∈ returns.val, row.val.length < Usize.max
  cfgProduced : control_flow.graph_successors code cut fold returns = .ok graph.succ
  targetValidity : ∀ (pc : Nat) (instruction : Inst), code.val[pc]? = some instruction →
    TargetsBounded instruction.op code.val.length
  returnValidity : ∀ (pc : Nat) (row : alloc.vec.Vec Usize), returns.val[pc]? = some row →
    ∀ destination ∈ row.val, destination.val < code.val.length
  reachLength : graph.reach.val.length = code.val.length
  destinations : Slice (alloc.vec.Vec ProgramKey)
  destinationLength : destinations.val.length = code.val.length
  destinationSource : SourceMemoryRows (code.val.map Inst.op) destinations.val
  registerSpans : ∀ operation ∈ code.val.map Inst.op, optionBounded (memoryWrites operation)
  index : PtxvMemoryWrite.register_index.RegisterIndex
  writes : alloc.vec.Vec (alloc.vec.Vec Usize)
  built : PtxvMemoryWrite.register_index.build_writes destinations = .ok (index, writes)
  graphWrites : graph.writes = writes
  graphCount : graph.register_count = index.count

theorem published_program_bounded (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (publication : ProgramPublication graph code cut fold returns) : GraphBounded graph := by
  have storage := graph_rows_fit_storage code.val cut fold.val returns.val
    publication.codeStorage publication.targetStorage publication.returnStorage
  exact actual_graph_is_bounded graph code cut fold returns storage publication.cfgProduced
    publication.targetValidity publication.returnValidity publication.reachLength publication.destinations
    publication.destinationLength publication.index publication.writes publication.built
    publication.graphWrites publication.graphCount

theorem published_program_analysis_and_queries_return_true
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (publication : ProgramPublication graph code cut fold returns) (start : Usize) (bit : U32)
    (startBound : start.val < code.val.length) (fits : AnalysisFits graph) :
    ∃ sets, PtxvMemoryWrite.cache_graph.may_write_sets graph start bit = .ok sets ∧
      ∀ (pc : Usize) (register : Reg),
        CfgMemoryWrittenBefore graph code.val cut fold.val returns.val bit start.val pc.val register →
        Std.bind (programKeyFromSource register) (fun identity =>
          Std.bind (PtxvMemoryWrite.register_index.RegisterIndex.get publication.index identity)
            (fun dense => PtxvMemoryWrite.cache_graph.may_at sets pc dense)) = .ok true := by
  have storage := graph_rows_fit_storage code.val cut fold.val returns.val
    publication.codeStorage publication.targetStorage publication.returnStorage
  have bounded := published_program_bounded graph code cut fold returns publication
  obtain ⟨sets, analyzed, queries⟩ := actual_cfg_memory_analysis_returns_and_query_is_true graph
    code cut fold returns storage publication.cfgProduced publication.destinations publication.destinationSource
    publication.registerSpans publication.index publication.writes publication.built publication.graphWrites
    start bit bounded (by rw [publication.reachLength]; exact startBound) fits
  refine ⟨sets, analyzed, ?_⟩
  intro pc register written
  obtain ⟨dense, identity, lookup, query⟩ := queries pc register written
  simp only [identity, bind_ok, lookup, query]

end Instruction.ControlFlow
