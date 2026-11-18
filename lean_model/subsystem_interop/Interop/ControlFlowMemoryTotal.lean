/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.ControlFlowMemoryInterop
import MemoryWrite.MayWriteQueryTotal

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

theorem actual_cfg_memory_analysis_returns_and_query_is_true
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (rowSpace : ∀ pc < code.val.length, (graphRows code.val cut fold.val returns.val)[pc]!.length ≤ Usize.max)
    (cfgProduced : control_flow.graph_successors code cut fold returns = .ok graph.succ)
    (destinations : Slice (alloc.vec.Vec ProgramKey))
    (source : SourceMemoryRows (code.val.map Inst.op) destinations.val)
    (boundedCode : ∀ operation ∈ code.val.map Inst.op, optionBounded (memoryWrites operation))
    (index : PtxvMemoryWrite.register_index.RegisterIndex)
    (writes : alloc.vec.Vec (alloc.vec.Vec Usize))
    (built : PtxvMemoryWrite.register_index.build_writes destinations = .ok (index, writes))
    (graphWrites : graph.writes = writes) (start : Usize) (bit : U32)
    (boundedGraph : GraphBounded graph) (startBound : start.val < graph.reach.val.length)
    (fits : AnalysisFits graph) :
    ∃ sets, PtxvMemoryWrite.cache_graph.may_write_sets graph start bit = .ok sets ∧
      ∀ (pc : Usize) (register : Reg),
        CfgMemoryWrittenBefore graph code.val cut fold.val returns.val bit start.val pc.val register →
        ∃ dense,
          programKeyFromSource register = .ok (programKey register) ∧
          PtxvMemoryWrite.register_index.RegisterIndex.get index (programKey register) = .ok dense ∧
          PtxvMemoryWrite.cache_graph.may_at sets pc dense = .ok true := by
  obtain ⟨sets, analyzed, _⟩ := MemoryWrite.spec_returns _ _
    (rust_may_write_sets_total graph start bit boundedGraph startBound fits)
  refine ⟨sets, analyzed, ?_⟩
  intro pc register written
  obtain ⟨identity, identityProduced, identityValue⟩ := specification_returns _ _ (rust_program_key register)
  have identityProduced : programKeyFromSource register = .ok (programKey register) := by
    simpa only [identityValue] using identityProduced
  have builtSpec := successful_build_writes destinations index writes built
  obtain ⟨dense, lookup, _⟩ := MemoryWrite.spec_returns _ _ (rust_get index (programKey register) builtSpec.1.1)
  obtain ⟨answer, queried, _⟩ := MemoryWrite.spec_returns _ _ (rust_may_at_total sets pc dense)
  have answerTrue := actual_cfg_memory_query_catches_every_path_write graph code cut fold returns
    rowSpace cfgProduced destinations source boundedCode index writes built graphWrites start pc register bit
    boundedGraph startBound sets analyzed written (programKey register) identityProduced dense lookup answer queried
  exact ⟨dense, identityProduced, lookup, by simpa only [answerTrue] using queried⟩

end Instruction.ControlFlow
