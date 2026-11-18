/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GraphSeedSource
import Interop.KernelGraphPublication
import MemoryWrite.GraphSource

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex MemoryWrite.BarrierReach
open Instruction.DestinationRows

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

structure KernelInputs (code : Slice Inst) (returns : Slice (alloc.vec.Vec Usize)) where
  codeStorage : code.val.length < Usize.max
  targetStorage : ∀ instruction ∈ code.val, TargetStorage instruction.op
  returnStorage : ∀ row ∈ returns.val, row.val.length < Usize.max
  targetValidity : ∀ (pc : Nat) (instruction : Inst), code.val[pc]? = some instruction →
    TargetsBounded instruction.op code.val.length
  returnValidity : ∀ (pc : Nat) (row : alloc.vec.Vec Usize), returns.val[pc]? = some row →
    ∀ destination ∈ row.val, destination.val < code.val.length
  registerSpans : ∀ operation ∈ code.val.map Inst.op, optionBounded (memoryWrites operation)

theorem source_successors_bounded (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool))
    (returns : Slice (alloc.vec.Vec Usize)) (input : KernelInputs code returns)
    (successors : alloc.vec.Vec (alloc.vec.Vec Usize))
    (produced : control_flow.graph_successors code cut fold returns = .ok successors) :
    EdgesBounded successors.val code.val.length := by
  have storage := graph_rows_fit_storage code.val cut fold.val returns.val
    input.codeStorage input.targetStorage input.returnStorage
  have exactRows := rust_graph_successors code cut fold returns storage
  simp only [produced, WP.spec_ok] at exactRows
  have lengthEq := congrArg List.length exactRows
  simp only [List.length_map, graph_rows_length] at lengthEq
  let graph : PtxvMemoryWrite.cache_graph.MayWriteGraph := {
    reach := alloc.vec.Vec.new _, succ := successors,
    writes := alloc.vec.Vec.new _, register_count := 0#usize }
  refine ⟨lengthEq, ?_⟩
  intro pc inside destination member
  have edge : (graphView graph 0#u32).edge pc destination.val := by
    refine ⟨successors.val[pc]!, ?_, destination, member, rfl⟩
    have rowBound : pc < successors.val.length := by omega
    rw [List.getElem?_eq_getElem rowBound, List.Inhabited_getElem_eq_getElem! _ _ rowBound]
  obtain ⟨instruction, instructionAt, admissible⟩ :=
    (actual_cfg_edges graph code cut fold returns storage produced 0#u32 pc destination.val).mp edge
  exact successors_bounded instruction.op instruction.guard
    (rowContext code.val.length cut fold.val returns.val pc)
    (input.targetValidity pc instruction instructionAt)
    (row_context_returns_bounded code.val.length cut fold.val returns.val input.returnValidity pc)
    destination.val admissible

def graphFromSource (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool))
    (returns : Slice (alloc.vec.Vec Usize)) : Result PtxvMemoryWrite.cache_graph.MayWriteGraph :=
  Std.bind (control_flow.graph_and_seeds code cut fold returns) (fun source =>
    PtxvMemoryWrite.graph_source.build source.successors source.seeds)

@[step]
theorem rust_graph_from_source (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool))
    (returns : Slice (alloc.vec.Vec Usize)) (input : KernelInputs code returns) :
    graphFromSource code cut fold returns ⦃ graph =>
      KernelSourceInputs graph code cut fold returns ⦄ := by
  have storage := graph_rows_fit_storage code.val cut fold.val returns.val
    input.codeStorage input.targetStorage input.returnStorage
  obtain ⟨source, sourceOk, sourceMeaning⟩ := MemoryWrite.spec_returns _ _
    (rust_graph_and_seeds code cut fold returns storage)
  obtain ⟨cfgOk, seedsOk, rowsExact, seedsExact⟩ := sourceMeaning
  have seedsLength := seedsExact.length.symm
  have bounded := source_successors_bounded code cut fold returns input source.successors cfgOk
  obtain ⟨graph, graphOk, graphMeaning⟩ := MemoryWrite.spec_returns _ _
    (MemoryWrite.GraphSource.rust_build source.successors source.seeds
      (by simpa only [seedsLength] using bounded))
  obtain ⟨succEq, writesEq, countEq, propagationOk, propagation⟩ := graphMeaning
  unfold graphFromSource
  rw [sourceOk]
  simp only [bind_ok, graphOk, WP.spec_ok]
  refine {
    codeStorage := input.codeStorage
    targetStorage := input.targetStorage
    returnStorage := input.returnStorage
    cfgProduced := by rw [succEq]; exact cfgOk
    targetValidity := input.targetValidity
    returnValidity := input.returnValidity
    reachLength := propagation.1.trans seedsLength
    reachProduced := ?_
    registerSpans := input.registerSpans }
  unfold reachFromSource
  rw [seedsOk]
  simp only [bind_ok]
  rw [succEq]
  exact propagationOk


theorem native_kernel_publication_analysis_and_queries_return_true
    {F : Type} (reader : destination_rows.LegacyDestinations F) (fallback : F)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (input : KernelInputs code returns)
    (legacy : LegacyReturns reader fallback code.val)
    (keyStorage : ∀ rows : alloc.vec.Vec (alloc.vec.Vec Reg),
      destination_rows.collect reader code fallback = .ok rows → BuildFits (programRows rows).val) :
    ∃ before rows keys index graph,
      graphFromSource code cut fold returns = .ok before ∧
      destination_rows.collect reader code fallback = .ok rows ∧
      keysFromSource rows = .ok keys ∧
      PtxvMemoryWrite.register_index.publish_graph before (alloc.vec.Vec.deref keys) = .ok (index, graph) ∧
      ∀ (mask : U32) (bit : Bit), HasBit mask bit → ∀ (start : Usize), start.val < code.val.length →
        AnalysisFits graph → ∃ sets,
          PtxvMemoryWrite.cache_graph.may_write_sets graph start mask = .ok sets ∧
          ∀ (pc : Usize) (register : Reg),
            RawMemoryWrittenBefore graph code.val cut fold.val returns.val mask start.val pc.val register →
            ToBarrier graph code.val cut fold.val returns.val mask bit pc.val →
            Std.bind (programKeyFromSource register) (fun identity =>
              Std.bind (PtxvMemoryWrite.register_index.RegisterIndex.get index identity)
                (fun dense => PtxvMemoryWrite.cache_graph.may_at sets pc dense)) = .ok true := by
  obtain ⟨before, sourceOk, source⟩ := MemoryWrite.spec_returns _ _
    (rust_graph_from_source code cut fold returns input)
  obtain ⟨rows, keys, index, graph, rowsOk, keysOk, published, meaning⟩ :=
    kernel_source_publication_analysis_and_queries_return_true reader fallback before code cut fold returns
      source legacy keyStorage
  exact ⟨before, rows, keys, index, graph, sourceOk, rowsOk, keysOk, published, meaning⟩

end Instruction.ControlFlow
