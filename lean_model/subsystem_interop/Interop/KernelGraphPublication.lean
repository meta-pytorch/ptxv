/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.DestinationRowsInterop
import Interop.ProgramGraphPublication

open Aeneas Aeneas.Std Result PtxvIr
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex MemoryWrite.BarrierReach
open Instruction.DestinationRows

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.ControlFlow

structure KernelSourceInputs (before : PtxvMemoryWrite.cache_graph.MayWriteGraph)
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
  registerSpans : ∀ operation ∈ code.val.map Inst.op, optionBounded (memoryWrites operation)

def KernelSourceInputs.publicationInputs
    {before : PtxvMemoryWrite.cache_graph.MayWriteGraph}
    {code : Slice Inst} {cut : U8} {fold : Slice (Option Bool)} {returns : Slice (alloc.vec.Vec Usize)}
    (source : KernelSourceInputs before code cut fold returns)
    (rows : alloc.vec.Vec (alloc.vec.Vec Reg)) (correct : Rows code.val rows.val)
    (fits : BuildFits (programRows rows).val) :
    ProgramSourceInputs before code cut fold returns := {
  codeStorage := source.codeStorage
  targetStorage := source.targetStorage
  returnStorage := source.returnStorage
  cfgProduced := source.cfgProduced
  targetValidity := source.targetValidity
  returnValidity := source.returnValidity
  reachLength := source.reachLength
  reachProduced := source.reachProduced
  destinations := alloc.vec.Vec.deref (programRows rows)
  destinationLength := by
    simp only [alloc.vec.Vec.deref, programRows, vecMap, alloc.vec.Vec.from_val, Slice.from_val, List.length_map]
    exact correct.length_eq.symm
  destinationSource := by
    simpa only [alloc.vec.Vec.deref, Slice.from_val] using program_rows_source code.val rows correct
  registerSpans := source.registerSpans
  buildFits := by simpa only [alloc.vec.Vec.deref, Slice.from_val] using fits
}

theorem kernel_source_publication_analysis_and_queries_return_true
    {F : Type} (reader : destination_rows.LegacyDestinations F) (fallback : F)
    (before : PtxvMemoryWrite.cache_graph.MayWriteGraph)
    (code : Slice Inst) (cut : U8) (fold : Slice (Option Bool)) (returns : Slice (alloc.vec.Vec Usize))
    (source : KernelSourceInputs before code cut fold returns)
    (legacy : LegacyReturns reader fallback code.val)
    (keyStorage : ∀ rows : alloc.vec.Vec (alloc.vec.Vec Reg),
      destination_rows.collect reader code fallback = .ok rows → BuildFits (programRows rows).val) :
    ∃ rows keys index graph,
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
  have bounded : ∀ instruction ∈ code.val, optionBounded (memoryWrites instruction.op) := by
    intro instruction member
    exact source.registerSpans instruction.op (List.mem_map.mpr ⟨instruction, member, rfl⟩)
  obtain ⟨rows, rowsOk, rowsCorrect⟩ := MemoryWrite.spec_returns _ _
    (rust_collect_total reader code fallback bounded legacy)
  obtain ⟨keys, keysOk, keysCorrect⟩ := MemoryWrite.spec_returns _ _ (rust_keys_from_source rows)
  let publication := source.publicationInputs rows rowsCorrect (keyStorage rows rowsOk)
  obtain ⟨index, graph, published, meaning⟩ := published_source_analysis_and_queries_return_true
    before code cut fold returns publication
  refine ⟨rows, keys, index, graph, rowsOk, keysOk, ?_, ?_⟩
  · rw [keysCorrect]
    exact published
  · exact meaning

end Instruction.ControlFlow
