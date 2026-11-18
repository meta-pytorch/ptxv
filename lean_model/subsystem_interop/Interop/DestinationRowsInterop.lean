/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.DestinationKeys
import Instruction.DestinationRowsTotal
import Interop.MemoryGraphInterop
import MemoryWrite.RawSourceRows

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.DestinationRows

def programRows (rows : alloc.vec.Vec (alloc.vec.Vec Reg)) :
    alloc.vec.Vec (alloc.vec.Vec ProgramKey) := vecMap (vecMap programKey) rows

def keysFromSource (rows : alloc.vec.Vec (alloc.vec.Vec Reg)) :
    Result (alloc.vec.Vec (alloc.vec.Vec ProgramKey)) :=
  Std.bind (destination_rows.keys (alloc.vec.Vec.deref rows)) PtxvMemoryWrite.source_rows.from_raw

theorem raw_to_program_rows (rows : alloc.vec.Vec (alloc.vec.Vec Reg)) :
    MemoryWrite.SourceRows.keyRows (rawRows (alloc.vec.Vec.deref rows)) = programRows rows := by
  apply alloc.vec.Vec.ext
  simp only [MemoryWrite.SourceRows.keyRows, MemoryWrite.SourceRows.vecMap,
    programRows, vecMap, rawRows, alloc.vec.Vec.deref, alloc.vec.Vec.from_val,
    List.map_map, Slice.from_val]
  apply List.map_congr_left
  intro row member
  apply alloc.vec.Vec.ext
  simp only [Function.comp_apply, MemoryWrite.SourceRows.vecMap, vecMap, alloc.vec.Vec.from_val,
    List.map_map]
  rfl

@[step]
theorem rust_keys_from_source (rows : alloc.vec.Vec (alloc.vec.Vec Reg)) :
    keysFromSource rows ⦃ result => result = programRows rows ⦄ := by
  unfold keysFromSource
  step with rust_keys (alloc.vec.Vec.deref rows) as ⟨raw, rawEq⟩
  rw [rawEq]
  apply WP.spec_mono (MemoryWrite.SourceRows.rust_from_raw (rawRows (alloc.vec.Vec.deref rows)))
  intro result resultEq
  exact resultEq.trans (raw_to_program_rows rows)

theorem selected_row_at {code : List Inst} {rows : List (alloc.vec.Vec Reg)}
    (correct : Rows code rows) (pc : Nat) (instruction : Inst)
    (atInstruction : code[pc]? = some instruction) :
    ∃ row, rows[pc]? = some row ∧ SelectedRow instruction row := by
  induction correct generalizing pc with
  | nil => simp at atInstruction
  | @cons instruction0 row0 code rows selected rest ih =>
      cases pc with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at atInstruction
          cases atInstruction
          exact ⟨row0, rfl, selected⟩
      | succ pc =>
          simp only [List.getElem?_cons_succ] at atInstruction ⊢
          exact ih pc atInstruction

theorem program_rows_source (code : List Inst) (rows : alloc.vec.Vec (alloc.vec.Vec Reg))
    (correct : Rows code rows.val) : SourceMemoryRows (code.map Inst.op) (programRows rows).val := by
  intro pc operation operationAt registers produced
  simp only [List.getElem?_map] at operationAt
  cases instructionAt : code[pc]? with
  | none => simp only [instructionAt, Option.map_none, reduceCtorEq] at operationAt
  | some instruction =>
      have same : instruction.op = operation := by
        simpa only [instructionAt, Option.map_some, Option.some.injEq] using operationAt
      obtain ⟨row, rowAt, selected⟩ := selected_row_at correct pc instruction instructionAt
      have selectedEq : row = registers := selected registers (by simpa only [same] using produced)
      refine ⟨vecMap programKey row, ?_, ?_⟩
      · simp only [programRows, vecMap, alloc.vec.Vec.from_val, List.getElem?_map,
          rowAt, Option.map_some]
      · simp only [vecMap, alloc.vec.Vec.from_val, selectedEq]

theorem actual_source_rows_pipeline_total {F : Type}
    (reader : destination_rows.LegacyDestinations F) (code : Slice Inst) (fallback : F)
    (bounded : ∀ instruction ∈ code.val, optionBounded (memoryWrites instruction.op))
    (legacy : LegacyReturns reader fallback code.val) :
    ∃ rows keys, destination_rows.collect reader code fallback = .ok rows ∧
      keysFromSource rows = .ok keys ∧ SourceMemoryRows (code.val.map Inst.op) keys.val ∧
      keys.val.length = code.val.length := by
  obtain ⟨rows, rowsOk, rowsCorrect⟩ := MemoryWrite.spec_returns _ _
    (rust_collect_total reader code fallback bounded legacy)
  obtain ⟨keys, keysOk, keysCorrect⟩ := MemoryWrite.spec_returns _ _ (rust_keys_from_source rows)
  refine ⟨rows, keys, rowsOk, keysOk, ?_, ?_⟩
  · rw [keysCorrect]
    exact program_rows_source code.val rows rowsCorrect
  · rw [keysCorrect]
    simp only [programRows, vecMap, alloc.vec.Vec.from_val, List.length_map]
    exact rowsCorrect.length_eq.symm

end Instruction.DestinationRows
