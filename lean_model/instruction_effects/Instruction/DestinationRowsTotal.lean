/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.DestinationRows
import Instruction.WriteRefinement

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.DestinationRows

def LegacyReturns {F : Type} (reader : destination_rows.LegacyDestinations F)
    (fallback : F) (code : List Inst) : Prop :=
  ∀ instruction ∈ code, memoryWrites instruction.op = none →
    ∃ row, reader.row fallback instruction.op = .ok row

theorem selection_returns {F : Type} (reader : destination_rows.LegacyDestinations F)
    (fallback : F) (instruction : Inst)
    (bounded : optionBounded (memoryWrites instruction.op))
    (legacy : memoryWrites instruction.op = none → ∃ row, reader.row fallback instruction.op = .ok row) :
    ∃ (produced : Option (alloc.vec.Vec Reg)) (row : alloc.vec.Vec Reg),
      register_effect.memory_writes instruction.op = .ok produced ∧
      (match produced with
      | none => reader.row fallback instruction.op
      | some registers => .ok registers) = .ok row := by
  obtain ⟨produced, read, meaning⟩ := spec_returns _ _ (memory_writes_spec instruction.op bounded)
  cases produced with
  | none =>
      obtain ⟨row, rowOk⟩ := legacy (by simpa only [registerObservation, Option.map_none] using meaning.symm)
      exact ⟨none, row, read, rowOk⟩
  | some registers => exact ⟨some registers, registers, read, rfl⟩

@[step]
theorem rust_collect_total {F : Type} (reader : destination_rows.LegacyDestinations F)
    (code : Slice Inst) (fallback : F)
    (bounded : ∀ instruction ∈ code.val, optionBounded (memoryWrites instruction.op))
    (legacy : LegacyReturns reader fallback code.val) :
    destination_rows.collect reader code fallback ⦃ rows => Rows code.val rows.val ⦄ := by
  have returns : WP.spec (destination_rows.collect reader code fallback) (fun _ => True) := by
    unfold destination_rows.collect destination_rows.collect_loop
    apply loop.spec_decr_nat
      (fun state : alloc.vec.Vec (alloc.vec.Vec Reg) × Usize => code.val.length - state.2.val)
      (fun state : alloc.vec.Vec (alloc.vec.Vec Reg) × Usize =>
        state.2.val ≤ code.val.length ∧ state.1.val.length = state.2.val)
    · rintro ⟨rows, pc⟩ ⟨pcBound, rowsLength⟩
      dsimp only at pcBound rowsLength
      unfold destination_rows.collect_loop.body
      by_cases inside : pc.val < code.val.length
      · have test : pc < Slice.len code := by scalar_tac
        simp only [test, ite_true]
        step with Slice.index_usize_spec code pc inside as ⟨instruction, instructionAt⟩
        have member : instruction ∈ code.val := by rw [instructionAt]; exact List.getElem_mem inside
        obtain ⟨produced, row, producedOk, rowOk⟩ := selection_returns reader fallback instruction
          (bounded instruction member) (legacy instruction member)
        rw [producedOk]
        simp only [bind_ok]
        cases produced <;> simp only at rowOk ⊢
        all_goals
          rw [rowOk]
          simp only [bind_ok]
          step with alloc.vec.Vec.push_spec rows row (by rw [rowsLength]; have := code.property; omega)
            as ⟨updated, updatedEq⟩
          step with Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac)
            as ⟨next, nextValue⟩
          refine ⟨by scalar_tac, ?_, by scalar_tac⟩
          simp only [updatedEq, List.length_append, List.length_singleton]
          scalar_tac
      · have test : ¬pc < Slice.len code := by scalar_tac
        simp only [test, ite_false, WP.spec_ok]
    · exact ⟨by scalar_tac, by simp [alloc.vec.Vec.with_capacity]⟩
  obtain ⟨rows, success, _⟩ := spec_returns _ _ returns
  rw [success, WP.spec_ok]
  exact successful_collect reader code fallback rows success

end Instruction.DestinationRows
