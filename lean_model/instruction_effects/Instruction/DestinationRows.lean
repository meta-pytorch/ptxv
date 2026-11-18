/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs
import Instruction.PartialSuccess

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.DestinationRows

def SelectedRow (instruction : Inst) (row : alloc.vec.Vec Reg) : Prop :=
  ∀ expected, register_effect.memory_writes instruction.op = .ok (some expected) → row = expected

def Rows (code : List Inst) (rows : List (alloc.vec.Vec Reg)) : Prop :=
  List.Forall₂ SelectedRow code rows

theorem successful_selection {F : Type}
    (reader : destination_rows.LegacyDestinations F) (fallback : F) (instruction : Inst)
    (produced : Option (alloc.vec.Vec Reg)) (row : alloc.vec.Vec Reg)
    (read : register_effect.memory_writes instruction.op = .ok produced)
    (selected : (match produced with
      | none => reader.row fallback instruction.op
      | some registers => .ok registers) = .ok row) : SelectedRow instruction row := by
  intro expected expectedOk
  have same : produced = some expected := Result.ok_injective (read.symm.trans expectedOk)
  rw [same] at selected
  exact (Result.ok_injective selected).symm

theorem successful_collect {F : Type} (reader : destination_rows.LegacyDestinations F)
    (code : Slice Inst) (fallback : F) (result : alloc.vec.Vec (alloc.vec.Vec Reg))
    (success : destination_rows.collect reader code fallback = .ok result) : Rows code.val result.val := by
  unfold destination_rows.collect destination_rows.collect_loop at success
  have execution := successful_loop_has_execution _ _ _ success
  apply execution.preserves
    (fun state : alloc.vec.Vec (alloc.vec.Vec Reg) × Usize =>
      state.2.val ≤ code.val.length ∧ Rows (code.val.take state.2.val) state.1.val)
    (fun rows : alloc.vec.Vec (alloc.vec.Vec Reg) => Rows code.val rows.val)
    ⟨by scalar_tac, List.Forall₂.nil⟩
  rintro ⟨rows, pc⟩ ⟨pcBound, prefixRows⟩ flow bodyOk
  dsimp only at pcBound prefixRows
  have rowLength : rows.val.length = pc.val := by
    have lengths := prefixRows.length_eq
    simp only [List.length_take, Nat.min_eq_left pcBound] at lengths
    omega
  unfold destination_rows.collect_loop.body at bodyOk
  by_cases inside : pc.val < code.val.length
  · have test : pc < Slice.len code := by scalar_tac
    simp only [test, ite_true] at bodyOk
    obtain ⟨instruction, instructionOk, readOk⟩ := bind_success _ _ _ bodyOk
    have instructionSpec := Slice.index_usize_spec code pc inside
    simp only [instructionOk, WP.spec_ok] at instructionSpec
    obtain ⟨produced, producedOk, selectedOk⟩ := bind_success _ _ _ readOk
    obtain ⟨row, rowOk, pushOk⟩ := bind_success _ _ _ selectedOk
    have selected := successful_selection reader fallback instruction produced row producedOk rowOk
    obtain ⟨updated, updatedOk, incrementOk⟩ := bind_success _ _ _ pushOk
    have updatedSpec := alloc.vec.Vec.push_spec rows row (by have := code.property; omega)
    simp only [updatedOk, WP.spec_ok] at updatedSpec
    obtain ⟨next, nextOk, returned⟩ := bind_success _ _ _ incrementOk
    have nextSpec := Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac)
    simp only [nextOk, WP.spec_ok] at nextSpec
    have equal := Result.ok_injective returned
    cases equal
    refine ⟨by scalar_tac, ?_⟩
    change Rows (code.val.take next.val) updated.val
    have nextVal : next.val = pc.val + 1 := by scalar_tac
    rw [nextVal, updatedSpec, List.take_succ_eq_append_getElem inside]
    exact List.rel_append prefixRows (.cons (by simpa only [instructionSpec] using selected) .nil)
  · have test : ¬pc < Slice.len code := by scalar_tac
    have complete : pc.val = code.val.length := by omega
    simp only [test, ite_false] at bodyOk
    have equal := Result.ok_injective bodyOk
    cases equal
    simpa only [complete, List.take_length] using prefixRows

end Instruction.DestinationRows
