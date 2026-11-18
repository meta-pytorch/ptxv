/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Source
import MemoryWrite.Representation

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite

theorem spec_returns {α : Type} (program : Result α) (post : α → Prop)
    (h : program ⦃ post ⦄) : ∃ value, program = Result.ok value ∧ post value := by
  cases program
  · rename_i value
    exact ⟨value, rfl, (WP.spec_ok value).mp h⟩
  · rename_i effect continuation
    exact False.elim ((WP.spec_vis effect continuation).mp h)
  · exact False.elim (WP.spec_div.mp h)

private theorem map_write_spec (cells : List Cell) (write : Key) (payload : Slice PayloadByte) :
    List.Forall₂ (fun before after => CellWriteSpec before after write payload)
      cells (cells.map (fun cell => modelCell cell write payload)) := by
  induction cells with
  | nil => exact .nil
  | cons cell cells ih => exact .cons (modelCell_spec cell write payload) ih

/-- The extracted Rust routine establishes the byte and history effects itself. -/
theorem apply_write_effect_preserves_representation {Input : Type}
    (sem : Semantics Input) (memory : Memory Input) (state : Slice Cell)
    (write : Key) (payload : Slice PayloadByte) (values : Input → Nat → Byte)
    (hstate : Represents sem memory state)
    (hspan : ValidWriteSpan state write payload.val.length)
    (hpayload : PayloadSound sem payload values) :
    ∃ next,
      apply_write_effect state write payload = .ok next ∧
      WriteSpec state next write payload ∧
      Represents sem (storeBytes memory write payload.val.length values) next ∧
      ValidWriteSpan next write payload.val.length ∧
      next.val.map Cell.key = state.val.map Cell.key := by
  obtain ⟨next, hrun, hnext⟩ := spec_returns _ _ (apply_write_effect_spec state write payload)
  refine ⟨next, hrun, ?_, ?_, ?_, ?_⟩
  · unfold WriteSpec
    rw [hnext]
    exact map_write_spec state.val write payload
  · intro cell hcell
    rw [hnext, List.mem_map] at hcell
    obtain ⟨before, hbefore, rfl⟩ := hcell
    exact modelCell_represents sem memory before write payload values
      (hstate before hbefore) hpayload
  · refine ⟨hspan.no_wrap, ?_⟩
    intro index hindex
    obtain ⟨cell, hcell, hloc⟩ := hspan.coverage index hindex
    refine ⟨modelCell cell write payload, ?_, ?_⟩
    · rw [hnext]
      exact List.mem_map.mpr ⟨cell, hcell, rfl⟩
    · rw [modelCell_key]
      exact hloc
  · rw [hnext, List.map_map]
    simp only [Function.comp_def, modelCell_key]

end MemoryWrite
