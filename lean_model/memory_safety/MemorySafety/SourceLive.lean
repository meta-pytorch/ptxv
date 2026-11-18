/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourcePrimitives

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false

namespace MemorySafety.Source

@[step]
theorem source_live_span (program : memory_safety.Program) (state : memory_safety.State)
    (request : memory_safety.Request)
    (shape : state.live.val.length = program.allocations.val.length) :
    memory_safety.access_valid_loop0 program.allocations state.live request.address request.width false 0#usize
      ⦃ result => result = true → LiveSpan (programModel program) (stateModel state) (requestModel request) ⦄ := by
  unfold memory_safety.access_valid_loop0
  apply loop.spec_decr_nat
    (fun current : Bool × Usize => program.allocations.val.length - current.2.val)
    (fun current : Bool × Usize => current.2.val ≤ program.allocations.val.length ∧
      (current.1 = true → LiveSpan (programModel program) (stateModel state) (requestModel request)))
  · intro current invariant
    rcases current with ⟨found, cursor⟩
    obtain ⟨cursorBound, previousFound⟩ := invariant
    dsimp only at cursorBound previousFound
    unfold memory_safety.access_valid_loop0.body
    by_cases inside : cursor.val < program.allocations.val.length
    · have sourceInside : cursor < alloc.vec.Vec.len program.allocations := by scalar_tac
      simp only [sourceInside, ite_true]
      step as ⟨row, rowEq⟩
      have rowIndex : row = program.allocations.val[cursor.val]'inside := rowEq
      have rowCheck :
          (if row.base ≤ request.address then do
            let offset ← request.address - row.base
            let live ← alloc.vec.Vec.index_usize state.live cursor
            if live then
              if offset ≤ row.bytes then do
                let remaining ← row.bytes - offset
                if request.width ≤ remaining then ok true else ok found
              else ok found
            else ok found
          else ok found)
            ⦃ result => result = true → LiveSpan (programModel program) (stateModel state) (requestModel request) ⦄ := by
        split
        · step as ⟨offset, offsetEq⟩
          step as ⟨live, liveEq⟩
          cases live with
          | false => simpa using previousFound
          | true =>
            simp only [ite_true]
            split
            · step as ⟨remaining, remainingEq⟩
              split
              · simp only [WP.spec_ok, true_implies]
                refine ⟨(allocationModel row, cursor.val), ?_, ?_, ?_, ?_⟩
                · apply List.mem_iff_getElem.mpr
                  refine ⟨cursor.val, by simp [programModel]; omega, ?_⟩
                  simp [programModel, List.getElem_zipIdx, rowIndex]
                · change state.live.val[cursor.val]? = some true
                  have liveInside : cursor.val < state.live.val.length := by omega
                  rw [List.getElem?_eq_getElem liveInside]
                  exact congrArg some liveEq.symm
                · simp only [allocationModel, requestModel]
                  scalar_tac
                · simp only [allocationModel, requestModel]
                  scalar_tac
              · simpa using previousFound
            · simpa using previousFound
        · simpa using previousFound
      step with rowCheck as ⟨nowFound, nowSafe⟩
      step as ⟨next, nextEq⟩
      exact ⟨by scalar_tac, nowSafe, by scalar_tac⟩
    · have sourceOutside : ¬cursor < alloc.vec.Vec.len program.allocations := by scalar_tac
      simpa [sourceOutside] using previousFound
  · simp

end MemorySafety.Source
