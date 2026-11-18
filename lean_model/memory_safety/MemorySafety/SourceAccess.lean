/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceLive
import MemorySafety.SourceInitialized
import MemorySafety.SourceRace

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false

namespace MemorySafety.Source

@[step]
theorem source_access_valid (program : memory_safety.Program) (state : memory_safety.State)
    (thread : Usize) (request : memory_safety.Request) :
    memory_safety.access_valid program state thread request
      ⦃ result => result = true →
        AccessSafe (programModel program) (stateModel state) thread.val (requestModel request) ⦄ := by
  by_cases shape : state.live.val.length = program.allocations.val.length
  · have sourceShape : alloc.vec.Vec.len state.live = alloc.vec.Vec.len program.allocations := by scalar_tac
    by_cases widthPositive : request.width = 0#u64
    · simp [memory_safety.access_valid, sourceShape, widthPositive]
    · by_cases alignmentPositive : request.alignment = 0#u64
      · simp [memory_safety.access_valid, sourceShape, widthPositive, alignmentPositive]
      · unfold memory_safety.access_valid
        simp only [sourceShape, bne_self_eq_false, Bool.false_eq_true, ite_false, widthPositive, alignmentPositive]
        step as ⟨remainder, remainderEq⟩
        by_cases aligned : remainder = 0#u64
        · simp only [aligned, bne_self_eq_false, Bool.false_eq_true, ite_false]
          step with UScalar.cast_inBounds_spec .U128 request.address (by scalar_tac) as ⟨address, addressEq⟩
          step with UScalar.cast_inBounds_spec .U128 request.width (by scalar_tac) as ⟨width, widthEq⟩
          step as ⟨finish, finishEq⟩
          step with UScalar.cast_inBounds_spec .U128 core.num.U64.MAX
            (by simp [core.num.U64.MAX, U64.rMax]; scalar_tac) as ⟨maximum, maximumEq⟩
          step as ⟨limit, limitEq⟩
          · simp only [U64.rMax] at maximumEq
            scalar_tac
          · by_cases nonwrapping : finish > limit
            · simp [nonwrapping]
            · simp only [nonwrapping, ite_false]
              step with source_live_span program state request shape as ⟨found, liveSpan⟩
              cases found with
              | false => simp
              | true =>
                simp only [ite_true]
                have live := liveSpan rfl
                have compose
                    (race : NoRace (stateModel state) (accessRecord thread.val (requestModel request)))
                    (initialized : kindModel request.kind = .load →
                      Initialized (programModel program) (stateModel state) (requestModel request)) :
                    AccessSafe (programModel program) (stateModel state) thread.val (requestModel request) := by
                  refine ⟨by simpa using shape, ?_, ?_, ?_, ?_, live, race, initialized⟩
                  · simp only [requestModel]
                    scalar_tac
                  · simp only [requestModel]
                    scalar_tac
                  · simp only [requestModel]
                    scalar_tac
                  · simp only [requestModel, addressLimit]
                    simp only [U64.rMax] at maximumEq
                    scalar_tac
                cases kind : request.kind with
                | Load =>
                  simp only [bind_ok, Bool.false_eq_true, ite_false]
                  step with source_access_races state.history thread request.address request.width false true
                    as ⟨valid, races⟩
                  step with source_read_bytes program state request valid as ⟨accepted, bytes⟩
                  have acceptedTrue : accepted = true := by assumption
                  have readBytes := bytes acceptedTrue
                  have noConflicts := races readBytes.1
                  apply compose
                  · simpa [NoRace, accessRecord, requestModel, kind, kindModel, AccessKind.isWrite,
                      stateModel, List.mem_map, recordModel] using noConflicts
                  · intro _
                    exact readBytes.2
                | Store defined =>
                  simp only [bind_ok, ite_true]
                  step with source_access_races state.history thread request.address request.width true true
                    as ⟨valid, races⟩
                  intro acceptedTrue
                  have noConflicts := races acceptedTrue
                  apply compose
                  · simpa [NoRace, accessRecord, requestModel, kind, kindModel, AccessKind.isWrite,
                      stateModel, List.mem_map, recordModel] using noConflicts
                  · simp [kind, kindModel]
        · simp [aligned]
  · have sourceShape : alloc.vec.Vec.len state.live ≠ alloc.vec.Vec.len program.allocations := by scalar_tac
    simp [memory_safety.access_valid, sourceShape]

end MemorySafety.Source
