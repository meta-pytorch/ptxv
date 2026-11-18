/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceAccess
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety.Source

theorem source_access_true (program : memory_safety.Program) (state : memory_safety.State)
    (thread : Usize) (request : memory_safety.Request)
    (computed : memory_safety.access_valid program state thread request = .ok true) :
    AccessSafe (programModel program) (stateModel state) thread.val (requestModel request) := by
  have specification := source_access_valid program state thread request
  simpa only [computed, WP.spec_ok, true_implies] using specification

theorem source_update_access (program : memory_safety.Program) (state : memory_safety.State)
    (thread : Usize) (request : memory_safety.Request) (next : memory_safety.State)
    (computed : memory_safety.update_memory program state (.Access thread request) = .ok next)
    (clean : next.bad = false) :
    AccessSafe (programModel program) (stateModel state) thread.val (requestModel request) ∧
      stateModel next = updateMemory (programModel program) (stateModel state)
        (.access thread.val (requestModel request)) := by
  simp only [memory_safety.update_memory, source_state_clone, bind_ok] at computed
  obtain ⟨valid, checked, remaining⟩ := source_bind_success _ _ _ computed
  cases valid with
  | false =>
    simp only [Bool.false_eq_true, ite_false, ok.injEq] at remaining
    rw [←remaining] at clean
    cases clean
  | true =>
    simp only [ite_true] at remaining
    have safe := source_access_true program state thread request checked
    refine ⟨safe, ?_⟩
    cases kind : request.kind with
    | Load =>
      simp only [kind, bind_ok] at remaining
      obtain ⟨history, pushed, result⟩ := source_bind_success _ _ _ remaining
      have historyEq := vec_push_success _ _ _ pushed
      simp only [ok.injEq] at result
      rw [←result]
      simp only [updateMemory, ite_eq_left safe]
      simp [requestModel, kindModel, kind, stateModel, accessRecord,
        AccessKind.isWrite, historyEq, recordModel]
    | Store defined =>
      simp only [kind, Std.bind_assoc, bind_ok] at remaining
      obtain ⟨writes, stored, remaining⟩ := source_bind_success _ _ _ remaining
      have writesEq := vec_push_success _ _ _ stored
      obtain ⟨history, pushed, result⟩ := source_bind_success _ _ _ remaining
      have historyEq := vec_push_success _ _ _ pushed
      simp only [ok.injEq] at result
      rw [←result]
      simp only [updateMemory, ite_eq_left safe]
      simp [requestModel, kindModel, kind, stateModel, accessRecord,
        AccessKind.isWrite, writesEq, historyEq, recordModel, writtenModel]

theorem source_update_free (program : memory_safety.Program) (state : memory_safety.State)
    (thread allocation : Usize) (next : memory_safety.State)
    (computed : memory_safety.update_memory program state (.Free thread allocation) = .ok next)
    (clean : next.bad = false) :
    FreeSafe (programModel program) (stateModel state) thread.val allocation.val ∧
      stateModel next = updateMemory (programModel program) (stateModel state)
        (.free thread.val allocation.val) := by
  simp only [memory_safety.update_memory, source_state_clone, bind_ok] at computed
  by_cases liveBound : allocation.val < state.live.val.length
  · have liveSource : ¬allocation ≥ alloc.vec.Vec.len state.live := by scalar_tac
    simp only [liveSource, ite_false] at computed
    by_cases allocationBound : allocation.val < program.allocations.val.length
    · have allocationSource : ¬allocation ≥ alloc.vec.Vec.len program.allocations := by scalar_tac
      simp only [allocationSource, ite_false, alloc.vec.Vec.index_slice_index] at computed
      obtain ⟨live, liveRead, remaining⟩ := source_bind_success _ _ _ computed
      cases live with
      | false =>
        simp only [Bool.false_eq_true, ite_false, ok.injEq] at remaining
        rw [←remaining] at clean
        cases clean
      | true =>
        simp only [ite_true] at remaining
        obtain ⟨block, blockRead, remaining⟩ := source_bind_success _ _ _ remaining
        obtain ⟨valid, checked, remaining⟩ := source_bind_success _ _ _ remaining
        cases valid with
        | false =>
          simp only [Bool.false_eq_true, ite_false, ok.injEq] at remaining
          rw [←remaining] at clean
          cases clean
        | true =>
          simp only [ite_true] at remaining
          have access := source_access_true program state thread _ checked
          have liveLookup := vec_index_success _ _ _ liveRead
          have blockLookup : (programModel program).allocations[allocation.val]? = some (allocationModel block) := by
            simp only [programModel, List.getElem?_map, vec_index_success _ _ _ blockRead, Option.map_some]
          have safe : FreeSafe (programModel program) (stateModel state) thread.val allocation.val := by
            simp only [FreeSafe, blockLookup]
            exact ⟨liveLookup, access⟩
          refine ⟨safe, ?_⟩
          obtain ⟨history, pushed, remaining⟩ := source_bind_success _ _ _ remaining
          have historyEq := vec_push_success _ _ _ pushed
          obtain ⟨pair, replaced, result⟩ := source_bind_success _ _ _ remaining
          rcases pair with ⟨oldLive, replace⟩
          have replacement := vec_index_mut_success state.live allocation oldLive replace
            (by simpa only [alloc.vec.Vec.index_mut_slice_index] using replaced)
          change Result.ok { state with live := replace false, history := history } = .ok next at result
          have same : { state with live := replace false, history := history } = next := by simpa only [ok.injEq] using result
          rw [←same, replacement.2]
          simp only [updateMemory, ite_eq_left safe, blockLookup]
          simp [stateModel, historyEq,
            accessRecord, freeRequest, allocationModel, recordModel, AccessKind.isWrite]
    · have allocationSource : allocation ≥ alloc.vec.Vec.len program.allocations := by scalar_tac
      simp only [allocationSource, ite_true, ok.injEq] at computed
      rw [←computed] at clean
      cases clean
  · have liveSource : allocation ≥ alloc.vec.Vec.len state.live := by scalar_tac
    simp only [liveSource, ite_true, ok.injEq] at computed
    rw [←computed] at clean
    cases clean

theorem source_update_memory_agrees (program : memory_safety.Program) (state : memory_safety.State)
    (event : memory_safety.Event) (next : memory_safety.State)
    (computed : memory_safety.update_memory program state event = .ok next)
    (clean : next.bad = false) :
    EventSafe (programModel program) (stateModel state) (eventModel event) ∧
      stateModel next = updateMemory (programModel program) (stateModel state) (eventModel event) := by
  cases event with
  | Access thread request => exact source_update_access program state thread request next computed clean
  | Free thread allocation => exact source_update_free program state thread allocation next computed clean
  | Control =>
    simp only [memory_safety.update_memory, source_state_clone, bind_ok, ok.injEq] at computed
    subst next
    exact ⟨True.intro, rfl⟩

theorem source_apply_move_agrees (program : memory_safety.Program) (state : memory_safety.State)
    (movement : memory_safety.Move) (next : memory_safety.State)
    (computed : memory_safety.apply_move program state movement = .ok next)
    (clean : next.bad = false) :
    EventSafe (programModel program) (stateModel state) (eventModel movement.event) ∧
      stateModel next = applyMove (programModel program) (stateModel state) (moveModel movement) := by
  unfold memory_safety.apply_move at computed
  obtain ⟨updated, updatedComputed, remaining⟩ := source_bind_success _ _ _ computed
  by_cases bad : updated.bad = true
  · simp only [bad, ite_true, ok.injEq] at remaining
    subst next
    rw [bad] at clean
    cases clean
  · have good : updated.bad = false := Bool.eq_false_iff.mpr bad
    simp only [good, Bool.false_eq_true, ite_false,
      vec_clone_exact core.clone.CloneUsize (by intro value; rfl), bind_ok, ok.injEq] at remaining
    have agrees := source_update_memory_agrees program state movement.event updated updatedComputed good
    refine ⟨agrees.1, ?_⟩
    have modelGood : (updateMemory (programModel program) (stateModel state) (eventModel movement.event)).bad = false := by
      rw [←agrees.2]
      exact good
    rw [←remaining]
    simp only [applyMove, moveModel, modelGood, Bool.false_eq_true, ite_false]
    rw [←agrees.2]
    rfl

end MemorySafety.Source
