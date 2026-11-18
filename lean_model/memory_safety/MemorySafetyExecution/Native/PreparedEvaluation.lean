/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Preparation

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

def PreparedEvaluation (program : CoreProgram) (state : CoreState) (actor : Usize)
    (commit : ptxv_memory_write.execution_core.Commit) : Prop :=
  ∃ thread instruction,
    state.threads.val[actor.val]? = some thread ∧ thread.running = true ∧
    program.code.val[thread.pc.val]? = some instruction ∧
    commit.actor = actor ∧ commit.next_pc.val = thread.pc.val + 1 ∧
    ptxv_memory_write.value_flow.evaluate thread.registers thread.ids instruction =
      .ok (.Ok commit.action)

private theorem successful_false {α : Type} (condition : Prop) [Decidable condition]
    (reason : Fault) (next : Result (core.result.Result α Fault)) (out : α)
    (accepted : (if condition then .ok (.Err reason) else next) = .ok (.Ok out)) :
    ¬condition ∧ next = .ok (.Ok out) := by
  by_cases h : condition <;> simp_all

private theorem successful_true {α : Type} (condition : Prop) [Decidable condition]
    (reason : Fault) (next : Result (core.result.Result α Fault)) (out : α)
    (accepted : (if condition then next else .ok (.Err reason)) = .ok (.Ok out)) :
    condition ∧ next = .ok (.Ok out) := by
  by_cases h : condition <;> simp_all

theorem source_prepare_evaluation (program : CoreProgram) (state : CoreState) (actor : Usize)
    (commit : ptxv_memory_write.execution_core.Commit)
    (accepted : ptxv_memory_write.execution_core.prepare program state actor = .ok (.Ok commit)) :
    PreparedEvaluation program state actor commit := by
  unfold ptxv_memory_write.execution_core.prepare at accepted
  obtain ⟨layout, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  obtain ⟨valid, accepted⟩ := successful_true _ _ _ _ accepted
  obtain ⟨actorBound, accepted⟩ := successful_false _ _ _ _ accepted
  obtain ⟨thread, selected, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  obtain ⟨running, accepted⟩ := successful_true _ _ _ _ accepted
  obtain ⟨pc, pcCast, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  obtain ⟨pcBound, accepted⟩ := successful_false _ _ _ _ accepted
  obtain ⟨notMaximum, accepted⟩ := successful_false _ _ _ _ accepted
  obtain ⟨nextPc, nextEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  obtain ⟨index, indexCast, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  obtain ⟨instruction, fetched, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  obtain ⟨result, evaluated, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  cases result with
  | Err fault => simp at accepted
  | Ok action =>
    have identity : ({ actor, next_pc := nextPc, action } : ptxv_memory_write.execution_core.Commit) = commit := by
      simpa only [ok.injEq, core.result.Result.Ok.injEq] using accepted
    subst commit
    have actorInside : actor.val < state.threads.val.length := by scalar_tac
    have threadValue : thread = state.threads.val[actor.val] := by
      change alloc.vec.Vec.index_usize state.threads actor = .ok thread at selected
      have spec := alloc.vec.Vec.index_usize_spec state.threads actor actorInside
      simpa only [selected, WP.spec_ok] using spec
    have castBound : thread.pc.val ≤ UScalar.max .Usize := by
      rcases Usize.bounds_eq with narrow | wide <;> scalar_tac
    have pcVal : pc.val = thread.pc.val := by
      have spec := UScalar.cast_inBounds_spec .Usize thread.pc castBound
      simpa only [pcCast, WP.spec_ok] using spec
    have indexVal : index.val = thread.pc.val := by
      have spec := UScalar.cast_inBounds_spec .Usize thread.pc castBound
      simpa only [indexCast, WP.spec_ok] using spec
    have instructionInside : index.val < program.code.val.length := by scalar_tac
    have instructionValue : instruction = program.code.val[index.val] := by
      change alloc.vec.Vec.index_usize program.code index = .ok instruction at fetched
      have spec := alloc.vec.Vec.index_usize_spec program.code index instructionInside
      simpa only [fetched, WP.spec_ok] using spec
    have nextVal : nextPc.val = thread.pc.val + 1 := by
      have facts := UScalar.add_equiv thread.pc 1#u32
      simp only [nextEq] at facts
      simpa using facts.2.1
    refine ⟨thread, instruction, ?_, running, ?_, rfl, nextVal, evaluated⟩
    · simpa only [threadValue] using List.getElem?_eq_getElem actorInside
    · have selected := List.getElem?_eq_getElem instructionInside
      simpa only [instructionValue, indexVal] using selected

end MemorySafetyExecution.Native
