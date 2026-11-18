/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

inductive CommitEffect (program : CoreProgram) (before : CoreState) :
    ptxv_memory_write.memory_safety.Event → CoreState → Prop where
  | control (after : CoreState) (frame : after.allocations = before.allocations) :
      CommitEffect program before .Control after
  | load (actor : Usize) (after : CoreState) (pointer : ptxv_memory_write.value_flow.Pointer)
      (width : U8) (access : ptxv_memory_write.execution_core.ResolvedAccess)
      (resolved : ptxv_memory_write.execution_core.resolve_access program before pointer width false =
        .ok (.Ok access)) (frame : after.allocations = before.allocations) :
      CommitEffect program before (.Access actor access.request) after
  | store (actor : Usize) (after : CoreState) (pointer : ptxv_memory_write.value_flow.Pointer)
      (width : U8) (access : ptxv_memory_write.execution_core.ResolvedAccess) (value : U64)
      (written : CoreState)
      (resolved : ptxv_memory_write.execution_core.resolve_access program before pointer width true =
        .ok (.Ok access))
      (stored : ptxv_memory_write.execution_core.store_value before access value = .ok written)
      (frame : after.allocations = written.allocations) :
      CommitEffect program before (.Access actor access.request) after

private theorem successful_actor {α : Type}
    (condition : Prop) [Decidable condition] (fault : Fault) (before : CoreState)
    (next : Result ((core.result.Result α Fault) × CoreState)) (out : α) (after : CoreState)
    (accepted : (if condition then .ok (.Err fault, before) else next) = .ok (.Ok out, after)) :
    ¬condition ∧ next = .ok (.Ok out, after) := by
  by_cases guard : condition <;> simp_all

private theorem successful_effect (left right : ptxv_memory_write.memory_safety.Event)
    (before after : CoreState)
    (same : (Result.ok (.Ok left, before) : Result ((core.result.Result
      ptxv_memory_write.memory_safety.Event Fault) × CoreState)) = .ok (.Ok right, after)) :
    left = right ∧ before = after := by
  simpa only [ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] using same

theorem source_commit_effect (program : CoreProgram) (before after : CoreState)
    (commit : ptxv_memory_write.execution_core.Commit) (event : ptxv_memory_write.memory_safety.Event)
    (accepted : ptxv_memory_write.execution_core.commit_action program before commit =
      .ok (.Ok event, after)) : CommitEffect program before event after := by
  rcases commit with ⟨actor, nextPc, action⟩
  unfold ptxv_memory_write.execution_core.commit_action at accepted
  obtain ⟨actorBound, accepted⟩ := successful_actor _ _ _ _ _ _ accepted
  cases action with
  | Next =>
    obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    rcases pair with ⟨thread, back⟩
    have accepted := successful_effect _ _ _ _ accepted
    rcases accepted with ⟨rfl, rfl⟩
    exact .control _ rfl
  | Halt =>
    obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    rcases pair with ⟨thread, back⟩
    have accepted := successful_effect _ _ _ _ accepted
    rcases accepted with ⟨rfl, rfl⟩
    exact .control _ rfl
  | Jump target =>
    obtain ⟨index, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨targetBound, accepted⟩ := successful_actor _ _ _ _ _ _ accepted
    obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    rcases pair with ⟨thread, back⟩
    have accepted := successful_effect _ _ _ _ accepted
    rcases accepted with ⟨rfl, rfl⟩
    exact .control _ rfl
  | WriteVirtual destination value =>
    obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    rcases pair with ⟨thread, back⟩
    obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    rcases pair with ⟨result, registers⟩
    cases result with
    | Err fault => simp at accepted
    | Ok unitValue =>
      obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      rcases pair with ⟨thread, back⟩
      have accepted := successful_effect _ _ _ _ accepted
      rcases accepted with ⟨rfl, rfl⟩
      exact .control _ rfl
  | WritePredicate destination value =>
    obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    rcases pair with ⟨thread, back⟩
    obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    rcases pair with ⟨result, registers⟩
    cases result with
    | Err fault => simp at accepted
    | Ok unitValue =>
      obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      rcases pair with ⟨thread, back⟩
      have accepted := successful_effect _ _ _ _ accepted
      rcases accepted with ⟨rfl, rfl⟩
      exact .control _ rfl
  | Load destination destinationBits bytes signed pointer =>
    obtain ⟨result, resolved, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    cases result with
    | Err fault => simp at accepted
    | Ok access =>
      obtain ⟨value, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      obtain ⟨result, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      cases result with
      | Err fault => simp at accepted
      | Ok mask =>
        obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        rcases pair with ⟨thread, back⟩
        obtain ⟨masked, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        rcases pair with ⟨result, registers⟩
        cases result with
        | Err fault => simp at accepted
        | Ok unitValue =>
          obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
          rcases pair with ⟨thread, back⟩
          have accepted := successful_effect _ _ _ _ accepted
          rcases accepted with ⟨rfl, rfl⟩
          exact .load actor _ pointer bytes access resolved rfl
  | Store bytes pointer value =>
    obtain ⟨result, resolved, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    cases result with
    | Err fault => simp at accepted
    | Ok access =>
      obtain ⟨written, stored, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      obtain ⟨pair, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      rcases pair with ⟨thread, back⟩
      have accepted := successful_effect _ _ _ _ accepted
      rcases accepted with ⟨rfl, rfl⟩
      exact .store actor _ pointer bytes access value written resolved stored rfl

end MemorySafetyExecution.Native
