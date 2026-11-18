/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.InputClosure

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

def InputSourceClosed (program : execution_core.Program)
    (certificate : Slice input_independent_execution.State) (state : input_independent_execution.State) : Prop :=
  checked_execution.good program state.concrete = .ok true ∧
    ∀ actor, execution_core.running state.concrete.execution actor = .ok true →
      ∃ next, input_independent_execution.step program state actor = .ok (some next) ∧
        next ∈ certificate.val

theorem source_input_verify_exact_closure (program : execution_core.Program)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate = .ok true) :
    (∃ initial, input_independent_execution.initial program = .ok (some initial) ∧
      initial ∈ certificate.val) ∧
      ∀ state ∈ certificate.val, InputSourceClosed program certificate state := by
  unfold input_independent_execution.verify at accepted
  obtain ⟨result, initialized, accepted⟩ := source_bind_success _ _ _ accepted
  cases result with
  | none => simp at accepted
  | some initial =>
    obtain ⟨present, included, accepted⟩ := source_bind_success _ _ _ accepted
    cases present with
    | false => simp at accepted
    | true =>
      simp only [ite_true] at accepted
      obtain ⟨concrete, _, accepted⟩ := source_bind_success _ _ _ accepted
      obtain ⟨verified, _, accepted⟩ := source_bind_success _ _ _ accepted
      cases verified with
      | false => simp at accepted
      | true =>
        simp only [ite_true] at accepted
        rw [source_input_verify_loop] at accepted
        have member : initial ∈ certificate.val := by
          rw [source_input_present certificate initial] at included
          simpa only [ok.injEq, decide_eq_true_eq] using included
        refine ⟨⟨initial, initialized, member⟩, ?_⟩
        intro state stateMember
        have checked := checkAll_acceptance _ _ accepted state (by simpa using stateMember)
        obtain ⟨good, closed⟩ := source_input_state_closed program certificate state checked
        refine ⟨good, ?_⟩
        intro actor running
        obtain ⟨next, stepped, present⟩ := closed actor running
        rw [source_input_present certificate next] at present
        exact ⟨next, stepped, by simpa only [ok.injEq, decide_eq_true_eq] using present⟩

inductive InputSourceReachable (program : execution_core.Program) : input_independent_execution.State → Prop
  | initial {state} : input_independent_execution.initial program = .ok (some state) →
      InputSourceReachable program state
  | step {before after actor} : InputSourceReachable program before →
      execution_core.running before.concrete.execution actor = .ok true →
      input_independent_execution.step program before actor = .ok (some after) →
      InputSourceReachable program after

theorem source_input_verify_reachable (program : execution_core.Program)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate = .ok true)
    {state : input_independent_execution.State} (reachable : InputSourceReachable program state) :
    state ∈ certificate.val ∧ InputSourceClosed program certificate state := by
  obtain ⟨⟨initial, initialized, initialMember⟩, closed⟩ :=
    source_input_verify_exact_closure program certificate accepted
  induction reachable with
  | @initial state computed =>
    have equal : initial = state := by simpa only [initialized, ok.injEq, Option.some.injEq] using computed
    subst state
    exact ⟨initialMember, closed initial initialMember⟩
  | @step before after actor reachable running computed ih =>
    obtain ⟨next, stepped, member⟩ := ih.2.2 actor running
    have equal : next = after := by simpa only [stepped, ok.injEq, Option.some.injEq] using computed
    subst after
    exact ⟨member, closed next member⟩

end MemorySafetyExecution.Source
