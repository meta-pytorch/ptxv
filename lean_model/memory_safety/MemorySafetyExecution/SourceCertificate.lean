/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.SourcePresent
import MemorySafetyExecution.SourceCertificateLoops
import MemorySafetyExecution.SourceSuccessors

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

def SourceClosed (program : execution_core.Program) (certificate : Slice checked_execution.State)
    (state : checked_execution.State) : Prop :=
  checked_execution.good program state = .ok true ∧ ∃ next,
    checked_execution.successors program state = .ok (some next) ∧
    ∀ after ∈ next.val, after ∈ certificate.val

theorem source_closed_acceptance (program : execution_core.Program)
    (certificate : Slice checked_execution.State) (state : checked_execution.State)
    (accepted : checked_execution.state_closed program certificate state = .ok true) :
    SourceClosed program certificate state := by
  unfold checked_execution.state_closed at accepted
  obtain ⟨good, checked, remaining⟩ := source_bind_success _ _ _ accepted
  cases good with
  | false => simp at remaining
  | true =>
    simp only [ite_true] at remaining
    obtain ⟨result, computed, remaining⟩ := source_bind_success _ _ _ remaining
    cases result with
    | none => simp at remaining
    | some next =>
      refine ⟨checked, next, computed, ?_⟩
      change checked_execution.state_closed_loop certificate next 0#usize = .ok true at remaining
      rw [source_closed_loop] at remaining
      intro after member
      have present := checkAll_acceptance _ _ remaining after (by simpa using member)
      rw [source_present] at present
      simpa using present

theorem source_verify_exact_closure (program : execution_core.Program)
    (certificate : Slice checked_execution.State)
    (accepted : checked_execution.verify program certificate = .ok true) :
    (∃ initial, checked_execution.initial program = .ok (some initial) ∧ initial ∈ certificate.val) ∧
      ∀ state ∈ certificate.val, SourceClosed program certificate state := by
  unfold checked_execution.verify at accepted
  obtain ⟨result, initialized, remaining⟩ := source_bind_success _ _ _ accepted
  cases result with
  | none => simp at remaining
  | some initial =>
    dsimp only at remaining
    simp only [source_present, bind_ok, decide_eq_true_eq] at remaining
    by_cases present : initial ∈ certificate.val
    · rw [ite_eq_left present] at remaining
      rw [source_verify_loop] at remaining
      refine ⟨⟨initial, initialized, present⟩, ?_⟩
      intro state member
      exact source_closed_acceptance program certificate state
        (checkAll_acceptance _ _ remaining state (by simpa using member))
    · rw [ite_eq_right present] at remaining
      simp at remaining

inductive SourceReachable (program : execution_core.Program) : checked_execution.State → Prop
  | initial {state} : checked_execution.initial program = .ok (some state) →
      SourceReachable program state
  | step {before after actor} : SourceReachable program before →
      execution_core.running before.execution actor = .ok true →
      checked_execution.step program before actor = .ok (some after) →
      SourceReachable program after

theorem source_verify_reachable (program : execution_core.Program)
    (certificate : Slice checked_execution.State)
    (accepted : checked_execution.verify program certificate = .ok true)
    {state : checked_execution.State} (reachable : SourceReachable program state) :
    state ∈ certificate.val ∧ checked_execution.good program state = .ok true := by
  obtain ⟨⟨initial, initialized, initialMember⟩, closed⟩ := source_verify_exact_closure program certificate accepted
  induction reachable with
  | @initial state computed =>
    have equal : initial = state := by simpa only [initialized, ok.injEq, Option.some.injEq] using computed
    subst state
    exact ⟨initialMember, (closed initial initialMember).1⟩
  | @step before after actor reachable enabled computed ih =>
    obtain ⟨_, next, selected, complete⟩ := closed before ih.1
    obtain ⟨other, stepped, present⟩ := source_successors_complete program before next selected actor enabled
    have equal : other = after := by simpa only [stepped, ok.injEq, Option.some.injEq] using computed
    subst after
    have member := complete other present
    exact ⟨member, (closed other member).1⟩

theorem source_verify_actor (program : execution_core.Program)
    (certificate : Slice checked_execution.State)
    (accepted : checked_execution.verify program certificate = .ok true)
    {state : checked_execution.State} (reachable : SourceReachable program state)
    (actor : Usize) (enabled : execution_core.running state.execution actor = .ok true) :
    ∃ after, checked_execution.step program state actor = .ok (some after) ∧
      after ∈ certificate.val ∧ checked_execution.good program after = .ok true := by
  obtain ⟨_, closed⟩ := source_verify_exact_closure program certificate accepted
  have member := (source_verify_reachable program certificate accepted reachable).1
  obtain ⟨_, next, selected, complete⟩ := closed state member
  obtain ⟨after, computed, present⟩ := source_successors_complete program state next selected actor enabled
  have member := complete after present
  exact ⟨after, computed, member, (closed after member).1⟩

theorem source_check_requires_certificate (program : execution_core.Program) (cap : Usize)
    (report : checked_execution.Report) (computed : checked_execution.check program cap = .ok report)
    (accepted : report.status = .Safe) :
    ∃ certificate, checked_execution.verify program certificate = .ok true := by
  unfold checked_execution.check at computed
  obtain ⟨candidate, _, remaining⟩ := source_bind_success _ _ _ computed
  cases candidate with
  | Unsafe states =>
    simp only [ok.injEq] at remaining
    rw [← remaining] at accepted
    cases accepted
  | Inconclusive states =>
    simp only [ok.injEq] at remaining
    rw [← remaining] at accepted
    cases accepted
  | Complete certificate =>
    obtain ⟨checked, verified, remaining⟩ := source_bind_success _ _ _ remaining
    cases checked with
    | false =>
      simp only [Bool.false_eq_true, ite_false, bind_ok, ok.injEq] at remaining
      rw [← remaining] at accepted
      cases accepted
    | true => exact ⟨_, verified⟩

end MemorySafetyExecution.Source
