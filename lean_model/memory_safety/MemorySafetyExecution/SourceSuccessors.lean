/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.SourceTransitions
import MemorySafety.SourceCertificate

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

theorem source_successors_loop_complete (program : execution_core.Program)
    (state : checked_execution.State) (result : alloc.vec.Vec checked_execution.State)
    (actor : Usize) (next : alloc.vec.Vec checked_execution.State)
    (computed : checked_execution.successors_loop program state result actor = .ok (some next)) :
    (∀ before ∈ result.val, before ∈ next.val) ∧
    ∀ selected : Usize, actor.val ≤ selected.val → selected.val < state.execution.threads.val.length →
      execution_core.running state.execution selected = .ok true →
      ∃ after, checked_execution.step program state selected = .ok (some after) ∧ after ∈ next.val := by
  generalize fuelEq : state.execution.threads.val.length - actor.val = fuel
  induction fuel using Nat.strong_induction_on generalizing result actor next with
  | h fuel ih =>
    have unroll : checked_execution.successors_loop program state result actor = (do
        let control ← checked_execution.successors_loop.body program state result actor
        match control with
        | .cont pair => checked_execution.successors_loop program state pair.1 pair.2
        | .done output => .ok output) := by
      unfold checked_execution.successors_loop
      rw [loop.eq_def]
      congr 1
      funext control
      cases control with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll] at computed
    by_cases inside : actor.val < state.execution.threads.val.length
    · have test : actor < alloc.vec.Vec.len state.execution.threads := by scalar_tac
      have bound : actor.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq state.execution.threads
        scalar_tac
      let following : Usize := Usize.ofNatCore (actor.val + 1) bound
      have followingVal : following.val = actor.val + 1 := by simp [following]
      have increment := usize_increment actor bound
      simp only [checked_execution.successors_loop.body, test, ite_true, Std.bind_assoc] at computed
      obtain ⟨running, enabled, remaining⟩ := source_bind_success _ _ _ computed
      cases running with
      | false =>
        simp only [Bool.false_eq_true, ite_false, increment, bind_ok] at remaining
        have inductionResult := ih (state.execution.threads.val.length - following.val) (by omega)
          result following next remaining rfl
        refine ⟨inductionResult.1, ?_⟩
        intro selected lower upper active
        by_cases same : selected = actor
        · subst selected
          rw [enabled] at active
          simp only [ok.injEq, Bool.false_eq_true] at active
        · have different : selected.val ≠ actor.val := by
            intro equal
            exact same (UScalar.eq_of_val_eq equal)
          exact inductionResult.2 selected (by omega) upper active
      | true =>
        simp only [ite_true, Std.bind_assoc] at remaining
        obtain ⟨choice, stepped, remaining⟩ := source_bind_success _ _ _ remaining
        cases choice with
        | none => simp at remaining
        | some after =>
          simp only [Std.bind_assoc] at remaining
          obtain ⟨pushed, appended, remaining⟩ := source_bind_success _ _ _ remaining
          have pushedVal := vec_push_success _ _ _ appended
          simp only [increment, bind_ok] at remaining
          have inductionResult := ih (state.execution.threads.val.length - following.val) (by omega)
            pushed following next remaining rfl
          refine ⟨?_, ?_⟩
          · intro before member
            apply inductionResult.1 before
            rw [pushedVal]
            exact List.mem_append_left _ member
          · intro selected lower upper active
            by_cases same : selected = actor
            · subst selected
              refine ⟨after, stepped, inductionResult.1 after ?_⟩
              rw [pushedVal]
              simp
            · have different : selected.val ≠ actor.val := by
                intro equal
                exact same (UScalar.eq_of_val_eq equal)
              exact inductionResult.2 selected (by omega) upper active
    · have test : ¬actor < alloc.vec.Vec.len state.execution.threads := by scalar_tac
      simp only [checked_execution.successors_loop.body, test, ite_false, bind_ok,
        ok.injEq, Option.some.injEq] at computed
      subst next
      refine ⟨fun _ member => member, ?_⟩
      intro selected lower upper _
      omega

theorem source_successors_complete (program : execution_core.Program)
    (state : checked_execution.State) (next : alloc.vec.Vec checked_execution.State)
    (computed : checked_execution.successors program state = .ok (some next))
    (actor : Usize) (enabled : execution_core.running state.execution actor = .ok true) :
    ∃ after, checked_execution.step program state actor = .ok (some after) ∧ after ∈ next.val := by
  have bound := source_running_bounds state.execution actor enabled
  have all := source_successors_loop_complete program state _ 0#usize next computed
  exact all.2 actor (by simp) bound enabled

end MemorySafetyExecution.Source
