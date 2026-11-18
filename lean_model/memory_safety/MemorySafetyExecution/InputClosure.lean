/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.InputPresent
import MemorySafetyExecution.SourceCertificateLoops
import MemorySafetyExecution.SourceTransitions

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 5000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

private theorem source_input_closed_loop (program : execution_core.Program)
    (certificate : Slice input_independent_execution.State) (state : input_independent_execution.State)
    (index : Usize)
    (computed : input_independent_execution.state_closed_loop program certificate
      state.concrete state.dependencies index = .ok true) :
    ∀ actor : Usize, index.val ≤ actor.val → actor.val < state.concrete.execution.threads.val.length →
      execution_core.running state.concrete.execution actor = .ok true →
      ∃ next, input_independent_execution.step program state actor = .ok (some next) ∧
        input_independent_execution.state_present certificate next = .ok true := by
  generalize fuelEq : state.concrete.execution.threads.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel inductionResult =>
    intro actor lower upper running
    have unroll : input_independent_execution.state_closed_loop program certificate
        state.concrete state.dependencies index = (do
        let result ← input_independent_execution.state_closed_loop.body program certificate
          state.concrete state.dependencies index
        match result with
        | .cont after => input_independent_execution.state_closed_loop program certificate state.concrete state.dependencies after
        | .done result => .ok result) := by
      unfold input_independent_execution.state_closed_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result <;> rfl
    rw [unroll] at computed
    have inside : index.val < state.concrete.execution.threads.val.length := by omega
    have test : index < alloc.vec.Vec.len state.concrete.execution.threads := by scalar_tac
    simp only [input_independent_execution.state_closed_loop.body, test, ite_true,
      Std.bind_assoc] at computed
    obtain ⟨active, observed, computed⟩ := source_bind_success _ _ _ computed
    have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
      have := alloc.vec.Vec.len_ineq state.concrete.execution.threads
      scalar_tac
    let after : Usize := Usize.ofNatCore (index.val + 1) bound
    have afterVal : after.val = index.val + 1 := rfl
    have increment := source_usize_increment index bound
    cases active with
    | false =>
      simp only [Bool.false_eq_true, ite_false, increment, bind_ok] at computed
      have different : actor ≠ index := by
        intro equal
        subst actor
        have impossible : false = true := by simpa only [observed, ok.injEq] using running
        cases impossible
      apply inductionResult (state.concrete.execution.threads.val.length - after.val)
        (by omega) after computed rfl actor (by
          have : index.val ≠ actor.val := by intro same; apply different; scalar_tac
          omega) upper running
    | true =>
      simp only [ite_true, Std.bind_assoc] at computed
      obtain ⟨result, stepped, computed⟩ := source_bind_success _ _ _ computed
      cases result with
      | none => simp at computed
      | some next =>
        simp only [Std.bind_assoc] at computed
        obtain ⟨present, included, computed⟩ := source_bind_success _ _ _ computed
        cases present with
        | false => simp at computed
        | true =>
          simp only [ite_true, increment, bind_ok] at computed
          by_cases same : actor = index
          · subst actor
            exact ⟨next, stepped, included⟩
          · apply inductionResult (state.concrete.execution.threads.val.length - after.val)
              (by omega) after computed rfl actor (by
                have : index.val ≠ actor.val := by intro equal; apply same; scalar_tac
                omega) upper running

theorem source_input_state_closed (program : execution_core.Program)
    (certificate : Slice input_independent_execution.State) (state : input_independent_execution.State)
    (computed : input_independent_execution.state_closed program certificate state = .ok true) :
    checked_execution.good program state.concrete = .ok true ∧
      ∀ actor, execution_core.running state.concrete.execution actor = .ok true →
        ∃ next, input_independent_execution.step program state actor = .ok (some next) ∧
          input_independent_execution.state_present certificate next = .ok true := by
  unfold input_independent_execution.state_closed at computed
  obtain ⟨good, tested, computed⟩ := source_bind_success _ _ _ computed
  cases good with
  | false => simp at computed
  | true =>
    refine ⟨tested, ?_⟩
    intro actor running
    exact source_input_closed_loop program certificate state 0#usize computed actor (by simp)
      (source_running_bounds state.concrete.execution actor running) running

theorem source_input_verify_loop (program : execution_core.Program)
    (certificate : Slice input_independent_execution.State) (index : Usize) :
    input_independent_execution.verify_loop program certificate index =
      checkAll (input_independent_execution.state_closed program certificate)
        (certificate.val.drop index.val) := by
  generalize fuelEq : certificate.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : input_independent_execution.verify_loop program certificate index = (do
        let result ← input_independent_execution.verify_loop.body program certificate index
        match result with
        | .cont after => input_independent_execution.verify_loop program certificate after
        | .done output => .ok output) := by
      unfold input_independent_execution.verify_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result <;> rfl
    rw [unroll]
    by_cases inside : index.val < certificate.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq certificate
        scalar_tac
      let after : Usize := Usize.ofNatCore (index.val + 1) bound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := usize_increment index bound
      have lookup : Slice.index_usize certificate index = .ok certificate.val[index.val] := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside]
      have test : index < Slice.len certificate := by scalar_tac
      simp only [input_independent_execution.verify_loop.body, test, ite_true,
        lookup, bind_ok, Std.bind_assoc, List.drop_eq_getElem_cons inside, checkAll]
      congr 1
      funext checked
      cases checked with
      | false => simp only [Bool.false_eq_true, ite_false, bind_ok]
      | true =>
        simp only [ite_true, increment, bind_ok]
        rw [ih (certificate.val.length - after.val) (by omega) after rfl]
        simp only [afterVal]
    · have test : ¬index < Slice.len certificate := by scalar_tac
      simp [input_independent_execution.verify_loop.body, test, checkAll,
        List.drop_eq_nil_of_le (show certificate.val.length ≤ index.val by omega)]

end MemorySafetyExecution.Source
