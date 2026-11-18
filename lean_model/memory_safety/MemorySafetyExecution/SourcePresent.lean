/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.StateEquality

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

deriving instance DecidableEq for memory_safety.WrittenSpan, memory_safety.AccessRecord,
  memory_safety.State, checked_execution.State

theorem source_monitor_equality (left right : memory_safety.State) :
    memory_safety.State.Insts.CoreCmpPartialEqState.eq left right = .ok (decide (left = right)) := by
  rw [source_state_equality]
  congr 1
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq]
  exact ⟨fun h => state_model_injective h, congrArg stateModel⟩

theorem checked_state_equality (left right : checked_execution.State) :
    left = right ↔ left.execution = right.execution ∧ left.monitor = right.monitor := by
  cases left
  cases right
  simp

theorem source_present_loop (states : Slice checked_execution.State) (state : checked_execution.State)
    (index : Usize) :
    checked_execution.state_present_loop states state index = .ok
      ((states.val.drop index.val).any (fun candidate => decide (candidate = state))) := by
  generalize fuelEq : states.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : checked_execution.state_present_loop states state index = (do
        let result ← checked_execution.state_present_loop.body states state index
        match result with
        | .cont after => checked_execution.state_present_loop states state after
        | .done output => .ok output) := by
      unfold checked_execution.state_present_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result <;> rfl
    rw [unroll]
    by_cases inside : index.val < states.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq states
        scalar_tac
      let after : Usize := Usize.ofNatCore (index.val + 1) bound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := usize_increment index bound
      let current : checked_execution.State := states.val[index.val]
      have lookup : Slice.index_usize states index = .ok current := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside, current]
      have test : index < Slice.len states := by scalar_tac
      have body : checked_execution.state_present_loop.body states state index = .ok
          (if current = state then .done true else .cont after) := by
        simp only [checked_execution.state_present_loop.body, test, ite_true, lookup,
          bind_ok, Native.source_state_equal]
        dsimp only [Native.CoreState]
        by_cases sameExecution : current.execution = state.execution <;>
          by_cases sameMonitor : current.monitor = state.monitor <;>
          simp only [source_monitor_equality, bind_ok, decide_eq_true_eq,
            checked_state_equality, sameExecution, sameMonitor, and_true, and_false,
            ite_true, ite_false, increment, bind_ok] <;>
          rfl
      rw [body]
      by_cases same : current = state
      · rw [ite_eq_left same]
        simp only [bind_ok, List.drop_eq_getElem_cons inside, List.any_cons]
        change Result.ok true = Result.ok (decide (current = state) || _)
        have yes : decide (current = state) = true := decide_eq_true same
        rw [yes]
        rfl
      · simp only [same, ite_false, bind_ok]
        rw [ih (states.val.length - after.val) (by omega) after rfl]
        rw [List.drop_eq_getElem_cons inside]
        simp only [List.any_cons, afterVal]
        change Result.ok _ = Result.ok (decide (current = state) || _)
        have no : decide (current = state) = false := decide_eq_false_iff_not.mpr same
        rw [no]
        rfl
    · have test : ¬index < Slice.len states := by scalar_tac
      simp [checked_execution.state_present_loop.body, test,
        List.drop_eq_nil_of_le (show states.val.length ≤ index.val by omega)]

theorem source_present (states : Slice checked_execution.State) (state : checked_execution.State) :
    checked_execution.state_present states state = .ok (decide (state ∈ states.val)) := by
  rw [checked_execution.state_present, source_present_loop]
  change Result.ok ((states.val.drop 0).any (fun candidate => decide (candidate = state))) = _
  simp only [List.drop_zero]
  congr 1
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq, List.any_eq_true, exists_eq_right]

end MemorySafetyExecution.Source
