/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.SourcePresent
import ValueFlowDependencies.SourceEquality

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

deriving instance DecidableEq for input_independent_execution.State

theorem input_state_equality (left right : input_independent_execution.State) :
    left = right ↔ left.concrete.execution = right.concrete.execution ∧
      left.concrete.monitor = right.concrete.monitor ∧ left.dependencies = right.dependencies := by
  cases left
  cases right
  simp only [input_independent_execution.State.mk.injEq, checked_state_equality, and_assoc]

theorem source_input_equality
    (left right : input_independent_execution.State) :
    input_independent_execution.state_equal left right = .ok (decide (left = right)) := by
  unfold input_independent_execution.state_equal
  rw [Native.source_state_equal]
  dsimp only [Native.CoreState]
  by_cases sameExecution : left.concrete.execution = right.concrete.execution <;>
    by_cases sameMonitor : left.concrete.monitor = right.concrete.monitor <;>
    simp only [source_monitor_equality, ValueFlowDependencies.Source.source_state_equal, bind_ok, decide_eq_true_eq,
      input_state_equality, sameExecution, sameMonitor,
      ite_true, ite_false] <;> rfl

theorem source_input_present_loop (states : Slice input_independent_execution.State) (state : input_independent_execution.State)
    (index : Usize) :
    input_independent_execution.state_present_loop states state index = .ok
      ((states.val.drop index.val).any (fun candidate => decide (candidate = state))) := by
  generalize fuelEq : states.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : input_independent_execution.state_present_loop states state index = (do
        let result ← input_independent_execution.state_present_loop.body states state index
        match result with
        | .cont after => input_independent_execution.state_present_loop states state after
        | .done output => .ok output) := by
      unfold input_independent_execution.state_present_loop
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
      let current : input_independent_execution.State := states.val[index.val]
      have lookup : Slice.index_usize states index = .ok current := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside, current]
      have test : index < Slice.len states := by scalar_tac
      have body : input_independent_execution.state_present_loop.body states state index = .ok
          (if current = state then .done true else .cont after) := by
        simp only [input_independent_execution.state_present_loop.body, test, ite_true, lookup,
          bind_ok, source_input_equality]
        by_cases same : current = state <;>
          simp only [same, decide_true, decide_false, Bool.false_eq_true, ite_true, ite_false, increment, bind_ok]; rfl
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
      simp [input_independent_execution.state_present_loop.body, test,
        List.drop_eq_nil_of_le (show states.val.length ≤ index.val by omega)]

theorem source_input_present (states : Slice input_independent_execution.State) (state : input_independent_execution.State)
:
    input_independent_execution.state_present states state = .ok (decide (state ∈ states.val)) := by
  rw [input_independent_execution.state_present, source_input_present_loop states state 0#usize]
  change Result.ok ((states.val.drop 0).any (fun candidate => decide (candidate = state))) = _
  simp only [List.drop_zero]
  congr 1
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq, List.any_eq_true, exists_eq_right]

end MemorySafetyExecution.Source
