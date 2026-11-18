/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicCancelSuffix
open Aeneas Aeneas.Std Result ControlFlow PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace AtomicCancel

inductive ReadFreeSuffix (code : Slice Inst) (register : Reg) : Usize → Prop where
  | stop (position : Usize) (instruction : Inst)
      (lookup : Slice.index_usize code position = .ok instruction)
      (safe : SafeInstruction register instruction) (stopped : Stops register instruction) :
      ReadFreeSuffix code register position
  | next (position following : Usize) (instruction : Inst)
      (lookup : Slice.index_usize code position = .ok instruction)
      (safe : SafeInstruction register instruction)
      (successor : (position + 1#usize : Result Usize) = .ok following)
      (rest : ReadFreeSuffix code register following) : ReadFreeSuffix code register position

theorem source_disabled_suffix (code : Slice Inst) (register : Reg) (position : Usize) :
    atomic_cancel.cancelled_return_loop code register.file register.idx position false false =
      .ok (false, false) := by
  unfold atomic_cancel.cancelled_return_loop
  rw [loop]
  by_cases bounded : position < Slice.len code <;>
    simp [atomic_cancel.cancelled_return_loop.body, bounded, bind_ok]

theorem source_finished_suffix (code : Slice Inst) (register : Reg) (position : Usize) :
    atomic_cancel.cancelled_return_loop code register.file register.idx position true true =
      .ok (true, true) := by
  unfold atomic_cancel.cancelled_return_loop
  rw [loop]
  by_cases bounded : position < Slice.len code <;>
    simp [atomic_cancel.cancelled_return_loop.body, bounded, bind_ok]

theorem checked_next_value (code : Slice Inst) (position next : Usize)
    (bounded : position.val < code.val.length)
    (successor : (position + 1#usize : Result Usize) = .ok next) :
    next.val = position.val + 1 := by
  have capacity := Slice.length_ineq code
  have result : (position + 1#usize : Result Usize) ⦃ value =>
      value.val = position.val + 1 ⦄ := by
    step
    assumption
  rw [successor] at result
  exact (WP.spec_ok next).mp result

theorem source_suffix_has_no_read_until_stop (code : Slice Inst) (register : Reg) (position : Usize)
    (success : atomic_cancel.cancelled_return_loop code register.file register.idx position
      true false = .ok (true, true)) : ReadFreeSuffix code register position := by
  generalize measure_eq : code.val.length - position.val = remaining
  induction remaining using Nat.strongRecOn generalizing position with
  | ind remaining induction =>
    unfold atomic_cancel.cancelled_return_loop at success
    rw [loop] at success
    obtain ⟨flow, body, success⟩ := bind_success _ _ _ |>.mp success
    have outcome := source_body_outcome code register position flow body
    cases outcome with
    | done => simp only [Result.ok.injEq, Prod.mk.injEq, Bool.false_eq_true, and_false] at success
    | rejected =>
      change atomic_cancel.cancelled_return_loop code register.file register.idx position
        false false = .ok (true, true) at success
      rw [source_disabled_suffix] at success
      simp at success
    | stopped instruction lookup safe stopped => exact .stop position instruction lookup safe stopped
    | advanced instruction next lookup safe bounded successor =>
      change atomic_cancel.cancelled_return_loop code register.file register.idx next
        true false = .ok (true, true) at success
      have next_value := checked_next_value code position next bounded successor
      have smaller : code.val.length - next.val < remaining := by omega
      exact .next position next instruction lookup safe successor
        (induction _ smaller next success rfl)
end AtomicCancel
