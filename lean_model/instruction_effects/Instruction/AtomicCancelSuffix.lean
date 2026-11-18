/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicCancelCaller
import Instruction.AtomicCancelReads
open Aeneas Aeneas.Std Result ControlFlow PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace AtomicCancel

structure SafeInstruction (register : Reg) (instruction : Inst) : Prop where
  scalar : instruction.uniform = false
  unguarded : Pred.is_always instruction.guard = .ok true
  privateScope : atomic_cancel.linear_private instruction.op = .ok true
  noRead : ¬baseRead instruction.op register

def Stops (register : Reg) (instruction : Inst) : Prop :=
  instruction.op = .Exit ∨ atomic_cancel.overwrites_return instruction.op register = .ok true

inductive BodyOutcome (code : Slice Inst) (register : Reg) (position : Usize) :
    ControlFlow (Usize × Bool × Bool) (Bool × Bool) → Prop where
  | done : BodyOutcome code register position (.done (true, false))
  | rejected : BodyOutcome code register position (.cont (position, false, false))
  | stopped (instruction : Inst)
      (lookup : Slice.index_usize code position = .ok instruction)
      (safe : SafeInstruction register instruction) (stop : Stops register instruction) :
      BodyOutcome code register position (.cont (position, true, true))
  | advanced (instruction : Inst) (next : Usize)
      (lookup : Slice.index_usize code position = .ok instruction)
      (safe : SafeInstruction register instruction)
      (bounded : position.val < code.val.length)
      (successor : (position + 1#usize : Result Usize) = .ok next) :
      BodyOutcome code register position (.cont (next, true, false))

theorem source_body_outcome (code : Slice Inst) (register : Reg) (position : Usize)
    (flow : ControlFlow (Usize × Bool × Bool) (Bool × Bool))
    (success : atomic_cancel.cancelled_return_loop.body code register.file register.idx position
      true false = .ok flow) : BodyOutcome code register position flow := by
  unfold atomic_cancel.cancelled_return_loop.body at success
  simp only [ite_true, Bool.false_eq_true, ite_false] at success
  by_cases bounded : position < Slice.len code
  · simp only [bounded, ite_true] at success
    obtain ⟨instruction, lookup, success⟩ := bind_success _ _ _ |>.mp success
    by_cases uniform : instruction.uniform = true
    · simp only [uniform, ite_true, Result.ok.injEq] at success
      subst flow
      exact .rejected
    · simp only [uniform, ite_false] at success
      have scalar : instruction.uniform = false := Bool.eq_false_of_not_eq_true uniform
      obtain ⟨unguarded, guard, success⟩ := bind_success _ _ _ |>.mp success
      cases unguarded with
      | false =>
        simp only [Bool.false_eq_true, ite_false, Result.ok.injEq] at success
        subst flow
        exact .rejected
      | true =>
        simp only [ite_true] at success
        obtain ⟨allowed, scope, success⟩ := bind_success _ _ _ |>.mp success
        cases allowed with
        | false =>
          simp only [Bool.false_eq_true, ite_false, Result.ok.injEq] at success
          subst flow
          exact .rejected
        | true =>
          simp only [ite_true] at success
          obtain ⟨reads, read, success⟩ := bind_success _ _ _ |>.mp success
          cases reads with
          | true =>
            simp only [ite_true, Result.ok.injEq] at success
            subst flow
            exact .rejected
          | false =>
            simp only [Bool.false_eq_true, ite_false] at success
            have safe : SafeInstruction register instruction :=
              ⟨scalar, guard, scope, source_refusal_means_no_base_read _ _ read⟩
            have bound : position.val < code.val.length := by scalar_tac
            cases hop : instruction.op <;>
              simp only [hop, bind_ok, Bool.false_eq_true, ite_false, ite_true] at success
            all_goals first
              | (have equal := Result.ok_injective success
                 subst flow
                 exact .stopped instruction lookup safe (.inl hop))
              | (obtain ⟨stop, stopped, success⟩ := bind_success _ _ _ |>.mp success
                 cases stop with
                 | true =>
                   simp only [ite_true, Result.ok.injEq] at success
                   subst flow
                   exact .stopped instruction lookup safe (.inr (by simpa only [hop] using stopped))
                 | false =>
                   simp only [Bool.false_eq_true, ite_false] at success
                   obtain ⟨next, successor, success⟩ := bind_success _ _ _ |>.mp success
                   have equal := Result.ok_injective success
                   subst flow
                   exact .advanced instruction next lookup safe bound successor)
  · simp only [bounded, ite_false, Result.ok.injEq] at success
    subst flow
    exact .done
end AtomicCancel
