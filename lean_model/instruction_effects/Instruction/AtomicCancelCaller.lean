/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicCancelRefinement
open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace AtomicCancel

theorem bind_success {α β : Type} (input : Result α) (f : α → Result β) (value : β) :
    Aeneas.Std.bind input f = .ok value ↔ ∃ x, input = .ok x ∧ f x = .ok value := by
  cases input <;> simp [bind_ok, bind_vis, bind_div]

def atomReturn (operation : Op) : Option Reg :=
  match operation with
  | .Atom _ _ _ _ _ destination _ _ _ => destination
  | _ => none

theorem cancelled_return_has_zero_consumer (front : Front) (code : Slice Inst) (pc : Usize)
    (accepted : atomic_cancel.cancelled_return front code pc = .ok true) :
    ∃ first next second register,
      Slice.index_usize code pc = .ok first ∧ atomReturn first.op = some register ∧
      (pc + 1#usize : Result Usize) = .ok next ∧ Slice.index_usize code next = .ok second ∧
      Shape second.op register ∧
      ∀ values, cancellationValue second.op values = some 0 := by
  unfold atomic_cancel.cancelled_return at accepted
  obtain ⟨different, hfront, accepted⟩ := bind_success _ _ _ |>.mp accepted
  cases different with
  | true => simp only [if_pos rfl, ite_true, Result.ok.injEq, Bool.false_eq_true] at accepted
  | false =>
    simp only [Bool.false_eq_true, if_false] at accepted
    by_cases outside : pc ≥ Slice.len code
    · simp only [outside, if_true, Result.ok.injEq, Bool.false_eq_true] at accepted
    · simp only [outside, if_false] at accepted
      obtain ⟨first, hfirst, accepted⟩ := bind_success _ _ _ |>.mp accepted
      by_cases uniform : first.uniform = true
      · simp only [uniform, if_true, Result.ok.injEq, Bool.false_eq_true] at accepted
      · simp only [uniform, if_false] at accepted
        obtain ⟨guard, hguard, accepted⟩ := bind_success _ _ _ |>.mp accepted
        cases guard with
        | false => simp only [Bool.false_eq_true, if_false, Result.ok.injEq] at accepted
        | true =>
          simp only [if_pos rfl, ite_true] at accepted
          cases hop : first.op <;> simp only [hop] at accepted
          all_goals try (simp only [Result.ok.injEq, Bool.false_eq_true] at accepted)
          rename_i kind space bytes address value destination sync predicate replacement
          cases destination with
          | none => simp only [Result.ok.injEq, Bool.false_eq_true] at accepted
          | some register =>
            obtain ⟨plain, hplain, accepted⟩ := bind_success _ _ _ |>.mp accepted
            cases plain with
            | false => simp only [Bool.false_eq_true, if_false, Result.ok.injEq] at accepted
            | true =>
              simp only [if_pos rfl, ite_true] at accepted
              obtain ⟨privateFile, hprivate, accepted⟩ := bind_success _ _ _ |>.mp accepted
              cases privateFile with
              | false => simp only [Bool.false_eq_true, if_false, Result.ok.injEq] at accepted
              | true =>
                simp only [if_pos rfl, ite_true] at accepted
                obtain ⟨next, hnext, accepted⟩ := bind_success _ _ _ |>.mp accepted
                by_cases past : next ≥ Slice.len code
                · simp only [past, if_true, Result.ok.injEq, Bool.false_eq_true] at accepted
                · simp only [past, if_false] at accepted
                  obtain ⟨second, hsecond, accepted⟩ := bind_success _ _ _ |>.mp accepted
                  by_cases uniform2 : second.uniform = true
                  · simp only [uniform2, if_true, Result.ok.injEq, Bool.false_eq_true] at accepted
                  · simp only [uniform2, if_false] at accepted
                    obtain ⟨guard2, hguard2, accepted⟩ := bind_success _ _ _ |>.mp accepted
                    cases guard2 with
                    | false => simp only [Bool.false_eq_true, if_false, Result.ok.injEq] at accepted
                    | true =>
                      simp only [if_pos rfl, ite_true] at accepted
                      obtain ⟨cancels, hcancels, accepted⟩ := bind_success _ _ _ |>.mp accepted
                      cases cancels with
                      | false => simp only [Bool.false_eq_true, if_false, Result.ok.injEq] at accepted
                      | true =>
                        exact ⟨first, next, second, register, hfirst, by simp [atomReturn, hop],
                          hnext, hsecond, cancels_true _ _ hcancels,
                          source_cancellation_value_zero _ _ hcancels⟩
end AtomicCancel
