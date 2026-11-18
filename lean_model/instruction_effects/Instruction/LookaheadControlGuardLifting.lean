/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.LookaheadControlFalseGuard
import Instruction.LookaheadControlFallthrough

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace LookaheadControl

theorem source_uniform_guard_retains_both_paths {R : Type}
    (readerInst : control_transfer.ControlReader R)
    (context : control_transfer.Context) (predicate : Pred) (operation : Op)
    (reader afterGuard afterTaken : R) (taken : control_transfer.Outcome)
    (supported : controls operation = true)
    (unknownGuard : readerInst.predicate reader predicate =
      .ok (.UniformUnknown, afterGuard))
    (targetResult : control_transfer.taken_operation readerInst context operation afterGuard =
      .ok (some taken, afterTaken))
    (fits : fallthroughFits taken) :
    control_transfer.from_operation readerInst context predicate operation reader ⦃ result =>
      result.2 = afterTaken ∧ ∃ outcome, result.1 = some outcome ∧
      permits context.pc.val (outcomeView outcome) (.at (context.pc.val + 1)) ∧
      ∀ observation, permits context.pc.val (outcomeView taken) observation →
        permits context.pc.val (outcomeView outcome) observation ⦄ := by
  cases operation <;>
    simp_all [controls, control_transfer.from_operation, bind_ok]
  all_goals
    step with source_with_fallthrough_preserves_both_paths taken context.pc fits
      as ⟨outcome, disabled, retained⟩
    exact ⟨outcome, rfl, disabled, retained⟩

end LookaheadControl
