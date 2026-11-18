/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs
import Instruction.LookaheadControlModel

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace LookaheadControl

def controls : Op → Bool
  | .Brx .. | .Call .. | .Ret .. => true
  | _ => false

theorem source_false_guard_skips_target_decode {R : Type}
    (readerInst : control_transfer.ControlReader R)
    (context : control_transfer.Context) (predicate : Pred) (operation : Op)
    (reader afterGuard : R)
    (supported : controls operation = true)
    (falseGuard : readerInst.predicate reader predicate =
      .ok (.Known 0#u64, afterGuard)) :
    control_transfer.from_operation readerInst context predicate operation reader =
      .ok (some .Next, afterGuard) := by
  cases operation <;>
    simp_all [controls, control_transfer.from_operation, bind_ok] <;> rfl

theorem source_false_guard_preserves_disabled_step {R : Type}
    (readerInst : control_transfer.ControlReader R)
    (context : control_transfer.Context) (predicate : Pred) (operation : Op)
    (reader afterGuard : R)
    (supported : controls operation = true)
    (falseGuard : readerInst.predicate reader predicate =
      .ok (.Known 0#u64, afterGuard)) :
    control_transfer.from_operation readerInst context predicate operation reader =
      .ok (some .Next, afterGuard) ∧
    permits context.pc.val .next (.at (context.pc.val + 1)) := by
  exact ⟨source_false_guard_skips_target_decode readerInst context predicate operation
    reader afterGuard supported falseGuard, rfl⟩

end LookaheadControl
