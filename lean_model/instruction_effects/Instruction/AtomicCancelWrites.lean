/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicCancelRefinement
open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace AtomicCancel

/-- An accepted stop must actually name this base destination. Native write/metadata semantics are separate. -/
def baseDestination (operation : Op) (register : Reg) : Prop :=
  match operation with
  | .Mov destination _ | .Int _ _ destination _ _ |
      .MulWide _ destination _ _ _ | .CvtInt destination _ _ _ _ => destination = register
  | .Ld destination count _ bytes _ _ _ _ => count > 0#u8 ∧ bytes > 0#u8 ∧ destination = register
  | .Atom _ _ _ _ _ destination _ _ _ => destination = some register
  | _ => False

theorem source_overwrite_exact (operation : Op) (register : Reg) :
    atomic_cancel.overwrites_return operation register = .ok true ↔ baseDestination operation register := by
  classical
  cases operation <;>
    simp only [atomic_cancel.overwrites_return, baseDestination, same_register_true,
      Result.ok.injEq, Bool.false_eq_true]
  case Atom _ _ _ _ _ destination _ _ _ =>
    cases destination <;> simp [same_register_true]
  case Ld destination count _ bytes _ _ _ _ =>
    by_cases n : count > 0#u8 <;> by_cases width : bytes > 0#u8 <;>
      simp [n, width, same_register_true]

end AtomicCancel
