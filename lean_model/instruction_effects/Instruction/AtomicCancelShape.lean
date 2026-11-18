/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs
import Instruction.AtomicCancelWord

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace AtomicCancel

def Shape (operation : Op) (register : Reg) : Prop :=
  ∃ kind destination, (kind = IntOp.Sub ∨ kind = IntOp.Xor) ∧
    destination.file = File.V ∧
    operation = Op.Int kind 32#u8 destination (Val.Reg register 1#u8) (Val.Reg register 1#u8)

theorem same_register_eq (left right : Reg) :
    atomic_cancel.same_register left right = .ok (by classical exact decide (left = right)) := by
  classical
  rcases left with ⟨lf, li⟩
  rcases right with ⟨rf, ri⟩
  cases lf <;> cases rf <;>
    simp [atomic_cancel.same_register, read_discriminant, File.read_discriminant, IScalar.hcast, lift]
  all_goals
    rcases System.Platform.numBits_eq with h | h <;>
      (rw [h]; simp [BitVec.signExtend])

theorem same_register_true (left right : Reg) :
    atomic_cancel.same_register left right = .ok true ↔ left = right := by
  classical
  simp [same_register_eq]

end AtomicCancel
