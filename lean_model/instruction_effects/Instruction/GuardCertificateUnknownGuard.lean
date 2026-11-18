/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateAssignment
import Instruction.GuardCertificateFrame
import Instruction.GuardCertificateWritten

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial
set_option autoImplicit false
set_option maxHeartbeats 3000000
namespace Instruction.GuardCertificate

/-- An unknown guard cannot preserve a written known fact merely because its bits agree. -/
theorem unknown_guard_written_fact_is_forgotten {C : Type}
    (reader : guard_certificate.Constants C) (context : guard_certificate.Evaluation C)
    (register : Reg) (writes : alloc.vec.Vec Reg) (result : Option U64)
    (unknown : guard_certificate.decision context.front context.facts
      context.instruction.guard = .ok none)
    (frameAt : guard_certificate.frame context.instruction.op = .ok (some writes))
    (writtenAt : guard_certificate.post_value_loop register writes 0#usize false = .ok true)
    (returned : guard_certificate.post_value reader context register = .ok result) :
    result = none := by
  unfold guard_certificate.post_value at returned
  obtain ⟨old, oldAt, tail⟩ := bind_success _ _ _ returned
  simp only [unknown, bind_ok, Bool.false_eq_true, ite_false, frameAt,
    core.option.Option.Insts.CoreOpsTry_traitTry.branch, writtenAt, ite_true] at tail
  obtain ⟨assigned, assignedAt, tail⟩ := bind_success _ _ _ tail
  obtain ⟨selected, selectedAt, tail⟩ := bind_success _ _ _ tail
  exact (Result.ok_injective tail).symm

end Instruction.GuardCertificate
