/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceSoundness

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem source_verify_certificate_requires_checked_program
    (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (certificate : Slice ptxv_memory_write.memory_safety.State)
    (accepted : strict_memory_admission.verify_certificate kernel launch certificate = .ok true) :
    ∃ program, strict_memory_admission.project kernel launch = .ok (.Ok program) ∧
      ptxv_memory_write.memory_safety.verify program certificate = .ok true := by
  unfold strict_memory_admission.verify_certificate at accepted
  obtain ⟨result, projected, verified⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  cases result with
  | Ok program => exact ⟨program, projected, verified⟩
  | Err reason => simp at verified

theorem accepted_certificate_checks_actual_program
    (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (certificate : Slice ptxv_memory_write.memory_safety.State)
    (accepted : strict_memory_admission.verify_certificate kernel launch certificate = .ok true) :
    ∃ program, strict_memory_admission.project kernel launch = .ok (.Ok program) ∧
      MemorySafety.MemorySafe (MemorySafety.Source.programModel program) := by
  obtain ⟨program, projected, verified⟩ :=
    source_verify_certificate_requires_checked_program kernel launch certificate accepted
  exact ⟨program, projected, MemorySafety.Source.accepted_rust_program_memory_safe program certificate verified⟩

theorem source_safe_status_requires_checked_certificate
    (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (certificate : Slice ptxv_memory_write.memory_safety.State)
    (accepted : strict_memory_status.status_for_certificate kernel launch certificate = .ok .Safe) :
    strict_memory_admission.verify_certificate kernel launch certificate = .ok true := by
  unfold strict_memory_status.status_for_certificate at accepted
  obtain ⟨result, verified, status⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  cases result with
  | false => simp at status
  | true => exact verified

end MemorySafetyIR
