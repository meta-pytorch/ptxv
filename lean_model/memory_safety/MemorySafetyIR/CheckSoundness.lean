/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchSoundness

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem report_is_safe_requires_safe (report : strict_memory.Report)
    (accepted : strict_memory.Report.is_safe report = .ok true) :
    report.status = .Safe := by
  unfold strict_memory.Report.is_safe at accepted
  cases found : report.status with
  | Safe => rfl
  | Unsafe => simp [found, strict_memory_status.Status.is_safe] at accepted
  | Inconclusive => simp [found, strict_memory_status.Status.is_safe] at accepted

theorem checked_safe_requires_certificate
    (kernel : ptxv_ir.Kernel) (launch : machine.Launch) (cap : Usize)
    (report : strict_memory.Report)
    (computed : strict_memory.check kernel launch cap = .ok report)
    (accepted : report.status = .Safe) :
    ∃ certificate, strict_memory_status.status_for_certificate kernel launch certificate = .ok .Safe := by
  unfold strict_memory.check at computed
  obtain ⟨projection, _, remaining⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
  cases projection with
  | Err reason =>
    simp only [strict_memory.Report.inconclusive, ok.injEq] at remaining
    rw [← remaining] at accepted
    cases accepted
  | Ok program =>
    obtain ⟨candidate, _, remaining⟩ := MemorySafety.Source.source_bind_success _ _ _ remaining
    cases candidate with
    | Unsafe states =>
      simp only [ok.injEq] at remaining
      rw [← remaining] at accepted
      cases accepted
    | Inconclusive states reason =>
      simp only [strict_memory.Report.inconclusive, ok.injEq] at remaining
      rw [← remaining] at accepted
      cases accepted
    | Complete certificate =>
      obtain ⟨status, verified, remaining⟩ := MemorySafety.Source.source_bind_success _ _ _ remaining
      obtain ⟨safe, _, remaining⟩ := MemorySafety.Source.source_bind_success _ _ _ remaining
      cases safe <;> simp only [ite_false, ite_true, Bool.false_eq_true, ok.injEq] at remaining <;>
        rw [← remaining] at accepted <;>
        have exact : status = .Safe := accepted <;>
        exact ⟨_, by simpa only [exact] using verified⟩

theorem check_report_is_safe_launch_memory_safe
    (kernel : ptxv_ir.Kernel) (launch : machine.Launch) (cap : Usize)
    (report : strict_memory.Report)
    (computed : strict_memory.check kernel launch cap = .ok report)
    (accepted : strict_memory.Report.is_safe report = .ok true) :
    LaunchRestrictions kernel launch ∧
      LaunchMemorySafe (launchIRProgram kernel launch) (launchOutputs launch) := by
  obtain ⟨certificate, checked⟩ := checked_safe_requires_certificate kernel launch cap report computed
    (report_is_safe_requires_safe report accepted)
  exact reported_safe_launch_memory_safe kernel launch certificate checked

end MemorySafetyIR
