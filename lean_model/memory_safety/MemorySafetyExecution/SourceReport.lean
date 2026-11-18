/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Soundness

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

theorem source_input_verify_concrete (program : execution_core.Program)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate = .ok true) :
    ∃ concrete, checked_execution.verify program concrete = .ok true := by
  unfold input_independent_execution.verify at accepted
  obtain ⟨result, _, accepted⟩ := source_bind_success _ _ _ accepted
  cases result with
  | none => simp at accepted
  | some initial =>
    obtain ⟨present, _, accepted⟩ := source_bind_success _ _ _ accepted
    cases present with
    | false => simp at accepted
    | true =>
      simp only [ite_true] at accepted
      obtain ⟨concrete, _, accepted⟩ := source_bind_success _ _ _ accepted
      obtain ⟨verified, checked, accepted⟩ := source_bind_success _ _ _ accepted
      cases verified with
      | false => simp at accepted
      | true => exact ⟨_, checked⟩

theorem source_report_safe_status (report : strict_memory_source.Report)
    (accepted : strict_memory_source.Report.is_safe report = .ok true) :
    report.status = .Safe := by
  unfold strict_memory_source.Report.is_safe at accepted
  cases found : report.status with
  | Safe => rfl
  | Unsafe => simp [found, strict_memory_status.Status.is_safe] at accepted
  | Inconclusive => simp [found, strict_memory_status.Status.is_safe] at accepted

theorem source_report_status_name_safe_iff (report : strict_memory_source.Report) :
    strict_memory_source.Report.status_name report = .ok (toStr "safe") ↔
      strict_memory_source.Report.is_safe report = .ok true := by
  cases found : report.status with
  | Safe => simp [strict_memory_source.Report.status_name, strict_memory_source.Report.is_safe,
      strict_memory_status.Status.is_safe, found]
  | Unsafe =>
    have different : (toStr "unsafe" : Str) ≠ toStr "safe" := by decide +kernel
    simp [strict_memory_source.Report.status_name, strict_memory_source.Report.is_safe,
      strict_memory_status.Status.is_safe, found, different]
  | Inconclusive =>
    have different : (toStr "inconclusive" : Str) ≠ toStr "safe" := by decide +kernel
    simp [strict_memory_source.Report.status_name, strict_memory_source.Report.is_safe,
      strict_memory_status.Status.is_safe, found, different]

theorem source_check_safe_certificate {F : Type _}
    (producerInterface : core.ops.function.FnOnce F
      (execution_core.Program × Usize) strict_memory_source.Candidate)
    (source entry : Slice U8) (manifest : ptxv_case.Manifest) (cap : Usize) (producer : F)
    (report : strict_memory_source.Report)
    (computed : strict_memory_source.check producerInterface source entry manifest cap producer = .ok report)
    (safe : report.status = .Safe) :
    ∃ program certificate,
      strict_memory_frontend_launch.build_program source entry manifest = .ok (some program) ∧
      input_independent_execution.verify program certificate = .ok true := by
  unfold strict_memory_source.check at computed
  obtain ⟨result, built, computed⟩ := source_bind_success _ _ _ computed
  cases result with
  | none =>
    simp only [strict_memory_source.Report.inconclusive, ok.injEq] at computed
    rw [← computed] at safe
    cases safe
  | some program =>
    obtain ⟨candidate, _, computed⟩ := source_bind_success _ _ _ computed
    cases candidate with
    | Unsafe states =>
      simp only [ok.injEq] at computed
      rw [← computed] at safe
      cases safe
    | Inconclusive states reason =>
      simp only [strict_memory_source.Report.inconclusive, ok.injEq] at computed
      rw [← computed] at safe
      cases safe
    | Complete certificate =>
      obtain ⟨verified, checked, computed⟩ := source_bind_success _ _ _ computed
      cases verified with
      | false =>
        simp only [Bool.false_eq_true, ite_false, strict_memory_source.Report.inconclusive,
          ok.injEq] at computed
        rw [← computed] at safe
        cases safe
      | true => exact ⟨program, certificate.deref, built, checked⟩

theorem source_check_safe_native_machine {F : Type _}
    (producerInterface : core.ops.function.FnOnce F
      (execution_core.Program × Usize) strict_memory_source.Candidate)
    (source entry : Slice U8) (manifest : ptxv_case.Manifest) (cap : Usize) (producer : F)
    (report : strict_memory_source.Report)
    (computed : strict_memory_source.check producerInterface source entry manifest cap producer = .ok report)
    (accepted : strict_memory_source.Report.is_safe report = .ok true) :
    ∃ program,
      strict_memory_frontend_launch.build_program source entry manifest = .ok (some program) ∧
      Native.MachineMemorySafe program := by
  obtain ⟨program, certificate, built, checked⟩ := source_check_safe_certificate producerInterface
    source entry manifest cap producer report computed (source_report_safe_status report accepted)
  obtain ⟨concrete, verified⟩ := source_input_verify_concrete program certificate checked
  exact ⟨program, built, Native.source_verify_machine_memory_safe program concrete verified⟩

end MemorySafetyExecution.Source
