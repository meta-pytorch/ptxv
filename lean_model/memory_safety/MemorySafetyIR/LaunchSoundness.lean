/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.SourceAcceptance
import MemorySafetyIR.SourceLaunchProjection
import MemorySafetyIR.SourceLaunchValidation

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem validated_launch_count (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (count : Usize) (trace : ValidatedLaunchTrace kernel launch count)
    (grid : machine.Grid)
    (read : alloc.vec.Vec.index_usize launch.grids 0#usize = .ok grid) :
    count.val = gridThreadCount grid ∧ outputsDefined launch := by
  obtain ⟨selected, ctas, block, selectedRead, ctasRead, blockRead, multiplied, outputsRead, _⟩ := trace
  have same : selected = grid := by
    rw [read] at selectedRead
    simpa only [ok.injEq] using selectedRead.symm
  subst selected
  have ctasExact := (source_dimensions_product grid.grid ctas ctasRead).1
  have blockExact := (source_dimensions_product grid.block block blockRead).1
  have productExact := Usize.checked_mul_bv_spec ctas block
  simp only [multiplied] at productExact
  refine ⟨?_, source_validate_outputs launch outputsRead⟩
  simpa only [gridThreadCount, ctasExact, blockExact] using productExact.2.1

theorem accepted_certificate_launch_memory_safe
    (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (certificate : Slice ptxv_memory_write.memory_safety.State)
    (accepted : strict_memory_admission.verify_certificate kernel launch certificate = .ok true) :
    LaunchRestrictions kernel launch ∧
      LaunchMemorySafe (launchIRProgram kernel launch) (launchOutputs launch) := by
  obtain ⟨program, projected, safe⟩ := accepted_certificate_checks_actual_program kernel launch certificate accepted
  obtain ⟨count, grid, code, validated, gridRead, correspondence, model⟩ :=
    source_projected_memory_model kernel launch program projected
  obtain ⟨trace, restrictions⟩ := source_validate_launch_trace kernel launch count validated
  obtain ⟨countExact, outputs⟩ := validated_launch_count kernel launch count trace grid gridRead
  have selected : launch.grids.val[0]? = some grid := by
    simpa using MemorySafety.Source.vec_index_success launch.grids 0#usize grid gridRead
  have launchCode : List.Forall₂ (fun instruction result => ∀ valuation,
      interpretInstruction (bindGlobals (launchIRProgram kernel launch).globals valuation) instruction = result)
      (launchIRProgram kernel launch).code code := by
    simpa only [launchIRProgram, selected] using correspondence
  have modelExact : MemorySafety.Source.programModel program =
      projectedProgram (launchIRProgram kernel launch) code := by
    rw [model, countExact]
    simp only [projectedProgram, launchIRProgram, selected]
  rw [modelExact] at safe
  exact ⟨restrictions, projected_launch_memory_safe _ _ code launchCode safe
    (validated_outputs_initially_defined kernel launch outputs)⟩

theorem reported_safe_launch_memory_safe
    (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (certificate : Slice ptxv_memory_write.memory_safety.State)
    (accepted : strict_memory_status.status_for_certificate kernel launch certificate = .ok .Safe) :
    LaunchRestrictions kernel launch ∧
      LaunchMemorySafe (launchIRProgram kernel launch) (launchOutputs launch) :=
  accepted_certificate_launch_memory_safe kernel launch certificate
    (source_safe_status_requires_checked_certificate kernel launch certificate accepted)


theorem reported_is_safe_launch_memory_safe
    (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (certificate : Slice ptxv_memory_write.memory_safety.State)
    (status : strict_memory_status.Status)
    (reported : strict_memory_status.status_for_certificate kernel launch certificate = .ok status)
    (accepted : strict_memory_status.Status.is_safe status = .ok true) :
    LaunchRestrictions kernel launch ∧
      LaunchMemorySafe (launchIRProgram kernel launch) (launchOutputs launch) := by
  cases status with
  | Safe => exact reported_safe_launch_memory_safe kernel launch certificate reported
  | Unsafe => simp [strict_memory_status.Status.is_safe] at accepted
  | Inconclusive => simp [strict_memory_status.Status.is_safe] at accepted

end MemorySafetyIR
