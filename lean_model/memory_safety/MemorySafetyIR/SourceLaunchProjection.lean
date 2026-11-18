/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchExecution
import MemorySafety.SourceVectors
import MemorySafetyIR.GlobalSource
import MemorySafetyIR.SourceLaunchAllocations
import MemorySafetyIR.SourceLaunchReplication

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

def ProjectionTrace (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (program : ptxv_memory_write.memory_safety.Program) : Prop :=
  ∃ count grid allocations specialized nregs code threads,
    strict_memory_admission.validate_launch kernel launch = .ok (.Ok count) ∧
    alloc.vec.Vec.index_usize launch.grids 0#usize = .ok grid ∧
    strict_memory_launch.project_allocations launch = .ok (.Ok allocations) ∧
    strict_memory_globals.specialize_code kernel.code.deref grid.cbank_relocs.deref = .ok (.Ok specialized) ∧
    strict_memory_admission.project_code specialized.deref nregs = .ok (.Ok code) ∧
    strict_memory_launch.replicate_code code.deref count = .ok threads ∧
    program = ⟨allocations, threads⟩

theorem source_projection_trace (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (program : ptxv_memory_write.memory_safety.Program)
    (accepted : strict_memory_admission.project kernel launch = .ok (.Ok program)) :
    ProjectionTrace kernel launch program := by
  unfold strict_memory_admission.project at accepted
  obtain ⟨validation, validated, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  cases validation with
  | Err reason => simp at accepted
  | Ok count =>
    simp only [alloc.vec.Vec.index_slice_index] at accepted
    obtain ⟨grid, gridComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨allocationResult, allocationsComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    cases allocationResult with
    | Err reason => simp at accepted
    | Ok allocations =>
      obtain ⟨specializedResult, specializedComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
      cases specializedResult with
      | Err reason => simp at accepted
      | Ok specialized =>
        obtain ⟨index, indexComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        obtain ⟨nregs, nregsComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        obtain ⟨codeResult, codeComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        cases codeResult with
        | Err reason => simp at accepted
        | Ok code =>
          obtain ⟨threads, threadsComputed, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
          simp only [ok.injEq, core.result.Result.Ok.injEq] at accepted
          exact ⟨count, grid, allocations, specialized, nregs, code, threads,
            validated, gridComputed, allocationsComputed, specializedComputed, codeComputed,
            threadsComputed, accepted.symm⟩

theorem instruction_memory_model_agrees (instruction : CheckedIRInstruction) :
    checkedInstructionMemoryModel instruction = MemorySafety.Source.instructionModel instruction := by
  cases instruction with
  | Access request => cases request.kind <;> rfl
  | _ => rfl

theorem vec_deref_eq_slice {α : Type} (values : alloc.vec.Vec α) :
    values.deref = values.slice := by
  apply Slice.ext
  simp [alloc.vec.Vec.deref, alloc.vec.Vec.val]

theorem source_projected_memory_model (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (program : ptxv_memory_write.memory_safety.Program)
    (accepted : strict_memory_admission.project kernel launch = .ok (.Ok program)) :
    ∃ count grid code,
      strict_memory_admission.validate_launch kernel launch = .ok (.Ok count) ∧
      alloc.vec.Vec.index_usize launch.grids 0#usize = .ok grid ∧
      List.Forall₂ (fun instruction result => ∀ valuation,
        interpretInstruction (bindGlobals grid.cbank_relocs.val valuation) instruction = result)
        kernel.code.val code ∧
      MemorySafety.Source.programModel program = ⟨launchAllocations launch, List.replicate count.val code⟩ := by
  obtain ⟨count, grid, allocations, specialized, nregs, code, threads,
    validated, gridComputed, allocationsComputed, specializedComputed, codeComputed,
    threadsComputed, programEq⟩ := source_projection_trace kernel launch program accepted
  have allocationsExact := source_project_allocations launch allocations allocationsComputed
  have replicated := source_replicate_code code.deref count threads threadsComputed
  have codeComputedSlice : strict_memory_admission.project_code specialized.slice nregs = .ok (.Ok code) := by
    simpa only [vec_deref_eq_slice] using codeComputed
  have footprint := source_global_code_interpretation_exact kernel.code.deref grid.cbank_relocs.deref
    nregs specialized code specializedComputed codeComputedSlice
  have modelEq : checkedInstructionMemoryModel = MemorySafety.Source.instructionModel :=
    funext instruction_memory_model_agrees
  rw [modelEq] at footprint
  refine ⟨count, grid, code.val.map MemorySafety.Source.instructionModel,
    validated, gridComputed, ?_, ?_⟩
  · simpa only [vec_deref_eq_slice, alloc.vec.Vec.val] using footprint
  · rw [programEq]
    change MemorySafety.Program.mk _ _ = _
    rw [allocationsExact]
    simpa only [vec_deref_eq_slice, alloc.vec.Vec.val] using congrArg (MemorySafety.Program.mk (launchAllocations launch)) replicated

end MemorySafetyIR
