/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.SourceSoundness
import MemorySafetyExecution.RunSafety
import MemorySafetyExecution.Native.ConstructionInstance
import MemorySafetyExecution.Native.InstanceComposition
import FrontendProgram.ModuleUnique

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

theorem source_memory_safe_every_source_program
    (source entry : List U8) (manifest : ptxv_case.Manifest)
    (safe : SourceMemorySafe source entry manifest) :
    ∀ program, PtxFrontendProgram.SourceProgram source entry manifest program →
      Native.FullMachineMemorySafe program ∧
      MachineModel.MemorySafeFrom (Native.Independent.programModel program)
        (Native.Independent.stateModel program.initial) ∧
      ∀ alternate, Native.MemoryInstance program alternate →
        Native.FullMachineMemorySafe alternate ∧
        MachineModel.MemorySafeFrom (Native.Independent.programModel alternate)
          (Native.Independent.stateModel alternate.initial) := by
  obtain ⟨original, ⟨parsed, grammar, constructed⟩, _, _, allInstances⟩ := safe
  intro program meaning
  obtain ⟨otherParsed, otherGrammar, otherConstructed⟩ := meaning
  have same := PtxFrontendProgram.module_meaning_unique source entry parsed otherParsed grammar otherGrammar
  subst otherParsed
  have instanceOf := PtxFrontendProgram.construction_memory_instance source entry manifest parsed
    original program grammar constructed otherConstructed
  have safeProgram := allInstances program instanceOf
  refine ⟨safeProgram.1, safeProgram.2, ?_⟩
  intro alternate alternateInstance
  exact allInstances alternate (instanceOf.trans alternateInstance)

theorem source_check_reports_safe_every_source_program {F : Type _}
    (producerInterface : core.ops.function.FnOnce F
      (ptxv_memory_write.execution_core.Program × Usize) strict_memory_source.Candidate)
    (source entry : Slice U8) (manifest : ptxv_case.Manifest) (cap : Usize) (producer : F)
    (report : strict_memory_source.Report)
    (computed : strict_memory_source.check producerInterface source entry manifest cap producer = .ok report)
    (accepted : strict_memory_source.Report.status_name report = .ok (toStr "safe")) :
    SourceMemorySafe source.val entry.val manifest ∧
    ∀ program, PtxFrontendProgram.SourceProgram source.val entry.val manifest program →
      Native.FullMachineMemorySafe program ∧
      MachineModel.MemorySafeFrom (Native.Independent.programModel program)
        (Native.Independent.stateModel program.initial) ∧
      ∀ alternate, Native.MemoryInstance program alternate →
        Native.FullMachineMemorySafe alternate ∧
        MachineModel.MemorySafeFrom (Native.Independent.programModel alternate)
          (Native.Independent.stateModel alternate.initial) := by
  have safe := source_check_reports_safe_full_memory_safety producerInterface source entry manifest cap
    producer report computed accepted
  exact ⟨safe, source_memory_safe_every_source_program source.val entry.val manifest safe⟩

theorem source_check_reports_safe_every_source_execution {F : Type _}
    (producerInterface : core.ops.function.FnOnce F
      (ptxv_memory_write.execution_core.Program × Usize) strict_memory_source.Candidate)
    (source entry : Slice U8) (manifest : ptxv_case.Manifest) (cap : Usize) (producer : F)
    (report : strict_memory_source.Report)
    (computed : strict_memory_source.check producerInterface source entry manifest cap producer = .ok report)
    (accepted : strict_memory_source.Report.status_name report = .ok (toStr "safe"))
    (program : Native.CoreProgram)
    (meaning : PtxFrontendProgram.SourceProgram source.val entry.val manifest program)
    (alternate : Native.CoreProgram) (instanceOf : Native.MemoryInstance program alternate)
    (states : Nat → MachineModel.State) (events : Nat → MemorySafety.Event)
    (execution : MachineModel.Execution (Native.Independent.programModel alternate)
      (Native.Independent.stateModel alternate.initial) states events)
    (first second : Nat) (ordered : first < second)
    (before after : MemorySafety.AccessRecord)
    (earlier : before ∈ readWriteRecords (events first))
    (later : after ∈ readWriteRecords (events second)) :
    ¬MemorySafety.Conflict before after := by
  have allPrograms := (source_check_reports_safe_every_source_program producerInterface source entry manifest
    cap producer report computed accepted).2
  have safe := (allPrograms program meaning).2.2 alternate instanceOf
  exact MachineModel.memory_safe_execution_no_conflict _ _ safe.2 states events execution first second
    ordered before after earlier later

end MemorySafetyExecution.Source
