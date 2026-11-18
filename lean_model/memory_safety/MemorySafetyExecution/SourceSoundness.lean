/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.SourceMemorySafety
import MemorySafetyExecution.SourceReport
import FrontendProgram.Source

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

theorem source_check_is_safe_full_memory_safety {F : Type _}
    (producerInterface : core.ops.function.FnOnce F
      (execution_core.Program × Usize) strict_memory_source.Candidate)
    (source entry : Slice U8) (manifest : ptxv_case.Manifest) (cap : Usize) (producer : F)
    (report : strict_memory_source.Report)
    (computed : strict_memory_source.check producerInterface source entry manifest cap producer=.ok report)
    (accepted : strict_memory_source.Report.is_safe report=.ok true) :
    SourceMemorySafe source.val entry.val manifest := by
  obtain ⟨program,certificate,built,checked⟩:=source_check_safe_certificate producerInterface
    source entry manifest cap producer report computed (source_report_safe_status report accepted)
  obtain ⟨parsed,parsedComputed,constructed⟩:=PtxFrontendProgram.source_build_construction source entry manifest program built
  have parsedMeaning:=PtxFrontendProgram.source_parse_source source entry parsed parsedComputed
  have construction:=constructed (PtxFrontendProgram.module_global_names_injective source.val entry.val parsed parsedMeaning)
  have allInstances:=Native.source_input_verify_all_instances source.val entry.val manifest parsed program construction certificate checked
  have allIndependent:=Native.Independent.source_input_verify_all_independent_instances source.val entry.val manifest parsed program construction certificate checked
  have original:=Native.source_memory_instance_original source.val entry.val manifest parsed program construction
  exact ⟨program,⟨parsed,parsedMeaning,construction⟩,allInstances program original,
    allIndependent program original,fun alternate instantiation=>
      ⟨allInstances alternate instantiation,allIndependent alternate instantiation⟩⟩

theorem source_check_reports_safe_full_memory_safety {F : Type _}
    (producerInterface : core.ops.function.FnOnce F
      (execution_core.Program × Usize) strict_memory_source.Candidate)
    (source entry : Slice U8) (manifest : ptxv_case.Manifest) (cap : Usize) (producer : F)
    (report : strict_memory_source.Report)
    (computed : strict_memory_source.check producerInterface source entry manifest cap producer=.ok report)
    (accepted : strict_memory_source.Report.status_name report=.ok (toStr "safe")) :
    SourceMemorySafe source.val entry.val manifest :=
  source_check_is_safe_full_memory_safety producerInterface source entry manifest cap producer report
    computed ((source_report_status_name_safe_iff report).mp accepted)

end MemorySafetyExecution.Source
