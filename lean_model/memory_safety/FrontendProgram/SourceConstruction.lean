/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceManifest
import FrontendProgram.SourceThreads
import FrontendProgram.SourceAllocations
import FrontendProgram.SourceOutputs
import FrontendProgram.SourceBufferInventory
import FrontendProgram.GrammarGlobalNames

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

theorem source_build_construction (source entry : Slice U8) (manifest : Manifest) (program : Program)
    (computed : strict_memory_frontend_launch.build_program source entry manifest=.ok (some program)) :
    ∃ parsed,strict_memory_frontend_parse.parse_source source entry=.ok (some parsed) ∧
      (GlobalNamesInjective source.val parsed → ConstructionMeaning source.val entry.val manifest parsed program) := by
  unfold strict_memory_frontend_launch.build_program at computed
  by_cases tooLarge : Slice.len source>1048576#usize
  · simp [tooLarge] at computed
  · simp only [tooLarge,ite_false] at computed
    obtain ⟨supported,supportedComputed,computed⟩:=source_bind_success _ _ _ computed
    cases supported with
    | false => simp at computed
    | true =>
      simp only [ite_true] at computed
      have manifestMeaning:=source_manifest_supported manifest supportedComputed
      obtain ⟨entryBytes,entryComputed,computed⟩:=source_bind_success _ _ _ computed
      obtain ⟨sameEntry,sameEntryComputed,computed⟩:=source_bind_success _ _ _ computed
      cases sameEntry with
      | false => simp at computed
      | true =>
        simp only [ite_true] at computed
        have compared:=source_bytes_equal entry entryBytes
        rw [sameEntryComputed] at compared
        have entryMeaning : utf8 manifest.entry=entry.val := by
          have equal : entry.val=entryBytes.val := by
            have equal : true=decide (entry.val=entryBytes.val) := by simpa only [WP.spec_ok] using compared
            exact of_decide_eq_true equal.symm
          exact (source_string_bytes manifest.entry entryBytes entryComputed).symm.trans equal.symm
        obtain ⟨parsed,parsedComputed,computed⟩:=source_bind_success _ _ _ computed
        cases parsed with
        | none => simp at computed
        | some parsed =>
          obtain ⟨buffers,buffersComputed,computed⟩:=source_bind_success _ _ _ computed
          cases buffers with
          | false => simp at computed
          | true =>
            simp only [ite_true] at computed
            obtain ⟨threads,threadsComputed,computed⟩:=source_bind_success _ _ _ computed
            cases threads with
            | none => simp at computed
            | some threads =>
              obtain ⟨allocations,allocationsComputed,computed⟩:=source_bind_success _ _ _ computed
              cases allocations with
              | none => simp at computed
              | some pair =>
                rcases pair with ⟨layouts,allocations⟩
                obtain ⟨outputs,outputsComputed,computed⟩:=source_bind_success _ _ _ computed
                cases outputs with
                | none => simp at computed
                | some outputs =>
                  have same : ({code:=parsed.code,layouts,initial:={threads,allocations},outputs}:Program)=program := by
                    simpa only [ok.injEq,Option.some.injEq] using computed
                  subst program
                  refine ⟨parsed,parsedComputed,?_⟩
                  intro names
                  obtain ⟨layoutMeaning,allocationCount,allocationMeaning,unique⟩:= source_initial_allocations source parsed manifest layouts allocations names allocationsComputed
                  obtain ⟨bufferMeanings,fresh⟩:=source_buffers_supported source parsed manifest buffersComputed
                  refine ⟨entryMeaning,manifestMeaning,⟨bufferMeanings,fresh,?_⟩,rfl,layoutMeaning,
                    allocationCount,allocationMeaning,source_initial_threads manifest parsed.scope.virtual_count
                      parsed.scope.predicate_count threads threadsComputed,
                    source_outputs source parsed manifest outputs outputsComputed⟩
                  intro i j object left right leftAt rightAt leftBacks rightBacks
                  have backing := leftBacks
                  obtain ⟨global,symbol,globalAt,_,_⟩:=backing
                  have inside:= (List.getElem?_eq_some_iff.mp globalAt).1
                  exact unique object inside i j left right leftAt rightAt leftBacks rightBacks

end PtxFrontendProgram
