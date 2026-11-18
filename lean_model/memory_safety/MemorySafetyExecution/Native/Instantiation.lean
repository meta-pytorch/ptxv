/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementSafety
import MemorySafetyExecution.Native.PlacementInitialState
import FrontendProgram.ConstructionInitial

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

/-- One program with arbitrary memory payloads and legal nonaliasing object bases. -/
structure MemoryInstance (original alternate : CoreProgram) : Prop where
  code : original.code=alternate.code
  outputs : original.outputs=alternate.outputs
  placements : ProgramPlacements original alternate
  initial : InitialBytesVariant original.initial alternate.initial

theorem source_memory_instance_original (source entry : List U8) (manifest : PtxFrontendProgram.Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (program : CoreProgram)
    (meaning : PtxFrontendProgram.ConstructionMeaning source entry manifest parsed program) :
    MemoryInstance program program := by
  have layouts:=meaning.2.2.2.2.1
  refine ⟨rfl,rfl,⟨rfl,?_,layouts.2.2⟩,⟨rfl,rfl,?_⟩⟩
  · intro index old fresh oldAt freshAt
    have same : old=fresh := by simpa only [oldAt,Option.some.injEq] using freshAt
    subst fresh
    refine ⟨rfl,rfl,?_⟩
    have inside : index<parsed.scope.globals.val.length := by
      have bound:=(List.getElem?_eq_some_iff.mp oldAt).1
      rw [layouts.1] at bound
      exact bound
    have globalAt:=List.getElem?_eq_getElem inside
    obtain ⟨actual,actualAt,_,alignment,aligned,bound⟩:=layouts.2.1 index parsed.scope.globals.val[index] globalAt
    have equal : actual=old := by simpa only [oldAt,Option.some.injEq] using actualAt.symm
    subst actual
    exact ⟨by simpa only [alignment] using aligned,bound⟩
  · intro index old fresh oldAt freshAt
    have same : old=fresh := by simpa only [oldAt,Option.some.injEq] using freshAt
    subst fresh
    exact ⟨rfl,rfl,rfl⟩

theorem source_input_verify_core_instances (source entry : List U8) (manifest : PtxFrontendProgram.Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (program : CoreProgram)
    (meaning : PtxFrontendProgram.ConstructionMeaning source entry manifest parsed program)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate=.ok true) :
    ∀ alternate,MemoryInstance program alternate →
      InitialRegisterCanonical alternate.initial ∧
      execution_core.initial_state alternate=.ok (.Ok alternate.initial) ∧
      FullCoreSafetyFrom alternate alternate.initial := by
  obtain ⟨⟨captured,initialized,_⟩,_⟩:=source_input_verify_exact_closure program certificate accepted
  obtain ⟨concreteInitialized,shadowInitialized⟩:=source_input_initial_parts program captured initialized
  obtain ⟨_,coreInitialized,_,_⟩:=source_initial_parts program captured.concrete concreteInitialized
  have exactInitial:=(source_initial_state_exact program captured.concrete.execution coreInitialized).1
  have capturedCore : execution_core.initial_state program=.ok (.Ok program.initial) := by
    simpa only [exactInitial] using coreInitialized
  have storage:=PtxFrontendProgram.construction_initial_storage source entry manifest parsed program meaning
  have canonical:=PtxFrontendProgram.construction_initial_registers source entry manifest parsed program meaning
  have noPointers:=PtxFrontendProgram.construction_initial_no_pointers source entry manifest parsed program meaning
  intro alternate instantiation
  have related : StateInputsRelated captured.dependencies program.initial alternate.initial :=
    source_initial_inputs_related program.initial alternate.initial captured.dependencies
      (by simpa only [exactInitial] using shadowInitialized) instantiation.initial storage canonical
  have capturedRelated : StateInputsRelated captured.dependencies captured.concrete.execution alternate.initial := by
    simpa only [exactInitial] using related
  have leftValid:=(source_initial_state_exact program program.initial capturedCore).2
  have leftSeparated:=(source_layout_valid_meaning program program.initial leftValid).2.2
  have rightSeparated:=placements_right_separated program alternate instantiation.placements
  have safe:=source_input_verify_full_safety_from program alternate certificate accepted instantiation.code instantiation.outputs
    instantiation.placements leftSeparated rightSeparated captured alternate.initial initialized capturedRelated
  have constructed:=source_initial_state_placements program alternate instantiation.placements instantiation.initial noPointers capturedCore
  have alternateCanonical : InitialRegisterCanonical alternate.initial := by
    simpa only [InitialRegisterCanonical,←instantiation.initial.threads] using canonical
  exact ⟨alternateCanonical,constructed,safe⟩


theorem source_input_verify_all_instances (source entry : List U8) (manifest : PtxFrontendProgram.Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (program : CoreProgram)
    (meaning : PtxFrontendProgram.ConstructionMeaning source entry manifest parsed program)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate=.ok true) :
    ∀ alternate,MemoryInstance program alternate → FullMachineMemorySafe alternate := by
  intro alternate instantiation
  obtain ⟨_,constructed,safe⟩:=source_input_verify_core_instances source entry manifest parsed program
    meaning certificate accepted alternate instantiation
  exact full_core_native_memory_safe alternate alternate.initial constructed safe

end MemorySafetyExecution.Native
