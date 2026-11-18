/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.AllocationGeometry
import MemorySafetyExecution.Native.CompatibleGeometry
import MemorySafetyExecution.Native.LayoutMeaning
import MemorySafetyExecution.Native.PlacementSeparation

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source
set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

@[step]
theorem source_layout_loop_geometry (program : CoreProgram) (allocations : alloc.vec.Vec execution_core.AllocationBytes)
    (cursor : Usize) (bound : cursor.val ≤ program.layouts.val.length)
    (lengths : program.layouts.val.length=allocations.val.length)
    (objects : ∀ (index : Nat) (layout : value_flow.AllocationLayout) (memory : execution_core.AllocationBytes),
      program.layouts.val[index]?=some layout → allocations.val[index]?=some memory →
      StorageExtent layout (alloc.vec.Vec.len memory.bytes) (alloc.vec.Vec.len memory.defined) ∧
      ValueFlow.LegalPlacement {bytes:=layout.bytes.val,alignment:=layout.alignment.val} layout.base.val)
    (separated : SeparatedLayouts program.layouts) :
    execution_core.layout_valid_loop program.code program.layouts program.initial program.outputs allocations true cursor
      ⦃ result => result=true ⦄ := by
  unfold execution_core.layout_valid_loop
  apply loop.spec_decr_nat (fun current : Bool × Usize => program.layouts.val.length-current.2.val)
    (fun current : Bool × Usize => current.2.val ≤ program.layouts.val.length ∧ current.1=true)
  · rintro ⟨valid,index⟩ ⟨before,flagTrue⟩
    dsimp only at before flagTrue
    subst valid
    unfold execution_core.layout_valid_loop.body
    by_cases inside : index.val < program.layouts.val.length
    · have test : index < alloc.vec.Vec.len program.layouts := by scalar_tac
      have memoryInside : index.val < allocations.val.length := by omega
      simp only [test,ite_true]
      step as ⟨layout,layoutAt⟩
      step as ⟨memory,memoryAt⟩
      have layoutSelected : program.layouts.val[index.val]?=some layout := by
        rw [List.getElem?_eq_getElem inside]
        simp only [layoutAt]
      have memorySelected : allocations.val[index.val]?=some memory := by
        rw [List.getElem?_eq_getElem memoryInside]
        simp only [memoryAt]
      obtain ⟨extent,legal⟩ := objects index.val layout memory layoutSelected memorySelected
      step with source_allocation_shape_geometry layout (alloc.vec.Vec.len memory.bytes)
        (alloc.vec.Vec.len memory.defined) extent legal as ⟨shape,shapeTrue⟩
      subst shape
      simp only [ite_true]
      step with source_allocation_compatible_geometry program index inside separated as ⟨compatible,compatibleTrue⟩
      subst compatible
      step as ⟨next,nextVal⟩
      exact ⟨by scalar_tac,by scalar_tac⟩
    · have test : ¬index < alloc.vec.Vec.len program.layouts := by scalar_tac
      simp [test]
  · exact ⟨bound,rfl⟩

theorem source_layout_valid_geometry (program : CoreProgram) (state : CoreState)
    (lengths : program.layouts.val.length=state.allocations.val.length)
    (objects : ∀ (index : Nat) (layout : value_flow.AllocationLayout) (memory : execution_core.AllocationBytes),
      program.layouts.val[index]?=some layout → state.allocations.val[index]?=some memory →
      StorageExtent layout (alloc.vec.Vec.len memory.bytes) (alloc.vec.Vec.len memory.defined) ∧
      ValueFlow.LegalPlacement {bytes:=layout.bytes.val,alignment:=layout.alignment.val} layout.base.val)
    (separated : SeparatedLayouts program.layouts) : execution_core.layout_valid program state=.ok true := by
  unfold execution_core.layout_valid
  have sameLength : alloc.vec.Vec.len program.layouts=alloc.vec.Vec.len state.allocations := by scalar_tac
  simp only [sameLength,decide_true]
  obtain ⟨result,computed,meaning⟩ := MemoryWrite.spec_returns _ _
    (source_layout_loop_geometry program state.allocations 0#usize (by scalar_tac) lengths objects separated)
  simpa only [meaning] using computed

theorem source_layout_valid_placements (leftProgram rightProgram : CoreProgram) (left right : CoreState)
    (placements : ProgramPlacements leftProgram rightProgram) (shapes : StorageShapesRelated left right)
    (captured : execution_core.layout_valid leftProgram left=.ok true) :
    execution_core.layout_valid rightProgram right=.ok true := by
  obtain ⟨leftLengths,extents,leftSeparated⟩ := source_layout_valid_meaning leftProgram left captured
  have rightLengths : rightProgram.layouts.val.length=right.allocations.val.length := by
    have layoutLengths:=placements.1
    have storageLengths:=shapes.1
    omega
  apply source_layout_valid_geometry rightProgram right rightLengths _ (placements_right_separated leftProgram rightProgram placements)
  intro index layout memory layoutAt memoryAt
  have layoutInside:=(List.getElem?_eq_some_iff.mp layoutAt).1
  have memoryInside:=(List.getElem?_eq_some_iff.mp memoryAt).1
  have oldLayoutInside : index < leftProgram.layouts.val.length := by have lengths:=placements.1;omega
  have oldMemoryInside : index < left.allocations.val.length := by have lengths:=shapes.1;omega
  let oldLayout:=leftProgram.layouts.val[index]
  let oldMemory:=left.allocations.val[index]
  have oldLayoutAt : leftProgram.layouts.val[index]?=some oldLayout := List.getElem?_eq_getElem oldLayoutInside
  have oldMemoryAt : left.allocations.val[index]?=some oldMemory := List.getElem?_eq_getElem oldMemoryInside
  obtain ⟨sameBytes,sameAlignment,legal⟩ := placements.2.1 index oldLayout layout oldLayoutAt layoutAt
  obtain ⟨bytesLength,definedEq,liveEq⟩ := shapes.2 index oldMemory memory oldMemoryAt memoryAt
  have oldExtent:=extents index oldLayout oldMemory oldLayoutAt oldMemoryAt
  refine ⟨?_,legal⟩
  rcases oldExtent with ⟨nonzero,size,defined,alignment⟩
  unfold StorageExtent
  refine ⟨?_,?_,?_,?_⟩
  · simpa only [sameBytes] using nonzero
  · rw [sameBytes]
    simpa using size.trans bytesLength
  · apply UScalar.eq_of_val_eq
    have definedLengths:=congrArg (fun values => values.val.length) definedEq
    have oldDefined:=congrArg (fun value : Usize => value.val) defined
    simpa using bytesLength.symm.trans (oldDefined.trans definedLengths)
  · simpa only [sameAlignment] using alignment

end MemorySafetyExecution.Native
