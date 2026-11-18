/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentModel

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent

theorem source_resolution_model (program : CoreProgram) (state : CoreState)
    (pointer : Pointer) (width : U8) (write : Bool) (access : ResolvedAccess)
    (computed : execution_core.resolve_access program state pointer width write=.ok (.Ok access)) :
    ∃memory, state.allocations.val[pointer.allocation.val]?=some memory ∧
      MachineModel.resolve (programModel program) (stateModel state)
        (ValueFlow.Source.pointerModel pointer) width.val write=
        some (MemorySafety.Source.requestModel access.request,memoryModel memory) := by
  obtain ⟨descriptor,layout,memory,layoutAt,memoryAt,widthKnown,offsetVal,addressVal,
    live,originBound,alignment,aligned,offsetAligned,baseAligned,physicalBound,
    storageBound,definedShape,initialized⟩ := source_resolve_range program state pointer width write access computed
  refine ⟨memory,memoryAt,?_⟩
  have initialized' : write=true ∨ ∀index∈List.range width.val,
      memory.defined.val[pointer.offset.val+index]?=some true := by
    cases write with
    | true => exact Or.inl rfl
    | false => exact Or.inr (fun index member => initialized (by simpa using descriptor.2.2.2.2)
        index (List.mem_range.mp member))
  have requestEq : MemorySafety.Source.requestModel access.request=
      (⟨layout.base.val+pointer.offset.val,width.val,width.val,
        if write then .store true else .load⟩:MemorySafety.Request) := by
    simp only [MemorySafety.Source.requestModel,addressVal,descriptor.2.2.1,descriptor.2.2.2.1,
      descriptor.2.2.2.2]
    cases write <;> rfl
  rw [requestEq]
  simp [MachineModel.resolve,programModel,stateModel,ValueFlow.Source.pointerModel,
    List.getElem?_map,layoutAt,memoryAt,layoutModel,memoryModel,ValueFlow.supportedWidth,
    widthKnown,live,originBound,alignment,aligned,offsetAligned,baseAligned,physicalBound,
    storageBound,definedShape,initialized']
  constructor
  · simpa only [Nat.reducePow] using physicalBound
  · intro falseWrite index inside
    exact initialized (by simpa [falseWrite] using descriptor.2.2.2.2) index inside

end MemorySafetyExecution.Native.Independent
