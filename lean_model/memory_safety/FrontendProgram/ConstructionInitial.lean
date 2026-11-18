/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Construction
import MemorySafetyExecution.Native.InputInitialRelation

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

theorem initial_allocation_storage_shape (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Nat) (allocation : Allocation)
    (meaning : InitialAllocation source parsed manifest object allocation) :
    allocation.defined.val.length=allocation.bytes.val.length := by
  obtain ⟨global,_,_,initial⟩:=meaning
  rcases initial with literal|backed
  · simp only [literal.2.1,literal.2.2,List.length_replicate]
  · obtain ⟨buffer,_,_,bytes,flags⟩:=backed
    simp only [bytes,flags,List.length_replicate]

theorem construction_initial_storage (source entry : List U8) (manifest : Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (program : Program)
    (meaning : ConstructionMeaning source entry manifest parsed program) :
    MemorySafetyExecution.Native.InitialStorageShape program.initial := by
  intro allocation member
  obtain ⟨object,atObject⟩:=List.mem_iff_getElem?.mp member
  exact initial_allocation_storage_shape source parsed manifest object allocation (meaning.2.2.2.2.2.2.1 object allocation atObject)

theorem construction_initial_registers (source entry : List U8) (manifest : Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (program : Program)
    (meaning : ConstructionMeaning source entry manifest parsed program) :
    MemorySafetyExecution.Native.InitialRegisterCanonical program.initial := by
  intro thread member
  obtain ⟨actor,atActor⟩:=List.mem_iff_getElem?.mp member
  have threads:=meaning.2.2.2.2.2.2.2.1
  obtain ⟨_,_,_,inventory⟩:=threads
  have initial:=inventory actor thread atActor
  have undefined:=initial.2.2.1.2.2.2.2.2
  simp only [ValueFlow.Storage.MachineRegistersVirtualMap.Canonical,undefined]
  exact ⟨List.Pairwise.nil,by simp⟩

theorem construction_initial_no_pointers (source entry : List U8) (manifest : Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (program : Program)
    (meaning : ConstructionMeaning source entry manifest parsed program) :
    ∀ thread∈program.initial.threads.val,∀ pointer∈thread.registers.pointers.val,pointer=none := by
  intro thread member pointer pointerMember
  obtain ⟨actor,atActor⟩:=List.mem_iff_getElem?.mp member
  have threads:=meaning.2.2.2.2.2.2.2.1
  obtain ⟨_,_,_,inventory⟩:=threads
  have initial:=inventory actor thread atActor
  have pointers:=initial.2.2.1.2.1
  rw [pointers] at pointerMember
  exact List.eq_of_mem_replicate pointerMember

end PtxFrontendProgram
