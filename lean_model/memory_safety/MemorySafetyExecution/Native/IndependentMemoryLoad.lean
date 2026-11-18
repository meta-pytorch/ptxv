/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentLoad
import MemorySafetyExecution.Native.IndependentResolution
import MemorySafetyExecution.Native.IndependentBanks

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent

theorem source_load_model (program : CoreProgram) (before after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (event : memory_safety.Event)
    (destination : U16) (bits bytes : U8) (signed : Bool) (pointer : Pointer)
    (canonical : Canonical before) (selected : before.threads.val[actor.val]?=some thread)
    (nextPc : pc.val=thread.pc.val+1)
    (computed : execution_core.commit_action program before
      {actor,next_pc:=pc,action:=.Load destination bits bytes signed pointer}=.ok (.Ok event,after)) :
    MachineModel.applyAction (programModel program) (stateModel before) actor.val (threadModel thread)
      (.load destination.val bits.val bytes.val signed (ValueFlow.Source.pointerModel pointer))=
      some (MemorySafety.Source.eventModel event,stateModel after) ∧ Canonical after := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  obtain ⟨access,value,mask,registers,resolved,loaded,maskRun,written,eventEq,afterEq⟩ :=
    source_load_commit_components program before after actor pc thread destination bits bytes signed pointer event inside selected computed
  obtain ⟨descriptor,layout,memory,layoutAt,memoryAt,widthKnown,offsetVal,addressVal,
    live,originBound,alignment,aligned,offsetAligned,baseAligned,physicalBound,
    storageBound,definedShape,initialized⟩ := source_resolve_range program before pointer bytes false access resolved
  obtain ⟨other,otherAt,resolution⟩ := source_resolution_model program before pointer bytes false access resolved
  have same : other=memory := Option.some.inj (otherAt.symm.trans memoryAt)
  subst other
  have small : access.width.val  ≤  8 := by rcases widthKnown with h|h|h|h <;> simp_all only [descriptor.2.1] <;> omega
  have nonzero : 0<access.width.val := by rcases widthKnown with h|h|h|h <;> simp_all only [descriptor.2.1] <;> omega
  have bounded : access.offset.val+access.width.val  ≤  memory.bytes.val.length := by simpa only [offsetVal,descriptor.2.1] using storageBound
  have loadSpec := source_load_value_model before access memory signed
    (by simpa only [descriptor.1] using memoryAt) bounded nonzero small
  rw [loaded] at loadSpec
  have valueEq : value.val=MachineModel.load (memoryModel memory) pointer.offset.val bytes.val signed := by
    simpa only [WP.spec_ok,offsetVal,descriptor.2.1] using loadSpec
  have maskSpec := ValueFlow.Source.source_width_mask bits
  rw [maskRun] at maskSpec
  simp only [WP.spec_ok,ValueFlow.Source.outcomeModel] at maskSpec
  have maskVal : mask.val=2^bits.val-1 := by
    by_cases supported : ValueFlow.supportedBits bits.val
    · simpa only [supported,ite_true,Option.some.injEq] using maskSpec
    · simp only [supported,ite_false,reduceCtorEq] at maskSpec
  have original := canonical thread (List.mem_of_getElem? selected)
  obtain ⟨word,rendered,bank,proper⟩ := source_virtual_bank_model program thread.registers registers
    destination (.Scalar (value &&& mask)) original written
  have wordEq : word=(⟨ValueFlow.truncate bits.val (MachineModel.load (memoryModel memory)
      pointer.offset.val bytes.val signed),0,none⟩:ValueFlow.Word) := by
    simpa only [MachineModel.render,ValueFlow.Source.valueModel,Option.some.injEq,
      UScalar.val_and,maskVal,Nat.and_two_pow_sub_one_eq_mod,ValueFlow.truncate,valueEq] using rendered.symm
  rw [wordEq] at bank
  constructor
  · rw [afterEq,model_thread_set]
    simp only [MachineModel.applyAction,resolution,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
    rw [eventEq]
    simp only [MemorySafety.Source.eventModel,threadModel,nextPc,bank,ValueFlow.Source.pointerModel]
  · rw [afterEq]
    exact canonical_thread_set before actor _ canonical proper

end MemorySafetyExecution.Native.Independent
