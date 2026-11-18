/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputInitialMetadata

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

structure InitialBytesVariant (left right : CoreState) : Prop where
  threads : left.threads=right.threads
  allocationsLength : left.allocations.val.length=right.allocations.val.length
  allocations : ∀ (index : Nat) (l r : execution_core.AllocationBytes), left.allocations.val[index]?=some l →
    right.allocations.val[index]?=some r → l.bytes.val.length=r.bytes.val.length ∧ l.defined=r.defined ∧ l.live=r.live

def InitialStorageShape (state : CoreState) : Prop :=
  ∀ allocation∈state.allocations.val, allocation.defined.val.length=allocation.bytes.val.length

def InitialRegisterCanonical (state : CoreState) : Prop :=
  ∀ thread∈state.threads.val, ValueFlow.Storage.MachineRegistersVirtualMap.Canonical thread.registers.raw.vund.val

private theorem initial_input_banks (registers : value_flow.Registers) (shadow : DependencyRegisters)
    (initial : inputRegistersModel shadow=initialInputRegistersModel registers) :
    InputBanksRelated (dependenciesModel shadow) (bankModel registers) (bankModel registers) := by
  have virtuals:=congrArg Prod.fst initial
  have predicates:=congrArg Prod.snd initial
  change shadow.virtuals.val=List.replicate registers.raw.regs.val.length (initialInputMetadata false) at virtuals
  change shadow.predicates.val=List.replicate (registers.raw.preds.val.length*64) (initialInputMetadata false) at predicates
  constructor
  · intro index tainted tracked
    simp only [dependenciesModel,metadataDependency,virtuals] at tracked
    by_cases inside : index<registers.raw.regs.val.length <;>
      simp [List.getElem?_replicate,inside,initialInputMetadata] at tracked
  · intro index tainted tracked
    simp only [dependenciesModel,metadataDependency,predicates] at tracked
    by_cases inside : index<registers.raw.preds.val.length*64 <;>
      simp [List.getElem?_replicate,inside,initialInputMetadata] at tracked

theorem source_initial_inputs_related (left right : CoreState) (shadow : value_flow_dependencies.State)
    (computed : input_independent_execution.initial_dependencies left=.ok (some shadow))
    (variant : InitialBytesVariant left right) (storage : InitialStorageShape left)
    (canonical : InitialRegisterCanonical left) : StateInputsRelated shadow left right := by
  obtain ⟨threadModels,memoryModels⟩:=source_initial_input_dependencies left shadow computed
  have threadLengths:=congrArg List.length threadModels
  simp only [List.length_map] at threadLengths
  have memoryLengths:=congrArg List.length memoryModels
  simp only [List.length_map] at memoryLengths
  refine ⟨by rw [variant.threads],variant.allocationsLength,threadLengths,memoryLengths,?_,?_⟩
  · intro index l r metadata leftAt rightAt metadataAt
    have same : l=r := by simpa only [←variant.threads,leftAt,Option.some.injEq] using rightAt
    subst r
    have metadataEq:=congrArg (fun values=>values[index]?) threadModels
    have modelEq : inputRegistersModel metadata=initialInputRegistersModel l.registers := by
      simpa only [List.getElem?_map,leftAt,metadataAt,Option.map_some,Option.some.injEq] using metadataEq
    have virtualsEq:=congrArg Prod.fst modelEq
    have predicatesEq:=congrArg Prod.snd modelEq
    have virtualsLength:=congrArg List.length virtualsEq
    have predicatesLength:=congrArg List.length predicatesEq
    simp only [inputRegistersModel,initialInputRegistersModel,List.length_replicate] at virtualsLength predicatesLength
    refine ⟨rfl,rfl,rfl,⟨?_,virtualsLength,predicatesLength,initial_input_banks l.registers metadata modelEq⟩⟩
    exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,canonical l (List.mem_of_getElem? leftAt)⟩
  · intro index l r metadata leftAt rightAt metadataAt
    obtain ⟨bytesLength,definitions,live⟩:=variant.allocations index l r leftAt rightAt
    have metadataEq:=congrArg (fun values=>values[index]?) memoryModels
    have valuesEq : metadata.val=l.defined.val.map initialInputMetadata := by
      simpa only [List.getElem?_map,leftAt,metadataAt,Option.map_some,Option.some.injEq] using metadataEq
    have shape:=storage l (List.mem_of_getElem? leftAt)
    refine ⟨bytesLength,definitions,live,?_,shape,?_⟩
    · rw [valuesEq,List.length_map,shape]
    · intro byte entry entryAt
      rw [valuesEq,List.getElem?_map] at entryAt
      cases flagAt : l.defined.val[byte]? with
      | none => simp [flagAt] at entryAt
      | some flag =>
        have equal : initialInputMetadata flag=entry := by simpa only [flagAt,Option.map_some,Option.some.injEq] using entryAt
        subst entry
        refine ⟨rfl,?_⟩
        intro _ impossible
        cases impossible

end MemorySafetyExecution.Native
