/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceDependencyModel
import ValueFlow.SourceInputWriters
import MemorySafetyExecution.Native.Enabledness

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

structure RegisterInputShape (left right : V.Registers) : Prop where
  virtualLength : left.raw.regs.val.length = right.raw.regs.val.length
  predicateLength : left.raw.preds.val.length = right.raw.preds.val.length
  pointers : left.pointers = right.pointers
  written : left.raw.written = right.raw.written
  definitions : left.raw.pdef = right.raw.pdef
  undefined : left.raw.vund = right.raw.vund
  canonical : ValueFlow.Storage.MachineRegistersVirtualMap.Canonical left.raw.vund.val

structure RegistersInputsRelated (dependencies : DependencyRegisters) (left right : V.Registers) : Prop where
  shape : RegisterInputShape left right
  virtualMetadataLength : dependencies.virtuals.val.length = left.raw.regs.val.length
  predicateMetadataLength : dependencies.predicates.val.length = left.raw.preds.val.length * 64
  banks : InputBanksRelated (dependenciesModel dependencies) (bankModel left) (bankModel right)

structure ThreadInputsRelated (dependencies : DependencyRegisters)
    (left right : execution_core.ThreadState) : Prop where
  pc : left.pc = right.pc
  running : left.running = right.running
  ids : left.ids = right.ids
  registers : RegistersInputsRelated dependencies left.registers right.registers

structure MemoryInputsRelated (metadata : alloc.vec.Vec Metadata)
    (left right : execution_core.AllocationBytes) : Prop where
  bytesLength : left.bytes.val.length = right.bytes.val.length
  definitions : left.defined = right.defined
  live : left.live = right.live
  metadataLength : metadata.val.length = left.bytes.val.length
  storageShape : left.defined.val.length = left.bytes.val.length
  bytes : ∀ (index : Nat) (entry : Metadata), metadata.val[index]? = some entry →
    left.defined.val[index]? = some entry.defined ∧
    (entry.defined = true → entry.tainted = false →
      left.bytes.val[index]? = right.bytes.val[index]?)

structure StateInputsRelated (dependencies : value_flow_dependencies.State) (left right : CoreState) : Prop where
  threadsLength : left.threads.val.length = right.threads.val.length
  allocationsLength : left.allocations.val.length = right.allocations.val.length
  threadMetadataLength : dependencies.threads.val.length = left.threads.val.length
  allocationMetadataLength : dependencies.memory.val.length = left.allocations.val.length
  threads : ∀ (index : Nat) (l r : execution_core.ThreadState) (shadow : DependencyRegisters),
    left.threads.val[index]? = some l → right.threads.val[index]? = some r →
    dependencies.threads.val[index]? = some shadow → ThreadInputsRelated shadow l r
  allocations : ∀ (index : Nat) (l r : execution_core.AllocationBytes) (shadow : alloc.vec.Vec Metadata),
    left.allocations.val[index]? = some l → right.allocations.val[index]? = some r →
    dependencies.memory.val[index]? = some shadow → MemoryInputsRelated shadow l r

theorem input_running_equal (dependencies : value_flow_dependencies.State) (left right : CoreState)
    (related : StateInputsRelated dependencies left right) (actor : Usize) :
    execution_core.running left actor = execution_core.running right actor := by
  have threadLengths := related.threadsLength
  have metadataLengths := related.threadMetadataLength
  unfold execution_core.running
  have lengths : alloc.vec.Vec.len left.threads = alloc.vec.Vec.len right.threads := by scalar_tac
  rw [lengths]
  by_cases inside : actor.val < right.threads.val.length
  · have test : actor < alloc.vec.Vec.len right.threads := by scalar_tac
    have leftInside : actor.val < left.threads.val.length := by omega
    have shadowInside : actor.val < dependencies.threads.val.length := by omega
    have threadRelation := related.threads actor.val left.threads.val[actor.val]
      right.threads.val[actor.val] dependencies.threads.val[actor.val]
      (List.getElem?_eq_getElem leftInside) (List.getElem?_eq_getElem inside)
      (List.getElem?_eq_getElem shadowInside)
    simp only [test, ite_true, alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
      alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem leftInside,
      List.getElem?_eq_getElem inside, bind_ok, threadRelation.running]
    rfl
  · have test : ¬actor < alloc.vec.Vec.len right.threads := by scalar_tac
    simp only [test, ite_false]

end MemorySafetyExecution.Native
