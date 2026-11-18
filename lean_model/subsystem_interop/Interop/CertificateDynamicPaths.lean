/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.CertificateCfg

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.GuardCertificate
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex MemoryWrite.BarrierReach
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.ControlFlow

inductive DynamicReach {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) (start : Nat) (initial : PartialRegisters) : Nat → PartialRegisters → Prop
  | start : DynamicReach certificate bank start initial start initial
  | peer {pc : Nat} {before after : PartialRegisters} :
      DynamicReach certificate bank start initial pc before →
      (∀ register, RegisterOwner.shared register.file = false → after register = before register) →
      DynamicReach certificate bank start initial pc after
  | next {pc destination : Nat} {before after : PartialRegisters} :
      DynamicReach certificate bank start initial pc before →
      DynamicStep certificate bank pc before destination after →
      DynamicReach certificate bank start initial destination after

theorem DynamicReach.execution {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) (start pc : Nat) (initial registers : PartialRegisters)
    (path : DynamicReach certificate bank start initial pc registers)
    (entry : ExecutionDefined certificate bank start initial) : ExecutionDefined certificate bank pc registers := by
  induction path with
  | start => exact entry
  | peer _ agreement ih => exact .peer ih agreement
  | next _ step ih => exact step.execution certificate bank _ _ _ _ ih

theorem DynamicReach.raw_reachable {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (bank : U32 → U8 → Option U64)
    (fold : alloc.vec.Vec (Option Bool)) (checked : CertifiedFolds reader certificate bank fold)
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph) (mask : U32)
    (start pc : Nat) (initial registers : PartialRegisters)
    (entry : ExecutionDefined certificate bank start initial)
    (path : DynamicReach certificate bank start initial pc registers) :
    Reachable (rawCfgView graph certificate.code.val 0#u8 fold.val certificate.returns.val mask) start pc := by
  induction path with
  | start => exact .start
  | peer _ _ ih => exact ih
  | @next source destination before after path step ih =>
    exact .next ih trivial (checked.dynamic_edge reader certificate bank fold graph mask source destination
      before after (path.execution certificate bank start source initial before entry) step)

def SelectedMemoryWrite {C : Type} (certificate : guard_certificate.Certificate C)
    (pc : Nat) (register : Reg) : Prop :=
  ∃ instruction, certificate.code.val[pc]? = some instruction ∧
    ∃ keys, memoryWrites instruction.op = some keys ∧ key register ∈ keys

inductive DynamicMemoryWrittenBefore {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) (start : Nat) (initial : PartialRegisters) :
    Nat → PartialRegisters → Reg → Prop
  | write {pc destination : Nat} {before after : PartialRegisters} {register : Reg} :
      DynamicReach certificate bank start initial pc before →
      (step : DynamicStep certificate bank pc before destination after) →
      step.truth = true → SelectedMemoryWrite certificate pc register →
      DynamicMemoryWrittenBefore certificate bank start initial destination after register
  | peer {pc : Nat} {before after : PartialRegisters} {register : Reg} :
      DynamicMemoryWrittenBefore certificate bank start initial pc before register →
      (∀ value, RegisterOwner.shared value.file = false → after value = before value) →
      DynamicMemoryWrittenBefore certificate bank start initial pc after register
  | carry {pc destination : Nat} {before after : PartialRegisters} {register : Reg} :
      DynamicMemoryWrittenBefore certificate bank start initial pc before register →
      DynamicStep certificate bank pc before destination after →
      DynamicMemoryWrittenBefore certificate bank start initial destination after register

theorem DynamicMemoryWrittenBefore.path {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) (start pc : Nat) (initial registers : PartialRegisters) (register : Reg)
    (written : DynamicMemoryWrittenBefore certificate bank start initial pc registers register) :
    DynamicReach certificate bank start initial pc registers := by
  induction written with
  | write path step _ _ => exact .next path step
  | peer _ agreement ih => exact .peer ih agreement
  | carry _ step ih => exact .next ih step

theorem DynamicMemoryWrittenBefore.raw_written {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (bank : U32 → U8 → Option U64)
    (fold : alloc.vec.Vec (Option Bool)) (checked : CertifiedFolds reader certificate bank fold)
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph) (mask : U32)
    (start pc : Nat) (initial registers : PartialRegisters) (register : Reg)
    (entry : ExecutionDefined certificate bank start initial)
    (written : DynamicMemoryWrittenBefore certificate bank start initial pc registers register) :
    RawMemoryWrittenBefore graph certificate.code.val 0#u8 fold.val certificate.returns.val mask start pc register := by
  induction written with
  | @write source destination before after register path step _ writes =>
    exact .write (path.raw_reachable reader certificate bank fold checked graph mask start source initial before entry)
      (checked.dynamic_edge reader certificate bank fold graph mask source destination before after
        (path.execution certificate bank start source initial before entry) step) writes
  | peer _ _ ih => exact ih
  | @carry source destination before after register written step ih =>
    exact .carry ih (checked.dynamic_edge reader certificate bank fold graph mask source destination before after
      ((written.path certificate bank start source initial before register).execution certificate bank start source
        initial before entry) step)

def DynamicToBarrier {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) (bit : Bit) (pc : Nat) (registers : PartialRegisters) : Prop :=
  ∃ destination after, DynamicReach certificate bank pc registers destination after ∧
    BarrierLabel certificate.code.val destination bit

theorem reachable_then_gate (graph : Graph) (label : Nat → Prop) (start pc : Nat)
    (path : Reachable graph start pc) : CanReach graph.edge label pc → CanReach graph.edge label start := by
  induction path with
  | start => exact id
  | next _ _ edge ih => intro gate; exact ih (.next edge gate)

theorem dynamic_to_barrier_is_static {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (bank : U32 → U8 → Option U64)
    (fold : alloc.vec.Vec (Option Bool)) (checked : CertifiedFolds reader certificate bank fold)
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph) (mask : U32) (bit : Bit)
    (pc : Nat) (registers : PartialRegisters) (entry : ExecutionDefined certificate bank pc registers)
    (gate : DynamicToBarrier certificate bank bit pc registers) :
    ToBarrier graph certificate.code.val 0#u8 fold.val certificate.returns.val mask bit pc := by
  obtain ⟨destination, after, path, label⟩ := gate
  have reachable := path.raw_reachable reader certificate bank fold checked graph mask pc destination registers after entry
  have gate := reachable_then_gate
    (rawCfgView graph certificate.code.val 0#u8 fold.val certificate.returns.val mask)
    (fun address => BarrierLabel certificate.code.val address bit) pc destination reachable (.here label)
  exact gate

end Instruction.ControlFlow
