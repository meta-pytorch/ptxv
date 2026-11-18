/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputState
import ValueFlow.SourceWriterExecution

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source
open ValueFlow.Storage.MachineRegisters ValueFlow.Storage.MachineRegistersVirtualMap

private theorem pointer_model_injective (left right : V.Pointer) (same : pointerModel left = pointerModel right) :
    left = right := by
  have allocation : left.allocation = right.allocation :=
    UScalar.eq_of_val_eq (congrArg ValueFlow.Pointer.allocation same)
  have offset : left.offset = right.offset :=
    UScalar.eq_of_val_eq (congrArg ValueFlow.Pointer.offset same)
  cases left
  cases right
  simp_all

private theorem related_pointer_tags (left right : V.Value) (tainted : Bool)
    (related : ValuesRelated tainted (valueModel left) (valueModel right)) :
    valuePointer left = valuePointer right := by
  cases tainted with
  | false =>
    simp only [ValuesRelated, Bool.false_eq_true, ite_false] at related
    cases left with
    | Scalar value => cases right <;> simp_all [valueModel, valuePointer]
    | Pointer pointer =>
      cases right with
      | Scalar value => cases related
      | Pointer other =>
        have same : pointerModel pointer = pointerModel other := by
          simpa only [valueModel, Value.pointer.injEq] using related
        simp only [valuePointer, pointer_model_injective pointer other same]
  | true =>
    obtain ⟨a, b, leftEq, rightEq⟩ := related
    cases left <;> cases right <;> simp_all [valueModel, valuePointer]

private theorem raw_virtual_shape (before after : machine_registers.LaneRegisters) (index : U16) (bits : U64)
    (canonical : Canonical before.vund.val)
    (bounds : index.val < before.regs.val.length ∧ index.val / 64 < before.written.val.length)
    (computed : machine_registers.write_virtual before index bits 0#u64 = .ok after) :
    after.regs.val = before.regs.val.set index.val bits ∧
    after.written.val = before.written.val.set (index.val / 64)
      (bitWord before.written.val[index.val / 64]! index true) ∧
    after.preds = before.preds ∧ after.pdef = before.pdef ∧
    after.vund.val = update before.vund.val index 0#u64 := by
  have spec := ValueFlow.Storage.MachineRegistersVirtual.source_write_virtual before index bits 0#u64
    bounds.1 bounds.2 (canonical_room before.vund.val canonical)
  rw [computed] at spec
  have meaning : after.regs.val = before.regs.val.set index.val bits ∧
      after.written.val = before.written.val.set (index.val / 64)
        (bitWord before.written.val[index.val / 64]! index true) ∧
      after.preds = before.preds ∧ after.pdef = before.pdef ∧
      ∃ position, ValueFlow.Storage.MachineRegistersVirtual.FirstAt before.vund.val index position ∧
        after.vund.val = ValueFlow.Storage.MachineRegistersVirtual.updateAt before.vund.val index 0#u64 position.val := by
    simpa only [WP.spec_ok, show (0#u64 != core.num.U64.MAX) = true from rfl] using spec
  obtain ⟨words, written, preds, pdef, position, located, undefined⟩ := meaning
  refine ⟨words, written, preds, pdef, ?_⟩
  rw [undefined]
  exact update_at_eq before.vund.val index 0#u64 position.val located

theorem input_virtual_write_shape (left right leftAfter rightAfter : V.Registers) (index : U16)
    (leftValue rightValue : V.Value) (layouts : Slice value_flow.AllocationLayout) (tainted : Bool)
    (shape : RegisterInputShape left right)
    (leftRun : value_flow.write_value left index leftValue layouts = .ok (.Ok (), leftAfter))
    (rightRun : value_flow.write_value right index rightValue layouts = .ok (.Ok (), rightAfter))
    (values : ValuesRelated tainted (valueModel leftValue) (valueModel rightValue)) :
    RegisterInputShape leftAfter rightAfter := by
  obtain ⟨l⟩ := source_write_value_components left leftAfter index leftValue layouts leftRun
  obtain ⟨r⟩ := source_write_value_components right rightAfter index rightValue layouts rightRun
  have rightCanonical : Canonical right.raw.vund.val := by simpa only [shape.undefined] using shape.canonical
  obtain ⟨lw, lm, lp, ld, lu⟩ := raw_virtual_shape left.raw l.raw index l.bits shape.canonical
    ⟨l.bounds.1, l.bounds.2.2⟩ l.written
  obtain ⟨rw, rm, rp, rd, ru⟩ := raw_virtual_shape right.raw r.raw index r.bits rightCanonical
    ⟨r.bounds.1, r.bounds.2.2⟩ r.written
  rw [l.afterEq, r.afterEq]
  constructor
  · simp only [lw, rw, List.length_set]
    exact shape.virtualLength
  · simpa only [lp, rp] using shape.predicateLength
  · rw [shape.pointers, related_pointer_tags leftValue rightValue tainted values]
  · apply alloc.vec.Vec.ext
    rw [lm, rm, shape.written]
  · rw [ld, rd, shape.definitions]
  · apply alloc.vec.Vec.ext
    rw [lu, ru, shape.undefined]
  · rw [lu]
    exact update_canonical left.raw.vund.val index 0#u64 shape.canonical

theorem input_predicate_write_shape (left right leftAfter rightAfter : V.Registers) (index : U16)
    (leftValue rightValue : Bool) (shape : RegisterInputShape left right)
    (leftRun : value_flow.write_predicate left index leftValue = .ok (.Ok (), leftAfter))
    (rightRun : value_flow.write_predicate right index rightValue = .ok (.Ok (), rightAfter)) :
    RegisterInputShape leftAfter rightAfter := by
  obtain ⟨lb, l, leftWritten, leftEq⟩ := source_write_predicate_components left leftAfter index leftValue leftRun
  obtain ⟨rb, r, rightWritten, rightEq⟩ := source_write_predicate_components right rightAfter index rightValue rightRun
  have lspec := ValueFlow.Storage.MachineRegisters.source_write_predicate left.raw index leftValue true lb.1 lb.2
  have rspec := ValueFlow.Storage.MachineRegisters.source_write_predicate right.raw index rightValue true rb.1 rb.2
  rw [leftWritten] at lspec
  rw [rightWritten] at rspec
  have lf : l.regs = left.raw.regs ∧ l.written = left.raw.written ∧ l.vund = left.raw.vund ∧
      l.preds.val = left.raw.preds.val.set (index.val / 64) (bitWord left.raw.preds.val[index.val / 64]! index leftValue) ∧
      l.pdef.val = left.raw.pdef.val.set (index.val / 64) (bitWord left.raw.pdef.val[index.val / 64]! index true) := by
    simpa only [WP.spec_ok] using lspec
  have rf : r.regs = right.raw.regs ∧ r.written = right.raw.written ∧ r.vund = right.raw.vund ∧
      r.preds.val = right.raw.preds.val.set (index.val / 64) (bitWord right.raw.preds.val[index.val / 64]! index rightValue) ∧
      r.pdef.val = right.raw.pdef.val.set (index.val / 64) (bitWord right.raw.pdef.val[index.val / 64]! index true) := by
    simpa only [WP.spec_ok] using rspec
  rw [leftEq, rightEq]
  constructor
  · simpa only [lf.1, rf.1] using shape.virtualLength
  · simp only [lf.2.2.2.1, rf.2.2.2.1, List.length_set]
    exact shape.predicateLength
  · exact shape.pointers
  · rw [lf.2.1, rf.2.1, shape.written]
  · apply alloc.vec.Vec.ext
    rw [lf.2.2.2.2, rf.2.2.2.2, shape.definitions]
  · rw [lf.2.2.1, rf.2.2.1, shape.undefined]
  · simpa only [lf.2.2.1] using shape.canonical

end MemorySafetyExecution.Native
