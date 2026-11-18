/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceWriteVirtual
import ValueFlow.SourceWritePredicate

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

open Storage.MachineRegistersVirtualMap

structure VirtualWriteComponents (before after : V.Registers) (index : U16)
    (value : V.Value) (layouts : Slice value_flow.AllocationLayout) where
  bits : U64
  raw : machine_registers.LaneRegisters
  bounds : index.val < before.raw.regs.val.length ∧ index.val < before.pointers.val.length ∧
    index.val / 64 < before.raw.written.val.length
  rendered : value_flow.render_value value layouts = .ok (.Ok bits)
  written : machine_registers.write_virtual before.raw index bits 0#u64 = .ok raw
  afterEq : after = { raw, pointers := before.pointers.set (core.convert.num.FromUsizeU16.from index) (valuePointer value) }

theorem source_write_value_components (before after : V.Registers) (index : U16)
    (value : V.Value) (layouts : Slice value_flow.AllocationLayout)
    (run : value_flow.write_value before index value layouts = .ok (.Ok (), after)) :
    Nonempty (VirtualWriteComponents before after index value layouts) := by
  unfold value_flow.write_value at run
  obtain ⟨writable, writableRun, run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
  cases writable with
  | false => simp at run
  | true =>
    have shapeSpec := source_writable_virtual before index
    have shape : true = (index.val < before.raw.regs.val.length && index.val < before.pointers.val.length &&
        index.val / 64 < before.raw.written.val.length) := by
      simpa only [writableRun, WP.spec_ok] using shapeSpec
    have bounds : index.val < before.raw.regs.val.length ∧ index.val < before.pointers.val.length ∧
        index.val / 64 < before.raw.written.val.length := by
      simpa only [Bool.and_eq_true, decide_eq_true_eq, and_assoc] using shape.symm
    obtain ⟨rendered, renderedRun, run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
    cases rendered with
    | Err error => simp at run
    | Ok bits =>
      obtain ⟨raw, written, run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
      have cast : (core.convert.num.FromUsizeU16.from index).val = index.val := by scalar_tac
      cases value with
      | Scalar scalar =>
        simp [lift, alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
          alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq, cast,
          List.getElem?_eq_getElem bounds.2.1] at run
        exact ⟨⟨bits, raw, bounds, renderedRun, written, run.symm⟩⟩
      | Pointer pointer =>
        simp [lift, alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
          alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq, cast,
          List.getElem?_eq_getElem bounds.2.1] at run
        exact ⟨⟨bits, raw, bounds, renderedRun, written, run.symm⟩⟩

theorem source_write_value_total (registers : V.Registers) (index : U16)
    (value : V.Value) (layouts : Slice value_flow.AllocationLayout) (bits : U64)
    (canonical : Canonical registers.raw.vund.val)
    (bounds : index.val < registers.raw.regs.val.length ∧ index.val < registers.pointers.val.length ∧
      index.val / 64 < registers.raw.written.val.length)
    (rendered : value_flow.render_value value layouts = .ok (.Ok bits)) :
    ∃ after, value_flow.write_value registers index value layouts = .ok (.Ok (), after) := by
  obtain ⟨writable, writableRun, shape⟩ := MemoryWrite.spec_returns _ _ (source_writable_virtual registers index)
  have writableTrue : writable = true := by simp [bounds] at shape; exact shape
  rw [writableTrue] at writableRun
  have rawSpec := Storage.MachineRegistersVirtual.source_write_virtual registers.raw index bits 0#u64
    bounds.1 bounds.2.2 (canonical_room registers.raw.vund.val canonical)
  obtain ⟨raw, written, _⟩ := MemoryWrite.spec_returns _ _ rawSpec
  refine ⟨{ raw, pointers := registers.pointers.set (core.convert.num.FromUsizeU16.from index) (valuePointer value) }, ?_⟩
  have cast : (core.convert.num.FromUsizeU16.from index).val = index.val := by scalar_tac
  cases value <;>
    simp [value_flow.write_value, writableRun, rendered, written, lift,
      alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
      cast, List.getElem?_eq_getElem bounds.2.1, valuePointer]

theorem source_write_predicate_components (before after : V.Registers) (index : U16) (value : Bool)
    (run : value_flow.write_predicate before index value = .ok (.Ok (), after)) :
    (index.val / 64 < before.raw.preds.val.length ∧ index.val / 64 < before.raw.pdef.val.length) ∧
    ∃ raw, machine_registers.write_predicate before.raw index value true = .ok raw ∧
      after = { before with raw } := by
  unfold value_flow.write_predicate at run
  obtain ⟨writable, writableRun, run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
  cases writable with
  | false => simp at run
  | true =>
    have shapeSpec := source_writable_predicate before index
    have bounds : index.val / 64 < before.raw.preds.val.length ∧ index.val / 64 < before.raw.pdef.val.length := by
      have shape : true = (index.val / 64 < before.raw.preds.val.length && index.val / 64 < before.raw.pdef.val.length) := by
        simpa only [writableRun, WP.spec_ok] using shapeSpec
      simpa only [Bool.and_eq_true, decide_eq_true_eq] using shape.symm
    obtain ⟨raw, written, run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
    exact ⟨bounds, raw, written, by simpa only [ok.injEq, Prod.mk.injEq, true_and] using run.symm⟩

theorem source_write_predicate_total (registers : V.Registers) (index : U16) (value : Bool)
    (bounds : index.val / 64 < registers.raw.preds.val.length ∧ index.val / 64 < registers.raw.pdef.val.length) :
    ∃ after, value_flow.write_predicate registers index value = .ok (.Ok (), after) := by
  obtain ⟨writable, writableRun, shape⟩ := MemoryWrite.spec_returns _ _ (source_writable_predicate registers index)
  have writableTrue : writable = true := by simp [bounds] at shape; exact shape
  rw [writableTrue] at writableRun
  obtain ⟨raw, written, _⟩ := MemoryWrite.spec_returns _ _
    (Storage.MachineRegisters.source_write_predicate registers.raw index value true bounds.1 bounds.2)
  exact ⟨{ registers with raw }, by simp only [value_flow.write_predicate, writableRun, bind_ok, ite_true, written]⟩

end ValueFlow.Source
