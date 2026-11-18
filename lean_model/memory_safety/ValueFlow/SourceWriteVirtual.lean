/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceReads
import ValueFlow.SourceUndefinedMap
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

open Storage.MachineRegisters Storage.MachineRegistersDefinedness
open Storage.MachineRegistersVirtualMap

def ValueWriteEffect (before after : V.Registers) (index : U16) (bits : U64)
    (pointer : Option V.Pointer) : Prop :=
  wordModel after index.val = some {
    bits := bits.val
    undefined := 0
    pointer := pointer.map pointerModel } ∧
  (∀ query, query ≠ index.val → wordModel after query = wordModel before query) ∧
  (∀ query, predicateModel after query = predicateModel before query) ∧
  Canonical after.raw.vund.val

 theorem source_raw_write_virtual (registers : V.Registers) (index : U16) (bits : U64)
    (pointer : Option V.Pointer) (raw : machine_registers.LaneRegisters)
    (canonical : Canonical registers.raw.vund.val)
    (room : registers.raw.vund.val.length < Usize.max)
    (bounds : index.val < registers.raw.regs.val.length ∧ index.val < registers.pointers.val.length ∧
      index.val / 64 < registers.raw.written.val.length)
    (run : machine_registers.write_virtual registers.raw index bits 0#u64 = .ok raw) :
    ValueWriteEffect registers
      { raw, pointers := registers.pointers.set (core.convert.num.FromUsizeU16.from index) pointer }
      index bits pointer := by
  have spec := source_write_virtual_map registers.raw index bits 0#u64 canonical bounds.1 bounds.2.2 room
  rw [run] at spec
  have facts : raw.regs.val = registers.raw.regs.val.set index.val bits ∧
      raw.written.val = registers.raw.written.val.set (index.val / 64)
        (bitWord registers.raw.written.val[index.val / 64]! index true) ∧
      raw.preds = registers.raw.preds ∧ raw.pdef = registers.raw.pdef ∧
      Canonical raw.vund.val ∧ (raw.vund.val.map Prod.fst).Nodup ∧
      lookup raw.vund.val index = none ∧
      ∀ other, other ≠ index → lookup raw.vund.val other = lookup registers.raw.vund.val other := by
    simpa only [WP.spec_ok, show (0#u64 != core.num.U64.MAX) = true from rfl,
      PartialMask, ne_eq, not_true_eq_false, false_and, ite_false] using spec
  rcases facts with ⟨words, markers, predicates, definitions, proper, unique, selected, others⟩
  have wordLength : raw.regs.val.length = registers.raw.regs.val.length := by rw [words]; simp
  have writtenLength : raw.written.val.length = registers.raw.written.val.length := by rw [markers]; simp
  have indexVal : (core.convert.num.FromUsizeU16.from index).val = index.val := by scalar_tac
  have writtenBit : bitAt raw.written.val index.val = true := by
    rw [markers]
    simp [bitAt, bounds.2.2, bit_word_selected]
  have undefinedZero : undefinedBits raw.vund.val index.val = 0 := by
    rw [undefined_lookup raw.vund.val index proper.1, selected]
    rfl
  constructor
  · rw [wordModel_bits]
    simp only [wordLength, writtenLength, alloc.vec.Vec.set, alloc.vec.Vec.from_val,
      List.length_set, indexVal, bounds, and_self, ite_true, writtenBit]
    simp [words, bounds.1, bounds.2.1, undefinedZero]
  constructor
  · intro query distinct
    have writtenFrame : bitAt raw.written.val query = bitAt registers.raw.written.val query := by
      rw [markers]
      exact bit_update_frame registers.raw.written.val index true bounds.2.2 query distinct
    have undefinedFrame := undefined_lookup_ext raw.vund.val registers.raw.vund.val
      proper.1 canonical.1 index others query distinct
    rw [wordModel_bits, wordModel_bits]
    simp only [wordLength, writtenLength, alloc.vec.Vec.set, alloc.vec.Vec.from_val,
      List.length_set, indexVal, writtenFrame, undefinedFrame]
    simp [words, Ne.symm distinct]
  constructor
  · intro query
    simp only [predicateModel, predicates, definitions]
  · exact proper

def valuePointer : V.Value → Option V.Pointer
  | .Scalar _ => none
  | .Pointer pointer => some pointer

theorem source_write_value_success (before after : V.Registers) (index : U16)
    (value : V.Value) (layouts : Slice value_flow.AllocationLayout)
    (canonical : Canonical before.raw.vund.val)
    (run : value_flow.write_value before index value layouts = .ok (.Ok (), after)) :
    ∃ bits, value_flow.render_value value layouts = .ok (.Ok bits) ∧
      ValueWriteEffect before after index bits (valuePointer value) := by
  unfold value_flow.write_value at run
  obtain ⟨writable, shapeRun, run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
  cases writable with
  | false => simp at run
  | true =>
    have spec := source_writable_virtual before index
    have shape : true = (index.val < before.raw.regs.val.length && index.val < before.pointers.val.length &&
        index.val / 64 < before.raw.written.val.length) := by
      simpa only [shapeRun, WP.spec_ok] using spec
    have bounds : index.val < before.raw.regs.val.length ∧ index.val < before.pointers.val.length ∧
        index.val / 64 < before.raw.written.val.length := by
      simpa only [Bool.and_eq_true, decide_eq_true_eq, and_assoc] using shape.symm
    obtain ⟨rendered, renderedRun, run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
    cases rendered with
    | Err error => simp at run
    | Ok bits =>
      obtain ⟨raw, rawRun, run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
      have effect := source_raw_write_virtual before index bits (valuePointer value) raw canonical
        (canonical_room before.raw.vund.val canonical) bounds rawRun
      have cast : (core.convert.num.FromUsizeU16.from index).val = index.val := by scalar_tac
      cases value with
      | Scalar scalar =>
        simp only [lift, bind_ok, alloc.vec.Vec.index_mut_slice_index] at run
        change (do
          let (_, back) ← alloc.vec.Vec.index_mut_usize before.pointers (core.convert.num.FromUsizeU16.from index)
          ok ((core.result.Result.Ok () : core.result.Result Unit value_flow.EvalError), ({ raw, pointers := back none } : V.Registers))) = .ok (.Ok (), after) at run
        simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
          cast, List.getElem?_eq_getElem bounds.2.1] at run
        have same : after = { raw, pointers := before.pointers.set (core.convert.num.FromUsizeU16.from index) none } := by
          simpa only [ok.injEq, Prod.mk.injEq, true_and] using run.symm
        exact ⟨bits, renderedRun, same ▸ effect⟩
      | Pointer pointer =>
        simp only [lift, bind_ok, alloc.vec.Vec.index_mut_slice_index] at run
        change (do
          let (_, back) ← alloc.vec.Vec.index_mut_usize before.pointers (core.convert.num.FromUsizeU16.from index)
          ok ((core.result.Result.Ok () : core.result.Result Unit value_flow.EvalError), ({ raw, pointers := back (some pointer) } : V.Registers))) = .ok (.Ok (), after) at run
        simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
          cast, List.getElem?_eq_getElem bounds.2.1] at run
        have same : after = { raw, pointers := before.pointers.set (core.convert.num.FromUsizeU16.from index) (some pointer) } := by
          simpa only [ok.injEq, Prod.mk.injEq, true_and] using run.symm
        exact ⟨bits, renderedRun, same ▸ effect⟩

end ValueFlow.Source
