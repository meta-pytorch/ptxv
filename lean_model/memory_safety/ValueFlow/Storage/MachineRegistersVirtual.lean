/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.Storage.MachineRegisters
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace ValueFlow.Storage.MachineRegistersVirtual

abbrev Entry := U16 × U64

def FirstAt (entries : List Entry) (key : U16) (position : Usize) : Prop :=
  position.val ≤ entries.length ∧
  (∀ i, i < position.val → entries[i]!.1.val < key.val) ∧
  (position.val < entries.length → key.val ≤ entries[position.val]!.1.val)

theorem source_search (registers : machine_registers.LaneRegisters) (key : U16) :
    machine_registers.set_virtual_undefined_loop0 registers key 0#usize ⦃ output =>
      match output with
      | (regs, preds, pdef, written, entries, position) =>
        regs = registers.regs ∧ preds = registers.preds ∧ pdef = registers.pdef ∧
        written = registers.written ∧ entries = registers.vund ∧
        FirstAt entries.val key position ⦄ := by
  unfold machine_registers.set_virtual_undefined_loop0
  apply loop.spec_decr_nat
    (fun position : Usize => registers.vund.val.length - position.val)
    (fun position : Usize => position.val ≤ registers.vund.val.length ∧
      ∀ i, i < position.val → registers.vund.val[i]!.1.val < key.val)
  · intro position invariant
    rcases invariant with ⟨bound, earlier⟩
    unfold machine_registers.set_virtual_undefined_loop0.body
    by_cases inside : position.val < registers.vund.val.length
    · have test : position < alloc.vec.Vec.len registers.vund := by scalar_tac
      simp only [test, ite_true]
      step as ⟨entryKey, entryMask, entryEq⟩
      have sameKey : entryKey.val = registers.vund.val[position.val]!.1.val := by
        have pairEq := congrArg Prod.fst entryEq
        simpa [List.Inhabited_getElem_eq_getElem! registers.vund.val position.val inside] using
          congrArg UScalar.val pairEq
      split
      · rename_i less
        have lessKey : registers.vund.val[position.val]!.1.val < key.val := by scalar_tac
        step as ⟨next, nextValue⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        intro i hi
        by_cases before : i < position.val
        · exact earlier i before
        · have same : i = position.val := by scalar_tac
          simpa [same] using lessKey
      · rename_i notLess
        simp only [WP.spec_ok, FirstAt, true_and]
        exact ⟨bound, earlier, by intro _; scalar_tac⟩
    · have test : ¬position < alloc.vec.Vec.len registers.vund := by scalar_tac
      simp only [test, ite_false, WP.spec_ok, FirstAt, true_and]
      exact ⟨bound, earlier, by omega⟩
  · refine ⟨by scalar_tac, ?_⟩
    intro i hi
    scalar_tac

def eraseFrom : List Entry → Nat → Nat → List Entry
  | [], _, _ => []
  | entry :: rest, skipIndex, index =>
    if index = skipIndex then eraseFrom rest skipIndex (index + 1)
    else entry :: eraseFrom rest skipIndex (index + 1)

def CopyInvariant (entries output retained : List Entry) (skipIndex index : Nat) : Prop :=
  index ≤ entries.length ∧ retained.length ≤ index ∧
    retained ++ eraseFrom (entries.drop index) skipIndex index = output

theorem source_erase_loop (entries : alloc.vec.Vec Entry) (skipIndex : Usize)
    (retained : alloc.vec.Vec Entry) (index : Usize) (output : List Entry)
    (initial : CopyInvariant entries.val output retained.val skipIndex.val index.val) :
    machine_registers.set_virtual_undefined_loop1 entries skipIndex retained index ⦃ result =>
      result.val = output ⦄ := by
  unfold machine_registers.set_virtual_undefined_loop1
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec Entry × Usize => entries.val.length - current.2.val)
    (fun current : alloc.vec.Vec Entry × Usize =>
      CopyInvariant entries.val output current.1.val skipIndex.val current.2.val)
  · rintro ⟨retained, index⟩ invariant
    change CopyInvariant entries.val output retained.val skipIndex.val index.val at invariant
    obtain ⟨bounded, lengthBound, remaining⟩ := invariant
    unfold machine_registers.set_virtual_undefined_loop1.body
    by_cases inside : index.val < entries.val.length
    · have test : index < alloc.vec.Vec.len entries := by scalar_tac
      simp only [test, ite_true]
      have drop : entries.val.drop index.val =
          entries.val[index.val] :: entries.val.drop (index.val + 1) :=
        List.drop_eq_getElem_cons inside
      by_cases distinct : index ≠ skipIndex
      · have tested : (index != skipIndex) = true := by simp [distinct]
        simp only [tested, ite_true]
        step as ⟨entry, entryEq⟩
        have dropRead : entries.val.drop index.val =
            entry :: entries.val.drop (index.val + 1) := by
          rw [List.drop_eq_getElem_cons inside, ←entryEq]
        step with alloc.vec.Vec.push_spec retained entry
          (by have := alloc.vec.Vec.len_ineq entries; omega) as ⟨updated, pushed⟩
        step as ⟨next, nextValue⟩
        refine ⟨⟨by scalar_tac, ?_, ?_⟩, by scalar_tac⟩
        · simp only [pushed, List.length_append, List.length_singleton]
          scalar_tac
        · have neValues : index.val ≠ skipIndex.val := by scalar_tac
          rw [dropRead, eraseFrom, ite_eq_right neValues] at remaining
          simpa only [pushed, nextValue, List.append_assoc, List.singleton_append] using remaining
      · have same : index = skipIndex := not_not.mp distinct
        simp only [same, bne_self_eq_false, Bool.false_eq_true, ite_false]
        step as ⟨next, nextValue⟩
        refine ⟨⟨by scalar_tac, by scalar_tac, ?_⟩, by scalar_tac⟩
        rw [drop, eraseFrom, ite_eq_left (by scalar_tac : index.val = skipIndex.val)] at remaining
        simpa only [nextValue, same] using remaining
    · have test : ¬index < alloc.vec.Vec.len entries := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      have empty : entries.val.drop index.val = [] := List.drop_eq_nil_iff.mpr (by omega)
      simpa only [empty, eraseFrom, List.append_nil] using remaining
  · exact initial

theorem erase_loops_equal (entries : alloc.vec.Vec Entry) (skipIndex : Usize)
    (retained : alloc.vec.Vec Entry) (index : Usize) :
    machine_registers.set_virtual_undefined_loop2 entries skipIndex retained index =
      machine_registers.set_virtual_undefined_loop1 entries skipIndex retained index := by
  rfl

theorem source_erase (entries : alloc.vec.Vec Entry) (skipIndex : Usize) :
    machine_registers.set_virtual_undefined_loop1 entries skipIndex (alloc.vec.Vec.new _) 0#usize
      ⦃ result => result.val = eraseFrom entries.val skipIndex.val 0 ⦄ := by
  apply source_erase_loop
  simp [CopyInvariant, alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from]

def hasKey (entries : List Entry) (key : U16) (position : Nat) : Bool :=
  match entries[position]? with
  | none => false
  | some entry => entry.1 == key

def updateAt (entries : List Entry) (key : U16) (mask : U64) (position : Nat) : List Entry :=
  if mask != 0#u64 && mask != core.num.U64.MAX then
    if hasKey entries key position then entries.set position (key, mask)
    else entries.insertIdx position (key, mask)
  else if hasKey entries key position then eraseFrom entries position 0
  else entries

def UndefinedUpdate (before after : machine_registers.LaneRegisters) (key : U16) (mask : U64) : Prop :=
  after.regs = before.regs ∧ after.preds = before.preds ∧ after.pdef = before.pdef ∧
  after.written = before.written ∧
  ∃ position, FirstAt before.vund.val key position ∧
    after.vund.val = updateAt before.vund.val key mask position.val

theorem source_set_undefined (registers : machine_registers.LaneRegisters)
    (key : U16) (mask : U64) (room : registers.vund.val.length < Usize.max) :
    machine_registers.set_virtual_undefined registers key mask ⦃ updated =>
      UndefinedUpdate registers updated key mask ⦄ := by
  obtain ⟨output, searchRun, searchPost⟩ := MemoryWrite.spec_returns _ _ (source_search registers key)
  rcases output with ⟨regs, preds, pdef, written, entries, position⟩
  rcases searchPost with ⟨rfl, rfl, rfl, rfl, rfl, located⟩
  unfold machine_registers.set_virtual_undefined
  rw [searchRun]
  simp only [bind_ok]
  change WP.spec (Std.bind (if position < alloc.vec.Vec.len registers.vund then do
      let entry ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (U16 × U64)) registers.vund position
      ok (decide (entry.1 = key))
    else ok false) _) _
  by_cases inside : position.val < registers.vund.val.length
  · have test : position < alloc.vec.Vec.len registers.vund := by scalar_tac
    simp only [test, ite_true]
    step as ⟨entry, entryEq⟩
    rcases entry with ⟨entryKey, entryMask⟩
    by_cases zero : mask = 0#u64
    · simp only [zero, bne_self_eq_false, Bool.false_eq_true, ite_false]
      by_cases equal : entryKey = key
      · simp only [equal, decide_true, ite_true]
        rw [erase_loops_equal]
        step with source_erase registers.vund position as ⟨updated, erased⟩
        refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
        simp [updateAt, hasKey, List.getElem?_eq_getElem inside, ←entryEq, erased, equal]
      · simp only [equal, decide_false, Bool.false_eq_true, ite_false, WP.spec_ok]
        refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
        simp [updateAt, hasKey, List.getElem?_eq_getElem inside, ←entryEq, equal]
    · have nz : (mask != 0#u64) = true := by simp [zero]
      simp only [nz, ite_true]
      by_cases full : mask = core.num.U64.MAX
      · simp only [full, bne_self_eq_false, Bool.false_eq_true, ite_false]
        by_cases equal : entryKey = key
        · simp only [equal, decide_true, ite_true]
          step with source_erase registers.vund position as ⟨updated, erased⟩
          refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
          simp [updateAt, hasKey, List.getElem?_eq_getElem inside, ←entryEq, erased, equal]
        · simp only [equal, decide_false, Bool.false_eq_true, ite_false, WP.spec_ok]
          refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
          simp [updateAt, hasKey, List.getElem?_eq_getElem inside, ←entryEq, equal]
      · have nf : (mask != core.num.U64.MAX) = true := by simp [full]
        simp only [nf, ite_true]
        by_cases equal : entryKey = key
        · simp only [equal, decide_true, ite_true]
          step as ⟨oldKey, oldMask, back, oldEq, backEq⟩
          refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
          rw [backEq]
          have oldKeyEq : oldKey = key := (congrArg Prod.fst (oldEq.trans entryEq.symm)).trans equal
          simp [oldKeyEq, updateAt, hasKey, List.getElem?_eq_getElem inside, zero, full,
            ←entryEq, equal, alloc.vec.Vec.set, alloc.vec.Vec.from_val]
        · simp only [equal, decide_false, Bool.false_eq_true, ite_false]
          have notEnd : ¬position = alloc.vec.Vec.len registers.vund := by scalar_tac
          simp only [notEnd, ite_false]
          step with alloc.vec.Vec.insert_spec registers.vund position (key, mask) inside room
            as ⟨updated, inserted⟩
          refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
          simp [updateAt, hasKey, List.getElem?_eq_getElem inside, zero, full,
            ←entryEq, equal, inserted]
  · have test : ¬position < alloc.vec.Vec.len registers.vund := by scalar_tac
    have atEnd : position.val = registers.vund.val.length := by exact Nat.le_antisymm located.1 (by omega)
    have endEq : position = alloc.vec.Vec.len registers.vund := by scalar_tac
    simp only [test, ite_false, bind_ok, Bool.false_eq_true]
    by_cases zero : mask = 0#u64
    · simp only [zero, bne_self_eq_false, Bool.false_eq_true, ite_false, WP.spec_ok]
      refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
      simp [updateAt, hasKey, atEnd]
    · have nz : (mask != 0#u64) = true := by simp [zero]
      simp only [nz, ite_true]
      by_cases full : mask = core.num.U64.MAX
      · simp only [full, bne_self_eq_false, Bool.false_eq_true, ite_false, WP.spec_ok]
        refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
        simp [updateAt, hasKey, atEnd]
      · have nf : (mask != core.num.U64.MAX) = true := by simp [full]
        simp only [nf, ite_true, endEq, ite_true]
        step with alloc.vec.Vec.push_spec registers.vund (key, mask) room as ⟨updated, pushed⟩
        refine ⟨rfl, rfl, rfl, rfl, position, located, ?_⟩
        simp [updateAt, hasKey, atEnd, zero, full, pushed]

theorem source_write_virtual (registers : machine_registers.LaneRegisters)
    (index : U16) (value undefined : U64)
    (wordBound : index.val < registers.regs.val.length)
    (writtenBound : index.val / 64 < registers.written.val.length)
    (room : registers.vund.val.length < Usize.max) :
    machine_registers.write_virtual registers index value undefined ⦃ updated =>
      updated.regs.val = registers.regs.val.set index.val value ∧
      updated.written.val = registers.written.val.set (index.val / 64)
        (MachineRegisters.bitWord registers.written.val[index.val / 64]! index
          (undefined != core.num.U64.MAX)) ∧
      updated.preds = registers.preds ∧ updated.pdef = registers.pdef ∧
      ∃ position, FirstAt registers.vund.val index position ∧
        updated.vund.val = updateAt registers.vund.val index undefined position.val ⦄ := by
  unfold machine_registers.write_virtual
  simp only [lift, bind_ok]
  have indexValue : (core.convert.num.FromUsizeU16.from index).val = index.val := by scalar_tac
  rw [MachineRegisters.source_set_word registers.regs
    (core.convert.num.FromUsizeU16.from index) value (by simpa [indexValue] using wordBound)]
  simp only [bind_ok]
  step with MachineRegisters.source_set_bit registers.written index
    (undefined != core.num.U64.MAX) writtenBound as ⟨written, writtenEq⟩
  step with (source_set_undefined
    { registers with
      regs := registers.regs.set (core.convert.num.FromUsizeU16.from index) value
      written := written } index undefined room) as ⟨updated, changed⟩
  rcases changed with ⟨words, predicates, definitions, markers, sparse⟩
  simp only [words, markers, predicates, definitions]
  exact ⟨by simp [alloc.vec.Vec.set, alloc.vec.Vec.from_val, indexValue], writtenEq,
    True.intro, True.intro, sparse⟩

theorem source_write_virtual_register_frame (registers : machine_registers.LaneRegisters)
    (index : U16) (value undefined : U64)
    (wordBound : index.val < registers.regs.val.length)
    (writtenBound : index.val / 64 < registers.written.val.length)
    (room : registers.vund.val.length < Usize.max) :
    machine_registers.write_virtual registers index value undefined ⦃ updated =>
      updated.regs.val[index.val]? = some value ∧
      ∀ other, other ≠ index.val → updated.regs.val[other]? = registers.regs.val[other]? ⦄ := by
  apply WP.spec_mono (source_write_virtual registers index value undefined wordBound writtenBound room)
  intro updated changed
  rw [changed.1]
  exact ⟨List.getElem?_set_self wordBound,
    fun _ distinct => List.getElem?_set_ne distinct.symm⟩

end ValueFlow.Storage.MachineRegistersVirtual
