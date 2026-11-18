/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRegistersVirtualMap

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineRegistersDefinedness

theorem bit_word_bits (old : U64) (index : U16) (enabled : Bool) :
    (MachineRegisters.bitWord old index enabled).bv =
      (old.bv &&& ~~~((1 : BitVec 64) <<< (index.val % 64))) |||
        if enabled then (1 : BitVec 64) <<< (index.val % 64) else 0 := rfl

theorem bit_word_selected (old : U64) (index : U16) (enabled : Bool) :
    (MachineRegisters.bitWord old index enabled).bv.getLsbD (index.val % 64) = enabled := by
  rw [bit_word_bits]
  have inside : index.val % 64 < 64 := Nat.mod_lt _ (by decide)
  cases enabled <;> simp [inside]

theorem bit_word_frame (old : U64) (index : U16) (enabled : Bool) (other : Nat)
    (inside : other < 64) (distinct : other ≠ index.val % 64) :
    (MachineRegisters.bitWord old index enabled).bv.getLsbD other = old.bv.getLsbD other := by
  rw [bit_word_bits]
  by_cases below : other < index.val % 64
  · cases enabled <;> simp [inside, below]
  · have different : other - index.val % 64 ≠ 0 := by omega
    cases enabled <;> simp [inside, different]

def bitAt (words : List U64) (index : Nat) : Bool :=
  words[index / 64]!.bv.getLsbD (index % 64)

theorem source_set_bit_selected (words : alloc.vec.Vec U64) (index : U16) (enabled : Bool)
    (inside : index.val / 64 < words.val.length) :
    machine_registers.set_bit words index enabled ⦃ updated =>
      bitAt updated.val index.val = enabled ⦄ := by
  apply WP.spec_mono (MachineRegisters.source_set_bit words index enabled inside)
  intro updated changed
  rw [changed]
  simp [bitAt, inside, bit_word_selected]

theorem bit_update_frame (words : List U64) (index : U16) (enabled : Bool)
    (inside : index.val / 64 < words.length) (other : Nat) (distinct : other ≠ index.val) :
    bitAt (words.set (index.val / 64)
      (MachineRegisters.bitWord words[index.val / 64]! index enabled)) other = bitAt words other := by
  by_cases sameWord : other / 64 = index.val / 64
  · have different : other % 64 ≠ index.val % 64 := by omega
    simp only [bitAt, sameWord]
    have read : (words.set (index.val / 64)
        (MachineRegisters.bitWord words[index.val / 64]! index enabled))[index.val / 64]! =
        MachineRegisters.bitWord words[index.val / 64]! index enabled := by simp [inside]
    rw [read]
    exact bit_word_frame _ index enabled _ (Nat.mod_lt _ (by decide)) different
  · simp [bitAt, sameWord, Ne.symm sameWord]

theorem source_set_bit_frame (words : alloc.vec.Vec U64) (index : U16) (enabled : Bool)
    (inside : index.val / 64 < words.val.length) :
    machine_registers.set_bit words index enabled ⦃ updated =>
      ∀ other, other ≠ index.val → bitAt updated.val other = bitAt words.val other ⦄ := by
  apply WP.spec_mono (MachineRegisters.source_set_bit words index enabled inside)
  intro updated changed
  rw [changed]
  exact bit_update_frame words.val index enabled inside

theorem source_write_virtual_definedness (registers : machine_registers.LaneRegisters)
    (index : U16) (value undefined : U64)
    (wordBound : index.val < registers.regs.val.length)
    (writtenBound : index.val / 64 < registers.written.val.length)
    (room : registers.vund.val.length < Usize.max) :
    machine_registers.write_virtual registers index value undefined ⦃ updated =>
      bitAt updated.written.val index.val = (undefined != core.num.U64.MAX) ⦄ := by
  apply WP.spec_mono
    (MachineRegistersVirtual.source_write_virtual registers index value undefined wordBound writtenBound room)
  intro updated changed
  rw [changed.2.1]
  simp [bitAt, writtenBound, bit_word_selected]

theorem source_write_predicate_flags (registers : machine_registers.LaneRegisters)
    (index : U16) (value defined : Bool)
    (predicateBound : index.val / 64 < registers.preds.val.length)
    (definedBound : index.val / 64 < registers.pdef.val.length) :
    machine_registers.write_predicate registers index value defined ⦃ updated =>
      bitAt updated.preds.val index.val = value ∧ bitAt updated.pdef.val index.val = defined ⦄ := by
  unfold machine_registers.write_predicate
  step with source_set_bit_selected registers.preds index value predicateBound as ⟨preds, values⟩
  step with source_set_bit_selected registers.pdef index defined definedBound as ⟨pdef, definitions⟩
  exact ⟨values, definitions⟩

theorem source_write_virtual_contract (registers : machine_registers.LaneRegisters)
    (index : U16) (value undefined : U64)
    (canonical : MachineRegistersVirtualMap.Canonical registers.vund.val)
    (wordBound : index.val < registers.regs.val.length)
    (writtenBound : index.val / 64 < registers.written.val.length)
    (room : registers.vund.val.length < Usize.max) :
    machine_registers.write_virtual registers index value undefined ⦃ updated =>
      updated.regs.val[index.val]? = some value ∧
      bitAt updated.written.val index.val = (undefined != core.num.U64.MAX) ∧
      updated.preds = registers.preds ∧ updated.pdef = registers.pdef ∧
      MachineRegistersVirtualMap.Canonical updated.vund.val ∧
      (updated.vund.val.map Prod.fst).Nodup ∧
      MachineRegistersVirtualMap.lookup updated.vund.val index =
        (if MachineRegistersVirtualMap.PartialMask undefined then some undefined else none) ∧
      ∀ other : U16, other ≠ index →
        updated.regs.val[other.val]? = registers.regs.val[other.val]? ∧
        bitAt updated.written.val other.val = bitAt registers.written.val other.val ∧
        MachineRegistersVirtualMap.lookup updated.vund.val other =
          MachineRegistersVirtualMap.lookup registers.vund.val other ⦄ := by
  apply WP.spec_mono (WP.spec_and
    (MachineRegistersVirtualMap.source_write_virtual_map registers index value undefined
      canonical wordBound writtenBound room)
    (source_write_virtual_definedness registers index value undefined wordBound writtenBound room))
  intro updated changed
  rcases changed with ⟨⟨words, markers, predicates, definitions, proper, unique, selected, others⟩, bit⟩
  refine ⟨?_, bit, predicates, definitions, proper, unique, selected, ?_⟩
  · rw [words]
    exact List.getElem?_set_self wordBound
  · intro other distinct
    have distinctValues : other.val ≠ index.val := by scalar_tac
    refine ⟨?_, ?_, others other distinct⟩
    · rw [words]
      exact List.getElem?_set_ne distinctValues.symm
    · rw [markers]
      exact bit_update_frame registers.written.val index (undefined != core.num.U64.MAX)
        writtenBound other.val distinctValues

end MemoryWrite.MachineRegistersDefinedness
