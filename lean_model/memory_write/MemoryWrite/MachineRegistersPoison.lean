/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRegistersWords
import MemoryWrite.MachineRegistersCoupling

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineRegisters

def clearWordAt (word : U64) (index : Nat) : U64 :=
  ⟨word.bv &&& ~~~((1 : BitVec 64) <<< (index % 64))⟩

theorem source_poison_uniform (registers : machine_registers.WarpRegisters)
    (warp : U32) (index : U16) (active : index.val < 255) :
    machine_registers.poison_uniform registers warp index ⦃ emitted updated =>
      updated.ur = registers.ur ∧ updated.up = registers.up ∧
      updated.up_def = registers.up_def ∧
      updated.ur_written.val = registers.ur_written.val.set (index.val / 64)
        (clearWordAt registers.ur_written.val[index.val / 64]! index.val) ∧
      emitted = some (uniformAccess warp) ⦄ := by
  have tested : ¬index >= 255#u16 := by scalar_tac
  simp only [machine_registers.poison_uniform, tested, ite_false, source_uniform_access, bind_ok]
  step as ⟨remainder, remainderValue⟩
  step as ⟨bit, bitValue, bitBits⟩
  step as ⟨clearMask, clearMaskValue⟩
  step as ⟨quotient, quotientValue⟩
  simp only [lift, bind_ok]
  step as ⟨old, oldValue⟩
  step as ⟨written, writtenValue⟩
  rw [writtenValue]
  simp only [Array.set_val_eq]
  have quotientAt : (core.convert.num.FromUsizeU16.from quotient).val = index.val / 64 := by scalar_tac
  rw [quotientAt]
  congr 1
  apply (UScalar.eq_equiv_bv_eq _ _).2
  have oldAt : old = registers.ur_written.val[index.val / 64]! := by
    rw [oldValue]
    simp only [quotientAt]
    exact List.Inhabited_getElem_eq_getElem! _ _ (by scalar_tac)
  simp only [clearWordAt, HAnd.hAnd, Complement.complement, UScalar.and,
    UScalar.not, oldAt, clearMaskValue, bitBits, remainderValue]
  rfl

def poisonWords (words : List U64) (index : Nat) : List U64 :=
  words.set (index / 64) (clearWordAt words[index / 64]! index)

theorem source_poison_scalar (registers : machine_registers.LaneRegisters) (index : U16) :
    machine_registers.poison_scalar registers index ⦃ updated =>
      updated.regs = registers.regs ∧ updated.preds = registers.preds ∧
      updated.pdef = registers.pdef ∧ updated.vund = registers.vund ∧
      updated.written.val = poisonWords registers.written.val index.val ⦄ := by
  unfold machine_registers.poison_scalar
  step as ⟨quotient, quotientValue⟩
  simp only [lift, bind_ok]
  have quotientAt : (core.convert.num.FromUsizeU16.from quotient).val = index.val / 64 := by scalar_tac
  by_cases inside : index.val / 64 < registers.written.val.length
  · have tested : core.convert.num.FromUsizeU16.from quotient < alloc.vec.Vec.len registers.written := by scalar_tac
    simp only [tested, ite_true]
    step as ⟨remainder, remainderValue⟩
    step as ⟨bit, bitValue, bitBits⟩
    step as ⟨old, back, oldValue, backValue⟩
    rw [backValue]
    simp only [alloc.vec.Vec.set, alloc.vec.Vec.from_val, poisonWords, quotientAt]
    congr 1
    apply (UScalar.eq_equiv_bv_eq _ _).2
    have oldAt : old = registers.written.val[index.val / 64]! := by
      rw [oldValue]
      simp only [quotientAt]
      exact List.Inhabited_getElem_eq_getElem! _ _ inside
    simp only [clearWordAt, HAnd.hAnd, Complement.complement, UScalar.and,
      UScalar.not, oldAt, bitBits, remainderValue]
    rfl
  · have tested : ¬core.convert.num.FromUsizeU16.from quotient < alloc.vec.Vec.len registers.written := by scalar_tac
    simp only [tested, ite_false, WP.spec_ok]
    simp [poisonWords, List.set_eq_of_length_le (by omega : registers.written.val.length ≤ index.val / 64)]

theorem source_poison_predicate (registers : machine_registers.LaneRegisters) (index : U16) :
    machine_registers.poison_predicate registers index ⦃ updated =>
      updated.regs = registers.regs ∧ updated.preds = registers.preds ∧
      updated.written = registers.written ∧ updated.vund = registers.vund ∧
      updated.pdef.val = poisonWords registers.pdef.val index.val ⦄ := by
  unfold machine_registers.poison_predicate
  step as ⟨quotient, quotientValue⟩
  simp only [lift, bind_ok]
  have quotientAt : (core.convert.num.FromUsizeU16.from quotient).val = index.val / 64 := by scalar_tac
  by_cases inside : index.val / 64 < registers.pdef.val.length
  · have tested : core.convert.num.FromUsizeU16.from quotient < alloc.vec.Vec.len registers.pdef := by scalar_tac
    simp only [tested, ite_true]
    step as ⟨remainder, remainderValue⟩
    step as ⟨bit, bitValue, bitBits⟩
    step as ⟨old, back, oldValue, backValue⟩
    rw [backValue]
    simp only [alloc.vec.Vec.set, alloc.vec.Vec.from_val, poisonWords, quotientAt]
    congr 1
    apply (UScalar.eq_equiv_bv_eq _ _).2
    have oldAt : old = registers.pdef.val[index.val / 64]! := by
      rw [oldValue]
      simp only [quotientAt]
      exact List.Inhabited_getElem_eq_getElem! _ _ inside
    simp only [clearWordAt, HAnd.hAnd, Complement.complement, UScalar.and,
      UScalar.not, oldAt, bitBits, remainderValue]
    rfl
  · have tested : ¬core.convert.num.FromUsizeU16.from quotient < alloc.vec.Vec.len registers.pdef := by scalar_tac
    simp only [tested, ite_false, WP.spec_ok]
    simp [poisonWords, List.set_eq_of_length_le (by omega : registers.pdef.val.length ≤ index.val / 64)]

end MemoryWrite.MachineRegisters
