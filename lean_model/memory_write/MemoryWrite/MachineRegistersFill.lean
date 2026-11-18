/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRegisters

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineRegistersFill

def shouldFill (written : List U64) (index : Nat) : Bool :=
  if index / 64 < written.length then
    ((written[index / 64]!.bv >>> (index % 64)) &&& 1) == 0
  else true

def fillLane (registers : machine_registers.LaneRegisters) (index : Usize) (value : U64) :
    machine_registers.LaneRegisters :=
  if shouldFill registers.written.val index.val then
    { registers with regs := registers.regs.set index value }
  else registers

def fillWarp (registers : machine_registers.WarpRegisters) (index : Usize) (value : U64) :
    machine_registers.WarpRegisters :=
  if shouldFill registers.ur_written.val index.val then
    { registers with ur := registers.ur.set index value }
  else registers

theorem source_fill_scalar (registers : machine_registers.LaneRegisters)
    (index : Usize) (value : U64) (inside : index.val < registers.regs.val.length) :
    machine_registers.fill_scalar_if_unwritten registers index value ⦃ updated =>
      updated = fillLane registers index value ⦄ := by
  unfold machine_registers.fill_scalar_if_unwritten
  step as ⟨word, wordValue⟩
  by_cases outside : registers.written.val.length ≤ index.val / 64
  · have tested : word ≥ alloc.vec.Vec.len registers.written := by scalar_tac
    simp only [tested, ite_true]
    step as ⟨old, back, oldEq, backEq⟩
    rw [backEq]
    simp [fillLane, shouldFill, show ¬index.val / 64 < registers.written.val.length by omega]
  · have wordInside : index.val / 64 < registers.written.val.length := by omega
    have tested : ¬word ≥ alloc.vec.Vec.len registers.written := by scalar_tac
    simp only [tested, ite_false]
    step as ⟨bits, bitsEq⟩
    step as ⟨remainder, remainderValue⟩
    step as ⟨shifted, shiftedValue, shiftedBits⟩
    simp only [lift, bind_ok]
    have bitsAt : bits = registers.written.val[index.val / 64]! := by
      have atWord : bits = registers.written.val[word.val]! :=
        bitsEq.trans (List.Inhabited_getElem_eq_getElem! _ _ (by omega))
      simpa only [wordValue] using atWord
    have condition : (shifted &&& 1#u64 = 0#u64) ↔
        shouldFill registers.written.val index.val = true := by
      simp only [shouldFill, wordInside, ite_true, beq_iff_eq]
      rw [UScalar.eq_equiv_bv_eq]
      simp only [HAnd.hAnd, UScalar.and, shiftedBits, remainderValue, bitsAt]
      rfl
    by_cases zero : shifted &&& 1#u64 = 0#u64
    · simp only [zero, ite_true]
      step as ⟨old, back, oldEq, backEq⟩
      rw [backEq]
      simp [fillLane, condition.mp zero]
    · simp only [zero, ite_false, WP.spec_ok]
      have noFill : shouldFill registers.written.val index.val = false := by
        cases h : shouldFill registers.written.val index.val <;> simp_all
      simp [fillLane, noFill]

theorem source_fill_uniform (registers : machine_registers.WarpRegisters)
    (index : Usize) (value : U64) (inside : index.val < registers.ur.val.length) :
    machine_registers.fill_uniform_if_unwritten registers index value ⦃ updated =>
      updated = fillWarp registers index value ⦄ := by
  unfold machine_registers.fill_uniform_if_unwritten
  step as ⟨word, wordValue⟩
  simp only [lift, bind_ok]
  by_cases outside : registers.ur_written.val.length ≤ index.val / 64
  · have tested : word ≥ Slice.len (Array.to_slice registers.ur_written) := by scalar_tac
    simp only [tested, ite_true]
    step as ⟨old, back, oldEq, backEq⟩
    rw [backEq]
    simp only [fillWarp, shouldFill,
      show ¬index.val / 64 < registers.ur_written.val.length by omega,
      ite_false, ite_true]
  · have wordInside : index.val / 64 < registers.ur_written.val.length := by omega
    have tested : ¬word ≥ Slice.len (Array.to_slice registers.ur_written) := by scalar_tac
    simp only [tested, ite_false]
    step as ⟨bits, bitsEq⟩
    step as ⟨remainder, remainderValue⟩
    step as ⟨shifted, shiftedValue, shiftedBits⟩
    have bitsAt : bits = registers.ur_written.val[index.val / 64]! := by
      have atWord : bits = registers.ur_written.val[word.val]! :=
        bitsEq.trans (List.Inhabited_getElem_eq_getElem! _ _ (by omega))
      simpa only [wordValue] using atWord
    have condition : (shifted &&& 1#u64 = 0#u64) ↔
        shouldFill registers.ur_written.val index.val = true := by
      simp only [shouldFill, wordInside, ite_true, beq_iff_eq]
      rw [UScalar.eq_equiv_bv_eq]
      simp only [HAnd.hAnd, UScalar.and, shiftedBits, remainderValue, bitsAt]
      rfl
    by_cases zero : shifted &&& 1#u64 = 0#u64
    · simp only [zero, ite_true]
      step as ⟨old, back, oldEq, backEq⟩
      rw [backEq]
      simp [fillWarp, condition.mp zero]
    · simp only [zero, ite_false, WP.spec_ok]
      have noFill : shouldFill registers.ur_written.val index.val = false := by
        cases h : shouldFill registers.ur_written.val index.val <;> simp_all
      simp [fillWarp, noFill]

theorem source_fill_scalar_metadata_frame (registers : machine_registers.LaneRegisters)
    (index : Usize) (value : U64) (inside : index.val < registers.regs.val.length) :
    machine_registers.fill_scalar_if_unwritten registers index value ⦃ updated =>
      updated.preds = registers.preds ∧ updated.pdef = registers.pdef ∧
      updated.written = registers.written ∧ updated.vund = registers.vund ∧
      (shouldFill registers.written.val index.val = false → updated = registers) ⦄ := by
  apply WP.spec_mono (source_fill_scalar registers index value inside)
  intro updated changed
  rw [changed]
  cases decision : shouldFill registers.written.val index.val <;> simp [fillLane, decision]

theorem source_fill_uniform_metadata_frame (registers : machine_registers.WarpRegisters)
    (index : Usize) (value : U64) (inside : index.val < registers.ur.val.length) :
    machine_registers.fill_uniform_if_unwritten registers index value ⦃ updated =>
      updated.ur_written = registers.ur_written ∧ updated.up = registers.up ∧
      updated.up_def = registers.up_def ∧
      (shouldFill registers.ur_written.val index.val = false → updated = registers) ⦄ := by
  apply WP.spec_mono (source_fill_uniform registers index value inside)
  intro updated changed
  rw [changed]
  cases decision : shouldFill registers.ur_written.val index.val <;> simp [fillWarp, decision]

end MemoryWrite.MachineRegistersFill
