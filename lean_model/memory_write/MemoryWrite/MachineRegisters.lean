/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineRegisters

def writeWord (words : List U64) (index : Nat) (value : U64) : List U64 :=
  words.set index value

theorem write_word_length (words : List U64) (index : Nat) (value : U64) :
    (writeWord words index value).length = words.length := by
  simp [writeWord]

theorem write_word_selected (words : List U64) (index : Nat) (value : U64)
    (inside : index < words.length) :
    (writeWord words index value)[index]? = some value := by
  simp [writeWord, inside]

theorem write_word_frame (words : List U64) (index other : Nat) (value : U64)
    (distinct : other ≠ index) :
    (writeWord words index value)[other]? = words[other]? := by
  simp [writeWord, distinct]

theorem source_set_word (words : alloc.vec.Vec U64) (index : Usize) (value : U64)
    (inside : index.val < words.val.length) :
    machine_registers.set_word words index value = .ok (words.set index value) := by
  simp [machine_registers.set_word, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]

theorem source_set_word_frame (words : alloc.vec.Vec U64) (index : Usize) (value : U64)
    (inside : index.val < words.val.length) :
    machine_registers.set_word words index value ⦃ updated =>
      updated.val.length = words.val.length ∧
      updated.val[index.val]? = some value ∧
      ∀ other, other ≠ index.val → updated.val[other]? = words.val[other]? ⦄ := by
  rw [source_set_word words index value inside]
  simp only [WP.spec_ok, alloc.vec.Vec.set, alloc.vec.Vec.from_val]
  constructor
  · simp
  constructor
  · exact List.getElem?_set_self inside
  intro other distinct
  exact List.getElem?_set_ne distinct.symm

def bitWord (old : U64) (index : U16) (enabled : Bool) : U64 :=
  let bit : BitVec 64 := (1 : BitVec 64) <<< (index.val % 64)
  ⟨(old.bv &&& ~~~bit) ||| if enabled then bit else 0⟩

@[step] theorem source_set_bit (words : alloc.vec.Vec U64) (index : U16) (enabled : Bool)
    (inside : index.val / 64 < words.val.length) :
    machine_registers.set_bit words index enabled ⦃ updated =>
      updated.val = words.val.set (index.val / 64) (bitWord words.val[index.val / 64]! index enabled) ⦄ := by
  unfold machine_registers.set_bit
  step as ⟨quotient, quotientValue⟩
  simp only [lift, bind_ok]
  step as ⟨remainder, remainderValue⟩
  step as ⟨bit, bitValue, bitBits⟩
  step as ⟨old, oldValue⟩
  split <;> simp only [bind_ok]
  all_goals
    step as ⟨oldAgain, back, oldAgainValue, backValue⟩
    rw [backValue]
    simp only [alloc.vec.Vec.set, alloc.vec.Vec.from_val]
    have quotientAt : (core.convert.num.FromUsizeU16.from quotient).val = index.val / 64 := by scalar_tac
    rw [quotientAt]
    congr 1
    apply (UScalar.eq_equiv_bv_eq _ _).2
    have oldAt : old = words.val[index.val / 64]! := by
      rw [oldValue]
      simp only [quotientAt]
      exact List.Inhabited_getElem_eq_getElem! _ _ inside
    simp only [bitWord, HAnd.hAnd, HOr.hOr, Complement.complement,
      UScalar.and, UScalar.or, UScalar.not, oldAt, bitBits, remainderValue]
    all_goals simp_all only [Bool.false_eq_true, ↓reduceIte]
    all_goals rfl

theorem source_write_predicate (registers : machine_registers.LaneRegisters)
    (index : U16) (value defined : Bool)
    (predicateBound : index.val / 64 < registers.preds.val.length)
    (definedBound : index.val / 64 < registers.pdef.val.length) :
    machine_registers.write_predicate registers index value defined ⦃ updated =>
      updated.regs = registers.regs ∧ updated.written = registers.written ∧
      updated.vund = registers.vund ∧
      updated.preds.val = registers.preds.val.set (index.val / 64)
        (bitWord registers.preds.val[index.val / 64]! index value) ∧
      updated.pdef.val = registers.pdef.val.set (index.val / 64)
        (bitWord registers.pdef.val[index.val / 64]! index defined) ⦄ := by
  unfold machine_registers.write_predicate
  step with source_set_bit registers.preds index value predicateBound as ⟨predicates, values⟩
  step with source_set_bit registers.pdef index defined definedBound as ⟨definitions, masks⟩
  simp [values, masks]

def slotUndefined (mask : Option U64) (stale : Bool) (slot : U8) : U32 :=
  if stale then core.num.U32.MAX else
    match mask with
    | none => 0#u32
    | some bits =>
      if slot.val < 2 then ⟨(bits.bv >>> (32 * slot.val)).zeroExtend 32⟩
      else if bits = 0#u64 then 0#u32 else core.num.U32.MAX

@[step] theorem source_slot_undefined (mask : Option U64) (stale : Bool) (slot : U8) :
    machine_registers.slot_undefined mask stale slot ⦃ result =>
      result = slotUndefined mask stale slot ⦄ := by
  cases stale with
  | true => simp [machine_registers.slot_undefined, slotUndefined]
  | false =>
    cases mask with
    | none => simp [machine_registers.slot_undefined, slotUndefined]
    | some bits =>
      by_cases low : slot.val < 2
      · have tested : slot < 2#u8 := by scalar_tac
        simp only [machine_registers.slot_undefined, Bool.false_eq_true, ↓reduceIte, tested]
        step as ⟨shift, shiftValue⟩
        step as ⟨shifted, shiftedValue, shiftedBits⟩
        apply (UScalar.eq_equiv_bv_eq _ _).2
        simp only [slotUndefined, Bool.false_eq_true, ↓reduceIte, low,
          UScalar.cast, shiftedBits, shiftValue]
        rfl
      · have tested : ¬slot < 2#u8 := by scalar_tac
        by_cases zero : bits = 0#u64 <;>
          simp [machine_registers.slot_undefined, slotUndefined, tested, low, zero]

theorem source_scalar_constant_frame
    (registers : machine_registers.LaneRegisters) (write : machine_registers.WordWrite)
    (constant : 255 ≤ write.index.val) :
    machine_registers.write_scalar registers write = .ok registers := by
  have tested : write.index ≥ 255#u16 := by scalar_tac
  simp [machine_registers.write_scalar, tested]

end MemoryWrite.MachineRegisters
