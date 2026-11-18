/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRegisters
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineRegisters

def bitWordAt (old : U64) (index : Nat) (enabled : Bool) : U64 :=
  let bit : BitVec 64 := (1 : BitVec 64) <<< (index % 64)
  ⟨(old.bv &&& ~~~bit) ||| if enabled then bit else 0⟩

def packedWord (value : U64) (undefined : U32) (slot : Nat) : U64 :=
  ⟨((value.bv >>> (32 * slot)) &&& (4294967295 : BitVec 64)) |||
    (if undefined = core.num.U32.MAX then (0 : BitVec 64) else (undefined.bv.zeroExtend 64 <<< 32))⟩

def scalarSlot (registers written : List U64) (write : machine_registers.WordWrite)
    (slot : U8) : List U64 × List U64 :=
  let index := write.index.val + slot.val
  let undefined := slotUndefined write.mask write.stale slot
  if index < 255 then
    (registers.set index (packedWord write.value undefined slot.val),
      written.set (index / 64) (bitWordAt written[index / 64]! index
        (undefined != core.num.U32.MAX)))
  else (registers, written)

theorem source_scalar_body (registers : machine_registers.LaneRegisters)
    (write : machine_registers.WordWrite) (count slot : U8)
    (base : write.index.val < 255) (slots : slot.val < 2) (active : slot.val < count.val)
    (wordsBound : 255 ≤ registers.regs.val.length)
    (bitsBound : 4 ≤ registers.written.val.length) :
    machine_registers.write_scalar_loop.body write.index write.value write.mask write.stale
      count registers slot ⦃ result =>
      ∃ updated next, result = .cont (updated, next) ∧ next.val = slot.val + 1 ∧
        updated.regs.val = (scalarSlot registers.regs.val registers.written.val write slot).1 ∧
        updated.written.val = (scalarSlot registers.regs.val registers.written.val write slot).2 ∧
        updated.preds = registers.preds ∧ updated.pdef = registers.pdef ∧
        updated.vund = registers.vund ⦄ := by
  unfold machine_registers.write_scalar_loop.body
  have tested : slot < count := by scalar_tac
  simp only [tested, ite_true, lift, bind_ok]
  step as ⟨index, indexValue⟩
  step with source_slot_undefined write.mask write.stale slot as ⟨undefined, undefinedValue⟩
  by_cases inside : index.val < 255
  · have within : index < 255#u16 := by scalar_tac
    simp only [within, ite_true]
    step as ⟨shift, shiftValue⟩
    step as ⟨shifted, shiftedValue, shiftedBits⟩
    have indexAt : index.val = write.index.val + slot.val := by scalar_tac
    have wordIndex : (core.convert.num.FromUsizeU16.from index).val = index.val := by scalar_tac
    have wordBound : (core.convert.num.FromUsizeU16.from index).val < registers.regs.val.length := by scalar_tac
    split
    · rename_i full
      simp only [bind_ok]
      rw [source_set_word _ _ _ wordBound]
      simp only [bind_ok]
      step with source_set_bit registers.written index
        (undefined != core.num.U32.MAX) (by scalar_tac) as ⟨written, writtenValue⟩
      step as ⟨next, nextValue⟩
      have packed : ((shifted &&& 4294967295#u64) ||| 0#u64) =
          packedWord write.value undefined slot.val := by
        apply (UScalar.eq_equiv_bv_eq _ _).2
        simp only [packedWord, full, ite_true, HAnd.hAnd, HOr.hOr,
          UScalar.and, UScalar.or, shiftedBits, shiftValue]
        rfl
      refine ⟨_, next, rfl, nextValue, ?_, ?_, rfl, rfl, rfl⟩
      · simp only [scalarSlot, ←indexAt, inside, ite_true,
          alloc.vec.Vec.set, alloc.vec.Vec.from_val, wordIndex, packed, undefinedValue]
      · simp only [scalarSlot, ←indexAt, inside, ite_true, writtenValue,
          bitWord, bitWordAt, undefinedValue]
    · rename_i notFull
      step as ⟨upper, upperValue, upperBits⟩
      rw [source_set_word _ _ _ wordBound]
      simp only [bind_ok]
      step with source_set_bit registers.written index
        (undefined != core.num.U32.MAX) (by scalar_tac) as ⟨written, writtenValue⟩
      step as ⟨next, nextValue⟩
      have packed : ((shifted &&& 4294967295#u64) ||| upper) =
          packedWord write.value undefined slot.val := by
        apply (UScalar.eq_equiv_bv_eq _ _).2
        simp only [packedWord, notFull, ite_false, HAnd.hAnd, HOr.hOr,
          UScalar.and, UScalar.or, shiftedBits, shiftValue, upperBits,
          core.convert.num.FromU64U32.from]
        rfl
      refine ⟨_, next, rfl, nextValue, ?_, ?_, rfl, rfl, rfl⟩
      · simp only [scalarSlot, ←indexAt, inside, ite_true,
          alloc.vec.Vec.set, alloc.vec.Vec.from_val, wordIndex, packed, undefinedValue]
      · simp only [scalarSlot, ←indexAt, inside, ite_true, writtenValue,
          bitWord, bitWordAt, undefinedValue]
  · have within : ¬index < 255#u16 := by scalar_tac
    have indexAt : index.val = write.index.val + slot.val := by scalar_tac
    simp only [within, ite_false, bind_ok]
    step as ⟨next, nextValue⟩
    refine ⟨_, next, rfl, nextValue, ?_, ?_, rfl, rfl, rfl⟩ <;>
      simp only [scalarSlot, ←indexAt, inside, ite_false]

theorem scalar_slot_lengths (registers written : List U64)
    (write : machine_registers.WordWrite) (slot : U8) :
    (scalarSlot registers written write slot).1.length = registers.length ∧
      (scalarSlot registers written write slot).2.length = written.length := by
  by_cases inside : write.index.val + slot.val < 255 <;> simp [scalarSlot, inside]

theorem source_scalar_loop_unroll (registers : machine_registers.LaneRegisters)
    (write : machine_registers.WordWrite) (count slot : U8) :
    machine_registers.write_scalar_loop registers write.index write.value write.mask
      write.stale count slot = (do
      let result ← machine_registers.write_scalar_loop.body write.index write.value
        write.mask write.stale count registers slot
      match result with
      | .cont (updated, next) => machine_registers.write_scalar_loop updated write.index write.value write.mask write.stale count next
      | .done updated => .ok updated) := by
  unfold machine_registers.write_scalar_loop
  rw [loop.eq_def]
  congr 1
  funext result
  cases result with
  | cont pair => rcases pair with ⟨updated, next⟩; rfl
  | done updated => rfl

theorem source_scalar_loop_done (registers : machine_registers.LaneRegisters)
    (write : machine_registers.WordWrite) (count slot : U8)
    (finished : count.val ≤ slot.val) :
    machine_registers.write_scalar_loop registers write.index write.value write.mask
      write.stale count slot = .ok registers := by
  rw [source_scalar_loop_unroll]
  have tested : ¬slot < count := by scalar_tac
  simp [machine_registers.write_scalar_loop.body, tested]

def scalarResult (registers written : List U64) (write : machine_registers.WordWrite) :
    List U64 × List U64 :=
  if 255 ≤ write.index.val then (registers, written) else
    let first := scalarSlot registers written write 0#u8
    if write.slots.val = 2 then scalarSlot first.1 first.2 write 1#u8 else first

theorem source_scalar_loop (registers : machine_registers.LaneRegisters)
    (write : machine_registers.WordWrite) (count : U8)
    (base : write.index.val < 255) (positive : 1 ≤ count.val) (bounded : count.val ≤ 2)
    (sameCount : (count.val = 2) ↔ (write.slots.val = 2))
    (wordsBound : 255 ≤ registers.regs.val.length)
    (bitsBound : 4 ≤ registers.written.val.length) :
    machine_registers.write_scalar_loop registers write.index write.value write.mask
      write.stale count 0#u8 ⦃ updated =>
      updated.regs.val = (scalarResult registers.regs.val registers.written.val write).1 ∧
      updated.written.val = (scalarResult registers.regs.val registers.written.val write).2 ∧
      updated.preds = registers.preds ∧ updated.pdef = registers.pdef ∧
      updated.vund = registers.vund ⦄ := by
  obtain ⟨firstResult, firstRun, firstPost⟩ := MemoryWrite.spec_returns _ _
    (source_scalar_body registers write count 0#u8 base (by scalar_tac)
      (by scalar_tac) wordsBound bitsBound)
  rcases firstPost with ⟨first, next, rfl, nextValue, firstRegs, firstWritten,
    firstPreds, firstPdef, firstVund⟩
  rw [source_scalar_loop_unroll, firstRun]
  simp only [bind_ok]
  have nextEq : next = 1#u8 := by scalar_tac
  rw [nextEq]
  by_cases two : count.val = 2
  · have firstWords : 255 ≤ first.regs.val.length := by
      rw [firstRegs, (scalar_slot_lengths _ _ _ _).1]
      exact wordsBound
    have firstBits : 4 ≤ first.written.val.length := by
      rw [firstWritten, (scalar_slot_lengths _ _ _ _).2]
      exact bitsBound
    obtain ⟨secondResult, secondRun, secondPost⟩ := MemoryWrite.spec_returns _ _
      (source_scalar_body first write count 1#u8 base (by scalar_tac)
        (by scalar_tac) firstWords firstBits)
    rcases secondPost with ⟨second, lastSlot, rfl, lastValue, secondRegs, secondWritten,
      secondPreds, secondPdef, secondVund⟩
    rw [source_scalar_loop_unroll, secondRun]
    simp only [bind_ok]
    rw [source_scalar_loop_done second write count lastSlot (by scalar_tac)]
    simp only [WP.spec_ok]
    have twoWrite : write.slots.val = 2 := sameCount.mp two
    simp only [scalarResult, Nat.not_le.mpr base, ite_false, twoWrite, ite_true,
      secondRegs, secondWritten, firstRegs, firstWritten,
      secondPreds, secondPdef, secondVund, firstPreds, firstPdef, firstVund,
      and_self]
  · rw [source_scalar_loop_done first write count 1#u8 (by scalar_tac)]
    simp only [WP.spec_ok]
    have notTwoWrite : write.slots.val ≠ 2 := fun equal => two (sameCount.mpr equal)
    simp only [scalarResult, Nat.not_le.mpr base, ite_false, notTwoWrite,
      firstRegs, firstWritten, firstPreds, firstPdef, firstVund, and_self]

theorem source_write_scalar (registers : machine_registers.LaneRegisters)
    (write : machine_registers.WordWrite) (slots : write.slots.val ≤ 2)
    (wordsBound : 255 ≤ registers.regs.val.length)
    (bitsBound : 4 ≤ registers.written.val.length) :
    machine_registers.write_scalar registers write ⦃ updated =>
      updated.regs.val = (scalarResult registers.regs.val registers.written.val write).1 ∧
      updated.written.val = (scalarResult registers.regs.val registers.written.val write).2 ∧
      updated.preds = registers.preds ∧ updated.pdef = registers.pdef ∧
      updated.vund = registers.vund ⦄ := by
  by_cases constant : 255 ≤ write.index.val
  · rw [source_scalar_constant_frame registers write constant]
    simp [scalarResult, constant]
  · have base : write.index.val < 255 := by omega
    have tested : ¬write.index >= 255#u16 := by scalar_tac
    have slotTest : write.slots <= 2#u8 := by scalar_tac
    simp only [machine_registers.write_scalar, tested, slotTest, decide_true, decide_false, Bool.false_or,
      massert_True, bind_ok, ite_false, lift]
    apply source_scalar_loop registers write
      (core.cmp.impls.OrdU8.max write.slots 1#u8) base
    · simp only [core.cmp.impls.OrdU8.max]
      split <;> scalar_tac
    · simp only [core.cmp.impls.OrdU8.max]
      split <;> scalar_tac
    · simp only [core.cmp.impls.OrdU8.max]
      split <;> scalar_tac
    · exact wordsBound
    · exact bitsBound

theorem scalar_slot_frame (registers written : List U64)
    (write : machine_registers.WordWrite) (slot : U8) (other : Nat)
    (distinct : other ≠ write.index.val + slot.val) :
    (scalarSlot registers written write slot).1[other]? = registers[other]? := by
  by_cases inside : write.index.val + slot.val < 255 <;>
    simp [scalarSlot, inside, List.getElem?_set_ne distinct.symm]

theorem scalar_result_frame (registers written : List U64)
    (write : machine_registers.WordWrite) (other : Nat)
    (first : other ≠ write.index.val) (second : other ≠ write.index.val + 1) :
    (scalarResult registers written write).1[other]? = registers[other]? := by
  by_cases constant : 255 ≤ write.index.val
  · simp [scalarResult, constant]
  · simp only [scalarResult, constant, ite_false]
    split
    · rw [scalar_slot_frame _ _ write 1#u8 other (by simpa using second)]
      exact scalar_slot_frame _ _ write 0#u8 other (by simpa using first)
    · exact scalar_slot_frame _ _ write 0#u8 other (by simpa using first)

theorem source_scalar_register_frame (registers : machine_registers.LaneRegisters)
    (write : machine_registers.WordWrite) (slots : write.slots.val ≤ 2)
    (wordsBound : 255 ≤ registers.regs.val.length)
    (bitsBound : 4 ≤ registers.written.val.length) :
    machine_registers.write_scalar registers write ⦃ updated =>
      ∀ other, other ≠ write.index.val → other ≠ write.index.val + 1 →
        updated.regs.val[other]? = registers.regs.val[other]? ⦄ := by
  obtain ⟨updated, run, post⟩ := MemoryWrite.spec_returns _ _
    (source_write_scalar registers write slots wordsBound bitsBound)
  rw [run]
  simp only [WP.spec_ok, post.1]
  exact scalar_result_frame registers.regs.val registers.written.val write

end MemoryWrite.MachineRegisters
