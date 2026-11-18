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

def undefinedUpper (undefined : U32) : BitVec 64 :=
  if undefined = core.num.U32.MAX then 0 else undefined.bv.zeroExtend 64 <<< 32

@[step] theorem source_undefined_upper (undefined : U32) :
    (if undefined = core.num.U32.MAX then .ok 0#u64 else
      core.convert.num.FromU64U32.from undefined <<< 32#i32) ⦃ upper =>
      upper.bv = undefinedUpper undefined ⦄ := by
  by_cases full : undefined = core.num.U32.MAX
  · simp [full, undefinedUpper]
  · simp only [full, ite_false]
    step as ⟨upper, upperValue, upperBits⟩
    rw [upperBits]
    simp only [undefinedUpper, full, ite_false, core.convert.num.FromU64U32.from]
    rfl

theorem source_uniform_body (registers : machine_registers.WarpRegisters)
    (write : machine_registers.WordWrite) (count slot : U8)
    (base : write.index.val < 255) (slots : slot.val < 2) (active : slot.val < count.val)
    (wordsBound : 255 ≤ registers.ur.val.length) :
    machine_registers.write_uniform_loop.body write.index write.value write.mask write.stale
      count registers slot ⦃ result =>
      ∃ updated next, result = .cont (updated, next) ∧ next.val = slot.val + 1 ∧
        updated.ur.val = (scalarSlot registers.ur.val registers.ur_written.val write slot).1 ∧
        updated.ur_written.val = (scalarSlot registers.ur.val registers.ur_written.val write slot).2 ∧
        updated.up = registers.up ∧ updated.up_def = registers.up_def ⦄ := by
  unfold machine_registers.write_uniform_loop.body
  have tested : slot < count := by scalar_tac
  simp only [tested, ite_true, lift, bind_ok]
  step as ⟨index, indexValue⟩
  step with source_slot_undefined write.mask write.stale slot as ⟨undefined, undefinedValue⟩
  by_cases inside : index.val < 255
  · have within : index < 255#u16 := by scalar_tac
    simp only [within, ite_true]
    step as ⟨shift, shiftValue⟩
    step as ⟨shifted, shiftedValue, shiftedBits⟩
    step with source_undefined_upper undefined as ⟨upper, upperBits⟩
    step as ⟨back, oldValue, backValue⟩
    rcases back with ⟨old, back⟩
    change back = registers.ur.set (core.convert.num.FromUsizeU16.from index) at backValue
    step as ⟨quotient, quotientValue⟩
    step as ⟨remainder, remainderValue⟩
    step as ⟨bit, bitValue, bitBits⟩
    step as ⟨oldWritten, oldWrittenValue⟩
    have indexAt : index.val = write.index.val + slot.val := by scalar_tac
    have wordIndex : (core.convert.num.FromUsizeU16.from index).val = index.val := by scalar_tac
    have quotientAt : (core.convert.num.FromUsizeU16.from quotient).val = index.val / 64 := by scalar_tac
    have packed : ((shifted &&& 4294967295#u64) ||| upper) =
        packedWord write.value undefined slot.val := by
      apply (UScalar.eq_equiv_bv_eq _ _).2
      simp only [packedWord, HAnd.hAnd, HOr.hOr, UScalar.and, UScalar.or,
        shiftedBits, shiftValue, upperBits, undefinedUpper]
      rfl
    have writtenAt : oldWritten = registers.ur_written.val[index.val / 64]! := by
      rw [oldWrittenValue]
      simp only [quotientAt]
      exact List.Inhabited_getElem_eq_getElem! _ _ (by scalar_tac)
    split <;> rename_i status <;> simp only [bind_ok]
    all_goals
      step as ⟨written, writtenValue⟩
      step as ⟨next, nextValue⟩
      refine ⟨_, next, rfl, nextValue, ?_, ?_, rfl, rfl⟩
      · rw [backValue]
        simp only [scalarSlot, ←indexAt, inside, ite_true, alloc.vec.Vec.set,
          alloc.vec.Vec.from_val, wordIndex, packed, undefinedValue]
      · rw [writtenValue]
        simp only [Array.set_val_eq, scalarSlot, ←indexAt, inside, ite_true, quotientAt]
        congr 1
        apply (UScalar.eq_equiv_bv_eq _ _).2
        simp only [bitWordAt, HAnd.hAnd, HOr.hOr, Complement.complement,
          UScalar.and, UScalar.or, UScalar.not, writtenAt, bitBits, remainderValue,
          ←undefinedValue, bne_iff_ne, ne_eq, status, not_true_eq_false,
          not_false_eq_true, ↓reduceIte]
        rfl
  · have within : ¬index < 255#u16 := by scalar_tac
    have indexAt : index.val = write.index.val + slot.val := by scalar_tac
    simp only [within, ite_false, bind_ok]
    step as ⟨next, nextValue⟩
    refine ⟨_, next, rfl, nextValue, ?_, ?_, rfl, rfl⟩ <;>
      simp only [scalarSlot, ←indexAt, inside, ite_false]

theorem source_uniform_loop_unroll (registers : machine_registers.WarpRegisters)
    (write : machine_registers.WordWrite) (count slot : U8) :
    machine_registers.write_uniform_loop registers write.index write.value write.mask
      write.stale count slot = (do
      let result ← machine_registers.write_uniform_loop.body write.index write.value
        write.mask write.stale count registers slot
      match result with
      | .cont (updated, next) => machine_registers.write_uniform_loop updated write.index write.value write.mask write.stale count next
      | .done updated => .ok updated) := by
  unfold machine_registers.write_uniform_loop
  rw [loop.eq_def]
  congr 1
  funext result
  cases result with
  | cont pair => rcases pair with ⟨updated, next⟩; rfl
  | done updated => rfl

theorem source_uniform_loop_done (registers : machine_registers.WarpRegisters)
    (write : machine_registers.WordWrite) (count slot : U8)
    (finished : count.val ≤ slot.val) :
    machine_registers.write_uniform_loop registers write.index write.value write.mask
      write.stale count slot = .ok registers := by
  rw [source_uniform_loop_unroll]
  have tested : ¬slot < count := by scalar_tac
  simp [machine_registers.write_uniform_loop.body, tested]

theorem source_uniform_loop (registers : machine_registers.WarpRegisters)
    (write : machine_registers.WordWrite) (count : U8)
    (base : write.index.val < 255) (positive : 1 ≤ count.val) (bounded : count.val ≤ 2)
    (sameCount : (count.val = 2) ↔ (write.slots.val = 2))
    (wordsBound : 255 ≤ registers.ur.val.length) :
    machine_registers.write_uniform_loop registers write.index write.value write.mask
      write.stale count 0#u8 ⦃ updated =>
      updated.ur.val = (scalarResult registers.ur.val registers.ur_written.val write).1 ∧
      updated.ur_written.val = (scalarResult registers.ur.val registers.ur_written.val write).2 ∧
      updated.up = registers.up ∧ updated.up_def = registers.up_def ⦄ := by
  obtain ⟨firstResult, firstRun, firstPost⟩ := MemoryWrite.spec_returns _ _
    (source_uniform_body registers write count 0#u8 base (by scalar_tac)
      (by scalar_tac) wordsBound)
  rcases firstPost with ⟨first, next, rfl, nextValue, firstRegs, firstWritten,
    firstUp, firstUpDef⟩
  rw [source_uniform_loop_unroll, firstRun]
  simp only [bind_ok]
  have nextEq : next = 1#u8 := by scalar_tac
  rw [nextEq]
  by_cases two : count.val = 2
  · have firstWords : 255 ≤ first.ur.val.length := by
      rw [firstRegs, (scalar_slot_lengths _ _ _ _).1]
      exact wordsBound
    obtain ⟨secondResult, secondRun, secondPost⟩ := MemoryWrite.spec_returns _ _
      (source_uniform_body first write count 1#u8 base (by scalar_tac)
        (by scalar_tac) firstWords)
    rcases secondPost with ⟨second, lastSlot, rfl, lastValue, secondRegs, secondWritten,
      secondUp, secondUpDef⟩
    rw [source_uniform_loop_unroll, secondRun]
    simp only [bind_ok]
    rw [source_uniform_loop_done second write count lastSlot (by scalar_tac)]
    simp only [WP.spec_ok]
    have twoWrite : write.slots.val = 2 := sameCount.mp two
    simp only [scalarResult, Nat.not_le.mpr base, ite_false, twoWrite, ite_true,
      secondRegs, secondWritten, firstRegs, firstWritten,
      secondUp, secondUpDef, firstUp, firstUpDef,
      and_self]
  · rw [source_uniform_loop_done first write count 1#u8 (by scalar_tac)]
    simp only [WP.spec_ok]
    have notTwoWrite : write.slots.val ≠ 2 := fun equal => two (sameCount.mpr equal)
    simp only [scalarResult, Nat.not_le.mpr base, ite_false, notTwoWrite,
      firstRegs, firstWritten, firstUp, firstUpDef, and_self]

theorem source_write_uniform (registers : machine_registers.WarpRegisters)
    (warp : U32) (write : machine_registers.WordWrite)
    (slots : write.slots.val ≤ 2) (wordsBound : 255 ≤ registers.ur.val.length) :
    machine_registers.write_uniform registers warp write ⦃ emitted updated =>
      updated.ur.val = (scalarResult registers.ur.val registers.ur_written.val write).1 ∧
      updated.ur_written.val = (scalarResult registers.ur.val registers.ur_written.val write).2 ∧
      updated.up = registers.up ∧ updated.up_def = registers.up_def ∧
      emitted = if 255 ≤ write.index.val then none else some (uniformAccess warp) ⦄ := by
  by_cases constant : 255 ≤ write.index.val
  · rw [source_uniform_constant_frame registers warp write constant]
    simp [scalarResult, constant]
  · have base : write.index.val < 255 := by omega
    have tested : ¬write.index >= 255#u16 := by scalar_tac
    have slotTest : write.slots <= 2#u8 := by scalar_tac
    simp only [machine_registers.write_uniform, tested, slotTest, decide_true,
      decide_false, Bool.false_or, massert_True, bind_ok, ite_false, lift]
    let count := core.cmp.impls.OrdU8.max write.slots 1#u8
    have positive : 1 ≤ count.val := by
      simp only [count, core.cmp.impls.OrdU8.max]
      split <;> scalar_tac
    have bounded : count.val ≤ 2 := by
      simp only [count, core.cmp.impls.OrdU8.max]
      split <;> scalar_tac
    have sameCount : (count.val = 2) ↔ (write.slots.val = 2) := by
      simp only [count, core.cmp.impls.OrdU8.max]
      split <;> scalar_tac
    step with source_uniform_loop registers write count base positive bounded sameCount
      wordsBound as ⟨updated, words, written, up, upDef⟩
    simp only [source_uniform_access, bind_ok, WP.spec_ok]
    exact ⟨words, written, up, upDef, by simp [constant]⟩

theorem source_uniform_word_transition (registers : machine_registers.WarpRegisters)
    (warp : U32) (write : machine_registers.WordWrite) (active : write.index.val < 255)
    (slots : write.slots.val ≤ 2) (wordsBound : 255 ≤ registers.ur.val.length) :
    machine_registers.write_uniform registers warp write ⦃ emitted updated =>
      updated.ur.val = (scalarResult registers.ur.val registers.ur_written.val write).1 ∧
      updated.ur_written.val = (scalarResult registers.ur.val registers.ur_written.val write).2 ∧
      updated.up = registers.up ∧ updated.up_def = registers.up_def ∧
      emitted.map accessEvent = some (uniformEvent warp) ⦄ := by
  step with source_write_uniform registers warp write slots wordsBound as
    ⟨emitted, updated, words, written, up, upDef, emission⟩
  have event : emitted = some (uniformAccess warp) := by
    simpa only [Nat.not_le.mpr active, ite_false] using emission
  exact ⟨words, written, up, upDef, by simp [event, uniform_access_event]⟩

end MemoryWrite.MachineRegisters
