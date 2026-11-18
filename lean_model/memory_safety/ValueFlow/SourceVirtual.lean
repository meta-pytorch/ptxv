/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceUndefined
import ValueFlow.SourceReads

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[step]
theorem source_read_virtual (registers : V.Registers) (index : U16) (bits : U8) :
    value_flow.read_virtual registers index bits ⦃ result =>
      outcomeModel valueModel result =
        if supportedBits bits.val then (wordModel registers index.val >>= readWord bits.val) else none ⦄ := by
  have oneVal : (1#u64).val = 1 := rfl
  have zeroVal : (0#u64).val = 0 := rfl
  unfold value_flow.read_virtual
  step as ⟨maskResult, maskMeaning⟩
  cases maskResult with
  | Err error =>
    have unsupported : supportedBits bits.val = false := by
      cases supported : supportedBits bits.val <;> simp_all [outcomeModel]
    simp [outcomeModel, unsupported]
  | Ok mask =>
    have supported : supportedBits bits.val = true := by
      cases supported : supportedBits bits.val <;> simp_all [outcomeModel]
    have maskVal : mask.val = 2 ^ bits.val - 1 := by
      simpa only [outcomeModel, supported, ite_true, Option.some.injEq] using maskMeaning
    step as ⟨writable, shape⟩
    cases writable with
    | false =>
      have outside : ¬(index.val < registers.raw.regs.val.length ∧
          index.val < registers.pointers.val.length ∧
          index.val / 64 < registers.raw.written.val.length) := by
        simpa [Bool.and_assoc] using shape.symm
      simp [outcomeModel, supported, wordModel, outside]
    | true =>
      have bounds : index.val < registers.raw.regs.val.length ∧
          index.val < registers.pointers.val.length ∧
          index.val / 64 < registers.raw.written.val.length := by
        simpa [Bool.and_assoc] using shape.symm
      simp only [ite_true]
      step as ⟨word, wordVal⟩
      simp only [lift, bind_ok]
      step as ⟨written, writtenEq⟩
      step as ⟨bit, bitVal⟩
      step as ⟨shifted, shiftedVal, shiftedBits⟩
      have flags : (shifted &&& 1#u64).val =
          (registers.raw.written.val[index.val / 64]?.getD 0#u64).val /
            2 ^ (index.val % 64) % 2 := by
        simp only [UScalar.val_and, oneVal, Nat.and_one_is_mod]
        rw [shiftedVal, Nat.shiftRight_eq_div_pow, bitVal, writtenEq]
        simp only [core.convert.num.FromUsizeU16.from_val_eq, wordVal,
          List.getD_getElem?, bounds.2.2, ↓reduceDIte]
      have remainder := Nat.mod_lt
        ((registers.raw.written.val[index.val / 64]?.getD 0#u64).val / 2 ^ (index.val % 64)) (by decide : 0 < 2)
      split
      · rename_i missing
        have absent : ¬(registers.raw.written.val[index.val / 64]?.getD 0#u64).val /
            2 ^ (index.val % 64) % 2 = 1 := by
          have valueZero : (shifted &&& 1#u64).val = 0 := by
            simpa only [zeroVal] using congrArg UScalar.val missing
          omega
        simp only [WP.spec_ok, outcomeModel, ite_true, wordModel, bounds,
          and_self, absent, ite_false]
        rfl
      · rename_i present
        have available : (registers.raw.written.val[index.val / 64]?.getD 0#u64).val /
            2 ^ (index.val % 64) % 2 = 1 := by
          have valueNonzero : (shifted &&& 1#u64).val ≠ 0 := by
            intro zero
            apply present
            apply UScalar.eq_of_val_eq
            simpa only [zeroVal] using zero
          omega
        step with source_read_undefined registers.raw.vund index 0#u64 0#usize (by simp) as
          ⟨undefined, undefinedMeaning⟩
        have undefinedVal : undefined.val = undefinedBits registers.raw.vund.val index.val := by
          change undefined.val = List.foldl (mergeUndefined index.val) 0 registers.raw.vund.val
          simpa only [List.drop_zero, zeroVal] using undefinedMeaning
        split
        · rename_i partiallyUndefined
          have undefinedNonzero : undefinedBits registers.raw.vund.val index.val &&&
              (2 ^ bits.val - 1) ≠ 0 := by
            have scalarNonzero : (undefined &&& mask).val ≠ 0 := by
              intro zero
              have different : undefined &&& mask ≠ 0#u64 := by
                simpa only [bne_iff_ne] using partiallyUndefined
              apply different
              apply UScalar.eq_of_val_eq
              simpa only [zeroVal] using zero
            simpa only [UScalar.val_and, undefinedVal, maskVal] using scalarNonzero
          simp only [WP.spec_ok, outcomeModel, supported, wordModel, bounds, and_self,
            available, ite_true, Option.bind_eq_bind, Option.bind_some, readWord, undefinedNonzero, decide_false,
            Bool.and_false, Bool.false_eq_true, ite_false]
        · rename_i complete
          have defined : undefinedBits registers.raw.vund.val index.val &&&
              (2 ^ bits.val - 1) = 0 := by
            have scalarZero : (undefined &&& mask).val = 0 := by
              have equal : undefined &&& mask = 0#u64 := by
                simpa only [bne_iff_ne, not_not] using complete
              simpa only [zeroVal] using congrArg UScalar.val equal
            simpa only [UScalar.val_and, undefinedVal, maskVal] using scalarZero
          step as ⟨pointer, pointerEq⟩
          cases pointer with
          | none =>
            step as ⟨value, valueEq⟩
            simp only [outcomeModel, valueModel, supported, ite_true, wordModel,
              bounds, and_self, available, Option.bind_eq_bind, Option.bind_some, readWord, defined]
            have pointerRead : registers.pointers.val[index.val]?.getD none = none := by
              simp only [List.getD_getElem?, bounds.2.1, ↓reduceDIte]
              simpa only [core.convert.num.FromUsizeU16.from_val_eq] using pointerEq.symm
            simp only [pointerRead, Option.map_none]
            congr 2
            simp only [UScalar.val_and, maskVal, Nat.and_two_pow_sub_one_eq_mod, truncate,
              List.getD_getElem?, bounds.1, ↓reduceDIte]
            simpa only [core.convert.num.FromUsizeU16.from_val_eq] using congrArg (fun v : U64 => v.val % 2 ^ bits.val) valueEq
          | some pointer =>
            dsimp only
            have pointerRead : registers.pointers.val[index.val]?.getD none = some pointer := by
              simp only [List.getD_getElem?, bounds.2.1, ↓reduceDIte]
              simpa only [core.convert.num.FromUsizeU16.from_val_eq] using pointerEq.symm
            have widthEq : (bits = 64#u8) ↔ bits.val = 64 := by
              rw [UScalar.eq_equiv]
              rfl
            split
            all_goals
              rename_i width
              simp only [WP.spec_ok, outcomeModel, valueModel, supported, ite_true, wordModel,
                bounds, and_self, available, Option.bind_eq_bind, Option.bind_some, readWord,
                defined, pointerRead, Option.map_some, Bool.true_and, decide_true]
              simp only [←widthEq, width, show (64#u8).val = 64 from rfl, ite_true, ite_false]

end ValueFlow.Source
