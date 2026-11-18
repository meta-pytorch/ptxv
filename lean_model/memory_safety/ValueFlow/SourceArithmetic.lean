/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceCompare

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[step]
theorem source_scalar_binary (operation : V.BinaryOperation) (bits : U8) (left right : U64) :
    value_flow.scalar_binary operation bits left right ⦃ result =>
      outcomeModel UScalar.val result =
        scalarBinary (binaryModel operation) bits.val left.val right.val ⦄ := by
  by_cases width32 : bits = 32#u8
  · subst bits
    cases operation <;>
      simp [value_flow.scalar_binary, lift, bind_ok, symbolic.binary_bits, outcomeModel,
        scalarBinary, binaryModel, scalarBits, truncate, UScalar.val_and,
        core.convert.num.FromU64U32.from_val_eq, UScalar.cast_val_eq,
        UScalar.size_def, UScalarTy.numBits]
    case Subtract =>
      have bound : right.val % 4294967296 < 4294967296 := Nat.mod_lt _ (by decide)
      have period : 18446744073709551616 - right.val % 4294967296 =
          4294967295 * 4294967296 + (4294967296 - right.val % 4294967296) := by omega
      have rhs : left.val % 4294967296 + 4294967296 - right.val % 4294967296 =
          left.val % 4294967296 + (4294967296 - right.val % 4294967296) :=
        Nat.add_sub_assoc (Nat.le_of_lt bound) _
      rw [rhs, period]
      simp [Nat.add_mod, Nat.mul_mod]
    case ShiftLeft =>
      split
      · rename_i tooLarge
        simp [show ¬ right.val % 4294967296 < 32 by omega]
      · rename_i inRange
        have valid : (right &&& 4294967295#u64).val < 64 := by
          simp only [UScalar.val_and, show (4294967295#u64).val = 4294967295 from rfl, nat_mask32]
          omega
        step as ⟨shifted, properties, shiftedBits⟩
        simp [outcomeModel, properties, Nat.shiftLeft_eq, UScalar.val_and,
          show (4294967295#u64).val = 4294967295 from rfl, nat_mask32,
          show U64.size = 18446744073709551616 by simp [U64.size, U64.numBits],
          Nat.mod_mod_of_dvd _ (by decide : 4294967296 ∣ 18446744073709551616),
          show right.val % 4294967296 < 32 by omega]
    case ShiftRightUnsigned =>
      split
      · rename_i tooLarge
        simp [show ¬ right.val % 4294967296 < 32 by omega]
      · rename_i inRange
        have valid : (right &&& 4294967295#u64).val < 64 := by
          simp only [UScalar.val_and, show (4294967295#u64).val = 4294967295 from rfl, nat_mask32]
          omega
        step as ⟨shifted, properties, shiftedBits⟩
        simp [outcomeModel, properties, Nat.shiftRight_eq_div_pow, UScalar.val_and,
          show (4294967295#u64).val = 4294967295 from rfl, nat_mask32,
          show right.val % 4294967296 < 32 by omega]

  · by_cases width64 : bits = 64#u8
    · subst bits
      cases operation <;>
        simp [value_flow.scalar_binary, lift, bind_ok, atomic_integer.replacement64, outcomeModel,
          scalarBinary, binaryModel, scalarBits, truncate, UScalar.val_and,
          core.convert.num.FromU64U32.from_val_eq, UScalar.cast_val_eq,
          UScalar.size_def, UScalarTy.numBits]
      case Subtract =>
        have bound : right.val ≤ 18446744073709551616 := by scalar_tac
        congr 1
        omega
      case And =>
        have bound : left.val &&& right.val < 18446744073709551616 := by
          have bounded := (left &&& right).bv.isLt
          change (left &&& right).val < 18446744073709551616 at bounded
          simpa only [UScalar.val_and] using bounded
        simp [Nat.mod_eq_of_lt bound]
      case Or =>
        have bound : left.val ||| right.val < 18446744073709551616 := by
          have bounded := (left ||| right).bv.isLt
          change (left ||| right).val < 18446744073709551616 at bounded
          simpa only [UScalar.val_or] using bounded
        simp [Nat.mod_eq_of_lt bound]
      case Xor =>
        have bound : left.val ^^^ right.val < 18446744073709551616 := by
          have bounded := (left ^^^ right).bv.isLt
          change (left ^^^ right).val < 18446744073709551616 at bounded
          simpa only [UScalar.val_xor] using bounded
        simp [Nat.mod_eq_of_lt bound]
      case ShiftLeft =>
        split
        · rename_i tooLarge
          simp [show ¬ right.val < 64 by omega]
        · rename_i inRange
          have valid : right.val < 64 := by omega
          step as ⟨shifted, properties, shiftedBits⟩
          simp [outcomeModel, properties, Nat.shiftLeft_eq,
            show U64.size = 18446744073709551616 by simp [U64.size, U64.numBits], show right.val < 64 by omega]
      case ShiftRightUnsigned =>
        split
        · rename_i tooLarge
          simp [show ¬ right.val < 64 by omega]
        · rename_i inRange
          have valid : right.val < 64 := by omega
          step as ⟨shifted, properties, shiftedBits⟩
          have bound : shifted.val < 18446744073709551616 := shifted.bv.isLt
          simp only [WP.spec_ok, outcomeModel, if_pos valid]
          rw [← Nat.shiftRight_eq_div_pow, ← properties, Nat.mod_eq_of_lt bound]

    · have no32 : bits.val ≠ 32 := by scalar_tac
      have no64 : bits.val ≠ 64 := by scalar_tac
      simp [value_flow.scalar_binary, width32, width64, outcomeModel, scalarBinary, scalarBits, no32, no64]

end ValueFlow.Source
