/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBindings.Model

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction.SegBinding

def checked {α : Type} (value : Option α) : Result α :=
  match value with
  | none => .fail .arrayOutOfBounds
  | some bits => .ok bits

theorem exact_eq_of_spec {α : Type} (computation : Result α) (value : α)
    (specification : computation ⦃ answer => answer = value ⦄) : computation = .ok value := by
  cases computation using Result.cases with
  | ret answer =>
    have h := (WP.spec_ok answer).mp specification
    rw [h]
  | vis effect continuation =>
    exact False.elim ((WP.spec_vis effect continuation).mp specification)
  | div => exact False.elim (WP.spec_div.mp specification)

theorem source_read_sval_exact (source : PtxvSim.seg.SVal)
    (args : Slice U32) (prior : Slice U64) :
    PtxvSim.seg.binding.read_sval source args prior = checked (svalBits args.val prior.val source) := by
  cases source with
  | Arg index =>
    have hi : (UScalar.cast .Usize index).val = index.val := by scalar_tac
    simp only [PtxvSim.seg.binding.read_sval, lift, bind_ok, Slice.index_usize,
      Slice.getElem?_Usize_eq, hi]
    cases h : args.val[index.val]? <;> simp [svalBits, checked, h]
  | Const bits => rfl
  | Op index =>
    have hi : (UScalar.cast .Usize index).val = index.val := by scalar_tac
    simp [PtxvSim.seg.binding.read_sval, lift, Slice.index_usize, hi, svalBits, checked]
    rfl

theorem source_predicate_register_exact (register : PtxvSim.ptxv_ir.Reg) (bits : U64) :
    PtxvSim.ptxv_ir.Reg.pred_bit register bits = .ok (registerPredicateBits register bits) := by
  rcases register with ⟨file, index⟩
  cases file <;> simp only [PtxvSim.ptxv_ir.Reg.pred_bit, registerPredicateBits]
  all_goals
    apply exact_eq_of_spec
    step with U64.ShiftRight_IScalar_spec bits 31#i32 (by decide) (by decide) as ⟨shifted, hshift⟩
    step as ⟨masked, hmask⟩
    congr 1
    rw [UScalar.eq_equiv]
    simp only [hmask, UScalar.val_and, hshift, show (1#u64).val = 1 by rfl, Nat.and_one_is_mod,
      Nat.shiftRight_eq_div_pow]

@[step] theorem source_read_sval (source : PtxvSim.seg.SVal)
    (args : Slice U32) (prior : Slice U64) (valid : svalValid args.val prior.val source) :
    PtxvSim.seg.binding.read_sval source args prior ⦃ bits =>
      some bits = svalBits args.val prior.val source ⦄ := by
  rw [source_read_sval_exact]
  cases source <;> simp_all [svalValid, svalBits, checked]

@[step] theorem source_file_equality (left right : PtxvSim.ptxv_ir.File) :
    PtxvSim.ptxv_ir.File.Insts.CoreCmpPartialEqFile.eq left right
      ⦃ answer => (answer = true ↔ left = right) ⦄ := by
  cases left <;> cases right <;>
    simp [PtxvSim.ptxv_ir.File.Insts.CoreCmpPartialEqFile.eq, PtxvSim.ptxv_ir.File.read_discriminant]

end Instruction.SegBinding
