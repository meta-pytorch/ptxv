/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Model
import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction

@[simp] theorem empty_register_vector : (alloc.vec.Vec.new Reg).val = [] := rfl

@[step] theorem file_eq (left right : File) :
    File.Insts.CoreCmpPartialEqFile.eq left right
      ⦃ result => result = decide (left = right) ⦄ := by
  cases left <;> cases right <;> simp [File.Insts.CoreCmpPartialEqFile.eq, File.read_discriminant ]

@[step] theorem atom_eq (left right : AtomOp) :
    AtomOp.Insts.CoreCmpPartialEqAtomOp.eq left right
      ⦃ result => result = decide (left = right) ⦄ := by
  cases left <;> cases right <;> simp [AtomOp.Insts.CoreCmpPartialEqAtomOp.eq, AtomOp.read_discriminant ]

@[step] theorem append_spec (initial : alloc.vec.Vec Reg) (first : Reg) (count : U16)
    (hindex : first.idx.val + count.val ≤ 65536)
    (hsize : initial.val.length + count.val ≤ Usize.max) :
    register_effect.append initial first count ⦃ result =>
      result.val.map key = initial.val.map key ++ consecutive first count.val ∧
      result.val.length = initial.val.length + count.val ⦄ := by
  unfold register_effect.append register_effect.append_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec Reg × U16 => count.val - state.2.val)
    (fun state : alloc.vec.Vec Reg × U16 =>
      state.2.val ≤ count.val ∧
      state.1.val.map key = initial.val.map key ++ consecutive first state.2.val ∧
      state.1.val.length = initial.val.length + state.2.val)
  · rintro ⟨result, index⟩ ⟨hbound, hmap, hlength⟩
    dsimp only at hbound hmap hlength
    unfold register_effect.append_loop.body
    by_cases hlt : index.val < count.val
    · have htest : index < count := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨position, hposition⟩; try scalar_tac
      step with alloc.vec.Vec.push_spec result { first with idx := position }
        (by scalar_tac) as ⟨pushed, hpushed⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · rw [hpushed, List.map_append, hmap]
        have hnextval : next.val = index.val + 1 := by scalar_tac
        rw [hnextval, consecutive_succ, List.append_assoc]
        congr 2
        simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true]
        have hpositionval : position.val = first.idx.val + index.val := by scalar_tac
        simp [key, hpositionval]
      · simp [hpushed, hlength]
        scalar_tac
      · scalar_tac
    · have htest : ¬index < count := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have heq : index.val = count.val := by omega
      exact ⟨by simpa [heq] using hmap, by simpa [heq] using hlength⟩
  · simp [consecutive]

@[step] theorem scalar_count_spec (file : File) (count : U8) :
    register_effect.scalar_count file count ⦃ result =>
      result.val = scalarSlots file count.val ∧ result.val ≤ 2 ⦄ := by
  cases file <;> simp [register_effect.scalar_count, File.Insts.CoreCmpPartialEqFile.eq,
    File.read_discriminant, scalarSlots, lift, bind_ok]

@[step] theorem slots_spec (initial : alloc.vec.Vec Reg) (value : Val)
    (hindex : Bounded (operandReads value))
    (hsize : initial.val.length + (operandReads value).length ≤ Usize.max) :
    register_effect.slots initial value ⦃ result =>
      result.val.map key = initial.val.map key ++ operandReads value ∧
      result.val.length = initial.val.length + (operandReads value).length ⦄ := by
  unfold register_effect.slots
  cases value with
  | Reg first elements =>
    step as ⟨count, hcount, hcountbound⟩
    step with append_spec initial first count
      (by rw [hcount]; exact consecutive_bounded first _ hindex)
      (by simpa [operandReads, hcount] using hsize) as ⟨result, hmap, hlength⟩
    simpa [operandReads, hcount] using And.intro hmap hlength
  | Imm _ | Const _ _ | SReg _ => simp [operandReads]

end Instruction
