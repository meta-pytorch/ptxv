/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Refinement
import Instruction.Bounds

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

@[step] theorem word_count_spec (bytes : U8) :
    register_effect.word_count bytes ⦃ result =>
      result.val = max (bytes.val / 4) 1 ∧ result.val ≤ 255 ⦄ := by
  unfold register_effect.word_count
  simp only [lift, bind_ok]
  step as ⟨quotient, hquotient⟩
  simp only [core.cmp.impls.OrdU16.max]
  split <;> constructor <;> scalar_tac

@[step] theorem store_count_spec (file : File) (bytes elements : U8) (asynchronous : Bool) :
    register_effect.store_count file bytes elements asynchronous ⦃ result =>
      result.val = storeSlots file bytes.val elements.val asynchronous ∧ result.val ≤ 255 ⦄ := by
  have heq : elements ≥ 2#u8 ↔ elements.val ≥ 2 := by scalar_tac
  have hb4eq : bytes > 4#u8 ↔ bytes.val > 4 := by scalar_tac
  have hb8eq : bytes > 8#u8 ↔ bytes.val > 8 := by scalar_tac
  cases file <;> cases asynchronous <;>
    by_cases he : elements.val ≥ 2 <;> by_cases hb4 : bytes.val > 4 <;>
    by_cases hb8 : bytes.val > 8
  all_goals simp [register_effect.store_count, File.Insts.CoreCmpPartialEqFile.eq,
    File.read_discriminant, heq, hb4eq, hb8eq, he, hb4, hb8, storeSlots]
  all_goals first
    | scalar_tac
    | (constructor <;> scalar_tac)
    | (step as ⟨count, hcount, hbound⟩; simp_all [scalarSlots] <;> scalar_tac)

@[step] theorem store_sources_spec (initial : alloc.vec.Vec Reg) (value : Val)
    (bytes : U8) (asynchronous : Bool)
    (hindex : Bounded (storeValueReads asynchronous bytes value))
    (hsize : initial.val.length + (storeValueReads asynchronous bytes value).length ≤ Usize.max) :
    register_effect.store_sources initial value bytes asynchronous ⦃ result =>
      result.val.map key = initial.val.map key ++ storeValueReads asynchronous bytes value ∧
      result.val.length = initial.val.length + (storeValueReads asynchronous bytes value).length ⦄ := by
  unfold register_effect.store_sources
  cases value with
  | Reg first elements =>
    step as ⟨count, hcount, hcountbound⟩
    step with append_spec initial first count
      (by rw [hcount]; exact consecutive_bounded first _ hindex)
      (by simpa [storeValueReads, hcount] using hsize) as ⟨result, hmap, hlength⟩
    simpa [storeValueReads, hcount] using And.intro hmap hlength
  | Imm _ | Const _ _ | SReg _ => simp [storeValueReads]

end Instruction
