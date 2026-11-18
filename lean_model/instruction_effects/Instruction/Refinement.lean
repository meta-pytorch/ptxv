/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SpanRefinement

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction

@[step] theorem vector_spec (initial : alloc.vec.Vec Reg) (value : Val)
    (hindex : Bounded (vectorReads value))
    (hsize : initial.val.length + (vectorReads value).length ≤ Usize.max) :
    register_effect.vector initial value ⦃ result =>
      result.val.map key = initial.val.map key ++ vectorReads value ∧
      result.val.length = initial.val.length + (vectorReads value).length ⦄ := by
  unfold register_effect.vector
  cases value with
  | Reg first count =>
    step as ⟨virtual, hvirtual⟩
    rw [hvirtual]
    simp only [decide_eq_true_eq]
    split
    · rename_i hf
      simp only [lift, bind_ok]
      let width := core.cmp.impls.OrdU16.max (UScalar.cast .U16 count) 1#u16
      have hwidth : width.val = max count.val 1 := by
        simp only [width, core.cmp.impls.OrdU16.max]
        split <;> scalar_tac
      step with append_spec initial first width
        (by rw [hwidth]; exact consecutive_bounded first _ (by simpa [vectorReads, hf] using hindex))
        (by simpa [vectorReads, hf, hwidth] using hsize) as ⟨result, hmap, hlength⟩
      simpa [vectorReads, hf, hwidth] using And.intro hmap hlength
    · rename_i hf
      step with slots_spec initial (.Reg first count)
        (by simpa [operandReads, vectorReads, hf] using hindex)
        (by simpa [operandReads, vectorReads, hf] using hsize) as ⟨result, hmap, hlength⟩
      simpa [operandReads, vectorReads, hf] using And.intro hmap hlength
  | Imm _ | Const _ _ | SReg _ => simp [register_effect.slots, operandReads, vectorReads]

@[step] theorem load_count_spec (file : File) (bytes elements : U8) :
    register_effect.load_count file bytes elements ⦃ result =>
      result.val = (if file = .V then
        if elements.val > 1 then elements.val else max (bytes.val / 8) 1
        else max (bytes.val / 4) 1) ∧ result.val ≤ 255 ⦄ := by
  unfold register_effect.load_count
  step as ⟨virtual, hvirtual⟩
  rw [hvirtual]
  simp only [decide_eq_true_eq]
  split
  · rename_i hf
    split
    · rename_i hn
      simp only [WP.spec_ok]
      constructor <;> scalar_tac
    · rename_i hn
      have hnval : ¬ elements.val > 1 := by scalar_tac
      simp only [hnval, ite_false, lift, bind_ok]
      step as ⟨quotient, hquotient⟩
      simp only [WP.spec_ok, core.cmp.impls.OrdU16.max]
      split <;> constructor <;> scalar_tac
  · rename_i hf
    simp only [hf, ite_false, lift, bind_ok]
    step as ⟨quotient, hquotient⟩
    simp only [WP.spec_ok, core.cmp.impls.OrdU16.max]
    split <;> constructor <;> scalar_tac

@[step] theorem load_writes_spec (destination : Reg) (elements bytes : U8)
    (space : Space) (address : Addr) (sync : Option MemSync) (signed nc : Bool)
    (hindex : Bounded (loadWrites destination bytes elements)) :
    register_effect.memory_writes (.Ld destination elements space bytes address sync signed nc)
      ⦃ result => ∃ registers, result = some registers ∧
        registers.val.map key = loadWrites destination bytes elements ⦄ := by
  simp only [register_effect.memory_writes]
  step as ⟨count, hcount, hbound⟩
  step with append_spec (alloc.vec.Vec.new Reg) destination count
    (by rw [hcount]; exact consecutive_bounded destination _ hindex)
    (by simp only [alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val]; scalar_tac)
    as ⟨registers, hmap, hlength⟩
  refine ⟨registers, rfl, ?_⟩
  simpa [hcount, loadWrites, alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from] using hmap

end Instruction
