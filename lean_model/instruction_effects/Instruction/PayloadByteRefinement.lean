/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadModel
import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

theorem payload_byte_cast (value : U64) :
    (UScalar.cast .U8 value).val = value.val % 256 := by
  simp only [UScalar.cast, BitVec.truncate_eq_setWidth, UScalar.val, BitVec.toNat_setWidth]
  rfl

@[step] theorem payload_write_window_spec (initial : alloc.vec.Vec U8)
    (first count : U32) (value : U64) (byte : U32) (output : alloc.vec.Vec U8)
    (hcount : count.val ≤ 8) (hbyte : byte.val ≤ count.val)
    (hlengthBound : initial.val.length ≤ 255)
    (hwithin : ∀ index < count.val, first.val + index < initial.val.length)
    (houtput : output.val.map UScalar.val =
      overwriteBytes (initial.val.map UScalar.val) first.val byte.val value.val) :
    payload.assemble_loop1_loop0 output first count value byte ⦃ result =>
      result.val.map UScalar.val =
        overwriteBytes (initial.val.map UScalar.val) first.val count.val value.val ⦄ := by
  unfold payload.assemble_loop1_loop0
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U8 × U32 => count.val - state.2.val)
    (fun state : alloc.vec.Vec U8 × U32 => state.2.val ≤ count.val ∧
      state.1.val.map UScalar.val =
        overwriteBytes (initial.val.map UScalar.val) first.val state.2.val value.val)
  · rintro ⟨current, index⟩ ⟨hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold payload.assemble_loop1_loop0.body
    by_cases hlt : index < count
    · simp only [hlt, ite_true]
      have hindex : index.val < count.val := by scalar_tac
      have hlength : current.val.length = initial.val.length := by
        have := congrArg List.length hcurrent
        simpa using this
      have hdestinationBound := hwithin index.val hindex
      step as ⟨shift, hshift⟩; try scalar_tac
      step as ⟨shifted, hshifted, hshiftedBv⟩; try scalar_tac
      step as ⟨destination, hdestination⟩; try scalar_tac
      try simp only [lift, bind_ok]
      step with alloc.vec.Vec.index_mut_usize_spec current (UScalar.cast .Usize destination)
        (by have := hwithin index.val hindex; scalar_tac)
        as ⟨old, back, hvalue, hback⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨?_, ?_, ?_⟩
      · scalar_tac
      · rw [hback, alloc.vec.Vec.set_val_eq, List.map_set, hcurrent]
        have hd : (UScalar.cast .Usize destination).val = first.val + index.val := by scalar_tac
        have hn : next.val = index.val + 1 := by scalar_tac
        have hs : shifted.val = value.val >>> (8 * index.val) := by
          simpa [hshift] using hshifted
        rw [hd, hn, overwrite_bytes_succ _ _ _ _ (by simpa using hwithin index.val hindex)]
        rw [payload_byte_cast, hs]
        rfl
      · scalar_tac
    · simp only [hlt, ite_false, WP.spec_ok]
      have heq : index.val = count.val := by scalar_tac
      simpa [heq] using hcurrent
  · exact ⟨hbyte, houtput⟩

end Instruction
