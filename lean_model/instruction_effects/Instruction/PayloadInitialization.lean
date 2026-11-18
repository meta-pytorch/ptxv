/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

@[step] theorem payload_initialize_spec (total : U8) (output : alloc.vec.Vec U8) (index : U8)
    (hi : index.val ≤ total.val) (houtput : output.val.map UScalar.val = List.replicate index.val 0) :
    payload.assemble_loop0 total output index ⦃ result =>
      result.val.map UScalar.val = List.replicate total.val 0 ⦄ := by
  unfold payload.assemble_loop0
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U8 × U8 => total.val - state.2.val)
    (fun state : alloc.vec.Vec U8 × U8 => state.2.val ≤ total.val ∧
      state.1.val.map UScalar.val = List.replicate state.2.val 0)
  · rintro ⟨current, byte⟩ ⟨hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold payload.assemble_loop0.body
    by_cases hlt : byte < total
    · simp only [hlt, ite_true]
      have hlength : current.val.length = byte.val := by
        have := congrArg List.length hcurrent
        simpa using this
      step with alloc.vec.Vec.push_spec current 0#u8 (by scalar_tac) as ⟨nextOutput, hnextOutput⟩
      step as ⟨nextByte, hnextByte⟩; try scalar_tac
      refine ⟨?_, ?_, ?_⟩
      · scalar_tac
      · rw [hnextOutput, List.map_append, hcurrent]
        have hn : nextByte.val = byte.val + 1 := by scalar_tac
        simp [hn, List.replicate_succ']
      · scalar_tac
    · simp only [hlt, ite_false, WP.spec_ok]
      have heq : byte.val = total.val := by scalar_tac
      simpa [heq] using hcurrent
  · exact ⟨hi, houtput⟩

end Instruction
