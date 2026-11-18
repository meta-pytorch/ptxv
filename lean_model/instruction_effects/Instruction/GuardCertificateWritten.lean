/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateLookup

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.GuardCertificate

@[step]
theorem rust_written (register : Reg) (writes : alloc.vec.Vec Reg) :
    guard_certificate.post_value_loop register writes 0#usize false
      ⦃ result => result = decide (register ∈ writes.val) ⦄ := by
  unfold guard_certificate.post_value_loop
  apply loop.spec_decr_nat
    (fun state : Usize × Bool => writes.val.length - state.1.val)
    (fun state : Usize × Bool => state.1.val ≤ writes.val.length ∧
      state.2 = decide (register ∈ writes.val.take state.1.val))
  · intro state invariant
    obtain ⟨bound, meaning⟩ := invariant
    obtain ⟨position, written⟩ := state
    unfold guard_certificate.post_value_loop.body
    dsimp only at *
    split
    · rename_i test
      have inside : position.val < writes.val.length := by
        simpa only [UScalar.lt_equiv, alloc.vec.Vec.len_val, alloc.vec.Vec.length] using test
      step with alloc.vec.Vec.index_usize_spec writes position inside as ⟨current, currentAt⟩
      rw [source_same_register]
      simp only [bind_ok]
      split
      · rename_i same
        simp only [decide_eq_true_eq] at same
        step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
          as ⟨next, nextAt⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        have nextValue : next.val = position.val + 1 := by scalar_tac
        rw [nextValue, List.take_succ_eq_append_getElem inside, ←currentAt]
        simp [same]
      · rename_i different
        simp only [decide_eq_true_eq] at different
        step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
          as ⟨next, nextAt⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        have nextValue : next.val = position.val + 1 := by scalar_tac
        rw [nextValue, List.take_succ_eq_append_getElem inside, ←currentAt]
        simpa [Ne.symm different] using meaning
    · rename_i outside
      have complete : position.val = writes.val.length := by
        have lower : writes.val.length ≤ position.val := by
          have outside' : ¬position.val < writes.val.length := by
            simpa only [UScalar.lt_equiv, alloc.vec.Vec.len_val, alloc.vec.Vec.length] using outside
          omega
        omega
      simpa only [WP.spec_ok, complete, List.take_length] using meaning
  · simp

end Instruction.GuardCertificate
