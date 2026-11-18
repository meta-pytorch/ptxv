/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceByte

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false

namespace MemorySafety.Source

@[step]
theorem source_read_bytes (program : memory_safety.Program) (state : memory_safety.State)
    (request : memory_safety.Request) (valid : Bool) :
    memory_safety.access_valid_loop2 program.allocations program.threads state.pcs state.live state.writes
      state.history state.bad request.address request.width valid 0#u64
      ⦃ result => result = true → valid = true ∧
        Initialized (programModel program) (stateModel state) (requestModel request) ⦄ := by
  unfold memory_safety.access_valid_loop2
  apply loop.spec_decr_nat
    (fun current : Bool × U64 => request.width.val - current.2.val)
    (fun current : Bool × U64 => current.2.val ≤ request.width.val ∧
      (current.1 = true → valid = true ∧
        ∀ offset, offset < current.2.val →
          byteDefined (programModel program) (stateModel state) (request.address.val + offset) = true))
  · intro current invariant
    rcases current with ⟨accepted, cursor⟩
    obtain ⟨cursorBound, checkedBytes⟩ := invariant
    dsimp only at cursorBound checkedBytes
    unfold memory_safety.access_valid_loop2.body
    by_cases inside : cursor.val < request.width.val
    · have sourceInside : cursor < request.width := by scalar_tac
      simp only [sourceInside, ite_true]
      cases accepted with
      | false => simp
      | true =>
        simp only [ite_true]
        step as ⟨sum, sumSpec⟩
        cases sum with
        | none =>
          simp only [bind_ok]
          step as ⟨next, nextEq⟩
          exact ⟨by scalar_tac, by simp, by scalar_tac⟩
        | some address =>
          step with source_byte_defined program state address as ⟨defined, definedEq⟩
          step as ⟨next, nextEq⟩
          refine ⟨by scalar_tac, ?_, by scalar_tac⟩
          intro nowDefined
          refine ⟨(checkedBytes rfl).1, ?_⟩
          intro offset before
          by_cases earlier : offset < cursor.val
          · exact (checkedBytes rfl).2 offset earlier
          · have sameOffset : offset = cursor.val := by scalar_tac
            subst offset
            have addressEq : address.val = request.address.val + cursor.val := sumSpec.2.1
            simpa [←addressEq, definedEq] using nowDefined
    · have sourceOutside : ¬cursor < request.width := by scalar_tac
      simp only [sourceOutside, ite_false, WP.spec_ok]
      have cursorEnd : cursor.val = request.width.val := by scalar_tac
      intro accepted
      refine ⟨(checkedBytes accepted).1, ?_⟩
      intro offset member
      exact (checkedBytes accepted).2 offset (by simpa [cursorEnd, requestModel] using List.mem_range.mp member)
  · refine ⟨by scalar_tac, ?_⟩
    intro accepted
    refine ⟨accepted, ?_⟩
    intro offset before
    scalar_tac

end MemorySafety.Source
