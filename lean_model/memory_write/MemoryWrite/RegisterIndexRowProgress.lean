/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexInternTotal

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

@[step]
theorem rust_build_row_returns (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (pc : Usize) (before : register_index.RegisterIndex)
    (pcBound : pc.val < destinations.val.length) (valid : Valid before)
    (countFits : before.count.val + (destinations.val[pc.val]!).val.length ≤ Usize.max)
    (pagesFit : before.pages.val.length + (destinations.val[pc.val]!).val.length ≤ Usize.max) :
    register_index.build_writes_loop0_loop0 destinations before pc (alloc.vec.Vec.new Usize) 0#usize
      ⦃ result => Valid result.1 ∧
        result.1.count.val ≤ before.count.val + (destinations.val[pc.val]!).val.length ∧
        result.1.pages.val.length ≤ before.pages.val.length + (destinations.val[pc.val]!).val.length ⦄ := by
  unfold register_index.build_writes_loop0_loop0
  apply loop.spec_decr_nat
    (fun state : register_index.RegisterIndex × alloc.vec.Vec Usize × Usize =>
      (destinations.val[pc.val]!).val.length - state.2.2.val)
    (fun state : register_index.RegisterIndex × alloc.vec.Vec Usize × Usize =>
      Valid state.1 ∧ state.2.2.val ≤ (destinations.val[pc.val]!).val.length ∧
      state.2.1.val.length = state.2.2.val ∧
      state.1.count.val ≤ before.count.val + state.2.2.val ∧
      state.1.pages.val.length ≤ before.pages.val.length + state.2.2.val)
  · rintro ⟨current, row, position⟩ ⟨currentValid, positionBound, rowLength, countBound, pageBound⟩
    dsimp only at positionBound rowLength countBound pageBound
    unfold register_index.build_writes_loop0_loop0.body
    step with Slice.index_usize_spec destinations pc pcBound as ⟨registers, registersEq⟩
    have registersAt : registers = destinations.val[pc.val]! := by
      rw [registersEq, List.Inhabited_getElem_eq_getElem! _ _ pcBound]
    by_cases inside : position.val < registers.val.length
    · have test : position < alloc.vec.Vec.len registers := by scalar_tac
      have insideSource : position.val < (destinations.val[pc.val]!).val.length := by
        simpa only [registersAt] using inside
      simp only [test, ite_true]
      step with alloc.vec.Vec.index_usize_spec registers position inside as ⟨key, keyEq⟩
      step with rust_intern_total current key currentValid (by omega) (by omega)
        as ⟨id, updated, interned, nextCount, nextPages⟩
      step with alloc.vec.Vec.push_spec row id (by scalar_tac) as ⟨appended, appendedEq⟩
      step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
        as ⟨next, nextValue⟩
      refine ⟨interned.1, by scalar_tac, ?_, by scalar_tac, by scalar_tac, by scalar_tac⟩
      · simp only [appendedEq, List.length_append, List.length_singleton]
        scalar_tac
    · have test : ¬position < alloc.vec.Vec.len registers := by scalar_tac
      have finished : position.val = (destinations.val[pc.val]!).val.length := by
        rw [registersAt] at inside
        omega
      simp only [test, ite_false, WP.spec_ok]
      exact ⟨currentValid, by omega, by omega⟩
  · exact ⟨valid, by scalar_tac, by simp, by scalar_tac, by scalar_tac⟩


end MemoryWrite.RegisterIndex
