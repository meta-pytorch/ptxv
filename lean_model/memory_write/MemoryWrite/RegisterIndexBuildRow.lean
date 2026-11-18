/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexBuildModel

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

theorem successful_build_row (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (pc : Usize) (before after : register_index.RegisterIndex)
    (row result : alloc.vec.Vec Usize) (position : Usize)
    (pcBound : pc.val < destinations.val.length) (valid : Valid before)
    (initialPrefix : RowPrefix before (destinations.val[pc.val]!).val row.val position.val)
    (success : register_index.build_writes_loop0_loop0 destinations before pc row position = .ok (after, result)) :
    Valid after ∧ Preserves before after ∧ RowRepresents after (destinations.val[pc.val]!).val result.val := by
  unfold register_index.build_writes_loop0_loop0 at success
  have execution := MemoryWrite.successful_loop_has_execution _ _ _ success
  apply execution.preserves
    (fun state : register_index.RegisterIndex × alloc.vec.Vec Usize × Usize =>
      Valid state.1 ∧ Preserves before state.1 ∧
        RowPrefix state.1 (destinations.val[pc.val]!).val state.2.1.val state.2.2.val)
    (fun state : register_index.RegisterIndex × alloc.vec.Vec Usize =>
      Valid state.1 ∧ Preserves before state.1 ∧ RowRepresents state.1 (destinations.val[pc.val]!).val state.2.val)
    ⟨valid, Preserves.refl before, initialPrefix⟩
  rintro ⟨current, currentRow, currentPosition⟩ ⟨currentValid, grows, currentPrefix⟩ flow bodyOk
  change RowPrefix current (destinations.val[pc.val]!).val currentRow.val currentPosition.val at currentPrefix
  unfold register_index.build_writes_loop0_loop0.body at bodyOk
  obtain ⟨registers, registerOk, testOk⟩ := MemoryWrite.bind_success _ _ _ bodyOk
  have registerSpec := Slice.index_usize_spec destinations pc pcBound
  simp only [registerOk, WP.spec_ok] at registerSpec
  have registersAt : registers = destinations.val[pc.val]! := by
    rw [registerSpec, List.Inhabited_getElem_eq_getElem! _ _ pcBound]
  by_cases inside : currentPosition.val < registers.val.length
  · have test : currentPosition < alloc.vec.Vec.len registers := by scalar_tac
    simp only [test, ite_true] at testOk
    obtain ⟨key, keyOk, internOk⟩ := MemoryWrite.bind_success _ _ _ testOk
    have keySpec := rust_vec_index registers currentPosition inside
    simp only [keyOk, WP.spec_ok] at keySpec
    have keyAt : key = (destinations.val[pc.val]!).val[currentPosition.val]! := by
      rw [keySpec, List.Inhabited_getElem_eq_getElem! _ _ inside, registersAt]
    obtain ⟨internResult, internedOk, pushOk⟩ := MemoryWrite.bind_success _ _ _ internOk
    rcases internResult with ⟨id, updated⟩
    have effect := successful_intern current updated key id currentValid internedOk
    obtain ⟨appended, appendedOk, incrementOk⟩ := MemoryWrite.bind_success _ _ _ pushOk
    have appendedValue := MemoryWrite.successful_vec_push currentRow id appended appendedOk
    obtain ⟨next, nextOk, returned⟩ := MemoryWrite.bind_success _ _ _ incrementOk
    have rawNextValue := MemoryWrite.successful_add currentPosition 1#usize next nextOk
    have nextValue : next.val = currentPosition.val + 1 := by scalar_tac
    have equal := Result.ok_injective returned
    cases equal
    refine ⟨effect.1, grows.trans effect.preserves, ?_⟩
    change RowPrefix updated (destinations.val[pc.val]!).val appended.val next.val
    have rowEffect := row_prefix_append current updated (destinations.val[pc.val]!).val
      currentRow.val currentPosition.val id currentPrefix (by simpa only [registersAt] using inside)
      (by simpa only [keyAt] using effect)
    simpa only [appendedValue, nextValue] using rowEffect
  · have test : ¬currentPosition < alloc.vec.Vec.len registers := by scalar_tac
    simp only [test, ite_false] at testOk
    have equal := Result.ok_injective testOk
    cases equal
    have finished : currentPosition.val = (destinations.val[pc.val]!).val.length := by
      have prefixBound := currentPrefix.1
      rw [registersAt] at inside
      omega
    exact ⟨currentValid, grows, currentPrefix.2.1.trans finished,
      fun earlier bound => currentPrefix.2.2 earlier (by omega)⟩

end MemoryWrite.RegisterIndex
