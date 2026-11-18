/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchModel
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem source_aliases_loop (aliases : Slice Bool) (found : Bool) (index : Usize)
    (computed : strict_memory_launch.aliases_present_loop aliases found index = .ok false) :
    found = false ∧ ∀ selectedAlias ∈ aliases.val.drop index.val, selectedAlias = false := by
  generalize fuelEq : aliases.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing found index with
  | h fuel ih =>
    have unroll : strict_memory_launch.aliases_present_loop aliases found index = (do
        let result ← strict_memory_launch.aliases_present_loop.body aliases found index
        match result with
        | .cont pair => strict_memory_launch.aliases_present_loop aliases pair.1 pair.2
        | .done result => .ok result) := by
      unfold strict_memory_launch.aliases_present_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done result => rfl
    rw [unroll] at computed
    by_cases inside : index.val < aliases.val.length
    · have test : index < Slice.len aliases := by scalar_tac
      have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq aliases
        scalar_tac
      let after : Usize := Usize.ofNatCore (index.val + 1) bound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := source_usize_increment index bound
      simp only [strict_memory_launch.aliases_present_loop.body, test, ite_true] at computed
      cases found with
      | true =>
        simp only [ite_true, bind_ok, increment] at computed
        have tail := ih (aliases.val.length - after.val) (by omega) true after computed rfl
        cases tail.1
      | false =>
        have read : Slice.index_usize aliases index = .ok aliases.val[index.val] := by
          simp [Slice.index_usize, List.getElem?_eq_getElem inside]
        simp only [Bool.false_eq_true, ite_false, read, bind_ok, increment] at computed
        have tail := ih (aliases.val.length - after.val) (by omega) aliases.val[index.val] after computed rfl
        refine ⟨rfl, ?_⟩
        rw [List.drop_eq_getElem_cons inside]
        intro selectedAlias member
        simp only [List.mem_cons] at member
        rcases member with equal | member
        · subst selectedAlias
          exact tail.1
        · apply tail.2 selectedAlias
          simpa only [afterVal] using member
    · have test : ¬index < Slice.len aliases := by scalar_tac
      simp only [strict_memory_launch.aliases_present_loop.body, test, ite_false, bind_ok,
        ok.injEq] at computed
      refine ⟨computed, ?_⟩
      simp [List.drop_eq_nil_of_le (show aliases.val.length ≤ index.val by omega)]

theorem source_aliases_present_false (aliases : Slice Bool)
    (computed : strict_memory_launch.aliases_present aliases = .ok false) :
    ∀ selectedAlias ∈ aliases.val, selectedAlias = false := by
  have remaining := (source_aliases_loop aliases false 0#usize computed).2
  simpa using remaining

end MemorySafetyIR
