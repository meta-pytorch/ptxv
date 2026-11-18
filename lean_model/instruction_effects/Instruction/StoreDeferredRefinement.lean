/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreRegisterMembership
import Instruction.StoreSourceRoles

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

def windowSource (window : Reg × U32 × U32) : Key := key window.1

def filterWindowSources (windows : List (Reg × U32 × U32)) (control : List Key) : List Key :=
  (windows.map windowSource).filter (fun reg => decide (reg ∉ control))

theorem filter_window_sources_succ (windows : List (Reg × U32 × U32)) (control : List Key)
    (index : Nat) (hi : index < windows.length) :
    filterWindowSources (windows.take (index + 1)) control =
      filterWindowSources (windows.take index) control ++
        (if windowSource windows[index] ∈ control then [] else [windowSource windows[index]]) := by
  unfold filterWindowSources
  rw [List.take_succ_eq_append_getElem hi, List.map_append, List.filter_append]
  by_cases hc : windowSource windows[index] ∈ control <;> simp [hc]

@[step] theorem deferred_registers_spec (windows : Slice (Reg × U32 × U32)) (control : Slice Reg) :
    store_window.deferred_registers windows control ⦃ result =>
      result.val.map key = filterWindowSources windows.val (control.val.map key) ∧
      result.val.length ≤ windows.val.length ⦄ := by
  unfold store_window.deferred_registers store_window.deferred_registers_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec Reg × Usize => windows.val.length - state.2.val)
    (fun state : alloc.vec.Vec Reg × Usize =>
      state.2.val ≤ windows.val.length ∧
      state.1.val.map key = filterWindowSources (windows.val.take state.2.val) (control.val.map key) ∧
      state.1.val.length ≤ state.2.val)
  · rintro ⟨result, index⟩ ⟨hbound, hmap, hlength⟩
    dsimp only at hbound hmap hlength
    unfold store_window.deferred_registers_loop.body
    by_cases hlt : index.val < windows.val.length
    · have htest : index < Slice.len windows := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨register, first, count, hwindow⟩
      have hentry : (register, first, count) = windows.val[index.val]'hlt := hwindow
      step with has_register_spec control register as ⟨found, hfound⟩
      have hselected :
          (if found then ok result else alloc.vec.Vec.push result register) ⦃ selected =>
            selected.val.map key = result.val.map key ++
              (if key register ∈ control.val.map key then [] else [key register]) ∧
            selected.val.length ≤ index.val + 1 ⦄ := by
        by_cases hc : key register ∈ control.val.map key
        · simp only [hfound, hc, decide_true, ite_true, List.append_nil, WP.spec_ok, true_and]
          omega
        · simp only [hfound, hc, decide_false, Bool.false_eq_true, ite_false]
          step with alloc.vec.Vec.push_spec result register (by scalar_tac) as ⟨pushed, hpushed⟩
          simp [hpushed]
          omega
      step with hselected as ⟨selected, hselectedMap, hselectedLength⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · have hn : next.val = index.val + 1 := by scalar_tac
        rw [hn, hselectedMap, hmap, filter_window_sources_succ windows.val (control.val.map key) index.val hlt]
        have hr : windowSource (windows.val[index.val]'hlt) = key register := by
          rw [← hentry]
          rfl
        rw [hr]
      · scalar_tac
      · scalar_tac
    · have htest : ¬index < Slice.len windows := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have hi : index.val = windows.val.length := by omega
      exact ⟨by simpa [hi] using hmap, by omega⟩
  · simp [filterWindowSources, empty_register_vector]

end Instruction
