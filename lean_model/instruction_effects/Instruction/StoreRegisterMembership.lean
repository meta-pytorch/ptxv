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

theorem key_eq_iff (left right : Reg) :
    key left = key right ↔ left.file = right.file ∧ left.idx = right.idx := by
  simp only [key, Key.mk.injEq]
  constructor
  · rintro ⟨hf, hi⟩
    exact ⟨hf, by scalar_tac⟩
  · rintro ⟨hf, hi⟩
    simp [hf, hi]

theorem prefix_membership_step (registers : List Reg) (needle : Reg) (index : Nat)
    (hindex : index < registers.length) :
    decide (key needle ∈ (registers.take (index + 1)).map key) =
      (decide (key needle ∈ (registers.take index).map key) ||
        decide (key registers[index] = key needle)) := by
  rw [List.take_succ_eq_append_getElem hindex]
  simp only [List.map_append, List.map_cons, List.map_nil, List.mem_append,
    List.mem_cons, List.not_mem_nil, or_false]
  by_cases hp : key needle ∈ (registers.take index).map key <;>
    by_cases he : key registers[index] = key needle <;> simp_all [eq_comm]

@[step] theorem has_register_spec (registers : Slice Reg) (needle : Reg) :
    store_window.has_register registers needle ⦃ result =>
      result = decide (key needle ∈ registers.val.map key) ⦄ := by
  unfold store_window.has_register store_window.has_register_loop
  apply loop.spec_decr_nat
    (fun state : Reg × Bool × Usize => registers.val.length - state.2.2.val)
    (fun state : Reg × Bool × Usize =>
      state.1 = needle ∧ state.2.2.val ≤ registers.val.length ∧
      state.2.1 = decide (key needle ∈ (registers.val.take state.2.2.val).map key))
  · rintro ⟨current, found, index⟩ ⟨hneedle, hbound, hfound⟩
    dsimp only at hneedle hbound hfound
    subst current
    unfold store_window.has_register_loop.body
    by_cases hlt : index.val < registers.val.length
    · have htest : index < Slice.len registers := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨candidate, hcandidate⟩
      have hmatch :
          (if found then ok (needle, true) else do
            let sameFile ← File.Insts.CoreCmpPartialEqFile.eq candidate.file needle.file
            let sameIndex ← if sameFile then ok (decide (candidate.idx = needle.idx)) else ok false
            ok (needle, sameIndex)) ⦃ pair =>
              pair.1 = needle ∧ pair.2 =
                (found || decide (key candidate = key needle)) ⦄ := by
        cases found
        · simp only [Bool.false_eq_true, ite_false, Bool.false_or]
          step with file_eq candidate.file needle.file as ⟨sameFile, hfile⟩
          rw [hfile]
          by_cases hf : candidate.file = needle.file
          · simp [hf, key_eq_iff]
          · simp [hf, key_eq_iff]
        · simp
      step with hmatch as ⟨nextNeedle, nextFound, hfirst, hflag⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨hfirst, ?_, ?_, ?_⟩
      · scalar_tac
      · have hn : next.val = index.val + 1 := by scalar_tac
        rw [hn, prefix_membership_step registers.val needle index.val hlt]
        rw [hflag, hfound, hcandidate]
      · scalar_tac
    · have htest : ¬index < Slice.len registers := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have hi : index.val = registers.val.length := by omega
      simpa [hi] using hfound
  · simp

end Instruction
