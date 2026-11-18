/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.BarrierSeed

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.BarrierSeed

def Rows (code : List Inst) (flags : List U32) : Prop :=
  List.Forall₂ (fun instruction word => SeedPost instruction.op word) code flags

theorem Rows.length {code : List Inst} {flags : List U32} (rows : Rows code flags) :
    code.length = flags.length := rows.length_eq

theorem Rows.at {code : List Inst} {flags : List U32} (rows : Rows code flags)
    (pc : Nat) (inside : pc < code.length) : SeedPost code[pc].op flags[pc]! := by
  have flagInside : pc < flags.length := by have := rows.length; omega
  simpa only [List.get_eq_getElem, List.Inhabited_getElem_eq_getElem! _ _ flagInside] using
    rows.get inside flagInside

@[step]
theorem rust_barrier_seeds (code : Slice Inst) :
    control_flow.barrier_seeds code ⦃ flags => Rows code.val flags.val ⦄ := by
  unfold control_flow.barrier_seeds control_flow.barrier_seeds_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U32 × Usize => code.val.length - state.2.val)
    (fun state => state.2.val ≤ code.val.length ∧ Rows (code.val.take state.2.val) state.1.val)
  · rintro ⟨flags, pc⟩ ⟨pcBound, prefixRows⟩
    dsimp only at pcBound prefixRows
    have flagLength : flags.val.length = pc.val := by
      have lengths := prefixRows.length
      simp only [List.length_take, Nat.min_eq_left pcBound] at lengths
      exact lengths.symm
    unfold control_flow.barrier_seeds_loop.body
    by_cases inside : pc.val < code.val.length
    · have test : pc < Slice.len code := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec code pc inside as ⟨instruction, instructionEq⟩
      step with rust_barrier_seed instruction.op as ⟨word, wordPost⟩
      step with alloc.vec.Vec.push_spec flags word (by have := code.property; omega)
        as ⟨updated, updatedValue⟩
      step with Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac) as ⟨next, nextValue⟩
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [nextValue, updatedValue, List.take_succ_eq_append_getElem inside]
      apply List.rel_append prefixRows
      exact .cons (by simpa only [instructionEq] using wordPost) .nil
    · have test : ¬pc < Slice.len code := by scalar_tac
      have complete : pc.val = code.val.length := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [complete, List.take_length] using prefixRows
  · exact ⟨by scalar_tac, List.Forall₂.nil⟩

end Instruction.BarrierSeed
