/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MaskedLoadModel
import Instruction.MaskedLoadWord
import Instruction.MaskedLoadBit

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

def LoadWriterContract {Writer State : Type} (writerInst : masked_load.RegisterWriter Writer)
    (observe : Writer → State) (valid : Writer → Prop)
    (transition : State → LoadObservation → State)
    (actions : List LoadObservation) : Prop :=
  ∀ writer, valid writer → ∀ destination value sourceByte,
    observedLoadWrite destination value sourceByte ∈ actions →
    writerInst.write writer destination value sourceByte ⦃ updated =>
      valid updated ∧ observe updated = transition (observe writer) (observedLoadWrite destination value sourceByte) ⦄

@[step] theorem masked_load_half_loop_spec {Writer State : Type}
    (writerInst : masked_load.RegisterWriter Writer) (first : Reg) (mask start : U8)
    (data : Array U8 32#usize) (writer : Writer) (rank : U16) (index : U8)
    (initial : State) (observe : Writer → State) (valid : Writer → Prop)
    (transition : State → LoadObservation → State) (hvalid : valid writer)
    (hstart : start.val + 4 ≤ 8) (hindex : index.val ≤ 4)
    (hrank : rank.val = (selectedLoadIndices mask start.val index.val).length)
    (hbounded : Bounded (consecutive first (halfWords mask start.val).length))
    (hwrite : LoadWriterContract writerInst observe valid transition
      (selectedLoadHalf first mask start.val (loadedMemoryWord data)))
    (hstate : observe writer = (selectedLoadPrefix first mask start.val (loadedMemoryWord data)
      index.val).foldl transition initial) :
    masked_load.half_loop writerInst first mask start data writer rank index ⦃ updated =>
      valid updated ∧ observe updated = (selectedLoadHalf first mask start.val (loadedMemoryWord data)).foldl
        transition initial ⦄ := by
  unfold masked_load.half_loop
  apply loop.spec_decr_nat
    (fun state : Reg × Writer × U16 × U8 => 4 - state.2.2.2.val)
    (fun state : Reg × Writer × U16 × U8 => state.1 = first ∧ valid state.2.1 ∧ state.2.2.2.val ≤ 4 ∧
      state.2.2.1.val = (selectedLoadIndices mask start.val state.2.2.2.val).length ∧
      observe state.2.1 = (selectedLoadPrefix first mask start.val (loadedMemoryWord data)
        state.2.2.2.val).foldl transition initial)
  · rintro ⟨currentFirst, current, rank, index⟩ ⟨hfirst, hcurrentValid, hbound, hrank, hcurrent⟩
    dsimp only at hfirst hbound hrank hcurrent
    subst currentFirst
    unfold masked_load.half_loop.body
    by_cases hlt : index.val < 4
    · have htest : index < 4#u8 := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨source, hsource⟩; try scalar_tac
      have hsourceval : source.val = start.val + index.val := by scalar_tac
      have hsourcelt : source.val < 8 := by omega
      step as ⟨shifted, hshiftval, hshiftbv⟩; try scalar_tac
      simp only [lift, bind_ok]
      rw [masked_load_bit_selected mask shifted source.val hshiftbv, hsourceval]
      cases hb : mask.bv.getLsbD (start.val + index.val) with
      | false =>
        simp only [Bool.false_eq_true, ite_false, bind_ok]
        step as ⟨nextIndex, hnextIndex⟩; try scalar_tac
        have hn : nextIndex.val = index.val + 1 := by scalar_tac
        refine ⟨hcurrentValid, ?_, ?_, ?_, ?_⟩
        · scalar_tac
        · simpa [hn, selected_load_indices_succ, hb] using hrank
        · simpa [hn, selected_load_prefix_succ, hb] using hcurrent
        · scalar_tac
      | true =>
        simp only [ite_true]
        have hranklt := selected_load_rank_bound mask start.val index.val hlt hb
        have hrankbound : rank.val < 4 := by
          have := selected_load_indices_length mask start.val index.val
          omega
        have hregister : first.idx.val + rank.val < 65536 := by
          apply hbounded ⟨first.file, first.idx.val + rank.val⟩
          apply List.mem_map.mpr
          refine ⟨rank.val, ?_, rfl⟩
          simpa [hrank] using hranklt
        step as ⟨destination, hdestination⟩; try scalar_tac
        step as ⟨nextRank, hnextRank⟩; try scalar_tac
        step with masked_load_word_spec data source hsourcelt as ⟨value, hvalue⟩
        step as ⟨sourceByte, hsourceByte⟩; try scalar_tac
        let action : LoadObservation :=
          ⟨⟨first.file, first.idx.val + (selectedLoadIndices mask start.val index.val).length⟩,
            4 * (start.val + index.val), loadedMemoryWord data (start.val + index.val)⟩
        have haction : observedLoadWrite { first with idx := destination } value sourceByte = action := by
          have hd : destination.val = first.idx.val + rank.val := by scalar_tac
          have hbval : sourceByte.val = 4 * (start.val + index.val) := by scalar_tac
          simp only [observedLoadWrite, key, hd, hbval, hvalue, hsourceval, hrank, action]
        have hmember : action ∈ selectedLoadHalf first mask start.val (loadedMemoryWord data) := by
          apply selected_load_prefix_subset first mask start.val (index.val + 1)
            (loadedMemoryWord data) (by omega) action
          rw [selected_load_prefix_succ, hb]
          simp only [ite_true, List.mem_append, List.mem_singleton]
          exact Or.inr rfl
        step with hwrite current hcurrentValid { first with idx := destination } value sourceByte
          (by rw [haction]; exact hmember) as ⟨updated, hupdatedValid, hupdated⟩
        rw [haction, hcurrent] at hupdated
        step as ⟨nextIndex, hnextIndex⟩; try scalar_tac
        have hn : nextIndex.val = index.val + 1 := by scalar_tac
        refine ⟨hupdatedValid, ?_, ?_, ?_, ?_⟩
        · scalar_tac
        · rw [hn, selected_load_indices_succ, hb]
          simp only [ite_true, List.length_append, List.length_singleton]
          scalar_tac
        · rw [hn, selected_load_prefix_succ, hb]
          simpa only [ite_true, List.foldl_append, List.foldl_cons, List.foldl_nil, action] using hupdated
        · scalar_tac
    · have htest : ¬index < 4#u8 := by scalar_tac
      simp only [htest, ite_false]
      have hi : index.val = 4 := by omega
      simpa [hi, selectedLoadHalf] using And.intro hcurrentValid hcurrent
  · exact ⟨rfl, hvalid, hindex, hrank, hstate⟩

end Instruction
