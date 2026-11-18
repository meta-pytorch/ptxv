/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MetadataDeliveryLoops

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

def destinationUpdateCommands (destination : Reg) (touched active guarded : List Usize) :
    List MetadataCommand :=
  clear_bitsCommands destination touched active guarded ++
  active_taintCommands destination active active guarded ++
  guard_taintCommands destination guarded active guarded

def historiesPrefix (destinations : List Reg) (touched active guarded : List Usize) (count : Nat) :
    List MetadataCommand :=
  (destinations.take count).flatMap fun destination => historyCommands destination touched active guarded

@[step] theorem metadata_histories_loop_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destinations : Slice Reg)
    (touched active guarded : Slice Usize) (writer : Writer) (index : Usize)
    (initial : State) (observe : Writer → State) (valid : Writer → Prop)
    (semantics : MetadataSemantics State) (hvalid : valid writer)
    (hcontract : MetadataWriterContract writerInst observe valid semantics)
    (hindex : index.val ≤ destinations.val.length)
    (hstate : observe writer = ( historiesPrefix destinations.val touched.val active.val guarded.val index.val).foldl
      (metadataStep semantics) initial) :
    metadata_delivery.histories_loop writerInst destinations touched active guarded writer index ⦃ updated =>
      valid updated ∧ observe updated = (historiesPrefix destinations.val touched.val active.val guarded.val
        destinations.val.length).foldl (metadataStep semantics) initial ⦄ := by
  unfold metadata_delivery.histories_loop
  apply loop.spec_decr_nat
    (fun state : Writer × Usize => destinations.val.length - state.2.val)
    (fun state : Writer × Usize => valid state.1 ∧ state.2.val ≤ destinations.val.length ∧
      observe state.1 = (historiesPrefix destinations.val touched.val active.val guarded.val state.2.val).foldl
        (metadataStep semantics) initial)
  · rintro ⟨current, position⟩ ⟨hcurrentValid, hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold metadata_delivery.histories_loop.body
    by_cases hlt : position.val < destinations.val.length
    · have htest : position < destinations.len := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨destination, hdestination⟩
      have hdestination : destination = destinations.val[position.val]'hlt := hdestination
      step with metadata_history_spec writerInst destination touched active guarded current
        observe valid semantics hcurrentValid hcontract as ⟨updated, hupdatedValid, hupdated⟩
      step as ⟨next, hnext⟩; try scalar_tac
      have hn : next.val = position.val + 1 := by scalar_tac
      refine ⟨hupdatedValid, by scalar_tac, ?_, by scalar_tac⟩
      rw [hn, historiesPrefix, List.take_succ_eq_append_getElem hlt,
        List.flatMap_append, List.flatMap_singleton, List.foldl_append]
      simpa only [← hdestination, hcurrent, historiesPrefix] using hupdated
    · have htest : ¬position < destinations.len := by scalar_tac
      have hi : position.val = destinations.val.length := by omega
      simpa [htest, hi] using And.intro hcurrentValid hcurrent
  · exact ⟨hvalid, hindex, hstate⟩

@[step] theorem metadata_histories_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destinations : Slice Reg)
    (touched active guarded : Slice Usize) (writer : Writer)
    (observe : Writer → State) (valid : Writer → Prop) (semantics : MetadataSemantics State)
    (hvalid : valid writer) (hcontract : MetadataWriterContract writerInst observe valid semantics) :
    metadata_delivery.histories writerInst destinations touched active guarded writer ⦃ updated =>
      valid updated ∧ observe updated = (historiesPrefix destinations.val touched.val active.val guarded.val
        destinations.val.length).foldl (metadataStep semantics) (observe writer) ⦄ := by
  unfold metadata_delivery.histories
  apply metadata_histories_loop_spec writerInst destinations touched active guarded writer 0#usize
    (observe writer) observe valid semantics hvalid hcontract
  · scalar_tac
  · simp [historiesPrefix]

def updatesPrefix (destinations : List Reg) (touched active guarded : List Usize) (count : Nat) :
    List MetadataCommand :=
  (destinations.take count).flatMap fun destination => destinationUpdateCommands destination touched active guarded

@[step] theorem metadata_updates_loop_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destinations : Slice Reg)
    (touched active guarded : Slice Usize) (writer : Writer) (index : Usize)
    (initial : State) (observe : Writer → State) (valid : Writer → Prop)
    (semantics : MetadataSemantics State) (hvalid : valid writer)
    (hcontract : MetadataWriterContract writerInst observe valid semantics)
    (hindex : index.val ≤ destinations.val.length)
    (hstate : observe writer = ( updatesPrefix destinations.val touched.val active.val guarded.val index.val).foldl
      (metadataStep semantics) initial) :
    metadata_delivery.updates_loop writerInst destinations touched active guarded writer index ⦃ updated =>
      valid updated ∧ observe updated = (updatesPrefix destinations.val touched.val active.val guarded.val
        destinations.val.length).foldl (metadataStep semantics) initial ⦄ := by
  unfold metadata_delivery.updates_loop
  apply loop.spec_decr_nat
    (fun state : Writer × Usize => destinations.val.length - state.2.val)
    (fun state : Writer × Usize => valid state.1 ∧ state.2.val ≤ destinations.val.length ∧
      observe state.1 = (updatesPrefix destinations.val touched.val active.val guarded.val state.2.val).foldl
        (metadataStep semantics) initial)
  · rintro ⟨current, position⟩ ⟨hcurrentValid, hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold metadata_delivery.updates_loop.body
    by_cases hlt : position.val < destinations.val.length
    · have htest : position < destinations.len := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨destination, hdestination⟩
      have hdestination : destination = destinations.val[position.val]'hlt := hdestination
      step with metadata_clear_bits_spec writerInst destination touched active guarded current
        observe valid semantics hcurrentValid hcontract as ⟨cleared, hclearedValid, hcleared⟩
      step with metadata_active_taint_spec writerInst destination active guarded cleared
        observe valid semantics hclearedValid hcontract as ⟨activeUpdated, hactiveValid, hactive⟩
      step with metadata_guard_taint_spec writerInst destination active guarded activeUpdated
        observe valid semantics hactiveValid hcontract as ⟨updated, hupdatedValid, hupdated⟩
      step as ⟨next, hnext⟩; try scalar_tac
      have hn : next.val = position.val + 1 := by scalar_tac
      refine ⟨hupdatedValid, by scalar_tac, ?_, by scalar_tac⟩
      rw [hn, updatesPrefix, List.take_succ_eq_append_getElem hlt,
        List.flatMap_append, List.flatMap_singleton, List.foldl_append]
      simpa only [← hdestination, hcurrent, updatesPrefix, destinationUpdateCommands, List.foldl_append, hcleared, hactive] using hupdated
    · have htest : ¬position < destinations.len := by scalar_tac
      have hi : position.val = destinations.val.length := by omega
      simpa [htest, hi] using And.intro hcurrentValid hcurrent
  · exact ⟨hvalid, hindex, hstate⟩

@[step] theorem metadata_updates_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destinations : Slice Reg)
    (touched active guarded : Slice Usize) (writer : Writer)
    (observe : Writer → State) (valid : Writer → Prop) (semantics : MetadataSemantics State)
    (hvalid : valid writer) (hcontract : MetadataWriterContract writerInst observe valid semantics) :
    metadata_delivery.updates writerInst destinations touched active guarded writer ⦃ updated =>
      valid updated ∧ observe updated = (updatesPrefix destinations.val touched.val active.val guarded.val
        destinations.val.length).foldl (metadataStep semantics) (observe writer) ⦄ := by
  unfold metadata_delivery.updates
  apply metadata_updates_loop_spec writerInst destinations touched active guarded writer 0#usize
    (observe writer) observe valid semantics hvalid hcontract
  · scalar_tac
  · simp [updatesPrefix]

end Instruction
