/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MetadataThreads

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

def observedMetadataAction : metadata_delivery.Action → MetadataAction
  | .History thread destination active guarded => .history thread.val (key destination) active guarded
  | .PrepareSources isLoad => .prepare isLoad
  | .ClearBits thread destination active guarded => .clearBits thread.val (key destination) active guarded
  | .SetActiveTaint thread destination guarded sourceTainted =>
    .activeTaint thread.val (key destination) guarded sourceTainted
  | .ForceGuardTaint thread destination active => .forceTaint thread.val (key destination) active

def MetadataWriterContract {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer)
    (observe : Writer → State) (valid : Writer → Prop)
    (semantics : MetadataSemantics State) : Prop :=
  (∀ writer, valid writer → ∀ action, writerInst.apply writer action ⦃ updated =>
    valid updated ∧ observe updated = semantics.apply (observe writer) (observedMetadataAction action) ⦄) ∧
  (∀ writer, valid writer → ∀ thread destination,
    writerInst.source_tainted writer thread destination ⦃ tainted =>
      tainted = semantics.sourceTainted (observe writer) thread.val (key destination) ⦄)

def historyCommands (destination : Reg) (threads active guarded : List Usize) : List MetadataCommand :=
  threads.map fun thread => .history thread.val (key destination) (thread.val ∈ threadNumbers active) (thread.val ∈ threadNumbers guarded)

@[step] theorem metadata_history_loop_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destination : Reg)
    (touched : Slice Usize) (active guarded : Slice Usize) (writer : Writer) (index : Usize)
    (initial : State) (observe : Writer → State) (valid : Writer → Prop)
    (semantics : MetadataSemantics State) (hvalid : valid writer)
    (hcontract : MetadataWriterContract writerInst observe valid semantics)
    (hindex : index.val ≤ touched.val.length)
    (hstate : observe writer = ((historyCommands destination touched.val active.val guarded.val).take index.val).foldl
      (metadataStep semantics) initial) :
    metadata_delivery.history_loop writerInst destination touched active guarded writer index ⦃ updated =>
      valid updated ∧ observe updated = (historyCommands destination touched.val active.val guarded.val).foldl
        (metadataStep semantics) initial ⦄ := by
  unfold metadata_delivery.history_loop
  apply loop.spec_decr_nat
    (fun state : Writer × Usize => touched.val.length - state.2.val)
    (fun state : Writer × Usize => valid state.1 ∧ state.2.val ≤ touched.val.length ∧
      observe state.1 = ((historyCommands destination touched.val active.val guarded.val).take state.2.val).foldl
        (metadataStep semantics) initial)
  · rintro ⟨current, position⟩ ⟨hcurrentValid, hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold metadata_delivery.history_loop.body
    by_cases hlt : position.val < touched.val.length
    · have htest : position < touched.len := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨thread, hthread⟩
      have hthread : thread = touched.val[position.val]'hlt := hthread
      step with metadata_contains_spec active thread as ⟨activeFlag, hactiveFlag⟩
      step with metadata_contains_spec guarded thread as ⟨guardedFlag, hguardedFlag⟩
      step with hcontract.1 current hcurrentValid (.History thread destination activeFlag guardedFlag)
        as ⟨updated, hupdatedValid, hupdated⟩
      have hstep : observe updated = metadataStep semantics (observe current)
          (.history thread.val (key destination) (thread.val ∈ threadNumbers active.val) (thread.val ∈ threadNumbers guarded.val)) := by
        simpa [metadataStep, observedMetadataAction, thread_numbers_member, hactiveFlag, hguardedFlag] using hupdated
      step as ⟨next, hnext⟩; try scalar_tac
      have hn : next.val = position.val + 1 := by scalar_tac
      refine ⟨hupdatedValid, by scalar_tac, ?_, by scalar_tac⟩
      rw [hn]
      have hc : position.val < (historyCommands destination touched.val active.val guarded.val).length := by
        simpa [historyCommands] using hlt
      rw [List.take_succ_eq_append_getElem hc, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      simpa only [historyCommands, List.getElem_map, ← hthread, hcurrent] using hstep
    · have htest : ¬position < touched.len := by scalar_tac
      have hi : position.val = touched.val.length := by omega
      simpa [htest, hi, historyCommands, ← List.map_take] using And.intro hcurrentValid hcurrent
  · exact ⟨hvalid, hindex, hstate⟩

@[step] theorem metadata_history_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destination : Reg)
    (touched : Slice Usize) (active guarded : Slice Usize) (writer : Writer)
    (observe : Writer → State) (valid : Writer → Prop) (semantics : MetadataSemantics State)
    (hvalid : valid writer) (hcontract : MetadataWriterContract writerInst observe valid semantics) :
    metadata_delivery.history writerInst destination touched active guarded writer ⦃ updated =>
      valid updated ∧ observe updated = (historyCommands destination touched.val active.val guarded.val).foldl
        (metadataStep semantics) (observe writer) ⦄ := by
  unfold metadata_delivery.history
  apply metadata_history_loop_spec writerInst destination touched active guarded writer 0#usize
    (observe writer) observe valid semantics hvalid hcontract
  · scalar_tac
  · simp

def clear_bitsCommands (destination : Reg) (threads active guarded : List Usize) : List MetadataCommand :=
  threads.map fun thread => .clearBits thread.val (key destination) (thread.val ∈ threadNumbers active) (thread.val ∈ threadNumbers guarded)

@[step] theorem metadata_clear_bits_loop_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destination : Reg)
    (touched : Slice Usize) (active guarded : Slice Usize) (writer : Writer) (index : Usize)
    (initial : State) (observe : Writer → State) (valid : Writer → Prop)
    (semantics : MetadataSemantics State) (hvalid : valid writer)
    (hcontract : MetadataWriterContract writerInst observe valid semantics)
    (hindex : index.val ≤ touched.val.length)
    (hstate : observe writer = ((clear_bitsCommands destination touched.val active.val guarded.val).take index.val).foldl
      (metadataStep semantics) initial) :
    metadata_delivery.clear_bits_loop writerInst destination touched active guarded writer index ⦃ updated =>
      valid updated ∧ observe updated = (clear_bitsCommands destination touched.val active.val guarded.val).foldl
        (metadataStep semantics) initial ⦄ := by
  unfold metadata_delivery.clear_bits_loop
  apply loop.spec_decr_nat
    (fun state : Writer × Usize => touched.val.length - state.2.val)
    (fun state : Writer × Usize => valid state.1 ∧ state.2.val ≤ touched.val.length ∧
      observe state.1 = ((clear_bitsCommands destination touched.val active.val guarded.val).take state.2.val).foldl
        (metadataStep semantics) initial)
  · rintro ⟨current, position⟩ ⟨hcurrentValid, hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold metadata_delivery.clear_bits_loop.body
    by_cases hlt : position.val < touched.val.length
    · have htest : position < touched.len := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨thread, hthread⟩
      have hthread : thread = touched.val[position.val]'hlt := hthread
      step with metadata_contains_spec active thread as ⟨activeFlag, hactiveFlag⟩
      step with metadata_contains_spec guarded thread as ⟨guardedFlag, hguardedFlag⟩
      step with hcontract.1 current hcurrentValid (.ClearBits thread destination activeFlag guardedFlag)
        as ⟨updated, hupdatedValid, hupdated⟩
      have hstep : observe updated = metadataStep semantics (observe current)
          (.clearBits thread.val (key destination) (thread.val ∈ threadNumbers active.val) (thread.val ∈ threadNumbers guarded.val)) := by
        simpa [metadataStep, observedMetadataAction, thread_numbers_member, hactiveFlag, hguardedFlag] using hupdated
      step as ⟨next, hnext⟩; try scalar_tac
      have hn : next.val = position.val + 1 := by scalar_tac
      refine ⟨hupdatedValid, by scalar_tac, ?_, by scalar_tac⟩
      rw [hn]
      have hc : position.val < (clear_bitsCommands destination touched.val active.val guarded.val).length := by
        simpa [clear_bitsCommands] using hlt
      rw [List.take_succ_eq_append_getElem hc, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      simpa only [clear_bitsCommands, List.getElem_map, ← hthread, hcurrent] using hstep
    · have htest : ¬position < touched.len := by scalar_tac
      have hi : position.val = touched.val.length := by omega
      simpa [htest, hi, clear_bitsCommands, ← List.map_take] using And.intro hcurrentValid hcurrent
  · exact ⟨hvalid, hindex, hstate⟩

@[step] theorem metadata_clear_bits_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destination : Reg)
    (touched : Slice Usize) (active guarded : Slice Usize) (writer : Writer)
    (observe : Writer → State) (valid : Writer → Prop) (semantics : MetadataSemantics State)
    (hvalid : valid writer) (hcontract : MetadataWriterContract writerInst observe valid semantics) :
    metadata_delivery.clear_bits writerInst destination touched active guarded writer ⦃ updated =>
      valid updated ∧ observe updated = (clear_bitsCommands destination touched.val active.val guarded.val).foldl
        (metadataStep semantics) (observe writer) ⦄ := by
  unfold metadata_delivery.clear_bits
  apply metadata_clear_bits_loop_spec writerInst destination touched active guarded writer 0#usize
    (observe writer) observe valid semantics hvalid hcontract
  · scalar_tac
  · simp

def active_taintCommands (destination : Reg) (threads _active guarded : List Usize) : List MetadataCommand :=
  threads.map fun thread => .activeTaint thread.val (key destination) (thread.val ∈ threadNumbers guarded)

@[step] theorem metadata_active_taint_loop_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destination : Reg)
    (active guarded : Slice Usize) (writer : Writer) (index : Usize)
    (initial : State) (observe : Writer → State) (valid : Writer → Prop)
    (semantics : MetadataSemantics State) (hvalid : valid writer)
    (hcontract : MetadataWriterContract writerInst observe valid semantics)
    (hindex : index.val ≤ active.val.length)
    (hstate : observe writer = ((active_taintCommands destination active.val active.val guarded.val).take index.val).foldl
      (metadataStep semantics) initial) :
    metadata_delivery.active_taint_loop writerInst destination active guarded writer index ⦃ updated =>
      valid updated ∧ observe updated = (active_taintCommands destination active.val active.val guarded.val).foldl
        (metadataStep semantics) initial ⦄ := by
  unfold metadata_delivery.active_taint_loop
  apply loop.spec_decr_nat
    (fun state : Writer × Usize => active.val.length - state.2.val)
    (fun state : Writer × Usize => valid state.1 ∧ state.2.val ≤ active.val.length ∧
      observe state.1 = ((active_taintCommands destination active.val active.val guarded.val).take state.2.val).foldl
        (metadataStep semantics) initial)
  · rintro ⟨current, position⟩ ⟨hcurrentValid, hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold metadata_delivery.active_taint_loop.body
    by_cases hlt : position.val < active.val.length
    · have htest : position < active.len := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨thread, hthread⟩
      have hthread : thread = active.val[position.val]'hlt := hthread
      step with metadata_contains_spec guarded thread as ⟨guardedFlag, hguardedFlag⟩
      step with hcontract.2 current hcurrentValid thread destination as ⟨sourceTainted, hsourceTainted⟩
      step with hcontract.1 current hcurrentValid (.SetActiveTaint thread destination guardedFlag sourceTainted)
        as ⟨updated, hupdatedValid, hupdated⟩
      have hstep : observe updated = metadataStep semantics (observe current)
          (.activeTaint thread.val (key destination) (thread.val ∈ threadNumbers guarded.val)) := by
        simpa [metadataStep, observedMetadataAction, thread_numbers_member, hguardedFlag, hsourceTainted] using hupdated
      step as ⟨next, hnext⟩; try scalar_tac
      have hn : next.val = position.val + 1 := by scalar_tac
      refine ⟨hupdatedValid, by scalar_tac, ?_, by scalar_tac⟩
      rw [hn]
      have hc : position.val < (active_taintCommands destination active.val active.val guarded.val).length := by
        simpa [active_taintCommands] using hlt
      rw [List.take_succ_eq_append_getElem hc, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      simpa only [active_taintCommands, List.getElem_map, ← hthread, hcurrent] using hstep
    · have htest : ¬position < active.len := by scalar_tac
      have hi : position.val = active.val.length := by omega
      simpa [htest, hi, active_taintCommands, ← List.map_take] using And.intro hcurrentValid hcurrent
  · exact ⟨hvalid, hindex, hstate⟩

@[step] theorem metadata_active_taint_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destination : Reg)
    (active guarded : Slice Usize) (writer : Writer)
    (observe : Writer → State) (valid : Writer → Prop) (semantics : MetadataSemantics State)
    (hvalid : valid writer) (hcontract : MetadataWriterContract writerInst observe valid semantics) :
    metadata_delivery.active_taint writerInst destination active guarded writer ⦃ updated =>
      valid updated ∧ observe updated = (active_taintCommands destination active.val active.val guarded.val).foldl
        (metadataStep semantics) (observe writer) ⦄ := by
  unfold metadata_delivery.active_taint
  apply metadata_active_taint_loop_spec writerInst destination active guarded writer 0#usize
    (observe writer) observe valid semantics hvalid hcontract
  · scalar_tac
  · simp

def guard_taintCommands (destination : Reg) (threads active _guarded : List Usize) : List MetadataCommand :=
  threads.map fun thread => .forceTaint thread.val (key destination) (thread.val ∈ threadNumbers active)

@[step] theorem metadata_guard_taint_loop_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destination : Reg)
    (active guarded : Slice Usize) (writer : Writer) (index : Usize)
    (initial : State) (observe : Writer → State) (valid : Writer → Prop)
    (semantics : MetadataSemantics State) (hvalid : valid writer)
    (hcontract : MetadataWriterContract writerInst observe valid semantics)
    (hindex : index.val ≤ guarded.val.length)
    (hstate : observe writer = ((guard_taintCommands destination guarded.val active.val guarded.val).take index.val).foldl
      (metadataStep semantics) initial) :
    metadata_delivery.guard_taint_loop writerInst destination active guarded writer index ⦃ updated =>
      valid updated ∧ observe updated = (guard_taintCommands destination guarded.val active.val guarded.val).foldl
        (metadataStep semantics) initial ⦄ := by
  unfold metadata_delivery.guard_taint_loop
  apply loop.spec_decr_nat
    (fun state : Writer × Usize => guarded.val.length - state.2.val)
    (fun state : Writer × Usize => valid state.1 ∧ state.2.val ≤ guarded.val.length ∧
      observe state.1 = ((guard_taintCommands destination guarded.val active.val guarded.val).take state.2.val).foldl
        (metadataStep semantics) initial)
  · rintro ⟨current, position⟩ ⟨hcurrentValid, hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold metadata_delivery.guard_taint_loop.body
    by_cases hlt : position.val < guarded.val.length
    · have htest : position < guarded.len := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨thread, hthread⟩
      have hthread : thread = guarded.val[position.val]'hlt := hthread
      step with metadata_contains_spec active thread as ⟨activeFlag, hactiveFlag⟩
      step with hcontract.1 current hcurrentValid (.ForceGuardTaint thread destination activeFlag)
        as ⟨updated, hupdatedValid, hupdated⟩
      have hstep : observe updated = metadataStep semantics (observe current)
          (.forceTaint thread.val (key destination) (thread.val ∈ threadNumbers active.val)) := by
        simpa [metadataStep, observedMetadataAction, thread_numbers_member, hactiveFlag] using hupdated
      step as ⟨next, hnext⟩; try scalar_tac
      have hn : next.val = position.val + 1 := by scalar_tac
      refine ⟨hupdatedValid, by scalar_tac, ?_, by scalar_tac⟩
      rw [hn]
      have hc : position.val < (guard_taintCommands destination guarded.val active.val guarded.val).length := by
        simpa [guard_taintCommands] using hlt
      rw [List.take_succ_eq_append_getElem hc, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      simpa only [guard_taintCommands, List.getElem_map, ← hthread, hcurrent] using hstep
    · have htest : ¬position < guarded.len := by scalar_tac
      have hi : position.val = guarded.val.length := by omega
      simpa [htest, hi, guard_taintCommands, ← List.map_take] using And.intro hcurrentValid hcurrent
  · exact ⟨hvalid, hindex, hstate⟩

@[step] theorem metadata_guard_taint_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (destination : Reg)
    (active guarded : Slice Usize) (writer : Writer)
    (observe : Writer → State) (valid : Writer → Prop) (semantics : MetadataSemantics State)
    (hvalid : valid writer) (hcontract : MetadataWriterContract writerInst observe valid semantics) :
    metadata_delivery.guard_taint writerInst destination active guarded writer ⦃ updated =>
      valid updated ∧ observe updated = (guard_taintCommands destination guarded.val active.val guarded.val).foldl
        (metadataStep semantics) (observe writer) ⦄ := by
  unfold metadata_delivery.guard_taint
  apply metadata_guard_taint_loop_spec writerInst destination active guarded writer 0#usize
    (observe writer) observe valid semantics hvalid hcontract
  · scalar_tac
  · simp

end Instruction
