/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceApplication
import MemorySafety.SourceInitial
import MemorySafety.Soundness
import MemorySafety.SourceTransitions
import MemorySafety.SourceSuccessors
import MemorySafety.SourceGather

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety.Source

theorem source_initial_success (program : memory_safety.Program) (start : memory_safety.State)
    (computed : memory_safety.initial program = .ok (some start)) :
    LayoutSafe (programModel program) ∧ stateModel start = initialState (programModel program) := by
  obtain ⟨result, actual, specification⟩ := MemoryWrite.spec_returns _ _ (source_initial program)
  have equal : result = some start := by simpa only [actual, ok.injEq] using computed
  exact specification start equal

theorem source_applied_moves_agree (program : memory_safety.Program) (state : memory_safety.State)
    (choices : List memory_safety.Move) (next : List memory_safety.State)
    (computed : List.Forall₂
      (fun movement after => memory_safety.apply_move program state movement = .ok after) choices next)
    (clean : ∀ after ∈ next, after.bad = false) :
    next.map stateModel = (choices.map moveModel).map (applyMove (programModel program) (stateModel state)) := by
  induction computed with
  | nil => rfl
  | @cons movement after choices next applied computed ih =>
    have agreement := source_apply_move_agrees program state movement after applied (clean after (by simp))
    have rest := ih (fun later member => clean later (by simp [member]))
    simp only [List.map_cons, agreement.2, rest]

theorem source_moves_agrees (program : memory_safety.Program) (state : memory_safety.State)
    (choices : alloc.vec.Vec memory_safety.Move)
    (computed : memory_safety.moves program state = .ok (some choices)) :
    moves (programModel program) (stateModel state) = some (choices.val.map moveModel) :=
  source_moves_agrees_of_thread program state
    (fun thread next actual => source_thread_moves_agrees program state thread next actual) choices computed

theorem source_successors_agrees (program : memory_safety.Program) (state : memory_safety.State)
    (next : alloc.vec.Vec memory_safety.State)
    (computed : memory_safety.successors program state = .ok (some next))
    (clean : ∀ after ∈ next.val, after.bad = false) :
    successors (programModel program) (stateModel state) = some (next.val.map stateModel) := by
  obtain ⟨choices, enumerated, applied⟩ := source_successors_execution program state next computed
  have choicesEq := source_moves_agrees program state choices enumerated
  simp only [successors, choicesEq, Option.map_some]
  congr 1
  exact (source_applied_moves_agree program state choices.val next.val applied clean).symm

theorem source_verified_model (program : memory_safety.Program)
    (certificate : Slice memory_safety.State)
    (accepted : memory_safety.verify program certificate = .ok true) :
    verify (programModel program) (certificate.val.map stateModel) = true := by
  obtain ⟨⟨start, initialized, initialMember⟩, closed⟩ := source_verify_exact_closure program certificate accepted
  obtain ⟨layout, initialEq⟩ := source_initial_success program start initialized
  have initialSystem : (system (programModel program)).initial = some [initialState (programModel program)] := by
    simp [system, initial, layout]
  change Certificate.verify (system (programModel program)) good (certificate.val.map stateModel) = true
  rw [Certificate.verify, initialSystem]
  simp only [Bool.and_eq_true]
  constructor
  · have present : initialState (programModel program) ∈ certificate.val.map stateModel := by
      rw [←initialEq]
      exact List.mem_map.mpr ⟨start, initialMember, rfl⟩
    simpa [Certificate.member] using present
  · apply List.all_eq_true.mpr
    intro modelState member
    obtain ⟨state, rawMember, rfl⟩ := List.mem_map.mp member
    obtain ⟨clean, next, computed, allPresent⟩ := closed state rawMember
    have nextClean : ∀ after ∈ next.val, after.bad = false := by
      intro after present
      exact (closed after (allPresent after present)).1
    have nextEq := source_successors_agrees program state next computed nextClean
    simp only [Certificate.validVertex, good, stateModel_bad, clean, Bool.not_false, Bool.true_and,
      system, nextEq, List.all_eq_true]
    intro after present
    obtain ⟨raw, rawPresent, rfl⟩ := List.mem_map.mp present
    simp only [Certificate.member, decide_eq_true_eq]
    exact List.mem_map.mpr ⟨raw, allPresent raw rawPresent, rfl⟩

theorem accepted_rust_program_memory_safe (program : memory_safety.Program)
    (certificate : Slice memory_safety.State)
    (accepted : memory_safety.verify program certificate = .ok true) :
    MemorySafe (programModel program) :=
  verified_memory_safe (programModel program) (certificate.val.map stateModel)
    (source_verified_model program certificate accepted)

end MemorySafety.Source
