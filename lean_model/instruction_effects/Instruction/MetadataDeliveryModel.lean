/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Model

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

inductive MetadataCommand where
  | history (thread : Nat) (destination : Key) (active guarded : Bool)
  | prepare (isLoad : Bool)
  | clearBits (thread : Nat) (destination : Key) (active guarded : Bool)
  | activeTaint (thread : Nat) (destination : Key) (guarded : Bool)
  | forceTaint (thread : Nat) (destination : Key) (active : Bool)
  deriving DecidableEq

inductive MetadataAction where
  | history (thread : Nat) (destination : Key) (active guarded : Bool)
  | prepare (isLoad : Bool)
  | clearBits (thread : Nat) (destination : Key) (active guarded : Bool)
  | activeTaint (thread : Nat) (destination : Key) (guarded sourceTainted : Bool)
  | forceTaint (thread : Nat) (destination : Key) (active : Bool)
  deriving DecidableEq

def metadataTouched (active guarded : List Nat) : List Nat :=
  active ++ guarded.filter fun thread => thread ∉ active

def metadataHistory (destinations : List Key) (touched active guarded : List Nat) :
    List MetadataCommand :=
  destinations.flatMap fun destination => touched.map fun thread =>
    .history thread destination (thread ∈ active) (thread ∈ guarded)

def metadataUpdates (destination : Key) (touched active guarded : List Nat) :
    List MetadataCommand :=
  touched.map (fun thread => .clearBits thread destination (thread ∈ active) (thread ∈ guarded)) ++
  active.map (fun thread => .activeTaint thread destination (thread ∈ guarded)) ++
  guarded.map (fun thread => .forceTaint thread destination (thread ∈ active))

def metadataCommands (isLoad : Bool) (destinations : List Key) (active guarded : List Nat) :
    List MetadataCommand :=
  metadataHistory destinations (metadataTouched active guarded) active guarded ++
  [.prepare isLoad] ++ destinations.flatMap fun destination =>
    metadataUpdates destination (metadataTouched active guarded) active guarded

def memoryMetadataCommands (operation : Op) (active guarded : List Nat) :
    Option (List MetadataCommand) :=
  (memoryWrites operation).map fun destinations =>
    metadataCommands (match operation with | .Ld .. => true | _ => false)
      destinations active guarded

structure MetadataSemantics (State : Type) where
  apply : State → MetadataAction → State
  sourceTainted : State → Nat → Key → Bool

def metadataStep {State : Type} (semantics : MetadataSemantics State)
    (state : State) (command : MetadataCommand) : State :=
  semantics.apply state <| match command with
  | .history thread destination active guarded => .history thread destination active guarded
  | .prepare isLoad => .prepare isLoad
  | .clearBits thread destination active guarded => .clearBits thread destination active guarded
  | .activeTaint thread destination guarded =>
    .activeTaint thread destination guarded (semantics.sourceTainted state thread destination)
  | .forceTaint thread destination active => .forceTaint thread destination active

@[simp] theorem metadata_touched_member (active guarded : List Nat) (thread : Nat) :
    thread ∈ metadataTouched active guarded ↔ thread ∈ active ∨ thread ∈ guarded := by
  simp only [metadataTouched, List.mem_append, List.mem_filter, decide_eq_true_eq]
  tauto

@[simp] theorem metadata_empty (isLoad : Bool) (active guarded : List Nat) :
    metadataCommands isLoad [] active guarded = [.prepare isLoad] := by
  simp [metadataCommands, metadataHistory]

theorem metadata_phase_order (isLoad : Bool) (destinations : List Key)
    (active guarded : List Nat) :
    metadataCommands isLoad destinations active guarded =
      metadataHistory destinations (metadataTouched active guarded) active guarded ++
      [.prepare isLoad] ++ destinations.flatMap
        (fun destination => metadataUpdates destination (metadataTouched active guarded) active guarded) := rfl


def MetadataCommand.target : MetadataCommand → Option (Nat × Key)
  | .history thread destination .. | .clearBits thread destination .. |
      .activeTaint thread destination .. | .forceTaint thread destination .. =>
    some (thread, destination)
  | .prepare _ => none

theorem metadata_target_coverage (isLoad : Bool) (destinations : List Key)
    (active guarded : List Nat) (command : MetadataCommand) (thread : Nat) (destination : Key)
    (hc : command ∈ metadataCommands isLoad destinations active guarded)
    (ht : command.target = some (thread, destination)) :
    destination ∈ destinations ∧ (thread ∈ active ∨ thread ∈ guarded) := by
  simp only [metadataCommands, List.mem_append, List.mem_singleton] at hc
  rcases hc with (hh | hp) | hu
  · simp only [metadataHistory, List.mem_flatMap, List.mem_map] at hh
    rcases hh with ⟨d, hd, t, hmember, rfl⟩
    simp only [MetadataCommand.target, Option.some.injEq, Prod.mk.injEq] at ht
    obtain ⟨rfl, rfl⟩ := ht
    exact ⟨hd, (metadata_touched_member active guarded t).mp hmember⟩
  · subst command
    simp [MetadataCommand.target] at ht
  · simp only [List.mem_flatMap, metadataUpdates, List.mem_append, List.mem_map] at hu
    rcases hu with ⟨d, hd, (⟨t, hm, rfl⟩ | ⟨t, hm, rfl⟩) | ⟨t, hm, rfl⟩⟩
    all_goals simp only [MetadataCommand.target, Option.some.injEq, Prod.mk.injEq] at ht
    all_goals obtain ⟨rfl, rfl⟩ := ht
    · exact ⟨hd, (metadata_touched_member active guarded t).mp hm⟩
    · exact ⟨hd, Or.inl hm⟩
    · exact ⟨hd, Or.inr hm⟩

end Instruction
