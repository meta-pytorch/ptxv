/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MetadataDeliveryFrame

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

def MetadataCommand.isPreparation : MetadataCommand → Bool
  | .prepare _ => true
  | _ => false

theorem metadata_prepare_once (isLoad : Bool) (destinations : List Key)
    (active guarded : List Nat) :
    (metadataCommands isLoad destinations active guarded).filter MetadataCommand.isPreparation = [.prepare isLoad] := by
  simp [metadataCommands, metadataHistory, metadataUpdates, MetadataCommand.isPreparation,
    List.filter_flatMap, List.filter_map, List.filter_cons, Function.comp_def] <;>
    induction destinations <;> simp_all

theorem metadata_history_coverage (isLoad : Bool) (destinations : List Key) (active guarded : List Nat)
    (destination : Key) (thread : Nat) (hdestination : destination ∈ destinations)
    (hthread : thread ∈ active ∨ thread ∈ guarded) :
    .history thread destination (thread ∈ active) (thread ∈ guarded) ∈
      metadataCommands isLoad destinations active guarded := by
  apply List.mem_append_left
  apply List.mem_append_left
  apply List.mem_flatMap.mpr
  refine ⟨destination, hdestination, ?_⟩
  exact List.mem_map_of_mem ((metadata_touched_member active guarded thread).mpr hthread)

theorem metadata_bits_coverage (isLoad : Bool) (destinations : List Key) (active guarded : List Nat)
    (destination : Key) (thread : Nat) (hdestination : destination ∈ destinations)
    (hthread : thread ∈ active ∨ thread ∈ guarded) :
    .clearBits thread destination (thread ∈ active) (thread ∈ guarded) ∈
      metadataCommands isLoad destinations active guarded := by
  apply List.mem_append_right
  apply List.mem_flatMap.mpr
  refine ⟨destination, hdestination, ?_⟩
  apply List.mem_append_left
  apply List.mem_append_left
  exact List.mem_map_of_mem ((metadata_touched_member active guarded thread).mpr hthread)

theorem metadata_active_coverage (isLoad : Bool) (destinations : List Key) (active guarded : List Nat)
    (destination : Key) (thread : Nat) (hdestination : destination ∈ destinations)
    (hthread : thread ∈ active) :
    .activeTaint thread destination (thread ∈ guarded) ∈ metadataCommands isLoad destinations active guarded := by
  apply List.mem_append_right
  apply List.mem_flatMap.mpr
  refine ⟨destination, hdestination, ?_⟩
  apply List.mem_append_left
  apply List.mem_append_right
  exact List.mem_map_of_mem hthread

theorem metadata_guard_coverage (isLoad : Bool) (destinations : List Key) (active guarded : List Nat)
    (destination : Key) (thread : Nat) (hdestination : destination ∈ destinations)
    (hthread : thread ∈ guarded) :
    .forceTaint thread destination (thread ∈ active) ∈ metadataCommands isLoad destinations active guarded := by
  apply List.mem_append_right
  apply List.mem_flatMap.mpr
  refine ⟨destination, hdestination, ?_⟩
  apply List.mem_append_right
  exact List.mem_map_of_mem hthread

end Instruction
