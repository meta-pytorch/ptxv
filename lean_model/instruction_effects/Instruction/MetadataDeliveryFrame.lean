/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MetadataDeliveryProducer

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
namespace Instruction

def MetadataAction.target : MetadataAction → Option (Nat × Key)
  | .history thread destination .. | .clearBits thread destination .. |
      .activeTaint thread destination .. | .forceTaint thread destination .. =>
    some (thread, destination)
  | .prepare _ => none

theorem metadata_step_frame {State Query Value : Type} (semantics : MetadataSemantics State)
    (cell : State → Query → Value) (related : Query → Nat → Key → Prop)
    (query : Query) (state : State) (command : MetadataCommand)
    (hframe : ∀ state action, (∀ thread destination, action.target = some (thread, destination) →
      ¬related query thread destination) → cell (semantics.apply state action) query = cell state query)
    (hseparate : ∀ thread destination, command.target = some (thread, destination) →
      ¬related query thread destination) :
    cell (metadataStep semantics state command) query = cell state query := by
  cases command <;> apply hframe <;> simpa [MetadataAction.target, MetadataCommand.target] using hseparate

theorem metadata_fold_frame {State Query Value : Type} (semantics : MetadataSemantics State)
    (cell : State → Query → Value) (related : Query → Nat → Key → Prop)
    (query : Query) (state : State) (commands : List MetadataCommand)
    (hframe : ∀ state action, (∀ thread destination, action.target = some (thread, destination) →
      ¬related query thread destination) → cell (semantics.apply state action) query = cell state query)
    (hseparate : ∀ command, command ∈ commands → ∀ thread destination,
      command.target = some (thread, destination) → ¬related query thread destination) :
    cell (commands.foldl (metadataStep semantics) state) query = cell state query := by
  induction commands generalizing state with
  | nil => rfl
  | cons command commands ih =>
    rw [List.foldl_cons, ih (metadataStep semantics state command)
      (fun item hmember => hseparate item (List.mem_cons_of_mem command hmember))]
    exact metadata_step_frame semantics cell related query state command hframe
      (hseparate command (List.mem_cons_self))

theorem metadata_commands_frame {State Query Value : Type} (semantics : MetadataSemantics State)
    (cell : State → Query → Value) (related : Query → Nat → Key → Prop)
    (query : Query) (state : State) (isLoad : Bool) (destinations : List Key) (active guarded : List Nat)
    (hframe : ∀ state action, (∀ thread destination, action.target = some (thread, destination) →
      ¬related query thread destination) → cell (semantics.apply state action) query = cell state query)
    (houtside : ∀ destination, destination ∈ destinations → ∀ thread,
      (thread ∈ active ∨ thread ∈ guarded) → ¬related query thread destination) :
    cell ((metadataCommands isLoad destinations active guarded).foldl (metadataStep semantics) state) query =
      cell state query := by
  apply metadata_fold_frame semantics cell related query state _ hframe
  intro command hmember thread destination htarget
  have hc := metadata_target_coverage isLoad destinations active guarded command thread destination hmember htarget
  exact houtside destination hc.1 thread hc.2

@[step] theorem actual_memory_metadata_frame {Writer State Query Value : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (operation : Op)
    (active guarded : Slice Usize) (writer : Writer) (observe : Writer → State)
    (valid : Writer → Prop) (semantics : MetadataSemantics State) (hvalid : valid writer)
    (hcontract : MetadataWriterContract writerInst observe valid semantics)
    (hbounded : optionBounded (memoryWrites operation))
    (hsize : active.val.length + guarded.val.length ≤ Usize.max)
    (cell : State → Query → Value) (related : Query → Nat → Key → Prop) (query : Query)
    (hframe : ∀ state action, (∀ thread destination, action.target = some (thread, destination) →
      ¬related query thread destination) → cell (semantics.apply state action) query = cell state query)
    (houtside : ∀ destinations, memoryWrites operation = some destinations → ∀ destination,
      destination ∈ destinations → ∀ thread, (thread ∈ threadNumbers active.val ∨ thread ∈ threadNumbers guarded.val) →
      ¬related query thread destination) :
    metadata_delivery.execute writerInst operation active guarded writer ⦃ result =>
      cell (observe result.2) query = cell (observe writer) query ⦄ := by
  step with actual_memory_metadata_delivery_spec writerInst operation active guarded writer observe valid semantics
    hvalid hcontract hbounded hsize as ⟨result, hresult⟩
  unfold MetadataDeliveryPost memoryMetadataCommands at hresult
  cases hmeaning : memoryWrites operation with
  | none =>
    simp only [hmeaning, Option.map_none] at hresult
    rw [hresult]
  | some destinations =>
    simp only [hmeaning, Option.map_some] at hresult
    rw [hresult.2.2]
    exact metadata_commands_frame semantics cell related query (observe writer) _ destinations
      (threadNumbers active.val) (threadNumbers guarded.val) hframe (houtside destinations hmeaning)

end Instruction
