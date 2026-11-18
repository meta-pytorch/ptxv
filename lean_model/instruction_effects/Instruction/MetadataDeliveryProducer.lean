/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MetadataDeliveryDestinations
import Instruction.WriteRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

def metadataIsLoad (operation : Op) : Bool :=
  match operation with | .Ld .. => true | _ => false

@[step] theorem metadata_is_load_spec (operation : Op) :
    metadata_delivery.is_load operation ⦃ result => result = metadataIsLoad operation ⦄ := by
  cases operation <;> simp [metadata_delivery.is_load, metadataIsLoad]

theorem metadata_histories_model (destinations : List Reg) (touched active guarded : List Usize) :
    historiesPrefix destinations touched active guarded destinations.length =
      metadataHistory (destinations.map key) (threadNumbers touched) (threadNumbers active) (threadNumbers guarded) := by
  simp only [historiesPrefix, List.take_length, metadataHistory, historyCommands,
    threadNumbers, List.flatMap_map, List.map_map, Function.comp_def]

theorem metadata_updates_model (destinations : List Reg) (touched active guarded : List Usize) :
    updatesPrefix destinations touched active guarded destinations.length =
      (destinations.map key).flatMap fun destination =>
        metadataUpdates destination (threadNumbers touched) (threadNumbers active) (threadNumbers guarded) := by
  simp only [updatesPrefix, List.take_length, metadataUpdates, destinationUpdateCommands,
    clear_bitsCommands, active_taintCommands, guard_taintCommands, threadNumbers,
    List.flatMap_map, List.map_map, Function.comp_def]

def MetadataDeliveryPost {Writer State : Type} (operation : Op) (active guarded : List Nat)
    (initial : Writer) (result : Bool × Writer) (observe : Writer → State) (valid : Writer → Prop)
    (semantics : MetadataSemantics State) : Prop :=
  match memoryMetadataCommands operation active guarded with
  | none => result = (false, initial)
  | some commands => result.1 = true ∧ valid result.2 ∧
      observe result.2 = commands.foldl (metadataStep semantics) (observe initial)

@[step] theorem actual_memory_metadata_delivery_spec {Writer State : Type}
    (writerInst : metadata_delivery.MetadataWriter Writer) (operation : Op)
    (active guarded : Slice Usize) (writer : Writer) (observe : Writer → State)
    (valid : Writer → Prop) (semantics : MetadataSemantics State) (hvalid : valid writer)
    (hcontract : MetadataWriterContract writerInst observe valid semantics)
    (hbounded : optionBounded (memoryWrites operation))
    (hsize : active.val.length + guarded.val.length ≤ Usize.max) :
    metadata_delivery.execute writerInst operation active guarded writer ⦃ result =>
      MetadataDeliveryPost operation (threadNumbers active.val) (threadNumbers guarded.val)
        writer result observe valid semantics ⦄ := by
  unfold metadata_delivery.execute
  step with memory_writes_spec operation hbounded as ⟨produced, hproduced⟩
  cases produced with
  | none =>
    have hn : memoryWrites operation = none := by simpa [registerObservation] using hproduced.symm
    simp [MetadataDeliveryPost, memoryMetadataCommands, hn]
  | some destinations =>
    have hd : memoryWrites operation = some (destinations.val.map key) := by
      simpa [registerObservation] using hproduced.symm
    step with metadata_touched_spec active guarded hsize as ⟨touched, htouched⟩
    step with metadata_histories_spec writerInst destinations.deref touched.deref active guarded writer
      observe valid semantics hvalid hcontract as ⟨historyUpdated, hhistoryValid, hhistory⟩
    step with metadata_is_load_spec operation as ⟨isLoad, hisLoad⟩
    step with hcontract.1 historyUpdated hhistoryValid (.PrepareSources isLoad)
      as ⟨prepared, hpreparedValid, hprepared⟩
    step with metadata_updates_spec writerInst destinations.deref touched.deref active guarded prepared
      observe valid semantics hpreparedValid hcontract as ⟨updated, hupdatedValid, hupdated⟩
    simp only [alloc.vec.Vec.deref, Slice.from_val] at hhistory hupdated
    simp only [MetadataDeliveryPost, memoryMetadataCommands, hd, Option.map_some]
    refine ⟨trivial, hupdatedValid, ?_⟩
    rw [hupdated, hprepared, hhistory]
    simp only [metadata_histories_model, metadata_updates_model, metadataCommands,
      htouched, List.foldl_append, List.foldl_cons, List.foldl_nil, observedMetadataAction,
      metadataStep, hisLoad, metadataIsLoad]
    rfl

end Instruction
