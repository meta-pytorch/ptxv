/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineActionBasic
import MemoryWrite.MachineActionFinish

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 1000000

namespace MemoryWrite.MachineAction

open MachineActionModel

theorem source_access_without_footprint (state : machine_action.ActionState)
    (slice : Bool) (event : machine_recorder.Access) (absent : state.fp = none) :
    machine_action.record_access state slice event =
      .ok ((if slice then some event else none), state) := by
  cases state with
  | mk fp recorder =>
    simp only at absent
    subst fp
    cases slice <;>
      simp [machine_action.record_access, source_footprint,
        MachineFootprint.source_record_context, machine_recorder.record_access,
        machine_recorder.route, core.option.Option.is_none, machine_action.apply_route]

theorem source_sync_slice_without_footprint (state : machine_action.ActionState)
    (resource : effects.Resource) (kind : effects.AccessKind) (absent : state.fp = none) :
    machine_action.record_access state true ⟨resource, kind, .SyncOnly⟩ =
      .ok (some ⟨resource, kind, .SyncOnly⟩, state) :=
  source_access_without_footprint state true ⟨resource, kind, .SyncOnly⟩ absent

theorem result_tag {α β : Type} (run : Result β) (tag output : α) (next : β)
    (success : (do let value ← run; .ok (tag, value)) = .ok (output, next)) :
    output = tag := by
  cases run with
  | ret value =>
    have same : (tag, value) = (output, next) := by simpa using success
    exact (congrArg Prod.fst same).symm
  | vis effect continuation => simp at success
  | div => simp at success

theorem model_access_slice (state : State) (slice : Bool)
    (event : machine_recorder.Access) (observed : Option machine_recorder.Access) (next : State)
    (success : recordAccess state slice event = .ok (observed, next)) :
    observed = if slice then some event else none := by
  have selected : (MachineRecorderModel.recordAccess state.recorder
      (recordContext state slice) (MachineRecorder.accessModel event)).1.slice.map sourceAccess =
        if slice then some event else none := by
    have pureSlice := MachineRecorderModel.access_slice_exact state.recorder
      (recordContext state slice).lower (MachineRecorder.accessModel event)
    simp only [MachineRecorderModel.recordAccess]
    rw [pureSlice]
    cases slice <;> rfl
  unfold recordAccess at success
  generalize pairEq : MachineRecorderModel.recordAccess state.recorder
    (recordContext state slice) (MachineRecorder.accessModel event) = pair at success selected
  rcases pair with ⟨delivery, recorder⟩
  simp only at success selected
  have tagged := result_tag
    (applyRoute { state with recorder } delivery.footprint)
    (delivery.slice.map sourceAccess) observed next success
  exact tagged.trans selected

theorem source_access_slice (state : machine_action.ActionState) (slice : Bool)
    (event : machine_recorder.Access) (observed : Option machine_recorder.Access)
    (next : machine_action.ActionState) (room : MachineRecorder.TraceRoom state.recorder)
    (success : machine_action.record_access state slice event = .ok (observed, next)) :
    observed = if slice then some event else none := by
  have refined := source_record_access state slice event room
  rw [success] at refined
  have modelSuccess : recordAccess (view state) slice event = .ok (observed, view next) := by
    simpa [mapPair] using refined.symm
  exact model_access_slice (view state) slice event observed (view next) modelSuccess

end MemoryWrite.MachineAction
