/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineActionModel

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 1000000

namespace MemoryWrite.MachineAction

open MachineActionModel

def mapState (run : Result machine_action.ActionState) : Result State := do
  let next ← run
  .ok (view next)

def mapPair {α : Type} (run : Result (α × machine_action.ActionState)) : Result (α × State) := do
  let (output, next) ← run
  .ok (output, view next)

theorem spec_result {α : Type} {run : Result α} {post : α → Prop}
    (spec : run ⦃ value => post value ⦄) : ∃ value, run = .ok value ∧ post value := by
  cases run with
  | ret value => exact ⟨value, rfl, by simpa using spec⟩
  | vis effect next => simp at spec
  | div => simp at spec

theorem source_footprint (state : machine_action.ActionState) :
    machine_action.footprint state = .ok state.fp := by
  cases h : state.fp <;> simp [machine_action.footprint, h]

theorem source_new : machine_action.new = .ok ⟨none, ⟨none⟩⟩ := by
  simp [machine_action.new, machine_recorder.new]

theorem source_set_footprint (state : machine_action.ActionState)
    (footprint : Option machine_footprint.Footprint) :
    machine_action.set_footprint state footprint = .ok { state with fp := footprint } := rfl

theorem source_take_footprint (state : machine_action.ActionState) :
    machine_action.take_footprint state = .ok (state.fp, { state with fp := none }) := rfl

theorem source_is_recording (state : machine_action.ActionState) :
    machine_action.is_recording state = .ok state.recorder.trace.isSome := rfl

theorem source_apply_route (state : machine_action.ActionState) (route : machine_recorder.Route) :
    mapState (machine_action.apply_route state route) =
      applyRoute (view state) (MachineRecorder.routeModel route) := by
  cases route with
  | Off => simp [mapState, machine_action.apply_route, applyRoute, view, MachineRecorder.routeModel]
  | Buffered => simp [mapState, machine_action.apply_route, applyRoute, view, MachineRecorder.routeModel]
  | Direct effect =>
    cases fpEq : state.fp with
    | none => simp [mapState, machine_action.apply_route, applyRoute, view,
        MachineRecorder.routeModel, fpEq]
    | some footprint =>
      cases effect with
      | Access resource kind =>
        cases kind <;>
          simp [mapState, machine_action.apply_route, ActionEffects.source_resource_range,
            effects.AccessKind.Insts.CoreCmpPartialEqAccessKind.eq,
            effects.AccessKind.read_discriminant, MachineFootprint.source_add,
            applyRoute, applyDirect, view, MachineRecorder.routeModel,
            MachineRecorder.directModel, fpEq, Std.bind_assoc]
      | Release key =>
        simp [mapState, machine_action.apply_route, applyRoute, applyDirect,
          MachineFootprint.appendKey, view, MachineRecorder.routeModel,
          MachineRecorder.directModel, fpEq, Std.bind_assoc]
      | Acquire key =>
        simp [mapState, machine_action.apply_route, applyRoute, applyDirect,
          MachineFootprint.appendKey, view, MachineRecorder.routeModel,
          MachineRecorder.directModel, fpEq, Std.bind_assoc]
      | Fault =>
        simp [mapState, machine_action.apply_route, applyRoute, applyDirect,
          view, MachineRecorder.routeModel, MachineRecorder.directModel, fpEq]

theorem source_begin (state : machine_action.ActionState)
    (room : MachineRecorder.TraceRoom state.recorder) :
    machine_action.begin state ⦃ started next =>
      (started, view next) = MachineActionModel.begin (view state) ⦄ := by
  simp only [machine_action.begin, source_footprint, MachineFootprint.source_begin_context, bind_ok]
  step with MachineRecorder.source_begin state.recorder
    ⟨state.fp.map MachineFootprint.beginEmpty⟩ room as ⟨started, recorder, same⟩
  simpa [MachineActionModel.begin, view, beginContext, MachineRecorder.beginContextModel] using
    congrArg (fun pair : Bool × MachineRecorderModel.Recorder =>
      (pair.1, State.mk state.fp pair.2)) same

theorem source_record_access (state : machine_action.ActionState) (slice : Bool)
    (event : machine_recorder.Access) (room : MachineRecorder.TraceRoom state.recorder) :
    mapPair (machine_action.record_access state slice event) =
      MachineActionModel.recordAccess (view state) slice event := by
  let context : machine_recorder.RecordContext :=
    ⟨state.fp.map (fun fp => MachineFootprint.recordMode fp.sync_only), slice⟩
  obtain ⟨⟨delivery, recorder⟩, executed, same⟩ := spec_result
    (MachineRecorder.source_record_access state.recorder context event room)
  dsimp only [context] at executed
  simp only [WP.uncurry'_pair] at same
  simp only [machine_action.record_access, source_footprint,
    MachineFootprint.source_record_context, bind_ok, executed]
  unfold mapPair
  conv_lhs =>
    arg 1
    change (do
      let next ← machine_action.apply_route { state with recorder } delivery.footprint
      .ok (delivery.slice, next))
  simp only [Std.bind_assoc, bind_ok]
  change (do
    let next ← machine_action.apply_route { state with recorder } delivery.footprint
    .ok (delivery.slice, view next)) = _
  unfold MachineActionModel.recordAccess
  change _ = (do
    let (delivery, recorder) := MachineRecorderModel.recordAccess
      (MachineRecorder.stateModel state.recorder) (MachineRecorder.recordContextModel context)
      (MachineRecorder.accessModel event)
    let next ← applyRoute { view state with recorder } delivery.footprint
    .ok (delivery.slice.map sourceAccess, next))
  rw [← same]
  simp only [MachineRecorder.deliveryModel, MachineRecorder.routeModel]
  have sliceRoundtrip : (delivery.slice.map MachineRecorder.accessModel).map sourceAccess = delivery.slice := by
    cases delivery.slice <;> rfl
  rw [sliceRoundtrip]
  have routed := source_apply_route { state with recorder } delivery.footprint
  have lifted := congrArg (fun run : Result State => do
    let next ← run
    .ok (delivery.slice, next)) routed
  unfold mapState at lifted
  change Aeneas.Std.bind
    (Aeneas.Std.bind (machine_action.apply_route { state with recorder } delivery.footprint)
      (fun next => .ok (view next)))
    (fun next => .ok (delivery.slice, next)) = _ at lifted
  rw [Std.bind_assoc] at lifted
  simp only [bind_ok] at lifted
  change Aeneas.Std.bind
    (machine_action.apply_route { state with recorder } delivery.footprint)
    (fun next => .ok (delivery.slice, view next)) =
    Aeneas.Std.bind
      (applyRoute (view { state with recorder }) (MachineRecorder.routeModel delivery.footprint))
      (fun next => .ok (delivery.slice, next))
  exact lifted

theorem source_record_release (state : machine_action.ActionState) (key : U64)
    (room : MachineRecorder.TraceRoom state.recorder) :
    mapState (machine_action.record_release state key) =
      MachineActionModel.recordRelease (view state) key := by
  let context : machine_recorder.RecordContext :=
    ⟨state.fp.map (fun fp => MachineFootprint.recordMode fp.sync_only), false⟩
  obtain ⟨⟨route, recorder⟩, executed, same⟩ := spec_result
    (MachineRecorder.source_record_release state.recorder context key room)
  dsimp only [context] at executed
  simp only [WP.uncurry'_pair] at same
  simp only [machine_action.record_release, source_footprint,
    MachineFootprint.source_record_context, bind_ok, executed]
  unfold MachineActionModel.recordRelease
  change _ = (do
    let (route, recorder) := MachineRecorderModel.recordRelease
      (MachineRecorder.stateModel state.recorder) (MachineRecorder.recordContextModel context) key
    applyRoute { view state with recorder } route)
  rw [← same]
  exact source_apply_route { state with recorder } route

theorem source_record_acquire (state : machine_action.ActionState) (key : U64)
    (room : MachineRecorder.TraceRoom state.recorder) :
    mapState (machine_action.record_acquire state key) =
      MachineActionModel.recordAcquire (view state) key := by
  let context : machine_recorder.RecordContext :=
    ⟨state.fp.map (fun fp => MachineFootprint.recordMode fp.sync_only), false⟩
  obtain ⟨⟨route, recorder⟩, executed, same⟩ := spec_result
    (MachineRecorder.source_record_acquire state.recorder context key room)
  dsimp only [context] at executed
  simp only [WP.uncurry'_pair] at same
  simp only [machine_action.record_acquire, source_footprint,
    MachineFootprint.source_record_context, bind_ok, executed]
  unfold MachineActionModel.recordAcquire
  change _ = (do
    let (route, recorder) := MachineRecorderModel.recordAcquire
      (MachineRecorder.stateModel state.recorder) (MachineRecorder.recordContextModel context) key
    applyRoute { view state with recorder } route)
  rw [← same]
  exact source_apply_route { state with recorder } route

theorem source_record_fault (state : machine_action.ActionState)
    (room : MachineRecorder.TraceRoom state.recorder) :
    mapState (machine_action.record_fault state) =
      MachineActionModel.recordFault (view state) := by
  let context : machine_recorder.RecordContext :=
    ⟨state.fp.map (fun fp => MachineFootprint.recordMode fp.sync_only), false⟩
  obtain ⟨⟨route, recorder⟩, executed, same⟩ := spec_result
    (MachineRecorder.source_record_fault state.recorder context room)
  dsimp only [context] at executed
  simp only [WP.uncurry'_pair] at same
  simp only [machine_action.record_fault, source_footprint,
    MachineFootprint.source_record_context, bind_ok, executed]
  unfold MachineActionModel.recordFault
  change _ = (do
    let (route, recorder) := MachineRecorderModel.recordFault
      (MachineRecorder.stateModel state.recorder) (MachineRecorder.recordContextModel context)
    applyRoute { view state with recorder } route)
  rw [← same]
  exact source_apply_route { state with recorder } route

theorem source_record_finding (state : machine_action.ActionState)
    (room : MachineRecorder.TraceRoom state.recorder) :
    mapState (machine_action.record_finding state) =
      MachineActionModel.recordFinding (view state) := by
  let context : machine_recorder.RecordContext :=
    ⟨state.fp.map (fun fp => MachineFootprint.recordMode fp.sync_only), false⟩
  obtain ⟨⟨route, recorder⟩, executed, same⟩ := spec_result
    (MachineRecorder.source_record_finding state.recorder context room)
  dsimp only [context] at executed
  simp only [WP.uncurry'_pair] at same
  simp only [machine_action.record_finding, source_footprint,
    MachineFootprint.source_record_context, bind_ok, executed]
  unfold MachineActionModel.recordFinding
  change _ = (do
    let (route, recorder) := MachineRecorderModel.recordFinding
      (MachineRecorder.stateModel state.recorder) (MachineRecorder.recordContextModel context)
    applyRoute { view state with recorder } route)
  rw [← same]
  exact source_apply_route { state with recorder } route

theorem source_record_arrival (state : machine_action.ActionState)
    (key : effects.PhaseKey) (step : barrier.ArrivalStep)
    (room : MachineRecorder.TraceRoom state.recorder) :
    machine_action.record_arrival state key step ⦃ accepted next =>
      (accepted, view next) = MachineActionModel.recordArrival (view state) key step ⦄ := by
  simp only [machine_action.record_arrival, source_footprint,
    MachineFootprint.source_record_context, bind_ok]
  step with MachineRecorder.source_record_arrival state.recorder
    ⟨state.fp.map (fun fp => MachineFootprint.recordMode fp.sync_only), false⟩ key step room as
    ⟨accepted, recorder, same⟩
  simpa [MachineActionModel.recordArrival, view, recordContext, MachineRecorder.recordContextModel] using
    congrArg (fun pair : Bool × MachineRecorderModel.Recorder =>
      (pair.1, State.mk state.fp pair.2)) same

end MemoryWrite.MachineAction
