/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRecorderModel
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 500000

namespace MemoryWrite.MachineRecorder

def stateModel (state : machine_recorder.Recorder) : MachineRecorderModel.Recorder :=
  ⟨state.trace.map alloc.vec.Vec.val⟩

def beginContextModel (context : machine_recorder.BeginContext) : MachineRecorderModel.BeginContext :=
  ⟨context.footprint_empty⟩

def recordContextModel (context : machine_recorder.RecordContext) : MachineRecorderModel.RecordContext :=
  ⟨context.mode, context.slice⟩

def finishContextModel (context : machine_recorder.FinishContext) : MachineRecorderModel.FinishContext :=
  ⟨context.footprint.map (fun footprint => ⟨footprint.mode, footprint.has_membership⟩)⟩

def accessModel (event : machine_recorder.Access) : MachineRecorderModel.Access :=
  ⟨event.resource, event.kind, event.visibility⟩

def directModel : machine_recorder.Direct → MachineRecorderModel.Direct
  | .Access resource kind => .access resource kind
  | .Release key => .release key
  | .Acquire key => .acquire key
  | .Fault => .fault

def routeModel : machine_recorder.Route → MachineRecorderModel.Route
  | .Off => .off
  | .Buffered => .buffered
  | .Direct direct => .direct (directModel direct)

def deliveryModel (delivery : machine_recorder.Delivery) : MachineRecorderModel.Delivery :=
  ⟨delivery.slice.map accessModel, routeModel delivery.footprint⟩

def TraceRoom (state : machine_recorder.Recorder) : Prop :=
  ∀ trace, state.trace = some trace → trace.val.length < Usize.max

theorem source_new : machine_recorder.new ⦃ state => stateModel state = MachineRecorderModel.empty ⦄ := by
  simp [machine_recorder.new, stateModel, MachineRecorderModel.empty]

theorem source_fresh_snapshot (state : machine_recorder.Recorder) :
    machine_recorder.fresh_snapshot state ⦃ snapshot => stateModel snapshot = MachineRecorderModel.empty ⦄ := by
  exact source_new

theorem source_begin (state : machine_recorder.Recorder)
    (context : machine_recorder.BeginContext) (room : TraceRoom state) :
    machine_recorder.begin state context ⦃ started next =>
      (started, stateModel next) = MachineRecorderModel.beginAction (stateModel state) (beginContextModel context) ⦄ := by
  cases state with
  | mk trace =>
    cases trace with
    | none =>
      cases context with
      | mk footprint =>
        cases footprint with
        | none =>
          simp [machine_recorder.begin, stateModel, beginContextModel,
            MachineRecorderModel.beginAction, MachineRecorderModel.BeginContext.lower,
            MachineRecorderModel.begin]
        | some clean =>
          cases clean <;>
            simp [machine_recorder.begin, stateModel, beginContextModel,
              MachineRecorderModel.beginAction, MachineRecorderModel.BeginContext.lower, MachineRecorderModel.begin,
              alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val]
    | some trace =>
      simp only [machine_recorder.begin]
      step with alloc.vec.Vec.push_spec trace .Finding (room trace rfl) as ⟨updated, appended⟩
      simp [stateModel, MachineRecorderModel.beginAction, MachineRecorderModel.begin, appended]

theorem source_route (state : machine_recorder.Recorder)
    (context : machine_recorder.RecordContext) (event : effects.Effect)
    (direct : Option machine_recorder.Direct) (room : TraceRoom state) :
    machine_recorder.route state context event direct ⦃ destination next =>
      (routeModel destination, stateModel next) =
        MachineRecorderModel.route (stateModel state) (recordContextModel context).lower event
          (direct.map directModel) ⦄ := by
  cases context with
  | mk mode slice =>
    cases mode with
    | none =>
      simp [machine_recorder.route, core.option.Option.is_none, routeModel,
        recordContextModel, MachineRecorderModel.RecordContext.lower, MachineRecorderModel.route]
    | some mode =>
      cases state with
      | mk trace =>
        cases trace with
        | none =>
          cases direct <;>
            simp [machine_recorder.route, core.option.Option.is_none, routeModel,
              stateModel, recordContextModel, MachineRecorderModel.RecordContext.lower, MachineRecorderModel.route]
        | some trace =>
          simp only [machine_recorder.route, core.option.Option.is_none, Option.isNone, Bool.false_eq_true,
            ↓reduceIte]
          step with alloc.vec.Vec.push_spec trace event (room trace rfl) as ⟨updated, appended⟩
          simp [routeModel, stateModel, recordContextModel, MachineRecorderModel.RecordContext.lower,
            MachineRecorderModel.route, appended]

theorem source_record_release (state : machine_recorder.Recorder)
    (context : machine_recorder.RecordContext) (key : U64) (room : TraceRoom state) :
    machine_recorder.record_release state context key ⦃ destination next =>
      (routeModel destination, stateModel next) =
        MachineRecorderModel.recordRelease (stateModel state) (recordContextModel context) key ⦄ := by
  simpa [machine_recorder.record_release, MachineRecorderModel.recordRelease, MachineRecorderModel.release, directModel] using
    source_route state context (.Release key) (some (.Release key)) room

theorem source_record_acquire (state : machine_recorder.Recorder)
    (context : machine_recorder.RecordContext) (key : U64) (room : TraceRoom state) :
    machine_recorder.record_acquire state context key ⦃ destination next =>
      (routeModel destination, stateModel next) =
        MachineRecorderModel.recordAcquire (stateModel state) (recordContextModel context) key ⦄ := by
  simpa [machine_recorder.record_acquire, MachineRecorderModel.recordAcquire, MachineRecorderModel.acquire, directModel] using
    source_route state context (.Acquire key) (some (.Acquire key)) room

theorem source_record_fault (state : machine_recorder.Recorder)
    (context : machine_recorder.RecordContext) (room : TraceRoom state) :
    machine_recorder.record_fault state context ⦃ destination next =>
      (routeModel destination, stateModel next) =
        MachineRecorderModel.recordFault (stateModel state) (recordContextModel context) ⦄ := by
  simpa [machine_recorder.record_fault, MachineRecorderModel.recordFault, MachineRecorderModel.fault, directModel] using
    source_route state context .Fault (some .Fault) room

theorem source_record_finding (state : machine_recorder.Recorder)
    (context : machine_recorder.RecordContext) (room : TraceRoom state) :
    machine_recorder.record_finding state context ⦃ destination next =>
      (routeModel destination, stateModel next) =
        MachineRecorderModel.recordFinding (stateModel state) (recordContextModel context) ⦄ := by
  simpa [machine_recorder.record_finding, MachineRecorderModel.recordFinding, MachineRecorderModel.finding] using
    source_route state context .Finding none room

theorem source_record_arrival (state : machine_recorder.Recorder)
    (context : machine_recorder.RecordContext) (key : effects.PhaseKey)
    (step : barrier.ArrivalStep) (room : TraceRoom state) :
    machine_recorder.record_arrival state context key step ⦃ accepted next =>
      (accepted, stateModel next) =
        MachineRecorderModel.recordArrival (stateModel state) (recordContextModel context) key step ⦄ := by
  cases context with
  | mk mode slice =>
    cases mode with
    | none =>
      simp [machine_recorder.record_arrival, core.option.Option.is_none,
        recordContextModel, MachineRecorderModel.RecordContext.lower, MachineRecorderModel.recordArrival, MachineRecorderModel.arrival]
    | some mode =>
      cases state with
      | mk trace =>
        cases trace with
        | none =>
          simp [machine_recorder.record_arrival, core.option.Option.is_none, stateModel,
            recordContextModel, MachineRecorderModel.RecordContext.lower, MachineRecorderModel.recordArrival, MachineRecorderModel.arrival]
        | some trace =>
          simp only [machine_recorder.record_arrival, core.option.Option.is_none, Option.isNone,
            Bool.false_eq_true, ↓reduceIte]
          step with alloc.vec.Vec.push_spec trace (.Arrival key step) (room trace rfl) as
            ⟨updated, appended⟩
          simp [stateModel, recordContextModel, MachineRecorderModel.RecordContext.lower,
            MachineRecorderModel.recordArrival, MachineRecorderModel.arrival, appended]

theorem source_record_access (state : machine_recorder.Recorder)
    (context : machine_recorder.RecordContext) (event : machine_recorder.Access)
    (room : TraceRoom state) :
    machine_recorder.record_access state context event ⦃ delivery next =>
      (deliveryModel delivery, stateModel next) =
        MachineRecorderModel.recordAccess (stateModel state) (recordContextModel context)
          (accessModel event) ⦄ := by
  cases context with
  | mk mode slice =>
    cases mode with
    | none =>
      simp only [machine_recorder.record_access, bind_ok]
      step with source_route state ⟨none, slice⟩
        (.Access event.resource event.kind event.visibility) none room as
        ⟨destination, next, routed⟩
      cases slice <;>
        simpa [deliveryModel, recordContextModel, accessModel,
          MachineRecorderModel.recordAccess, MachineRecorderModel.access,
          MachineRecorderModel.RecordContext.lower] using (And.intro (congrArg Prod.fst routed) (congrArg Prod.snd routed))
    | some mode =>
      simp only [machine_recorder.record_access, MachineRecorderModel.source_included]
      cases selected : MachineRecorderModel.included mode event.visibility event.resource.space
      all_goals
        simp only [Bool.false_eq_true, ↓reduceIte, bind_ok]
      · step with source_route state ⟨some mode, slice⟩
          (.Access event.resource event.kind event.visibility) none room as
          ⟨destination, next, routed⟩
        cases slice <;>
          simpa [deliveryModel, recordContextModel, accessModel,
            MachineRecorderModel.recordAccess, MachineRecorderModel.access,
            MachineRecorderModel.RecordContext.lower, selected] using (And.intro (congrArg Prod.fst routed) (congrArg Prod.snd routed))
      · step with source_route state ⟨some mode, slice⟩
          (.Access event.resource event.kind event.visibility)
          (some (.Access event.resource event.kind)) room as ⟨destination, next, routed⟩
        cases slice <;>
          simpa [deliveryModel, recordContextModel, accessModel, directModel,
            MachineRecorderModel.recordAccess, MachineRecorderModel.access,
            MachineRecorderModel.RecordContext.lower, selected] using (And.intro (congrArg Prod.fst routed) (congrArg Prod.snd routed))

end MemoryWrite.MachineRecorder
