/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRecorderFinish
import MemoryWrite.MachineFootprintContexts
import MemoryWrite.MachineFootprintMerge

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 1000000

namespace MemoryWrite.MachineActionModel

structure State where
  fp : Option machine_footprint.Footprint
  recorder : MachineRecorderModel.Recorder

def view (state : machine_action.ActionState) : State :=
  ⟨state.fp, MachineRecorder.stateModel state.recorder⟩

def empty : State := ⟨none, MachineRecorderModel.empty⟩

def beginContext (state : State) : MachineRecorderModel.BeginContext :=
  ⟨state.fp.map MachineFootprint.beginEmpty⟩

def recordContext (state : State) (slice : Bool) : MachineRecorderModel.RecordContext :=
  ⟨state.fp.map (fun fp => MachineFootprint.recordMode fp.sync_only), slice⟩

def finishContext (state : State) : MachineRecorderModel.FinishContext :=
  ⟨state.fp.map (fun fp => ⟨MachineFootprint.recordMode fp.sync_only,
    (fp.r.val ++ fp.w.val).any (fun range => decide (range.1 = 18#u8))⟩)⟩

def begin (state : State) : Bool × State :=
  let (started, recorder) := MachineRecorderModel.beginAction state.recorder (beginContext state)
  (started, { state with recorder })

def sourceAccess (event : MachineRecorderModel.Access) : machine_recorder.Access :=
  ⟨event.resource, event.kind, event.visibility⟩

def applyDirect (fp : machine_footprint.Footprint) :
    MachineRecorderModel.Direct → Result machine_footprint.Footprint
  | .access resource kind => MachineFootprint.addRef fp
      (ActionEffects.resourceTag resource.space, resource.owner, resource.start, resource.end)
      (match kind with | .Read => false | .Write => true)
  | .release key => MachineFootprint.appendKey false fp key
  | .acquire key => MachineFootprint.appendKey true fp key
  | .fault => .ok { fp with all := true }

def applyRoute (state : State) : MachineRecorderModel.Route → Result State
  | .off | .buffered => .ok state
  | .direct effect => do
      match state.fp with
      | none => .fail .panic
      | some footprint =>
        let fp ← applyDirect footprint effect
        .ok { state with fp := some fp }

def recordAccess (state : State) (slice : Bool) (event : machine_recorder.Access) :
    Result (Option machine_recorder.Access × State) := do
  let (delivery, recorder) := MachineRecorderModel.recordAccess state.recorder
    (recordContext state slice) (MachineRecorder.accessModel event)
  let next ← applyRoute { state with recorder } delivery.footprint
  .ok (delivery.slice.map sourceAccess, next)

def recordRelease (state : State) (key : U64) : Result State := do
  let (route, recorder) := MachineRecorderModel.recordRelease state.recorder (recordContext state false) key
  applyRoute { state with recorder } route

def recordAcquire (state : State) (key : U64) : Result State := do
  let (route, recorder) := MachineRecorderModel.recordAcquire state.recorder (recordContext state false) key
  applyRoute { state with recorder } route

def recordFault (state : State) : Result State := do
  let (route, recorder) := MachineRecorderModel.recordFault state.recorder (recordContext state false)
  applyRoute { state with recorder } route

def recordFinding (state : State) : Result State := do
  let (route, recorder) := MachineRecorderModel.recordFinding state.recorder (recordContext state false)
  applyRoute { state with recorder } route

def recordArrival (state : State) (key : effects.PhaseKey) (step : barrier.ArrivalStep) : Bool × State :=
  let (accepted, recorder) := MachineRecorderModel.recordArrival state.recorder (recordContext state false) key step
  (accepted, { state with recorder })

def finish (state : State) (forceWrite : Bool) : Result State := do
  let (summary, recorder) ← MachineRecorderModel.finishAction state.recorder (finishContext state) forceWrite
  match summary with
  | none => .ok { state with recorder }
  | some summary =>
    match state.fp with
    | none => .fail .panic
    | some footprint =>
      let fp ← MachineFootprint.nativeMerge footprint summary
      .ok ⟨some fp, recorder⟩

end MemoryWrite.MachineActionModel
