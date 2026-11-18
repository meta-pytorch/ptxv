/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.EffectsFold
import MemoryWrite.EffectsSticky

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineRecorderModel

structure FootprintContext where
  mode : effects.FoldMode
  empty : Bool
  hasMembership : Bool

structure Context where
  footprint : Option FootprintContext
  slice : Bool

structure Recorder where
  trace : Option (List effects.Effect)

structure Access where
  resource : effects.Resource
  kind : effects.AccessKind
  visibility : effects.Visibility

inductive Direct where
  | access (resource : effects.Resource) (kind : effects.AccessKind)
  | release (key : U64)
  | acquire (key : U64)
  | fault

inductive Route where
  | off
  | buffered
  | direct (effect : Direct)

structure Delivery where
  slice : Option Access
  footprint : Route

def empty : Recorder := ⟨none⟩

def included (mode : effects.FoldMode) (visibility : effects.Visibility)
    (space : effects.ResourceSpace) : Bool :=
  match mode, visibility with
  | .Full, .SyncOnly => false
  | .SyncOnly, _ =>
      match space with
      | .Shared | .Global | .Tmem | .MbarPhase | .WarpPc => false
      | _ => true
  | _, _ => true

theorem source_included (mode : effects.FoldMode) (visibility : effects.Visibility)
    (space : effects.ResourceSpace) :
    effects.included mode visibility space = .ok (included mode visibility space) := by
  cases mode <;> cases visibility <;> cases space <;>
    simp [effects.included, included,
      effects.FoldMode.Insts.CoreCmpPartialEqFoldMode.eq,
      effects.Visibility.Insts.CoreCmpPartialEqVisibility.eq,
      effects.FoldMode.read_discriminant, effects.Visibility.read_discriminant]

def begin (state : Recorder) (context : Context) : Bool × Recorder :=
  match state.trace with
  | some events => (false, ⟨some (events ++ [.Finding])⟩)
  | none =>
      match context.footprint with
      | some footprint =>
          if footprint.empty then (true, ⟨some []⟩) else (false, state)
      | none => (false, state)

def route (state : Recorder) (context : Context) (event : effects.Effect)
    (direct : Option Direct) : Route × Recorder :=
  match context.footprint with
  | none => (.off, state)
  | some _ =>
      match state.trace with
      | some events => (.buffered, ⟨some (events ++ [event])⟩)
      | none =>
          match direct with
          | some event => (.direct event, state)
          | none => (.off, state)

def access (state : Recorder) (context : Context) (event : Access) : Delivery × Recorder :=
  let direct := match context.footprint with
    | some footprint =>
        if included footprint.mode event.visibility event.resource.space then
          some (.access event.resource event.kind)
        else none
    | none => none
  let (destination, next) := route state context
    (.Access event.resource event.kind event.visibility) direct
  (⟨if context.slice then some event else none, destination⟩, next)

def release (state : Recorder) (context : Context) (key : U64) : Route × Recorder :=
  route state context (.Release key) (some (.release key))

def acquire (state : Recorder) (context : Context) (key : U64) : Route × Recorder :=
  route state context (.Acquire key) (some (.acquire key))

def fault (state : Recorder) (context : Context) : Route × Recorder :=
  route state context .Fault (some .fault)

def finding (state : Recorder) (context : Context) : Route × Recorder :=
  route state context .Finding none

def arrival (state : Recorder) (context : Context)
    (key : effects.PhaseKey) (step : barrier.ArrivalStep) : Bool × Recorder :=
  match context.footprint, state.trace with
  | some _, some events => (true, ⟨some (events ++ [.Arrival key step])⟩)
  | _, _ => (false, state)

def summarize (mode : effects.FoldMode) (trace : List effects.Effect) : Result effects.Summary := do
  let initial ← effects.begin mode
  let complete ← ActionEffects.sequence trace initial
  effects.finish complete

def finish (state : Recorder) (context : Context) (forceWrite : Bool) :
    Result (Option effects.Summary × Recorder) :=
  match state.trace, context.footprint with
  | some events, some footprint => do
      let trace := if forceWrite || footprint.hasMembership then events ++ [.Finding] else events
      let summary ← summarize footprint.mode trace
      .ok (some summary, empty)
  | _, _ => .ok (none, empty)

theorem begin_nested (events : List effects.Effect) (context : Context) :
    begin ⟨some events⟩ context = (false, ⟨some (events ++ [.Finding])⟩) := rfl

theorem begin_without_footprint (slice : Bool) :
    begin empty ⟨none, slice⟩ = (false, empty) := rfl

theorem begin_dirty (footprint : FootprintContext) (slice : Bool)
    (dirty : footprint.empty = false) :
    begin empty ⟨some footprint, slice⟩ = (false, empty) := by
  simp [begin, empty, dirty]

theorem begin_clean (footprint : FootprintContext) (slice : Bool)
    (clean : footprint.empty = true) :
    begin empty ⟨some footprint, slice⟩ = (true, ⟨some []⟩) := by
  simp [begin, empty, clean]

theorem access_slice_exact (state : Recorder) (context : Context) (event : Access) :
    (access state context event).1.slice = if context.slice then some event else none := rfl

theorem access_without_footprint (state : Recorder) (event : Access) :
    access state ⟨none, true⟩ event = (⟨some event, .off⟩, state) := rfl

theorem sync_access_without_footprint (state : Recorder) (resource : effects.Resource)
    (kind : effects.AccessKind) :
    access state ⟨none, true⟩ ⟨resource, kind, .SyncOnly⟩ =
      (⟨some ⟨resource, kind, .SyncOnly⟩, .off⟩, state) := rfl

theorem access_buffered (events : List effects.Effect) (footprint : FootprintContext)
    (slice : Bool) (event : Access) :
    access ⟨some events⟩ ⟨some footprint, slice⟩ event =
      (⟨if slice then some event else none, .buffered⟩,
        ⟨some (events ++ [.Access event.resource event.kind event.visibility])⟩) := rfl

theorem access_direct (footprint : FootprintContext) (slice : Bool) (event : Access) :
    access empty ⟨some footprint, slice⟩ event =
      (⟨if slice then some event else none,
        if included footprint.mode event.visibility event.resource.space then
          .direct (.access event.resource event.kind) else .off⟩, empty) := by
  by_cases selected : included footprint.mode event.visibility event.resource.space
  all_goals simp [access, route, empty, selected]

theorem arrival_requires_buffer (state : Recorder) (context : Context)
    (key : effects.PhaseKey) (step : barrier.ArrivalStep)
    (accepted : (arrival state context key step).1 = true) :
    context.footprint.isSome = true ∧ state.trace.isSome = true := by
  cases context with
  | mk footprint slice =>
    cases footprint <;> cases state with
    | mk trace => cases trace <;> simp_all [arrival]

theorem arrival_without_buffer (context : Context) (key : effects.PhaseKey)
    (step : barrier.ArrivalStep) : arrival empty context key step = (false, empty) := by
  cases h : context.footprint <;> simp [arrival, empty, h]

theorem fault_buffered_preserves_prefix (events : List effects.Effect)
    (footprint : FootprintContext) (slice : Bool) :
    fault ⟨some events⟩ ⟨some footprint, slice⟩ =
      (.buffered, ⟨some (events ++ [.Fault])⟩) := rfl

theorem buffered_fault_observations (state : effects.FoldState)
    (events : List effects.Effect) :
    ActionEffects.sequence (events ++ [.Fault]) state = (do
      let observed ← ActionEffects.sequence events state
      .ok { observed with
        summary := { observed.summary with all := true }, invalidated := true }) := by
  simp [ActionEffects.sequence_append, ActionEffects.sequence, ActionEffects.source_fault]

theorem finish_without_footprint (state : Recorder) (slice forceWrite : Bool) :
    finish state ⟨none, slice⟩ forceWrite = .ok (none, empty) := by
  cases h : state.trace <;> simp [finish, h]

theorem finish_idle (context : Context) (forceWrite : Bool) :
    finish empty context forceWrite = .ok (none, empty) := by
  cases context.footprint <;> rfl

theorem finish_consumes (state : Recorder) (context : Context) (forceWrite : Bool)
    (summary : Option effects.Summary) (next : Recorder)
    (success : finish state context forceWrite = .ok (summary, next)) : next.trace = none := by
  unfold finish at success
  split at success
  · rename_i events footprint _ _
    cases computed : summarize footprint.mode
      (if forceWrite || footprint.hasMembership then events ++ [.Finding] else events) <;>
      simp_all [empty]
    exact congrArg Recorder.trace success.2.symm
  · simpa [empty] using (congrArg (fun pair => pair.2.trace) (Result.ok_injective success)).symm

theorem summarize_source (mode : effects.FoldMode) (trace : Slice effects.Effect) :
    effects.fold mode trace = summarize mode trace.val := by
  simp [effects.fold, summarize, ActionEffects.source_fold_sequence]

structure Observations where
  reads : List (U8 × U32 × U64 × U64) := []
  writes : List (U8 × U32 × U64 × U64) := []
  releases : List U64 := []
  acquires : List U64 := []
  all : Bool := false

def range (resource : effects.Resource) : U8 × U32 × U64 × U64 :=
  (ActionEffects.resourceTag resource.space, resource.owner, resource.start, resource.end)

def observeDirect (state : Observations) : Direct → Observations
  | .access resource .Read => { state with reads := state.reads ++ [range resource] }
  | .access resource .Write => { state with writes := state.writes ++ [range resource] }
  | .release key => { state with releases := state.releases ++ [key] }
  | .acquire key => { state with acquires := state.acquires ++ [key] }
  | .fault => { state with all := true }

def immediate (mode : effects.FoldMode) : effects.Effect → Option Direct
  | .Access resource kind visibility =>
      if included mode visibility resource.space then some (.access resource kind) else none
  | .Release key => some (.release key)
  | .Acquire key => some (.acquire key)
  | .Fault => some .fault
  | .Finding | .Arrival _ _ => none

def admissible : effects.Effect → Prop
  | .Arrival _ _ => False
  | _ => True

def observe (mode : effects.FoldMode) (state : Observations) : effects.Effect → Observations
  | .Access resource kind visibility =>
      if included mode visibility resource.space then observeDirect state (.access resource kind)
      else state
  | .Release key => { state with releases := state.releases ++ [key] }
  | .Acquire key => { state with acquires := state.acquires ++ [key] }
  | .Fault => { state with all := true }
  | .Finding | .Arrival _ _ => state

def observeTrace (mode : effects.FoldMode) (state : Observations)
    (events : List effects.Effect) : Observations := events.foldl (observe mode) state

def observeImmediate (mode : effects.FoldMode) (state : Observations)
    (events : List effects.Effect) : Observations :=
  (events.filterMap (immediate mode)).foldl observeDirect state

theorem direct_buffered_equivalent (mode : effects.FoldMode) (state : Observations)
    (events : List effects.Effect) (allowed : ∀ event ∈ events, admissible event) :
    observeImmediate mode state events = observeTrace mode state events := by
  induction events generalizing state with
  | nil => rfl
  | cons event events ih =>
    have rest : ∀ item ∈ events, admissible item := by
      intro item member
      exact allowed item (by simp [member])
    have current := allowed event (by simp)
    cases event with
    | Access resource kind visibility =>
      by_cases selected : included mode visibility resource.space
      · simpa [observeImmediate, observeTrace, immediate, observe, selected] using
          ih (observeDirect state (.access resource kind)) rest
      · simpa [observeImmediate, observeTrace, immediate, observe, selected] using ih state rest
    | Release key =>
      simpa [observeImmediate, observeTrace, immediate, observe, observeDirect] using
        ih { state with releases := state.releases ++ [key] } rest
    | Acquire key =>
      simpa [observeImmediate, observeTrace, immediate, observe, observeDirect] using
        ih { state with acquires := state.acquires ++ [key] } rest
    | Fault =>
      simpa [observeImmediate, observeTrace, immediate, observe, observeDirect] using
        ih { state with all := true } rest
    | Finding =>
      change observeImmediate mode state events = observeTrace mode state events
      exact ih state rest
    | Arrival _ _ => exact False.elim current

theorem record_context_ignores_lifecycle_flags (state : Recorder)
    (left right : FootprintContext) (slice : Bool) (event : Access)
    (sameMode : left.mode = right.mode) :
    access state ⟨some left, slice⟩ event = access state ⟨some right, slice⟩ event := by
  simp only [access, route, sameMode]

theorem finish_ignores_empty_flag (state : Recorder)
    (left right : FootprintContext) (slice forceWrite : Bool)
    (sameMode : left.mode = right.mode) (sameMembership : left.hasMembership = right.hasMembership) :
    finish state ⟨some left, slice⟩ forceWrite = finish state ⟨some right, slice⟩ forceWrite := by
  cases h : state.trace <;> simp [finish, h, sameMode, sameMembership]

structure BeginContext where
  footprintEmpty : Option Bool

structure RecordContext where
  mode : Option effects.FoldMode
  slice : Bool

structure FinishFootprint where
  mode : effects.FoldMode
  hasMembership : Bool

structure FinishContext where
  footprint : Option FinishFootprint

def BeginContext.lower (context : BeginContext) : Context :=
  ⟨context.footprintEmpty.map (fun empty => ⟨.Full, empty, false⟩), false⟩

def RecordContext.lower (context : RecordContext) : Context :=
  ⟨context.mode.map (fun mode => ⟨mode, false, false⟩), context.slice⟩

def FinishContext.lower (context : FinishContext) : Context :=
  ⟨context.footprint.map (fun footprint => ⟨footprint.mode, false, footprint.hasMembership⟩), false⟩

def beginAction (state : Recorder) (context : BeginContext) : Bool × Recorder :=
  begin state context.lower

def recordAccess (state : Recorder) (context : RecordContext) (event : Access) : Delivery × Recorder :=
  access state context.lower event

def recordRelease (state : Recorder) (context : RecordContext) (key : U64) : Route × Recorder :=
  release state context.lower key

def recordAcquire (state : Recorder) (context : RecordContext) (key : U64) : Route × Recorder :=
  acquire state context.lower key

def recordFault (state : Recorder) (context : RecordContext) : Route × Recorder :=
  fault state context.lower

def recordFinding (state : Recorder) (context : RecordContext) : Route × Recorder :=
  finding state context.lower

def recordArrival (state : Recorder) (context : RecordContext)
    (key : effects.PhaseKey) (step : barrier.ArrivalStep) : Bool × Recorder :=
  arrival state context.lower key step

def finishAction (state : Recorder) (context : FinishContext) (forceWrite : Bool) :
    Result (Option effects.Summary × Recorder) :=
  finish state context.lower forceWrite

end MemoryWrite.MachineRecorderModel
