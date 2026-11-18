/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRecorderModel

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RecorderObservation

open MachineRecorderModel

def projection (summary : effects.Summary) : Observations :=
  ⟨summary.reads.val, summary.writes.val, summary.releases.val, summary.acquires.val, summary.all⟩

def Fits (state : Observations) (spare : Nat) : Prop :=
  state.reads.length + spare ≤ Usize.max ∧ state.writes.length + spare ≤ Usize.max ∧
  state.releases.length + spare ≤ Usize.max ∧ state.acquires.length + spare ≤ Usize.max

theorem observe_fits (mode : effects.FoldMode) (state : Observations)
    (event : effects.Effect) (spare : Nat) (room : Fits state (spare + 1)) :
    Fits (observe mode state event) spare := by
  cases event with
  | Access resource kind visibility =>
    cases selected : included mode visibility resource.space <;> cases kind <;>
      simp [observe, observeDirect, selected, Fits] at * <;> omega
  | Release key | Acquire key | Fault | Finding | Arrival _ _ =>
    simp [observe, Fits] at *; omega

theorem source_poison_total (space : effects.ResourceSpace) (kind : effects.AccessKind) :
    effects.poisons_arrivals space kind ⦃ _ => True ⦄ := by
  cases space <;> cases kind <;>
    simp [effects.poisons_arrivals,
      effects.ResourceSpace.Insts.CoreCmpPartialEqResourceSpace.eq,
      effects.AccessKind.Insts.CoreCmpPartialEqAccessKind.eq,
      effects.ResourceSpace.read_discriminant, effects.AccessKind.read_discriminant]

theorem source_apply (state : effects.FoldState) (event : effects.Effect)
    (allowed : admissible event) (room : Fits (projection state.summary) 1) :
    effects.apply state event ⦃ next =>
      next.mode = state.mode ∧ next.arrivals = state.arrivals ∧
      projection next.summary = observe state.mode (projection state.summary) event ⦄ := by
  have reads : state.summary.reads.val.length < Usize.max := by
    simp only [Fits, projection] at room
    omega
  have writes : state.summary.writes.val.length < Usize.max := by
    simp only [Fits, projection] at room
    omega
  have releases : state.summary.releases.val.length < Usize.max := by
    simp only [Fits, projection] at room
    omega
  have acquires : state.summary.acquires.val.length < Usize.max := by
    simp only [Fits, projection] at room
    omega
  cases event with
  | Access resource kind visibility =>
    unfold effects.apply
    step with source_poison_total resource.space kind as ⟨poison⟩
    simp only [source_included, bind_ok]
    cases selected : included state.mode visibility resource.space
    · simp [selected, projection, observe]
    · simp only [ite_true, ActionEffects.source_resource_range, bind_ok]
      cases kind with
      | Read =>
        step with alloc.vec.Vec.push_spec state.summary.reads
          (ActionEffects.resourceTag resource.space, resource.owner, resource.start, resource.end)
          reads as ⟨updated, appended⟩
        simp [projection, observe, observeDirect, range, selected, appended]
      | Write =>
        step with alloc.vec.Vec.push_spec state.summary.writes
          (ActionEffects.resourceTag resource.space, resource.owner, resource.start, resource.end)
          writes as ⟨updated, appended⟩
        simp [projection, observe, observeDirect, range, selected, appended]
  | Release key =>
    unfold effects.apply
    step with alloc.vec.Vec.push_spec state.summary.releases key releases as ⟨updated, appended⟩
    simp [projection, observe, appended]
  | Acquire key =>
    unfold effects.apply
    step with alloc.vec.Vec.push_spec state.summary.acquires key acquires as ⟨updated, appended⟩
    simp [projection, observe, appended]
  | Fault => simp [ActionEffects.source_fault, projection, observe]
  | Finding => simp [ActionEffects.source_finding, projection, observe]
  | Arrival _ _ => exact False.elim allowed

theorem fits_mono (state : Observations) (large small : Nat)
    (room : Fits state large) (bounded : small ≤ large) : Fits state small := by
  simp only [Fits] at *
  omega

theorem source_sequence (events : List effects.Effect) (state : effects.FoldState)
    (allowed : ∀ event ∈ events, admissible event)
    (room : Fits (projection state.summary) events.length) :
    ActionEffects.sequence events state ⦃ next =>
      next.mode = state.mode ∧ next.arrivals = state.arrivals ∧
      projection next.summary = observeTrace state.mode (projection state.summary) events ⦄ := by
  induction events generalizing state with
  | nil => simp [ActionEffects.sequence, observeTrace]
  | cons event events ih =>
    have thisAllowed : admissible event := allowed event (by simp)
    have restAllowed : ∀ other ∈ events, admissible other := by
      intro other member
      exact allowed other (by simp [member])
    have oneRoom : Fits (projection state.summary) 1 :=
      fits_mono _ _ _ room (by simp)
    simp only [ActionEffects.sequence]
    step with source_apply state event thisAllowed oneRoom as
      ⟨middle, modeEq, arrivalsEq, observedEq⟩
    have restRoom : Fits (projection middle.summary) events.length := by
      rw [observedEq]
      apply observe_fits
      simpa only [List.length_cons, Nat.add_comm] using room
    step with ih middle restAllowed restRoom as ⟨next, nextMode, nextArrivals, observed⟩
    refine ⟨nextMode.trans modeEq, nextArrivals.trans arrivalsEq, ?_⟩
    simpa only [observeTrace, List.foldl_cons, observedEq, modeEq] using observed

theorem source_finish_no_arrivals (state : effects.FoldState)
    (empty : state.arrivals.val = []) : effects.finish state = .ok state.summary := by
  unfold effects.finish effects.finish_loop
  rw [loop.eq_def]
  have test : ¬0#usize < alloc.vec.Vec.len state.arrivals := by
    have length : state.arrivals.val.length = 0 := by rw [empty]; rfl
    scalar_tac
  simp [effects.finish_loop.body, test]

def initial (mode : effects.FoldMode) : effects.FoldState :=
  ⟨mode,
    ⟨alloc.vec.Vec.new _, alloc.vec.Vec.new _, alloc.vec.Vec.new _, alloc.vec.Vec.new _, false⟩,
    alloc.vec.Vec.new _, false⟩

theorem source_summarize (mode : effects.FoldMode) (events : List effects.Effect)
    (allowed : ∀ event ∈ events, admissible event) (room : events.length ≤ Usize.max) :
    summarize mode events ⦃ summary =>
      projection summary = observeTrace mode {} events ⦄ := by
  have capacity : Fits (projection (initial mode).summary) events.length := by
    simpa [Fits, projection, initial, alloc.vec.Vec.new, alloc.vec.Vec.val,
      alloc.vec.Vec.from, Slice.from_val] using And.intro room (And.intro room (And.intro room room))
  simp only [summarize, effects.begin, bind_ok]
  change (do
    let complete ← ActionEffects.sequence events (initial mode)
    effects.finish complete) ⦃ summary => projection summary = observeTrace mode {} events ⦄
  step with source_sequence events (initial mode) allowed capacity as
    ⟨complete, modeEq, arrivalsEq, observed⟩
  have noArrivals : complete.arrivals.val = [] := by
    rw [arrivalsEq]
    simp [initial, alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val]
  rw [source_finish_no_arrivals complete noArrivals]
  simpa [initial, projection, alloc.vec.Vec.new, alloc.vec.Vec.val,
    alloc.vec.Vec.from, Slice.from_val] using observed

theorem actual_fold_direct_equivalent (mode : effects.FoldMode) (trace : Slice effects.Effect)
    (allowed : ∀ event ∈ trace.val, admissible event) :
    effects.fold mode trace ⦃ summary =>
      projection summary = observeImmediate mode {} trace.val ⦄ := by
  rw [summarize_source]
  apply WP.spec_mono (source_summarize mode trace.val allowed (Slice.length_ineq trace))
  intro summary observed
  rw [direct_buffered_equivalent mode {} trace.val allowed]
  exact observed

end MemoryWrite.RecorderObservation
