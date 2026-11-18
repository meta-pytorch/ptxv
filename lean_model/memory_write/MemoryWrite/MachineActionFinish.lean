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
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineActionFinish

open MachineActionModel

theorem source_footprint (state : machine_action.ActionState) :
    machine_action.footprint state = .ok state.fp := by
  cases fpEq : state.fp <;> simp [machine_action.footprint, fpEq]

theorem source_finish (state : machine_action.ActionState) (forceWrite : Bool)
    (room : MachineRecorder.TraceRoom state.recorder) :
    (do
      let next ← machine_action.finish state forceWrite
      .ok (view next)) = finish (view state) forceWrite := by
  simp only [machine_action.finish, source_footprint, bind_ok,
    MachineFootprint.source_finish_context]
  let context : machine_recorder.FinishContext := ⟨state.fp.map (fun fp =>
    ⟨MachineFootprint.recordMode fp.sync_only,
      (fp.r.val ++ fp.w.val).any (fun range => decide (range.1 = 18#u8))⟩)⟩
  have contextEq : MachineRecorder.finishContextModel context = finishContext (view state) := by
    cases fpEq : state.fp <;>
      simp [MachineRecorder.finishContextModel, finishContext, context, view, fpEq]
  unfold finish
  rw [← contextEq]
  simp only [view]
  rw [← MachineRecorder.source_finish state.recorder context forceWrite room]
  simp only [Std.bind_assoc]
  congr 1
  funext result
  rcases result with ⟨summary, recorder⟩
  cases summary with
  | none => simp_all [view]
  | some summary =>
    cases fpEq : state.fp with
    | none => simp
    | some fp =>
      simp_all [MachineFootprint.source_merge_summary, Std.bind_assoc, view]

theorem source_range_clone (ranges : alloc.vec.Vec MachineFootprint.Range) :
    alloc.vec.CloneVec.clone (BuiltinClone MachineFootprint.Range) ranges
      ⦃ next => next = ranges ⦄ := by
  simp only [alloc.vec.CloneVec.clone, BuiltinClone]
  step with Slice.clone_spec (fun _ _ => rfl) as ⟨next, same⟩
  cases ranges
  simp_all

theorem source_key_clone (keys : alloc.vec.Vec U64) :
    alloc.vec.CloneVec.clone core.clone.CloneU64 keys ⦃ next => next = keys ⦄ := by
  simp only [alloc.vec.CloneVec.clone]
  step with Slice.clone_spec (fun _ _ => rfl) as ⟨next, same⟩
  cases keys
  simp_all

theorem source_footprint_clone (fp : machine_footprint.Footprint) :
    machine_footprint.Footprint.Insts.CoreCloneClone.clone fp ⦃ next => next = fp ⦄ := by
  simp only [machine_footprint.Footprint.Insts.CoreCloneClone.clone,
    core.clone.impls.CloneBool.clone, lift, bind_ok]
  step with source_range_clone fp.r as ⟨reads, readsEq⟩
  step with source_range_clone fp.w as ⟨writes, writesEq⟩
  step with source_key_clone fp.rel as ⟨releases, releasesEq⟩
  step with source_key_clone fp.acq as ⟨acquires, acquiresEq⟩
  simp_all

theorem source_snapshot (state : machine_action.ActionState) :
    machine_action.snapshot state ⦃ next =>
      next.fp = state.fp ∧ next.recorder.trace = none ⦄ := by
  simp only [machine_action.snapshot]
  cases fpEq : state.fp with
  | none =>
    simp [machine_recorder.fresh_snapshot, machine_recorder.new]
  | some fp =>
    step with source_footprint_clone fp as ⟨next, same⟩
    simp [same, machine_recorder.fresh_snapshot, machine_recorder.new]


theorem finish_consumes (state next : State) (forceWrite : Bool)
    (success : finish state forceWrite = .ok next) : next.recorder.trace = none := by
  unfold finish at success
  cases computed : MachineRecorderModel.finishAction state.recorder (finishContext state) forceWrite with
  | ret pair =>
    rcases pair with ⟨summary, recorder⟩
    have consumed := MachineRecorderModel.finish_consumes state.recorder
      (finishContext state).lower forceWrite summary recorder computed
    cases summary with
    | none =>
      simp only [computed, bind_ok] at success
      have same := Result.ok_injective success
      simpa [← same] using consumed
    | some summary =>
      cases fpEq : state.fp with
      | none => simp [computed, fpEq] at success
      | some fp =>
        cases merged : MachineFootprint.nativeMerge fp summary <;>
          simp_all
        simpa [← success] using consumed
  | vis effect continuation => simp [computed] at success
  | div => simp [computed] at success

theorem source_finish_consumes (state next : machine_action.ActionState) (forceWrite : Bool)
    (room : MachineRecorder.TraceRoom state.recorder)
    (success : machine_action.finish state forceWrite = .ok next) :
    next.recorder.trace = none := by
  have equality := source_finish state forceWrite room
  rw [success] at equality
  simp only [bind_ok] at equality
  have consumed := finish_consumes (view state) (view next) forceWrite equality.symm
  simpa [view, MachineRecorder.stateModel] using consumed

theorem source_finish_without_footprint (state : machine_action.ActionState) (forceWrite : Bool)
    (absent : state.fp = none) :
    machine_action.finish state forceWrite = .ok { state with recorder := ⟨none⟩ } := by
  simp [machine_action.finish, source_footprint, absent,
    MachineFootprint.source_finish_context, MachineRecorder.source_finish_without_footprint]

theorem source_finish_idle (state : machine_action.ActionState) (forceWrite : Bool)
    (idle : state.recorder.trace = none) :
    machine_action.finish state forceWrite = .ok state := by
  rcases state with ⟨fp, ⟨trace⟩⟩
  simp only at idle
  subst trace
  simp [machine_action.finish, source_footprint, MachineFootprint.source_finish_context,
    MachineRecorder.source_finish_idle]


theorem finish_summary_frame (state : State) (forceWrite : Bool)
    (fp : machine_footprint.Footprint) (summary : effects.Summary)
    (recorder : MachineRecorderModel.Recorder)
    (footprint : state.fp = some fp)
    (computed : MachineRecorderModel.finishAction state.recorder (finishContext state) forceWrite =
      .ok (some summary, recorder))
    (readRoom : fp.r.val.length + summary.reads.val.length + summary.writes.val.length < Usize.max)
    (writeRoom : fp.w.val.length + summary.reads.val.length + summary.writes.val.length < Usize.max)
    (relRoom : fp.rel.val.length + summary.releases.val.length + summary.acquires.val.length < Usize.max)
    (acqRoom : fp.acq.val.length + summary.releases.val.length + summary.acquires.val.length < Usize.max) :
    finish state forceWrite ⦃ next => ∃ updated,
      next.fp = some updated ∧ next.recorder.trace = none ∧
      updated.sync_only = fp.sync_only ∧ updated.all = (fp.all || summary.all) ∧
      updated.rel.val = fp.rel.val ++ summary.releases.val ∧
      updated.acq.val = fp.acq.val ++ summary.acquires.val ⦄ := by
  have consumed := MachineRecorderModel.finish_consumes state.recorder
    (finishContext state).lower forceWrite (some summary) recorder computed
  simp [finish, computed, footprint]
  rw [← MachineFootprint.source_merge_summary]
  step with MachineFootprint.source_merge_summary_keys fp summary
    readRoom writeRoom relRoom acqRoom as ⟨updated, mode, all, rel, acq⟩
  exact ⟨updated, rfl, consumed, mode, all, rel, acq⟩

end MemoryWrite.MachineActionFinish
