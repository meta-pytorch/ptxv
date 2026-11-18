/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockTick
import MemoryWrite.RustClockPublication

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

universe u

namespace MemoryWrite.SparseClock

theorem slice_data_value {T : Type u} (base : alloc.vec.Vec T) :
    base.slice.val = base.val := rfl

def dataModel (data : Data) : State := ⟨baseModel data.base.val, entryModel data.own⟩

@[step]
theorem rust_single (actor stamp : U32) :
    clock.single actor stamp ⦃ result =>
      baseModel result.1.val = [(actor.val, stamp.val)] ∧ result.2 = (0#u32, 0#u32) ⦄ := by
  unfold clock.single
  step as ⟨base, pushed⟩
  · simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new, alloc.vec.Vec.val,
      alloc.vec.Vec.from, Slice.from_val]
    scalar_tac
  · simp_all [baseModel, entryModel, alloc.vec.Vec.with_capacity, alloc.vec.Vec.new,
      alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val]

@[step]
theorem rust_install_relocation {T : Type u} (base next : T) (own : U32 × U32) :
    clock.install_relocation base own next ⦃ result =>
      result.1 = next ∧ result.2 = (0#u32, 0#u32) ⦄ := by
  simp [clock.install_relocation]

@[step]
theorem rust_relocate (base : Slice (U32 × U32)) (own : U32 × U32) (actor : U32)
    (sorted : Sorted (baseModel base.val)) (room : base.val.length < Usize.max) :
    clock.relocate base own actor ⦃ result =>
      match result with
      | none => relocate ⟨baseModel base.val, entryModel own⟩ actor.val =
          ⟨baseModel base.val, entryModel own⟩
      | some next => relocate ⟨baseModel base.val, entryModel own⟩ actor.val =
          ⟨baseModel next.val, (0, 0)⟩ ⦄ := by
  rcases own with ⟨owner, stamp⟩
  have unfolded : clock.relocate base (owner, stamp) actor =
      (if stamp != 0#u32 then
        if owner != actor then do
          let next ← clock.entries base (owner, stamp)
          ok (some next)
        else ok none
      else ok none) := rfl
  rw [unfolded]
  by_cases zero : stamp = 0#u32
  · simp [zero, relocate, entryModel]
  · have nonzero : stamp.val ≠ 0 := by intro h; apply zero; scalar_tac
    by_cases same : owner = actor
    · simp [same, relocate, entryModel]
    · have different := UScalar.neq_to_neq_val.mp same
      simp only [bne_iff_ne, ne_eq, zero, same, not_false_eq_true, ite_true]
      step with rust_entries_bounded base (owner, stamp) sorted room as ⟨next, projected⟩
      simp [relocate, entryModel, nonzero, different, projected]

@[step]
theorem rust_tick_own (base : Slice (U32 × U32)) (own : U32 × U32) (actor : U32)
    (sorted : Sorted (baseModel base.val))
    (headroom : value (baseModel base.val) (entryModel own) actor.val < U32.max) :
    clock.tick_own base own actor ⦃ next => entryModel next =
      (actor.val, value (baseModel base.val) (entryModel own) actor.val + 1) ⦄ := by
  unfold clock.tick_own
  step with rust_get base own actor sorted as ⟨stamp, projected⟩
  step as ⟨next, nextValue⟩; try scalar_tac
  have hnext : next.val = value (baseModel base.val) (entryModel own) actor.val + 1 := by
    rw [nextValue, projected]
  exact congrArg (Prod.mk actor.val) hnext

end MemoryWrite.SparseClock

namespace MemoryWrite.SparseClock

def prepareData (data : Data) (actor : U32) : Result Data := do
  let plan ← clock.relocate data.base.slice data.own actor
  match plan with
  | none => pure data
  | some next =>
    let (base, own) ← clock.install_relocation data.base data.own next
    pure ⟨base, own⟩

@[step]
theorem source_prepare_data (data : Data) (actor : U32)
    (sorted : (dataModel data).WellFormed) (room : data.base.val.length < Usize.max) :
    prepareData data actor ⦃ result => dataModel result = relocate (dataModel data) actor.val ⦄ := by
  unfold prepareData
  step with rust_relocate data.base.slice data.own actor sorted room as ⟨plan, planned⟩
  cases plan with
  | none => simpa only [WP.spec_ok, dataModel, slice_data_value] using planned.symm
  | some next =>
    step with rust_install_relocation data.base next data.own as ⟨base, own, installed, cleared⟩
    rw [installed, cleared]
    simpa [dataModel, entryModel, slice_data_value] using planned.symm

def tickData (data : Data) (actor : U32) : Result Data := do
  let prepared ← prepareData data actor
  let own ← clock.tick_own prepared.base.slice prepared.own actor
  pure { prepared with own }

@[step]
theorem source_tick_data (data : Data) (actor : U32)
    (sorted : (dataModel data).WellFormed) (room : data.base.val.length < Usize.max)
    (headroom : projectData data actor.val < U32.max) :
    tickData data actor ⦃ result => dataModel result = tick (dataModel data) actor.val ⦄ := by
  unfold tickData
  step with source_prepare_data data actor sorted room as ⟨prepared, preparedModel⟩
  have preparedSorted : (dataModel prepared).WellFormed := by
    rw [preparedModel]
    exact relocation_preserves_sortedness (dataModel data) actor.val sorted
  have sameClock : projectData prepared = projectData data := by
    change (dataModel prepared).project = (dataModel data).project
    rw [preparedModel, relocation_preserves_projection]
  step with rust_tick_own prepared.base.slice prepared.own actor preparedSorted
      (by simpa only [projectData, slice_data_value] using
        (show projectData prepared actor.val < U32.max by rw [sameClock]; exact headroom)) as ⟨own, ownModel⟩
  unfold tick
  rw [← preparedModel]
  exact congrArg (State.mk (baseModel prepared.base.val)) ownModel

theorem source_tick_increments_one_actor_and_preserves_sortedness (data : Data) (actor : U32)
    (sorted : (dataModel data).WellFormed) (room : data.base.val.length < Usize.max)
    (headroom : projectData data actor.val < U32.max) :
    tickData data actor ⦃ result =>
      projectData result = BarrierReduction.update (projectData data) actor.val
        (projectData data actor.val + 1) ∧ (dataModel result).WellFormed ⦄ := by
  apply WP.spec_mono (source_tick_data data actor sorted room headroom)
  intro result projected
  constructor
  · change (dataModel result).project = _
    rw [projected]
    exact tick_increments_exactly_one_actor (dataModel data) actor.val
  · rw [projected]
    exact tick_preserves_sortedness (dataModel data) actor.val sorted

theorem source_tick_preserves_constructed_history (data : Data) (actor : U32)
    (constructed : Constructed (dataModel data)) (room : data.base.val.length < Usize.max)
    (headroom : projectData data actor.val < U32.max) :
    tickData data actor ⦃ result => Constructed (dataModel result) ⦄ := by
  apply WP.spec_mono (source_tick_data data actor
    (constructed_clocks_have_sorted_bases _ constructed) room headroom)
  intro result projected
  rw [projected]
  exact .ticked constructed actor.val

end MemoryWrite.SparseClock

namespace MemoryWrite.SparseClock

theorem source_empty_has_constructed_history :
    clock.empty ⦃ result => Constructed (dataModel ⟨result.1, result.2⟩) ⦄ := by
  apply WP.spec_mono rust_empty
  intro result projected
  rcases result with ⟨base, own⟩
  rcases projected with ⟨emptyBase, emptyOwn⟩
  change own = (0#u32, 0#u32) at emptyOwn
  rw [emptyOwn]
  simpa [dataModel, baseModel, emptyBase, entryModel, emptyState] using Constructed.empty

theorem source_single_has_constructed_history (actor stamp : U32) :
    clock.single actor stamp ⦃ result => Constructed (dataModel ⟨result.1, result.2⟩) ⦄ := by
  apply WP.spec_mono (rust_single actor stamp)
  intro result projected
  rcases result with ⟨base, own⟩
  rcases projected with ⟨singleBase, emptyOwn⟩
  rw [emptyOwn]
  simpa [dataModel, singleBase, entryModel, singleState] using Constructed.single actor.val stamp.val

theorem source_join_preserves_constructed_history (left right : Data) (sharedBase : Bool)
    (hl : Constructed (dataModel left)) (hr : Constructed (dataModel right))
    (room : left.base.val.length + right.base.val.length + 1 < Usize.max) :
    clock.join_plan left.base.slice left.own right.base.slice right.own sharedBase ⦃ plan =>
      Constructed (dataModel (applySourcePlan left right plan)) ⦄ := by
  apply WP.spec_mono (rust_join_plan_bounded left.base.slice right.base.slice left.own right.own
    sharedBase (constructed_clocks_have_sorted_bases _ hl)
    (constructed_clocks_have_sorted_bases _ hr) room)
  intro plan projected
  have baseProjection := apply_source_plan_base left right plan
  rw [projected] at baseProjection
  have ownProjection := apply_source_plan_own left right plan
  have exactState : dataModel (applySourcePlan left right plan) =
      joinState (dataModel left) (dataModel right) sharedBase := by
    simp only [dataModel, joinState]
    rw [baseProjection, ownProjection]
    simp only [slice_data_value]
  rw [exactState]
  exact .joined hl hr sharedBase

theorem source_get_from_constructed_history (data : Data) (actor : U32)
    (constructed : Constructed (dataModel data)) :
    clock.get data.base.slice data.own actor ⦃ result => result.val = projectData data actor.val ⦄ :=
  rust_get data.base.slice data.own actor (constructed_clocks_have_sorted_bases _ constructed)

end MemoryWrite.SparseClock
