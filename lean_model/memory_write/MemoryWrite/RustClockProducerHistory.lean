/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustClockFilter
import MemoryWrite.RustClockUpdates
import MemoryWrite.RustClockReadRelease

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

theorem source_result_satisfies {T : Type} (command : Result T) (result : T)
    (post : T → Prop) (run : command = .ok result) (verified : command ⦃ post ⦄) : post result := by
  obtain ⟨actual, executed, property⟩ := spec_returns _ _ verified
  rw [run] at executed
  have equal := Result.ok_injective executed
  simpa only [equal] using property

inductive ProducerHistory : Data → Prop
  | empty (base : alloc.vec.Vec (U32 × U32)) (own : U32 × U32)
      (run : clock.empty = .ok (base, own)) : ProducerHistory ⟨base, own⟩
  | single (actor stamp : U32) (base : alloc.vec.Vec (U32 × U32)) (own : U32 × U32)
      (run : clock.single actor stamp = .ok (base, own)) : ProducerHistory ⟨base, own⟩
  | clone {before after : Data} (origin : ProducerHistory before)
      (contents : after.base.val = before.base.val) (own : after.own = before.own) :
      ProducerHistory after
  | own_update {before : Data} (origin : ProducerHistory before) (own : U32 × U32) :
      ProducerHistory ⟨before.base, own⟩
  | materialized {before : Data} (origin : ProducerHistory before)
      (after : alloc.vec.Vec (U32 × U32)) (room : before.base.val.length < Usize.max)
      (run : clock.entries before.base.slice before.own = .ok after) :
      ProducerHistory ⟨after, (0#u32, 0#u32)⟩
  | prepared {before : Data} (origin : ProducerHistory before) (actor : U32) (after : Data)
      (room : before.base.val.length < Usize.max) (run : prepareData before actor = .ok after) :
      ProducerHistory after
  | joined {left right : Data} (leftOrigin : ProducerHistory left) (rightOrigin : ProducerHistory right)
      (shared : Bool) (plan : clock.JoinPlan)
      (same : shared = true → left.base.val = right.base.val)
      (room : left.base.val.length + right.base.val.length + 1 < Usize.max)
      (run : clock.join_plan left.base.slice left.own right.base.slice right.own shared = .ok plan) :
      ProducerHistory (applySourcePlan left right plan)
  | shared {C : Type} {before : Data} (origin : ProducerHistory before)
      (getter : clock.OwnEntry C) (clocks : Slice C) (project : C → Entry)
      (reads : ∀ c ∈ clocks.val, getter.own_entry c ⦃ own => entryModel own = project c ⦄)
      (base : alloc.vec.Vec (U32 × U32)) (own : U32 × U32)
      (room : before.base.val.length + clocks.val.length < Usize.max)
      (run : clock.join_shared getter before.base.slice clocks = .ok (base, own)) :
      ProducerHistory ⟨base, own⟩
  | met {left right : Data} (leftOrigin : ProducerHistory left) (rightOrigin : ProducerHistory right)
      (base : alloc.vec.Vec (U32 × U32)) (own : U32 × U32)
      (room : left.base.val.length + right.base.val.length + 2 < Usize.max)
      (run : clock.meet_clock left.base.slice left.own right.base.slice right.own = .ok (base, own)) :
      ProducerHistory ⟨base, own⟩
  | sorted_updates {before : Data} (origin : ProducerHistory before)
      (updates : Slice (U32 × U32)) (ordered : WeakSorted (baseModel updates.val))
      (base : alloc.vec.Vec (U32 × U32)) (own : U32 × U32)
      (room : before.base.val.length + 1 + updates.val.length < Usize.max)
      (run : clock.join_sorted_updates before.base.slice before.own updates = .ok (base, own)) :
      ProducerHistory ⟨base, own⟩
  | read_release {before : Data} (origin : ProducerHistory before)
      (lower reader done : U32) (base : alloc.vec.Vec (U32 × U32)) (own : U32 × U32)
      (room : before.base.val.length + 2 < Usize.max)
      (run : clock.read_release_clock before.base.slice before.own lower reader done = .ok (base, own)) :
      ProducerHistory ⟨base, own⟩
  | filtered {K : Type} {before : Data} (origin : ProducerHistory before)
      (getter : clock.KeepEntry K) (keep : K) (model : Entry → Bool)
      (reads : ∀ entry ∈ before.base.val,
        getter.keep_entry keep entry ⦃ selected => selected = model (entryModel entry) ⦄)
      (base : alloc.vec.Vec (U32 × U32)) (room : before.base.val.length < Usize.max)
      (run : clock.filter_entries getter before.base.slice keep = .ok base) :
      ProducerHistory ⟨base, (0#u32, 0#u32)⟩
  | uncovered {before seen : Data} (origin : ProducerHistory before) (seenOrigin : ProducerHistory seen)
      (base : alloc.vec.Vec (U32 × U32)) (own : U32 × U32)
      (room : before.base.val.length + 1 < Usize.max)
      (run : clock.uncovered_clock before.base.slice before.own seen.base.slice seen.own = .ok (some (base, own))) :
      ProducerHistory ⟨base, own⟩
  | tail {before : Data} (origin : ProducerHistory before) (lower : U32)
      (base : alloc.vec.Vec (U32 × U32)) (room : before.base.val.length < Usize.max)
      (run : clock.tail before.base.slice before.own lower = .ok base) :
      ProducerHistory ⟨base, (0#u32, 0#u32)⟩
  | repacked {before : Data} (origin : ProducerHistory before)
      (base : alloc.vec.Vec (U32 × U32)) (own : U32 × U32)
      (run : clock.from_sorted before.base = .ok (base, own)) : ProducerHistory ⟨base, own⟩

theorem producer_histories_have_sorted_bases (data : Data) (history : ProducerHistory data) :
    Sorted (baseModel data.base.val) := by
  induction history with
  | empty base own run =>
    have post := source_result_satisfies clock.empty (base, own) _ run rust_empty
    simp [post.1, baseModel, Sorted]
  | single actor stamp base own run =>
    have post := source_result_satisfies (clock.single actor stamp) (base, own) _ run (rust_single actor stamp)
    simp only [post.1, Sorted, List.pairwise_singleton]
  | clone origin contents own ih => simpa only [contents] using ih
  | own_update origin own ih => exact ih
  | materialized origin after room run ih =>
    have post := source_result_satisfies _ after _ run
      (rust_entries_bounded _ _ ih room)
    rw [post]
    exact overlay_sorted _ _ ih
  | @prepared before origin actor after room run ih =>
    have post := source_result_satisfies _ after _ run (source_prepare_data _ actor ih room)
    have ordered := relocation_preserves_sortedness (dataModel before) actor.val ih
    change (relocate (dataModel before) actor.val).WellFormed at ordered
    rw [←post] at ordered
    exact ordered
  | joined leftOrigin rightOrigin shared plan same room run ihl ihr =>
    have post := source_result_satisfies _ plan _ run
      (source_join_refines_pointwise_max _ _ shared ihl ihr same room)
    exact post.2.1
  | shared origin getter clocks project reads base own room run ih =>
    have post := source_result_satisfies _ (base, own) _ run
      (rust_join_shared getter _ clocks project reads ih room)
    exact post.2.2
  | met leftOrigin rightOrigin base own room run ihl ihr =>
    have post := source_result_satisfies _ (base, own) _ run
      (rust_meet_clock _ _ _ _ ihl ihr room)
    rw [post.1]
    exact meet_entries_sorted _ _ (overlay_sorted _ _ ihl)
  | sorted_updates origin updates ordered base own room run ih =>
    have post := source_result_satisfies _ (base, own) _ run
      (rust_join_sorted_updates _ updates _ ih room)
    rw [post.1]
    exact merge_updates_sorted _ _ (overlay_sorted _ _ ih) ordered
  | read_release origin lower reader done base own room run ih =>
    have post := source_result_satisfies _ (base, own) _ run
      (rust_read_release_clock _ _ lower reader done ih room)
    rw [post.1]
    exact release_entries_sorted _ _ _ (select_tail_sorted _ _ _ ih)
  | filtered origin getter keep model reads base room run ih =>
    obtain ⟨result, executed, projected, ordered⟩ :=
      source_filter_preserves_order getter _ keep model reads ih room
    rw [run] at executed
    have equal := Result.ok_injective executed
    simpa only [←equal] using ordered
  | uncovered origin seenOrigin base own room run ih ihSeen =>
    have post := source_result_satisfies _ (some (base, own)) _ run
      (rust_uncovered_clock _ _ _ _ ih ihSeen room)
    rw [post.1]
    exact novel_entries_sorted _ _ (overlay_sorted _ _ ih)
  | tail origin lower base room run ih =>
    have post := source_result_satisfies _ base _ run (rust_tail _ _ lower ih room)
    rw [post]
    exact select_tail_sorted _ _ _ ih
  | repacked origin base own run ih =>
    have post := source_result_satisfies _ (base, own) _ run (rust_from_sorted _)
    have contents : base = _ := post.1
    simpa only [contents] using ih

end MemoryWrite.SparseClock
