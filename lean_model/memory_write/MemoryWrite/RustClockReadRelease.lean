/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockReadRelease
import MemoryWrite.RustClockFilter
import MemoryWrite.RustClockUpdates

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

@[step] theorem rust_read_bit : clock.READ_BIT ⦃ bit => bit.val = readBit ⦄ := by
  unfold clock.READ_BIT
  step as ⟨bit, projected⟩
  simp only [readBit]
  scalar_tac

@[step] theorem rust_read_actor (actor : U32) :
    clock.read_actor actor ⦃ reader => reader.val = readActor actor.val ⦄ := by
  unfold clock.read_actor
  step with rust_read_bit as ⟨bit, projected⟩
  by_cases low : actor.val < readBit
  · have tested : actor < bit := by scalar_tac
    simp only [tested, ite_true]
    step as ⟨reader, readerValue⟩; try (simp only [readBit] at *; scalar_tac)
    simp only [readActor, low, ite_true]
    scalar_tac
  · have tested : ¬actor < bit := by scalar_tac
    simp only [tested, ite_false, WP.spec_ok, readActor, low, ite_false]

def ReadPartitionPrefix (source fresh inherited : List Entry) (index : Nat) : Prop :=
  index ≤ source.length ∧ fresh = freshReads (source.take index) ∧
    inherited = inheritedReads (source.take index)

@[step] theorem rust_read_partition_loop (source fresh inherited : alloc.vec.Vec (U32 × U32))
    (index : Usize) (room : source.val.length < Usize.max)
    (invariant : ReadPartitionPrefix (baseModel source.val) (baseModel fresh.val)
      (baseModel inherited.val) index.val) :
    clock.read_release_clock_loop source fresh inherited index ⦃ result =>
      baseModel result.1.val = freshReads (baseModel source.val) ∧
      baseModel result.2.val = inheritedReads (baseModel source.val) ⦄ := by
  unfold clock.read_release_clock_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (U32 × U32) × alloc.vec.Vec (U32 × U32) × Usize =>
      source.val.length - current.2.2.val)
    (fun current : alloc.vec.Vec (U32 × U32) × alloc.vec.Vec (U32 × U32) × Usize =>
      ReadPartitionPrefix (baseModel source.val) (baseModel current.1.val)
        (baseModel current.2.1.val) current.2.2.val)
  · intro current hcurrent
    rcases current with ⟨fresh, inherited, index⟩
    change ReadPartitionPrefix (baseModel source.val) (baseModel fresh.val)
      (baseModel inherited.val) index.val at hcurrent
    obtain ⟨bounded, freshModel, inheritedModel⟩ := hcurrent
    simp only [baseModel_length] at bounded
    have partition := read_partition_length ((baseModel source.val).take index.val)
    rw [←freshModel, ←inheritedModel] at partition
    simp only [baseModel_length, List.length_take, Nat.min_eq_left bounded] at partition
    unfold clock.read_release_clock_loop.body
    by_cases running : index.val < source.val.length
    · have tested : index < alloc.vec.Vec.len source := by scalar_tac
      simp only [tested, ite_true]
      step as ⟨actor, stamp, read⟩
      have prefixStep : (baseModel source.val).take (index.val + 1) =
          (baseModel source.val).take index.val ++ [entryModel (actor, stamp)] := by
        have step := List.take_succ_eq_append_getElem (show index.val < (baseModel source.val).length by simpa using running)
        have projected := baseModel_getElem source.val index.val running
        rw [←read] at projected
        simpa only [←projected] using step
      step with rust_read_bit as ⟨bit, bitValue⟩
      by_cases low : actor.val < readBit
      · have testedLow : actor < bit := by scalar_tac
        simp only [testedLow, ite_true]
        step with rust_read_actor actor as ⟨reader, readerValue⟩
        step with alloc.vec.Vec.push_spec fresh (reader, stamp) (by omega) as ⟨updated, pushed⟩
        step as ⟨next, nextValue⟩
        have increment : next.val = index.val + 1 := by scalar_tac
        refine ⟨⟨by simp only [baseModel_length]; omega, ?_, ?_⟩, by omega⟩
        · rw [increment, prefixStep, pushed, base_model_append, base_model_cons, base_model_nil, freshModel]
          simp [freshReads, readerValue, readActor, entryModel, low]
        · have high : ¬readBit ≤ actor.val := by omega
          simpa [increment, prefixStep, inheritedReads, entryModel, high] using inheritedModel
      · have testedLow : ¬actor < bit := by scalar_tac
        simp only [testedLow, ite_false]
        step with alloc.vec.Vec.push_spec inherited (actor, stamp) (by omega) as ⟨updated, pushed⟩
        step as ⟨next, nextValue⟩
        have increment : next.val = index.val + 1 := by scalar_tac
        refine ⟨⟨by simp only [baseModel_length]; omega, ?_, ?_⟩, by omega⟩
        · simpa [increment, prefixStep, freshReads, entryModel, low] using freshModel
        · have high : readBit ≤ actor.val := by omega
          rw [increment, prefixStep, pushed, base_model_append, base_model_cons, base_model_nil, inheritedModel]
          simp [inheritedReads, entryModel, high]
    · have tested : ¬index < alloc.vec.Vec.len source := by scalar_tac
      simp only [tested, ite_false, WP.spec_ok]
      have finished : (baseModel source.val).take index.val = baseModel source.val :=
        List.take_of_length_le (by simp only [baseModel_length]; omega)
      exact ⟨by simpa only [finished] using freshModel,
        by simpa only [finished] using inheritedModel⟩
  · exact invariant

@[step] theorem rust_read_release_clock (base : Slice (U32 × U32)) (own : U32 × U32)
    (lower reader done : U32) (ordered : Sorted (baseModel base.val))
    (room : base.val.length + 2 < Usize.max) :
    clock.read_release_clock base own lower reader done ⦃ result =>
      baseModel result.1.val = releaseEntries (selectTail (baseModel base.val) (entryModel own) lower.val)
        reader.val done.val ∧ result.2 = (0#u32, 0#u32) ⦄ := by
  unfold clock.read_release_clock
  step with rust_tail base own lower ordered (by omega) as ⟨source, tailModel⟩
  have length := select_tail_length (baseModel base.val) (entryModel own) lower.val
  rw [←tailModel] at length
  simp only [baseModel_length] at length
  step with rust_read_partition_loop source (alloc.vec.Vec.new (U32 × U32))
    (alloc.vec.Vec.new (U32 × U32)) 0#usize (by omega)
    (by simp [ReadPartitionPrefix, alloc.vec.Vec.new, alloc.vec.Vec.from,
      alloc.vec.Vec.val, Slice.from_val, freshReads, inheritedReads])
    as ⟨fresh, inherited, freshModel, inheritedModel⟩
  have partition := read_partition_length (baseModel source.val)
  rw [←freshModel, ←inheritedModel] at partition
  simp only [baseModel_length] at partition
  step with rust_merge_entries_bounded (alloc.vec.Vec.deref fresh) (alloc.vec.Vec.deref inherited)
    (by simp only [alloc.vec.Vec.deref, Slice.from_val]; omega) as ⟨merged, mergedModel⟩
  simp only [alloc.vec.Vec.deref, Slice.from_val] at mergedModel
  have tailSorted : Sorted (baseModel source.val) := by
    rw [tailModel]
    exact select_tail_sorted _ _ _ ordered
  have mergeSorted : Sorted (baseModel merged.val) := by
    rw [mergedModel, freshModel, inheritedModel]
    exact merge_entries_sorted _ _ (fresh_reads_sorted _ tailSorted) (inherited_reads_sorted _ tailSorted)
  have mergeLength := merge_entries_length (baseModel fresh.val) (baseModel inherited.val)
  rw [←mergedModel] at mergeLength
  simp only [baseModel_length] at mergeLength
  step with rust_read_actor reader as ⟨readKey, readKeyModel⟩
  step with rust_overlay_entry_bounded merged readKey done mergeSorted (by omega)
    as ⟨overlaid, overlaidModel⟩
  step with rust_from_sorted overlaid as ⟨result, imported, ownCleared⟩
  refine ⟨?_, ownCleared⟩
  rw [imported, overlaidModel, mergedModel, freshModel, inheritedModel, tailModel]
  simp only [releaseEntries, entryModel, readKeyModel]

theorem source_read_release_preserves_order_and_forwards_reads
    (base : Slice (U32 × U32)) (own : U32 × U32) (lower reader done : U32)
    (ordered : Sorted (baseModel base.val)) (room : base.val.length + 2 < Usize.max) :
    ∃ result, clock.read_release_clock base own lower reader done = .ok result ∧
      Sorted (baseModel result.1.val) ∧ result.2 = (0#u32, 0#u32) ∧
      value (baseModel result.1.val) (entryModel result.2) =
        readReleaseValue (value (baseModel base.val) (entryModel own)) lower.val reader.val done.val := by
  obtain ⟨⟨baseOut, ownOut⟩, run, projected, cleared⟩ :=
    spec_returns _ _ (rust_read_release_clock base own lower reader done ordered room)
  refine ⟨(baseOut, ownOut), run, ?_, cleared, ?_⟩
  · rw [projected]
    exact release_entries_sorted _ _ _ (select_tail_sorted _ _ _ ordered)
  · rw [cleared, projected]
    have meaning := release_entries_refines_read_projection (baseModel base.val) (entryModel own)
      lower.val reader.val done.val
    funext actor
    simpa [value, entryModel] using congrFun meaning actor

theorem source_read_actor_matches_native_bit_policy (actor : U32) :
    ∃ result, clock.read_actor actor = .ok result ∧ result.val = actor.val ||| readBit := by
  obtain ⟨result, run, projected⟩ := spec_returns _ _ (rust_read_actor actor)
  refine ⟨result, run, ?_⟩
  rw [projected]
  exact read_actor_matches_high_bit actor.val (by unfold readBit; scalar_tac)

end MemoryWrite.SparseClock
