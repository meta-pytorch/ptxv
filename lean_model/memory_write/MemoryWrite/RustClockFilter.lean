/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockFilter
import MemoryWrite.RustClockMeet

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

universe u

namespace MemoryWrite.SparseClock

def FilterPrefix (entries output result : List (U32 × U32))
    (keep : (U32 × U32) → Bool) (index : Nat) : Prop :=
  index ≤ entries.length ∧ result.length ≤ index ∧
    result ++ (entries.drop index).filter keep = output

@[step]
theorem rust_filter_loop {K : Type u} (getter : clock.KeepEntry K)
    (entries : Slice (U32 × U32)) (keep : K) (model : (U32 × U32) → Bool)
    (output : List (U32 × U32)) (result : alloc.vec.Vec (U32 × U32)) (index : Usize)
    (reads : ∀ entry ∈ entries.val, getter.keep_entry keep entry ⦃ selected => selected = model entry ⦄)
    (room : entries.val.length < Usize.max)
    (initial : FilterPrefix entries.val output result.val model index.val) :
    clock.filter_entries_loop getter entries keep result index ⦃ done => done.val = output ⦄ := by
  unfold clock.filter_entries_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (U32 × U32) × Usize => entries.val.length - current.2.val)
    (fun current : alloc.vec.Vec (U32 × U32) × Usize =>
      FilterPrefix entries.val output current.1.val model current.2.val)
  · intro current invariant
    rcases current with ⟨result, index⟩
    change FilterPrefix entries.val output result.val model index.val at invariant
    obtain ⟨bounded, lengthBound, remaining⟩ := invariant
    unfold clock.filter_entries_loop.body
    by_cases running : index.val < entries.val.length
    · have tested : index < Slice.len entries := by scalar_tac
      simp only [tested, ite_true]
      step with Slice.index_usize_spec entries index running as ⟨entry, read⟩
      have member : entry ∈ entries.val := by rw [read]; exact List.getElem_mem running
      step with reads entry member as ⟨selected, projected⟩
      have drop : entries.val.drop index.val = entry :: entries.val.drop (index.val + 1) := by
        rw [List.drop_eq_getElem_cons running, ←read]
      cases chosen : selected with
      | true =>
        simp only [ite_true]
        step with alloc.vec.Vec.push_spec result entry (by omega) as ⟨updated, pushed⟩
        step as ⟨next, nextValue⟩
        have increment : next.val = index.val + 1 := by scalar_tac
        have included : model entry = true := by simpa only [chosen] using projected.symm
        refine ⟨⟨by omega, ?_, ?_⟩, by omega⟩
        · simp only [pushed, List.length_append, List.length_singleton]; omega
        · rw [drop, List.filter_cons, included] at remaining
          simpa only [pushed, increment, List.append_assoc, List.singleton_append,
            ite_true] using remaining
      | false =>
        simp only [Bool.false_eq_true, ite_false]
        step as ⟨next, nextValue⟩
        have increment : next.val = index.val + 1 := by scalar_tac
        have excluded : model entry = false := by simpa only [chosen] using projected.symm
        refine ⟨⟨by omega, by omega, ?_⟩, by omega⟩
        rw [drop, List.filter_cons, excluded] at remaining
        simpa only [increment, Bool.false_eq_true, ite_false] using remaining
    · have tested : ¬index < Slice.len entries := by scalar_tac
      simp only [tested, ite_false, WP.spec_ok]
      have empty : entries.val.drop index.val = [] := List.drop_eq_nil_iff.mpr (by omega)
      simpa only [empty, List.filter_nil, List.append_nil] using remaining
  · exact initial

@[step]
theorem rust_filter_entries {K : Type u} (getter : clock.KeepEntry K)
    (entries : Slice (U32 × U32)) (keep : K) (model : (U32 × U32) → Bool)
    (reads : ∀ entry ∈ entries.val, getter.keep_entry keep entry ⦃ selected => selected = model entry ⦄)
    (room : entries.val.length < Usize.max) :
    clock.filter_entries getter entries keep ⦃ result => result.val = entries.val.filter model ⦄ := by
  unfold clock.filter_entries
  apply rust_filter_loop getter entries keep model (entries.val.filter model)
    (alloc.vec.Vec.new _) 0#usize reads room
  simp [FilterPrefix, alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val]

@[step]
theorem rust_suffix (base : Slice (U32 × U32)) (start : Usize)
    (bounded : start.val ≤ base.val.length) :
    core.slice.index.SliceIndexRangeUsizeSlice.index
      { start, «end» := Slice.len base } base ⦃ result => result.val = base.val.drop start.val ⦄ := by
  have checked : start ≤ Slice.len base ∧ Slice.len base ≤ base.length := by scalar_tac
  simp only [core.slice.index.SliceIndexRangeUsizeSlice.index, checked]
  simp [List.slice, Slice.len]

@[step]
theorem rust_tail (base : Slice (U32 × U32)) (own : U32 × U32) (lower : U32)
    (sorted : Sorted (baseModel base.val)) (room : base.val.length < Usize.max) :
    clock.tail base own lower ⦃ result =>
      baseModel result.val = selectTail (baseModel base.val) (entryModel own) lower.val ⦄ := by
  rcases own with ⟨owner, stamp⟩
  unfold clock.tail
  step with rust_locate base lower sorted as ⟨start, found, located⟩
  step with rust_suffix base start (by simpa using located.1) as ⟨suffix, selected⟩
  step with alloc.slice.Slice.to_vec_spec (BuiltinClone (U32 × U32)) suffix
    (by intro x hx; rfl) as ⟨result, clone⟩
  have contents : result.val = base.val.drop start.val := by
    rw [←selected]; exact congrArg Slice.val clone.symm
  have cut : baseModel result.val =
      (baseModel base.val).filter (fun entry => decide (lower.val ≤ entry.1)) := by
    rw [contents, baseModel, List.map_drop]
    exact (located_filter_eq_drop (baseModel base.val) lower.val start.val found located).symm
  have ordered : Sorted (baseModel result.val) := by
    rw [cut]; exact filter_preserves_sortedness _ _ sorted
  by_cases included : owner >= lower
  · simp only [included, ite_true]
    step with rust_overlay_entry_bounded result owner stamp ordered
      (by rw [contents, List.length_drop]; omega) as ⟨updated, projected⟩
    have bound : lower.val ≤ owner.val := by scalar_tac
    simpa only [selectTail, entryModel, bound, ite_true, cut] using projected
  · have bound : ¬lower.val ≤ owner.val := by scalar_tac
    simp only [included, ite_false, WP.spec_ok, selectTail, entryModel, bound, cut]

@[step]
theorem rust_from_sorted (entries : alloc.vec.Vec (U32 × U32)) :
    clock.from_sorted entries ⦃ result => result.1 = entries ∧ result.2 = (0#u32, 0#u32) ⦄ := by
  simp [clock.from_sorted]

theorem source_tail_preserves_order_and_projects_threshold
    (base : Slice (U32 × U32)) (own : U32 × U32) (lower : U32)
    (sorted : Sorted (baseModel base.val)) (room : base.val.length < Usize.max) :
    ∃ result, clock.tail base own lower = .ok result ∧ Sorted (baseModel result.val) ∧
      entriesValue (baseModel result.val) = fun actor =>
        if lower.val ≤ actor then value (baseModel base.val) (entryModel own) actor else 0 := by
  obtain ⟨result, run, projected⟩ := spec_returns _ _ (rust_tail base own lower sorted room)
  refine ⟨result, run, ?_, ?_⟩
  · rw [projected]; exact select_tail_sorted _ _ _ sorted
  · rw [projected]; funext actor; exact select_tail_value _ _ _ _

theorem source_filter_preserves_order {K : Type u} (getter : clock.KeepEntry K)
    (entries : Slice (U32 × U32)) (keep : K) (model : Entry → Bool)
    (reads : ∀ entry ∈ entries.val, getter.keep_entry keep entry ⦃ selected => selected = model (entryModel entry) ⦄)
    (sorted : Sorted (baseModel entries.val)) (room : entries.val.length < Usize.max) :
    ∃ result, clock.filter_entries getter entries keep = .ok result ∧
      baseModel result.val = (baseModel entries.val).filter model ∧ Sorted (baseModel result.val) := by
  obtain ⟨result, run, projected⟩ := spec_returns _ _
    (rust_filter_entries getter entries keep (fun entry => model (entryModel entry)) reads room)
  refine ⟨result, run, ?_, ?_⟩
  · simp only [projected, baseModel, List.filter_map, Function.comp_def]
  · rw [projected]
    have ordered : entries.val.Pairwise (fun a b => a.1.val < b.1.val) := by
      simpa only [Sorted, baseModel, List.pairwise_map, entryModel] using sorted
    simpa only [Sorted, baseModel, List.pairwise_map, entryModel] using
      ordered.filter (fun entry => model (entryModel entry))

@[step]
theorem rust_uncovered_reader (reader : clock.Uncovered) (entry : U32 × U32)
    (sorted : Sorted (baseModel reader.base.val)) :
    clock.Uncovered.Insts.Ptxv_memory_writeClockKeepEntry.keep_entry reader entry
      ⦃ selected => selected = decide
        (value (baseModel reader.base.val) (entryModel reader.own) entry.1.val < entry.2.val) ⦄ := by
  rcases entry with ⟨actor, stamp⟩
  unfold clock.Uncovered.Insts.Ptxv_memory_writeClockKeepEntry.keep_entry
  step with rust_get reader.base reader.own actor sorted as ⟨known, projected⟩
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq]
  scalar_tac

@[step]
theorem rust_uncovered_clock (base seen : Slice (U32 × U32)) (own seenOwn : U32 × U32)
    (sorted : Sorted (baseModel base.val)) (seenSorted : Sorted (baseModel seen.val))
    (room : base.val.length + 1 < Usize.max) :
    clock.uncovered_clock base own seen seenOwn ⦃ result =>
      match result with
      | none => novelEntries (overlay (baseModel base.val) (entryModel own))
          (value (baseModel seen.val) (entryModel seenOwn)) = []
      | some (selected, nextOwn) =>
          baseModel selected.val = novelEntries (overlay (baseModel base.val) (entryModel own))
            (value (baseModel seen.val) (entryModel seenOwn)) ∧
          nextOwn = (0#u32, 0#u32) ∧ selected.val ≠ [] ⦄ := by
  unfold clock.uncovered_clock
  step with rust_entries_bounded base own sorted (by omega) as ⟨materialized, materializedModel⟩
  have lengthBound := overlay_length (baseModel base.val) (entryModel own)
  rw [←materializedModel] at lengthBound
  simp only [baseModel_length] at lengthBound
  let reader : clock.Uncovered := ⟨seen, seenOwn⟩
  let selectedModel : (U32 × U32) → Bool := fun entry => decide
    (value (baseModel seen.val) (entryModel seenOwn) entry.1.val < entry.2.val)
  step with rust_filter_entries clock.Uncovered.Insts.Ptxv_memory_writeClockKeepEntry
      (alloc.vec.Vec.deref materialized) reader selectedModel
      (by intro entry member; exact rust_uncovered_reader reader entry seenSorted)
      (by simp only [alloc.vec.Vec.deref, Slice.from_val]; omega) as ⟨selected, filtered⟩
  have projected : baseModel selected.val =
      novelEntries (overlay (baseModel base.val) (entryModel own))
        (value (baseModel seen.val) (entryModel seenOwn)) := by
    rw [filtered]
    simp only [alloc.vec.Vec.deref, Slice.from_val]
    rw [←materializedModel]
    simp only [baseModel, novelEntries, List.filter_map, entryModel, Function.comp_def]
    rfl
  by_cases empty : alloc.vec.Vec.len selected = 0#usize
  · simp only [empty, ite_true, WP.spec_ok]
    have emptyList : selected.val = [] := by
      apply List.length_eq_zero_iff.mp
      have lengthZero := congrArg (fun n : Usize => n.val) empty
      simpa using lengthZero
    rw [←projected, emptyList]
    rfl
  · have nonempty : selected.val ≠ [] := by
      intro h
      apply empty
      have lengthZero : selected.val.length = 0 := by simp only [h, List.length_nil]
      scalar_tac
    simp only [empty, ite_false, clock.from_sorted, bind_ok, WP.spec_ok]
    exact ⟨projected, trivial, nonempty⟩

theorem source_uncovered_preserves_order_and_new_clock_values
    (base seen : Slice (U32 × U32)) (own seenOwn : U32 × U32)
    (sorted : Sorted (baseModel base.val)) (seenSorted : Sorted (baseModel seen.val))
    (room : base.val.length + 1 < Usize.max) :
    ∃ result, clock.uncovered_clock base own seen seenOwn = .ok result ∧
      match result with
      | none => novelEntries (overlay (baseModel base.val) (entryModel own))
          (value (baseModel seen.val) (entryModel seenOwn)) = []
      | some (selected, nextOwn) => Sorted (baseModel selected.val) ∧ nextOwn = (0#u32, 0#u32) ∧
          entriesValue (baseModel selected.val) = fun actor =>
            let incoming := value (baseModel base.val) (entryModel own) actor
            if value (baseModel seen.val) (entryModel seenOwn) actor < incoming then incoming else 0 := by
  obtain ⟨result, run, projected⟩ := spec_returns _ _
    (rust_uncovered_clock base seen own seenOwn sorted seenSorted room)
  refine ⟨result, run, ?_⟩
  cases result with
  | none => exact projected
  | some pair =>
    rcases pair with ⟨selected, nextOwn⟩
    refine ⟨?_, projected.2.1, ?_⟩
    · rw [projected.1]
      exact novel_entries_sorted _ _ (overlay_sorted _ _ sorted)
    · rw [projected.1]
      funext actor
      simpa only [overlay_value] using novel_entries_value
        (overlay (baseModel base.val) (entryModel own))
        (value (baseModel seen.val) (entryModel seenOwn)) actor (overlay_sorted _ _ sorted)

theorem source_uncovered_none_iff_no_new_entries
    (base seen : Slice (U32 × U32)) (own seenOwn : U32 × U32)
    (sorted : Sorted (baseModel base.val)) (seenSorted : Sorted (baseModel seen.val))
    (room : base.val.length + 1 < Usize.max) :
    ∃ result, clock.uncovered_clock base own seen seenOwn = .ok result ∧
      (result = none ↔ novelEntries (overlay (baseModel base.val) (entryModel own))
        (value (baseModel seen.val) (entryModel seenOwn)) = []) := by
  obtain ⟨result, run, projected⟩ := spec_returns _ _
    (rust_uncovered_clock base seen own seenOwn sorted seenSorted room)
  refine ⟨result, run, ?_⟩
  cases result with
  | none => simp only [true_iff]; exact projected
  | some pair =>
    rcases pair with ⟨selected, nextOwn⟩
    simp only [Option.some_ne_none, false_iff]
    intro empty
    apply projected.2.2
    have noEntries : baseModel selected.val = [] := projected.1.trans empty
    simpa only [baseModel, List.map_eq_nil_iff] using noEntries

end MemoryWrite.SparseClock
