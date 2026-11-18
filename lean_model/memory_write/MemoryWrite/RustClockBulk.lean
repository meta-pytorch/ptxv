/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockBulk
import MemoryWrite.RustClockTick

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

universe u

namespace MemoryWrite.SparseClock

def OwnPrefix {C : Type u} (clocks : List C) (project : C → Entry)
    (output : List Entry) (result : List Entry) (index initialLength : Nat) : Prop :=
  index ≤ clocks.length ∧ Sorted result ∧ result.length ≤ initialLength + index ∧
    foldOwns result ((clocks.drop index).map project) = output

@[step]
theorem rust_join_shared_loop {C : Type u} (getter : clock.OwnEntry C)
    (clocks : Slice C) (project : C → Entry) (output : List Entry)
    (result : alloc.vec.Vec (U32 × U32)) (index : Usize) (initialLength : Nat)
    (reads : ∀ c ∈ clocks.val, getter.own_entry c ⦃ own => entryModel own = project c ⦄)
    (room : initialLength + clocks.val.length < Usize.max)
    (invariant : OwnPrefix clocks.val project output (baseModel result.val) index.val initialLength) :
    clock.join_shared_loop getter clocks result index ⦃ done =>
      baseModel done.val = output ∧ Sorted (baseModel done.val) ⦄ := by
  unfold clock.join_shared_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (U32 × U32) × Usize => clocks.val.length - current.2.val)
    (fun current : alloc.vec.Vec (U32 × U32) × Usize =>
      OwnPrefix clocks.val project output (baseModel current.1.val) current.2.val initialLength)
  · intro current hcurrent
    rcases current with ⟨result, index⟩
    obtain ⟨bounded, sorted, lengthBound, remaining⟩ := hcurrent
    unfold clock.join_shared_loop.body
    by_cases running : index.val < clocks.val.length
    · have tested : index < Slice.len clocks := by scalar_tac
      simp only [tested, ite_true]
      step with Slice.index_usize_spec clocks index running as ⟨c, read⟩
      have member : c ∈ clocks.val := by rw [read]; exact List.getElem_mem running
      step with reads c member as ⟨actor, stamp, projected⟩
      step with rust_overlay_entry_bounded result actor stamp sorted (by simp only [baseModel_length] at lengthBound; omega)
        as ⟨updated, updatedModel⟩
      step as ⟨next, nextValue⟩
      have increment : next.val = index.val + 1 := by scalar_tac
      have drop : clocks.val.drop index.val = c :: clocks.val.drop (index.val + 1) := by
        rw [List.drop_eq_getElem_cons running, ←read]
      refine ⟨⟨by omega, ?_, ?_, ?_⟩, by omega⟩
      · rw [updatedModel]; exact overlay_sorted _ _ sorted
      · have grow := overlay_length (baseModel result.val) (entryModel (actor, stamp))
        rw [←updatedModel] at grow
        simp only [baseModel_length] at grow lengthBound ⊢
        omega
      · rw [drop, List.map_cons, foldOwns] at remaining
        simpa only [increment, updatedModel, projected] using remaining
    · have tested : ¬index < Slice.len clocks := by scalar_tac
      simp only [tested, ite_false, WP.spec_ok]
      have empty : clocks.val.drop index.val = [] := List.drop_eq_nil_iff.mpr (by omega)
      exact ⟨by simpa only [empty, List.map_nil, foldOwns] using remaining, sorted⟩
  · exact invariant

@[step]
theorem rust_join_shared {C : Type u} (getter : clock.OwnEntry C)
    (base : Slice (U32 × U32)) (clocks : Slice C) (project : C → Entry)
    (reads : ∀ c ∈ clocks.val, getter.own_entry c ⦃ own => entryModel own = project c ⦄)
    (sorted : Sorted (baseModel base.val))
    (room : base.val.length + clocks.val.length < Usize.max) :
    clock.join_shared getter base clocks ⦃ result =>
      baseModel result.1.val = foldOwns (baseModel base.val) (clocks.val.map project) ∧
      result.2 = (0#u32, 0#u32) ∧ Sorted (baseModel result.1.val) ⦄ := by
  unfold clock.join_shared
  step with alloc.slice.Slice.to_vec_spec (BuiltinClone (U32 × U32)) base
    (by intro x hx; rfl) as ⟨result, clone⟩
  have contents : result.val = base.val := congrArg Slice.val clone.symm
  step with rust_join_shared_loop getter clocks project
      (foldOwns (baseModel base.val) (clocks.val.map project)) result 0#usize base.val.length
      reads room (by simpa [OwnPrefix, contents] using sorted) as ⟨done, projected, ordered⟩
  exact ⟨projected, ordered⟩

theorem source_shared_join_refines_pointwise_maximum {C : Type u} (getter : clock.OwnEntry C)
    (base : Slice (U32 × U32)) (clocks : Slice C) (project : C → Entry)
    (reads : ∀ c ∈ clocks.val, getter.own_entry c ⦃ own => entryModel own = project c ⦄)
    (sorted : Sorted (baseModel base.val))
    (room : base.val.length + clocks.val.length < Usize.max) :
    ∃ result, clock.join_shared getter base clocks = .ok result ∧
      Sorted (baseModel result.1.val) ∧
      value (baseModel result.1.val) (entryModel result.2) =
        BarrierReduction.join (entriesValue (baseModel base.val))
          (entriesValue (clocks.val.map project)) := by
  obtain ⟨⟨done, own⟩, run, projected, cleared, ordered⟩ :=
    spec_returns _ _ (rust_join_shared getter base clocks project reads sorted room)
  refine ⟨(done, own), run, ordered, ?_⟩
  rw [cleared, projected]
  have equality := fold_owns_accumulates_pointwise_maximum (baseModel base.val) (clocks.val.map project)
  funext actor
  simpa [value, entryModel] using congrFun equality actor

end MemoryWrite.SparseClock
