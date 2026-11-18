/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockMeet
import MemoryWrite.RustClockBulk

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

def rawMeet : List (U32 × U32) → List (U32 × U32) → List (U32 × U32)
  | [], _ => []
  | _, [] => []
  | first :: left, second :: right =>
    if first.1 < second.1 then rawMeet left (second :: right)
    else if second.1 < first.1 then rawMeet (first :: left) right
    else if core.cmp.impls.OrdU32.min first.2 second.2 = 0#u32 then rawMeet left right
    else (first.1, core.cmp.impls.OrdU32.min first.2 second.2) :: rawMeet left right
termination_by left right => left.length + right.length

theorem raw_meet_model (left right : List (U32 × U32)) :
    baseModel (rawMeet left right) = meetEntries (baseModel left) (baseModel right) := by
  fun_induction rawMeet left right with
  | case1 right => simp [baseModel, meetEntries]
  | case2 left h => cases left <;> simp [baseModel, meetEntries]
  | case3 first left second right hl ih =>
    have h : first.1.val < second.1.val := by scalar_tac
    simpa [baseModel, entryModel, meetEntries, h] using ih
  | case4 first left second right hn hl ih =>
    have hn' : ¬first.1.val < second.1.val := by scalar_tac
    have h : second.1.val < first.1.val := by scalar_tac
    simpa [baseModel, entryModel, meetEntries, hn', h] using ih
  | case5 first left second right hn hn' hz ih =>
    have h : ¬first.1.val < second.1.val := by scalar_tac
    have h' : ¬second.1.val < first.1.val := by scalar_tac
    have zero : min first.2.val second.2.val = 0 := by scalar_tac
    simpa [baseModel, entryModel, meetEntries, h, h', zero] using ih
  | case6 first left second right hn hn' hz ih =>
    have h : ¬first.1.val < second.1.val := by scalar_tac
    have h' : ¬second.1.val < first.1.val := by scalar_tac
    have nonzero : min first.2.val second.2.val ≠ 0 := by scalar_tac
    simpa [baseModel, entryModel, meetEntries, h, h', nonzero] using
      congrArg (List.cons (entryModel (first.1, core.cmp.impls.OrdU32.min first.2 second.2))) ih

def MeetPrefix (left right output result : List (U32 × U32)) (i j : Nat) : Prop :=
  i ≤ left.length ∧ j ≤ right.length ∧ result.length ≤ i + j ∧
    result ++ rawMeet (left.drop i) (right.drop j) = output

@[step]
theorem rust_meet_loop (left right : Slice (U32 × U32))
    (output : List (U32 × U32)) (result : alloc.vec.Vec (U32 × U32)) (i j : Usize)
    (room : left.val.length + right.val.length < Usize.max)
    (initial : MeetPrefix left.val right.val output result.val i.val j.val) :
    clock.meet_entries_loop left right result i j ⦃ done => done.val = output ⦄ := by
  unfold clock.meet_entries_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (U32 × U32) × Usize × Usize =>
      left.val.length - current.2.1.val + (right.val.length - current.2.2.val))
    (fun current : alloc.vec.Vec (U32 × U32) × Usize × Usize =>
      MeetPrefix left.val right.val output current.1.val current.2.1.val current.2.2.val)
  · intro current invariant
    rcases current with ⟨result, i, j⟩
    change MeetPrefix left.val right.val output result.val i.val j.val at invariant
    obtain ⟨leftBound, rightBound, lengthBound, remaining⟩ := invariant
    unfold clock.meet_entries_loop.body
    by_cases leftRunning : i.val < left.val.length
    · have leftTest : i < Slice.len left := by scalar_tac
      simp only [leftTest, ite_true]
      by_cases rightRunning : j.val < right.val.length
      · have rightTest : j < Slice.len right := by scalar_tac
        simp only [rightTest, ite_true]
        step with Slice.index_usize_spec left i leftRunning as ⟨a, leftRead⟩
        rcases a with ⟨p, cp⟩
        step with Slice.index_usize_spec right j rightRunning as ⟨q, cq, rightRead⟩
        change (if p < q then do
            let ni ← i + 1#usize
            ok (ControlFlow.cont (result, ni, j))
          else if q < p then do
            let nj ← j + 1#usize
            ok (ControlFlow.cont (result, i, nj))
          else do
            let stamp ← lift (core.cmp.impls.OrdU32.min cp cq)
            let updated ← if stamp > 0#u32 then alloc.vec.Vec.push result (p, stamp) else ok result
            let ni ← i + 1#usize
            let nj ← j + 1#usize
            ok (ControlFlow.cont (updated, ni, nj))) ⦃ _ ⦄
        have leftDrop : left.val.drop i.val = (p, cp) :: left.val.drop (i.val + 1) := by
          rw [List.drop_eq_getElem_cons leftRunning, ←leftRead]
        have rightDrop : right.val.drop j.val = (q, cq) :: right.val.drop (j.val + 1) := by
          rw [List.drop_eq_getElem_cons rightRunning, ←rightRead]
        by_cases less : p < q
        · simp only [less, ite_true]
          step as ⟨ni, nextI⟩
          have increment : ni.val = i.val + 1 := by scalar_tac
          refine ⟨⟨by omega, rightBound, by omega, ?_⟩, by omega⟩
          rw [leftDrop, rightDrop, rawMeet, ite_eq_left less] at remaining
          simpa only [increment, rightDrop] using remaining
        · simp only [less, ite_false]
          by_cases greater : q < p
          · simp only [greater, ite_true]
            step as ⟨nj, nextJ⟩
            have increment : nj.val = j.val + 1 := by scalar_tac
            refine ⟨⟨leftBound, by omega, by omega, ?_⟩, by omega⟩
            rw [leftDrop, rightDrop, rawMeet, ite_eq_right less, ite_eq_left greater] at remaining
            simpa only [increment, leftDrop] using remaining
          · simp only [greater, ite_false]
            step as ⟨stamp, minimum⟩
            have minEq : stamp = core.cmp.impls.OrdU32.min cp cq := by scalar_tac
            by_cases positive : stamp > 0#u32
            · simp only [positive, ite_true]
              step with alloc.vec.Vec.push_spec result (p, stamp) (by omega) as ⟨updated, pushed⟩
              step as ⟨ni, nextI⟩
              step as ⟨nj, nextJ⟩
              have incrementI : ni.val = i.val + 1 := by scalar_tac
              have incrementJ : nj.val = j.val + 1 := by scalar_tac
              have nonzero : core.cmp.impls.OrdU32.min cp cq ≠ 0#u32 := by scalar_tac
              refine ⟨⟨by omega, by omega, ?_, ?_⟩, by omega⟩
              · simp only [pushed, List.length_append, List.length_singleton]; omega
              · rw [leftDrop, rightDrop, rawMeet, ite_eq_right less, ite_eq_right greater,
                  ite_eq_right nonzero] at remaining
                simpa only [pushed, incrementI, incrementJ, List.append_assoc,
                  List.singleton_append, minEq] using remaining
            · simp only [positive, ite_false]
              step as ⟨ni, nextI⟩
              step as ⟨nj, nextJ⟩
              have incrementI : ni.val = i.val + 1 := by scalar_tac
              have incrementJ : nj.val = j.val + 1 := by scalar_tac
              have zero : core.cmp.impls.OrdU32.min cp cq = 0#u32 := by scalar_tac
              refine ⟨⟨by omega, by omega, by omega, ?_⟩, by omega⟩
              rw [leftDrop, rightDrop, rawMeet, ite_eq_right less, ite_eq_right greater,
                ite_eq_left zero] at remaining
              simpa only [incrementI, incrementJ] using remaining
      · have rightTest : ¬j < Slice.len right := by scalar_tac
        simp only [rightTest, ite_false, WP.spec_ok]
        have empty : right.val.drop j.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        have meetEmpty : rawMeet (left.val.drop i.val) [] = [] := by
          cases left.val.drop i.val <;> simp [rawMeet]
        simpa only [empty, meetEmpty, List.append_nil] using remaining
    · have leftTest : ¬i < Slice.len left := by scalar_tac
      simp only [leftTest, ite_false, WP.spec_ok]
      have empty : left.val.drop i.val = [] := List.drop_eq_nil_iff.mpr (by omega)
      simpa only [empty, rawMeet, List.append_nil] using remaining
  · exact initial

@[step]
theorem rust_meet_entries (left right : Slice (U32 × U32))
    (room : left.val.length + right.val.length < Usize.max) :
    clock.meet_entries left right ⦃ result =>
      baseModel result.val = meetEntries (baseModel left.val) (baseModel right.val) ⦄ := by
  unfold clock.meet_entries
  step with rust_meet_loop left right (rawMeet left.val right.val)
      (alloc.vec.Vec.new _) 0#usize 0#usize room (by simp [MeetPrefix, alloc.vec.Vec.new,
        alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val]) as ⟨result, model⟩
  rw [model, raw_meet_model]


@[step]
theorem rust_meet_clock (left right : Slice (U32 × U32)) (leftOwn rightOwn : U32 × U32)
    (leftSorted : Sorted (baseModel left.val)) (rightSorted : Sorted (baseModel right.val))
    (room : left.val.length + right.val.length + 2 < Usize.max) :
    clock.meet_clock left leftOwn right rightOwn ⦃ result =>
      baseModel result.1.val = meetEntries
        (overlay (baseModel left.val) (entryModel leftOwn))
        (overlay (baseModel right.val) (entryModel rightOwn)) ∧
      result.2 = (0#u32, 0#u32) ⦄ := by
  unfold clock.meet_clock
  step with rust_entries_bounded left leftOwn leftSorted (by omega) as ⟨leftEntries, leftModel⟩
  step with rust_entries_bounded right rightOwn rightSorted (by omega) as ⟨rightEntries, rightModel⟩
  have leftLength := overlay_length (baseModel left.val) (entryModel leftOwn)
  have rightLength := overlay_length (baseModel right.val) (entryModel rightOwn)
  rw [←leftModel] at leftLength
  rw [←rightModel] at rightLength
  simp only [baseModel_length] at leftLength rightLength
  step with rust_meet_entries (alloc.vec.Vec.deref leftEntries) (alloc.vec.Vec.deref rightEntries)
      (by simp only [alloc.vec.Vec.deref, Slice.from_val]; omega) as ⟨result, model⟩
  simpa only [alloc.vec.Vec.deref, Slice.from_val, leftModel, rightModel] using model

theorem source_meet_refines_pointwise_minimum
    (left right : Slice (U32 × U32)) (leftOwn rightOwn : U32 × U32)
    (leftSorted : Sorted (baseModel left.val)) (rightSorted : Sorted (baseModel right.val))
    (room : left.val.length + right.val.length + 2 < Usize.max) :
    ∃ result, clock.meet_clock left leftOwn right rightOwn = .ok result ∧
      Sorted (baseModel result.1.val) ∧
      value (baseModel result.1.val) (entryModel result.2) = fun actor =>
        min (value (baseModel left.val) (entryModel leftOwn) actor)
            (value (baseModel right.val) (entryModel rightOwn) actor) := by
  obtain ⟨⟨base, own⟩, run, projected, cleared⟩ :=
    spec_returns _ _ (rust_meet_clock left right leftOwn rightOwn leftSorted rightSorted room)
  refine ⟨(base, own), run, ?_, ?_⟩
  · rw [projected]
    exact meet_entries_sorted _ _ (overlay_sorted _ _ leftSorted)
  · have model := meet_entries_value
      (overlay (baseModel left.val) (entryModel leftOwn))
      (overlay (baseModel right.val) (entryModel rightOwn))
      (overlay_sorted _ _ leftSorted) (overlay_sorted _ _ rightSorted)
    rw [cleared]
    have zeroOwn : value (baseModel base.val) (entryModel (0#u32, 0#u32)) =
        entriesValue (baseModel base.val) := by
      funext actor; simp [value, entryModel]
    rw [zeroOwn, projected, model]
    simp only [overlay_value]


end MemoryWrite.SparseClock
