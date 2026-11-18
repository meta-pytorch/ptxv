/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ClockUpdates
import MemoryWrite.RustClockMerge
import MemoryWrite.RustClockFilter

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

@[simp] theorem base_model_append (left right : List (U32 × U32)) :
    baseModel (left ++ right) = baseModel left ++ baseModel right := by simp [baseModel]
@[simp] theorem base_model_cons (entry : U32 × U32) (rest : List (U32 × U32)) :
    baseModel (entry :: rest) = entryModel entry :: baseModel rest := rfl
@[simp] theorem base_model_nil : baseModel [] = [] := rfl

theorem baseModel_set (base : List (U32 × U32)) (index : Nat) (entry : U32 × U32) :
    baseModel (base.set index entry) = (baseModel base).set index (entryModel entry) := by
  simp [baseModel, List.map_set]

@[step] theorem rust_push_coalesced (result : alloc.vec.Vec (U32 × U32))
    (entry : U32 × U32) (room : result.val.length < Usize.max) :
    clock.push_coalesced result entry ⦃ updated =>
      baseModel updated.val = pushCoalesced (baseModel result.val) (entryModel entry) ⦄ := by
  rcases entry with ⟨owner, time⟩
  unfold clock.push_coalesced
  by_cases positive : 0 < result.val.length
  · have tested : alloc.vec.Vec.len result > 0#usize := by scalar_tac
    simp only [tested, ite_true]
    step as ⟨last, lastValue⟩
    have lastIndex : last.val = result.val.length - 1 := by scalar_tac
    have bounded : last.val < result.val.length := by omega
    step as ⟨actor, stamp, read⟩
    have actorValue : actor = result.val[last.val].1 := congrArg Prod.fst read
    have stampValue : stamp = result.val[last.val].2 := congrArg Prod.snd read
    have modelLast : (baseModel result.val)[result.val.length - 1]'(by simp; omega) =
        entryModel (actor, stamp) := by
      have projected := baseModel_getElem result.val last.val bounded
      rw [←read] at projected
      simpa only [lastIndex] using projected
    by_cases same : actor = owner
    · simp only [same, ite_true]
      step as ⟨maximum, maximumValue⟩
      step as ⟨actorAgain, stampAgain, back, readAgain, backValue⟩
      rw [backValue]
      simp only [alloc.vec.Vec.set, alloc.vec.Vec.from, alloc.vec.Vec.val, Slice.from_val]
      change baseModel (result.val.set last.val (actorAgain, maximum)) = _
      have again : actorAgain = actor := by
        have h := congrArg Prod.fst readAgain
        exact h.trans actorValue.symm
      change baseModel (result.val.set last.val (actorAgain, maximum)) =
        pushCoalesced (baseModel result.val) (entryModel (owner, time))
      rw [baseModel_set, again]
      have maxValue : maximum.val = max stamp.val time.val := by scalar_tac
      simp only [pushCoalesced, baseModel_length, positive, ↓reduceDIte, modelLast,
        entryModel, same, ↓reduceIte, lastIndex, maxValue]
    · simp only [same, ite_false]
      step with alloc.vec.Vec.push_spec result (owner, time) room as ⟨updated, pushed⟩
      have different : actor.val ≠ owner.val := by scalar_tac
      rw [pushed]
      simp only [base_model_append, base_model_cons, base_model_nil, pushCoalesced,
        baseModel_length, positive, ↓reduceDIte, modelLast, entryModel, different, ↓reduceIte]
  · have tested : ¬alloc.vec.Vec.len result > 0#usize := by scalar_tac
    simp only [tested, ite_false]
    step with alloc.vec.Vec.push_spec result (owner, time) room as ⟨updated, pushed⟩
    simp [pushed, baseModel, pushCoalesced, positive]

@[step] theorem rust_extend_entries (result : alloc.vec.Vec (U32 × U32))
    (suffix : Slice (U32 × U32)) (room : result.val.length + suffix.val.length ≤ Usize.max) :
    alloc.vec.Vec.extend_from_slice (BuiltinClone (U32 × U32)) result suffix ⦃ updated =>
      updated.val = result.val ++ suffix.val ⦄ := by
  unfold alloc.vec.Vec.extend_from_slice
  have checked : result.length + suffix.length ≤ Usize.max := room
  simp only [checked, ↓reduceDIte]
  step with List.clone_spec (clone := (BuiltinClone (U32 × U32)).clone) (l := suffix.val)
    (by intro x hx; rfl) as ⟨copied, contents, length⟩
  simp [contents, alloc.vec.Vec.from, alloc.vec.Vec.val, Slice.from_val]

@[step] theorem rust_append_sorted_suffix (result : alloc.vec.Vec (U32 × U32))
    (suffix : Slice (U32 × U32)) (ordered : Sorted (baseModel suffix.val))
    (room : result.val.length + suffix.val.length < Usize.max) :
    clock.append_sorted_suffix result suffix ⦃ updated =>
      baseModel updated.val = coalesceInto (baseModel result.val) (baseModel suffix.val) ⦄ := by
  unfold clock.append_sorted_suffix
  by_cases positive : 0 < suffix.val.length
  · have tested : Slice.len suffix > 0#usize := by scalar_tac
    simp only [tested, ite_true]
    step with Slice.index_usize_spec suffix 0#usize positive as ⟨entry, read⟩
    step with rust_push_coalesced result entry (by omega) as ⟨updated, coalesced⟩
    step with rust_suffix suffix 1#usize (by scalar_tac) as ⟨rest, restValue⟩
    have growth := push_coalesced_length (baseModel result.val) (entryModel entry)
    rw [←coalesced] at growth
    simp only [baseModel_length] at growth
    have restLength : rest.val.length = suffix.val.length - 1 := by simp [restValue]
    step with rust_extend_entries updated rest (by omega) as ⟨done, appended⟩
    have shape : baseModel suffix.val = entryModel entry :: baseModel rest.val := by
      rw [restValue]
      have raw := List.drop_eq_getElem_cons positive
      have raw := congrArg baseModel raw
      simpa [read, baseModel] using raw
    rw [appended, base_model_append, coalesced, shape]
    exact (coalesce_into_sorted_suffix _ _ _ (by rwa [←shape])).symm
  · have tested : ¬Slice.len suffix > 0#usize := by scalar_tac
    have empty : suffix.val = [] := List.length_eq_zero_iff.mp (by omega)
    simp [tested, empty, baseModel, coalesceInto]

def UpdatesPrefix (left right : List (U32 × U32)) (output : List Entry) (result : List (U32 × U32)) (i j : Nat) : Prop :=
  i ≤ left.length ∧ j ≤ right.length ∧ result.length ≤ i + j ∧
    coalesceInto (baseModel result) (baseModel (rawMerge (left.drop i) (right.drop j))) = output

@[step] theorem rust_updates_loop (left right : Slice (U32 × U32))
    (output : List Entry) (result : alloc.vec.Vec (U32 × U32)) (i j : Usize)
    (leftSorted : Sorted (baseModel left.val))
    (hsize : left.val.length + right.val.length < Usize.max)
    (hp : UpdatesPrefix left.val right.val output result.val i.val j.val) :
    clock.merge_sorted_updates_loop left right result i j ⦃ done => baseModel done.val = output ⦄ := by
  unfold clock.merge_sorted_updates_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (U32 × U32) × Usize × Usize =>
      left.val.length - current.2.1.val + (right.val.length - current.2.2.val))
    (fun current : alloc.vec.Vec (U32 × U32) × Usize × Usize =>
      UpdatesPrefix left.val right.val output current.1.val current.2.1.val current.2.2.val)
  · intro current hcurrent
    rcases current with ⟨result, i, j⟩
    change UpdatesPrefix left.val right.val output result.val i.val j.val at hcurrent
    obtain ⟨hib, hjb, hlength, htail⟩ := hcurrent
    unfold clock.merge_sorted_updates_loop.body
    by_cases hi : i.val < left.val.length
    · have hit : i < Slice.len left := by scalar_tac
      simp only [hit, ite_true]
      by_cases hj : j.val < right.val.length
      · have hjt : j < Slice.len right := by scalar_tac
        simp only [hjt, ite_true]
        step with Slice.index_usize_spec left i hi as ⟨a, hleft⟩
        rcases a with ⟨p, cp⟩
        step with Slice.index_usize_spec right j hj as ⟨b, hright⟩
        rcases b with ⟨q, cq⟩
        change (if p = q then do
            let maximum ← lift (core.cmp.impls.OrdU32.max cp cq)
            let updated ← clock.push_coalesced result (p, maximum)
            let ni ← i + 1#usize
            let nj ← j + 1#usize
            pure (ControlFlow.cont (updated, ni, nj))
          else if p < q then do
            let updated ← clock.push_coalesced result (p, cp)
            let ni ← i + 1#usize
            pure (ControlFlow.cont (updated, ni, j))
          else do
            let updated ← clock.push_coalesced result (q, cq)
            let nj ← j + 1#usize
            pure (ControlFlow.cont (updated, i, nj))) ⦃ _ ⦄
        have hdropL : left.val.drop i.val = (p, cp) :: left.val.drop (i.val + 1) := by
          rw [List.drop_eq_getElem_cons hi, ←hleft]
        have hdropR : right.val.drop j.val = (q, cq) :: right.val.drop (j.val + 1) := by
          rw [List.drop_eq_getElem_cons hj, ←hright]
        have hcapacity : result.val.length < Usize.max := by omega
        by_cases heq : p = q
        · simp only [heq, ite_true]
          step as ⟨maximum, hmaximum⟩
          step with rust_push_coalesced result (q, maximum) hcapacity as ⟨updated, hupdated⟩
          step as ⟨ni, hni⟩
          step as ⟨nj, hnj⟩
          have hniVal : ni.val = i.val + 1 := by scalar_tac
          have hnjVal : nj.val = j.val + 1 := by scalar_tac
          refine ⟨⟨by omega, by omega, ?_, ?_⟩, by omega⟩
          · have grow := push_coalesced_length (baseModel result.val) (entryModel (q, maximum))
            rw [←hupdated] at grow
            simp only [baseModel_length] at grow
            omega
          · rw [hdropL, hdropR, rawMerge, ite_eq_left heq] at htail
            simpa [hupdated, hniVal, hnjVal, base_model_cons, base_model_nil, coalesceInto, entryModel, hmaximum, heq] using htail
        · simp only [heq, ite_false]
          by_cases hlt : p < q
          · simp only [hlt, ite_true]
            step with rust_push_coalesced result (p, cp) hcapacity as ⟨updated, hupdated⟩
            step as ⟨ni, hni⟩
            have hniVal : ni.val = i.val + 1 := by scalar_tac
            refine ⟨⟨by omega, hjb, ?_, ?_⟩, by omega⟩
            · have grow := push_coalesced_length (baseModel result.val) (entryModel (p, cp))
              rw [←hupdated] at grow
              simp only [baseModel_length] at grow
              omega
            · rw [hdropL, hdropR, rawMerge, ite_eq_right heq, ite_eq_left hlt] at htail
              simpa [hupdated, hniVal, hdropR, base_model_cons, base_model_nil, coalesceInto, entryModel] using htail
          · simp only [hlt, ite_false]
            step with rust_push_coalesced result (q, cq) hcapacity as ⟨updated, hupdated⟩
            step as ⟨nj, hnj⟩
            have hnjVal : nj.val = j.val + 1 := by scalar_tac
            refine ⟨⟨hib, by omega, ?_, ?_⟩, by omega⟩
            · have grow := push_coalesced_length (baseModel result.val) (entryModel (q, cq))
              rw [←hupdated] at grow
              simp only [baseModel_length] at grow
              omega
            · rw [hdropL, hdropR, rawMerge, ite_eq_right heq, ite_eq_right hlt] at htail
              simpa [hupdated, hnjVal, hdropL, base_model_cons, base_model_nil, coalesceInto, entryModel] using htail
      · have hjt : ¬j < Slice.len right := by scalar_tac
        simp only [hjt, ite_false]
        step with rust_suffix left i hib as ⟨suffix, selected⟩
        have suffixOrdered : Sorted (baseModel suffix.val) := by
          rw [selected]
          simpa only [baseModel, List.map_drop] using (show Sorted ((baseModel left.val).drop i.val) from leftSorted.drop)
        have suffixLength : suffix.val.length = left.val.length - i.val := by simp [selected]
        step with rust_append_sorted_suffix result suffix suffixOrdered (by omega)
          as ⟨done, appended⟩
        have hdropR : right.val.drop j.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        rw [appended, selected]
        simpa only [hdropR, raw_merge_nil_right] using htail
    · have hit : ¬i < Slice.len left := by scalar_tac
      simp only [hit, ite_false]
      by_cases hj : j.val < right.val.length
      · have hjt : j < Slice.len right := by scalar_tac
        simp only [hjt, ite_true]
        step with Slice.index_usize_spec right j hj as ⟨b, hright⟩
        rcases b with ⟨q, cq⟩
        step with rust_push_coalesced result (q, cq) (by omega) as ⟨updated, hupdated⟩
        step as ⟨nj, hnj⟩
        have hnjVal : nj.val = j.val + 1 := by scalar_tac
        have hdropL : left.val.drop i.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        have hdropR : right.val.drop j.val = (q, cq) :: right.val.drop (j.val + 1) := by
          rw [List.drop_eq_getElem_cons hj, ←hright]
        refine ⟨⟨hib, by omega, ?_, ?_⟩, by omega⟩
        · have grow := push_coalesced_length (baseModel result.val) (entryModel (q, cq))
          rw [←hupdated] at grow
          simp only [baseModel_length] at grow
          omega
        · simpa [hdropL, hdropR, hupdated, hnjVal, rawMerge, base_model_cons, base_model_nil, coalesceInto, entryModel] using htail
      · have hjt : ¬j < Slice.len right := by scalar_tac
        simp only [hjt, ite_false, WP.spec_ok]
        have hdropL : left.val.drop i.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        have hdropR : right.val.drop j.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        simpa [hdropL, hdropR, rawMerge, base_model_nil, coalesceInto] using htail
  · exact hp


@[step] theorem rust_merge_sorted_updates (base updates : Slice (U32 × U32))
    (ordered : Sorted (baseModel base.val)) (room : base.val.length + updates.val.length < Usize.max) :
    clock.merge_sorted_updates base updates ⦃ result =>
      baseModel result.val = mergeUpdates (baseModel base.val) (baseModel updates.val) ⦄ := by
  unfold clock.merge_sorted_updates
  step as ⟨capacity, capacityValue⟩
  step with rust_updates_loop base updates
    (mergeUpdates (baseModel base.val) (baseModel updates.val))
    (alloc.vec.Vec.with_capacity (U32 × U32) capacity) 0#usize 0#usize ordered room
    (by simp [UpdatesPrefix, alloc.vec.Vec.with_capacity, alloc.vec.Vec.new,
      alloc.vec.Vec.from, alloc.vec.Vec.val, Slice.from_val, mergeUpdates, raw_merge_model])
    as ⟨result, projected⟩
  exact projected

@[step] theorem rust_join_sorted_updates (base updates : Slice (U32 × U32))
    (own : U32 × U32) (ordered : Sorted (baseModel base.val))
    (room : base.val.length + 1 + updates.val.length < Usize.max) :
    clock.join_sorted_updates base own updates ⦃ result =>
      baseModel result.1.val = mergeUpdates (overlay (baseModel base.val) (entryModel own))
        (baseModel updates.val) ∧ result.2 = (0#u32, 0#u32) ⦄ := by
  unfold clock.join_sorted_updates
  step with rust_entries_bounded base own ordered (by omega) as ⟨entries, materialized⟩
  have growth := overlay_length (baseModel base.val) (entryModel own)
  rw [←materialized] at growth
  simp only [baseModel_length] at growth
  step with rust_merge_sorted_updates (alloc.vec.Vec.deref entries) updates
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val, materialized] using overlay_sorted _ _ ordered)
      (by simp only [alloc.vec.Vec.deref, Slice.from_val]; omega) as ⟨result, projected⟩
  simpa only [alloc.vec.Vec.deref, Slice.from_val, materialized] using projected

theorem source_batched_own_merge_refines_pointwise_maximum (base updates : Slice (U32 × U32))
    (own : U32 × U32) (ordered : Sorted (baseModel base.val))
    (updatesOrdered : WeakSorted (baseModel updates.val))
    (room : base.val.length + 1 + updates.val.length < Usize.max) :
    ∃ result, clock.join_sorted_updates base own updates = .ok result ∧
      Sorted (baseModel result.1.val) ∧ result.2 = (0#u32, 0#u32) ∧
      value (baseModel result.1.val) (entryModel result.2) =
        BarrierReduction.join (value (baseModel base.val) (entryModel own))
          (entriesValue (baseModel updates.val)) := by
  obtain ⟨⟨entries, cleared⟩, run, projected, ownCleared⟩ :=
    spec_returns _ _ (rust_join_sorted_updates base updates own ordered room)
  refine ⟨(entries, cleared), run, ?_, ownCleared, ?_⟩
  · rw [projected]
    exact merge_updates_sorted _ _ (overlay_sorted _ _ ordered) updatesOrdered
  · rw [ownCleared, projected]
    have meaning := merge_updates_value (overlay (baseModel base.val) (entryModel own))
      (baseModel updates.val)
    rw [overlay_value] at meaning
    funext actor
    simpa [value, entryModel] using congrFun meaning actor

end MemoryWrite.SparseClock
