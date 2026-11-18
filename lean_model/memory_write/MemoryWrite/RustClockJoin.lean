/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustClockMerge
import MemoryWrite.RustClockScan

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

@[step] theorem rust_merge_clock_bounded (left right : Slice (U32 × U32))
    (rightOwn : U32 × U32) (hr : Sorted (baseModel right.val))
    (hsize : left.val.length + right.val.length + 1 < Usize.max) :
    clock.merge_clock left right rightOwn ⦃ result =>
      baseModel result.val = mergeEntries (baseModel left.val)
        (overlay (baseModel right.val) (entryModel rightOwn)) ⦄ := by
  rcases rightOwn with ⟨owner, time⟩
  unfold clock.merge_clock
  change (if time = 0#u32 then clock.merge_entries left right
    else do
      let folded ← clock.entries right (owner,time)
      clock.merge_entries left (alloc.vec.Vec.deref folded)) ⦃ _ ⦄
  by_cases hz : time = 0#u32
  · simp only [hz, ite_true]
    step with rust_merge_entries_bounded left right (by omega) as ⟨result, hresult⟩
    simpa [entryModel, overlay] using hresult
  · simp only [hz, ite_false]
    step with rust_entries_bounded right (owner,time) hr (by omega) as ⟨folded, hfolded⟩
    have hlength : folded.val.length ≤ right.val.length + 1 := by
      have h := overlay_length (baseModel right.val) (entryModel (owner,time))
      rw [←hfolded] at h
      simpa only [baseModel_length] using h
    step with rust_merge_entries_bounded left (alloc.vec.Vec.deref folded) (by
      simpa only [alloc.vec.Vec.deref, Slice.from_val] using
        (show left.val.length + folded.val.length < Usize.max by omega)) as ⟨result, hresult⟩
    simpa only [alloc.vec.Vec.deref, Slice.from_val, hfolded] using hresult

def planModel : clock.JoinPlan → JoinPlan
  | .Keep => .keep
  | .ShareRight => .shareRight
  | .Replace entries => .replace (baseModel entries.val)

@[step] theorem rust_join_plan_bounded (left right : Slice (U32 × U32))
    (leftOwn rightOwn : U32 × U32) (sharedBase : Bool)
    (hl : Sorted (baseModel left.val)) (hr : Sorted (baseModel right.val))
    (hsize : left.val.length + right.val.length + 1 < Usize.max) :
    clock.join_plan left leftOwn right rightOwn sharedBase ⦃ result =>
      planModel result = joinPlan (baseModel left.val) (entryModel leftOwn)
        (baseModel right.val) (entryModel rightOwn) sharedBase ⦄ := by
  rcases rightOwn with ⟨owner, time⟩
  by_cases hzero : time = 0#u32
  · have hvzero : time.val = 0 := by scalar_tac
    cases sharedBase with
    | true =>
      simp [clock.join_plan, core.slice.Slice.is_empty, Slice.length, hzero,
        joinPlan, entryModel, planModel, WP.spec_ok]
    | false =>
      by_cases hrnil : right.val = []
      · simp [clock.join_plan, core.slice.Slice.is_empty, Slice.length, hrnil, hzero,
          joinPlan, entryModel, baseModel, planModel, WP.spec_ok]
      · have hrmodel : baseModel right.val ≠ [] := by simpa [baseModel] using hrnil
        simp [clock.join_plan, core.slice.Slice.is_empty, Slice.length, hrnil, hzero]
        step with rust_base_covers left right hl hr as ⟨covered, hcovered⟩
        cases covered with
        | true =>
          simp only [ite_true, WP.spec_ok]
          simp [planModel, joinPlan, entryModel, hrmodel, ←hcovered]
        | false =>
          simp only [Bool.false_eq_true, ite_false]
          step with rust_base_covers right left hr hl as ⟨reverse, hreverse⟩
          cases reverse with
          | true =>
            simp only [ite_true, WP.spec_ok]
            simp [planModel, joinPlan, entryModel, hrmodel,
              ←hcovered, ←hreverse]
          | false =>
            simp only [Bool.false_eq_true, ite_false]
            step with rust_merge_clock_bounded left right (owner, 0#u32) hr hsize as ⟨result, hresult⟩
            simpa [planModel, joinPlan, entryModel, hrmodel,
              ←hcovered, ←hreverse] using congrArg JoinPlan.replace hresult
  · have hvnonzero : time.val ≠ 0 := by intro h; apply hzero; scalar_tac
    cases sharedBase with
    | false =>
      simp [clock.join_plan, core.slice.Slice.is_empty, Slice.length, hzero]
      step with rust_lookup left owner hl as ⟨known, hknown⟩
      split
      · rename_i hc
        have hcval : time.val ≤ entriesValue (baseModel left.val) owner.val := by scalar_tac
        step with rust_base_covers left right hl hr as ⟨covered, hcovered⟩
        cases covered with
        | true =>
          simp only [ite_true, WP.spec_ok]
          simp [planModel, joinPlan, entryModel, hvnonzero, hcval, ←hcovered]
        | false =>
          simp only [Bool.false_eq_true, ite_false]
          step with rust_merge_clock_bounded left right (owner, time) hr hsize as ⟨result, hresult⟩
          simpa [planModel, joinPlan, entryModel, hvnonzero, hcval, ←hcovered] using
            congrArg JoinPlan.replace hresult
      · rename_i hc
        have hcval : ¬time.val ≤ entriesValue (baseModel left.val) owner.val := by scalar_tac
        step with rust_merge_clock_bounded left right (owner, time) hr hsize as ⟨result, hresult⟩
        simpa [planModel, joinPlan, entryModel, hvnonzero, hcval] using
          congrArg JoinPlan.replace hresult
    | true =>
      simp [clock.join_plan, core.slice.Slice.is_empty, Slice.length, hzero]
      step with rust_get left leftOwn owner hl as ⟨known, hknown⟩
      split
      · rename_i hcovered
        try simp only [WP.spec_ok]
        have hc : time.val ≤ value (baseModel left.val) (entryModel leftOwn) owner.val := by scalar_tac
        dsimp only [entryModel] at hc
        simp [planModel, joinPlan, entryModel, hvnonzero, hc]
      · rename_i hcovered
        have hc : ¬time.val ≤ value (baseModel left.val) (entryModel leftOwn) owner.val := by scalar_tac
        step with rust_lookup left owner hl as ⟨baseKnown, hbaseKnown⟩
        have htest : ¬time.val ≤ baseKnown.val := by
          unfold value at hc
          scalar_tac
        try simp only [htest, ite_false]
        step with rust_merge_clock_bounded left right (owner, time) hr hsize as ⟨result, hresult⟩
        have hb : ¬time.val ≤ entriesValue (baseModel left.val) owner.val := by
          unfold value at hc; omega
        dsimp only [entryModel] at hc
        simpa [planModel, joinPlan, entryModel, hvnonzero, hc, hb] using congrArg JoinPlan.replace hresult

@[step] theorem rust_join_plan_value (left right : Slice (U32 × U32))
    (leftOwn rightOwn : U32 × U32) (sharedBase : Bool)
    (hl : Sorted (baseModel left.val)) (hr : Sorted (baseModel right.val))
    (hsize : left.val.length + right.val.length + 1 < Usize.max)
    (hshared : sharedBase = true → left.val = right.val) :
    clock.join_plan left leftOwn right rightOwn sharedBase ⦃ result =>
      value (applyPlan (baseModel left.val) (baseModel right.val) (planModel result))
          (entryModel leftOwn) =
        BarrierReduction.join (value (baseModel left.val) (entryModel leftOwn))
          (value (baseModel right.val) (entryModel rightOwn)) ∧
      Sorted (applyPlan (baseModel left.val) (baseModel right.val) (planModel result)) ⦄ := by
  step with rust_join_plan_bounded left right leftOwn rightOwn sharedBase hl hr hsize as
    ⟨result, hresult⟩
  rw [hresult]
  exact ⟨join_plan_value _ _ _ _ _ (fun h => congrArg baseModel (hshared h)),
    join_plan_sorted _ _ _ _ _ hl hr⟩

end MemoryWrite.SparseClock
