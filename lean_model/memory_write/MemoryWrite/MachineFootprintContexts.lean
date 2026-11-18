/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineFootprint

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineFootprint

def recordMode (sync : Bool) : effects.FoldMode := if sync then .SyncOnly else .Full

theorem source_mode (sync : Bool) : machine_footprint.mode sync = .ok (recordMode sync) := by
  cases sync <;> rfl

def beginEmpty (fp : machine_footprint.Footprint) : Bool :=
  !fp.all && decide (fp.r.val = []) && decide (fp.w.val = []) &&
    decide (fp.rel.val = []) && decide (fp.acq.val = [])

theorem length_zero {α : Type} (values : alloc.vec.Vec α) :
    (alloc.vec.Vec.len values = 0#usize) ↔ values.val = [] := by
  constructor
  · intro same
    have h := congrArg UScalar.val same
    simpa using h
  · intro empty
    apply UScalar.eq_of_val_eq
    simp [empty]

theorem deref_value {α : Type} (values : alloc.vec.Vec α) :
    (alloc.vec.Vec.deref values).val = values.val := by
  simp [alloc.vec.Vec.deref, alloc.vec.Vec.val]

theorem source_begin_context (fp : Option machine_footprint.Footprint) :
    machine_footprint.begin_context fp = .ok { footprint_empty := fp.map beginEmpty } := by
  cases fp with
  | none => simp [machine_footprint.begin_context, core.option.Option.map]
  | some fp =>
    simp only [machine_footprint.begin_context, core.option.Option.map,
      machine_footprint.begin_context.closure.Insts.CoreOpsFunctionFnOnceTupleShared0FootprintBool.call_once, length_zero]
    cases all : fp.all <;> by_cases r : fp.r.val = [] <;> by_cases w : fp.w.val = [] <;>
      by_cases rel : fp.rel.val = [] <;> by_cases acq : fp.acq.val = [] <;>
        simp [all, r, w, rel, acq, beginEmpty]

theorem source_record_context (fp : Option machine_footprint.Footprint) (slice : Bool) :
    machine_footprint.record_context fp slice =
      .ok { mode := fp.map (fun x => recordMode x.sync_only), slice } := by
  cases fp with
  | none => simp [machine_footprint.record_context, core.option.Option.map]
  | some fp =>
    simp [machine_footprint.record_context, core.option.Option.map, source_mode,
      machine_footprint.record_context.closure.Insts.CoreOpsFunctionFnOnceTupleShared0FootprintFoldMode.call_once]

theorem source_membership_loop (ranges : Slice Range) (index : Usize) :
    machine_footprint.has_membership_loop ranges index =
      .ok ((ranges.val.drop index.val).any (fun x => decide (x.1 = 18#u8))) := by
  generalize fuelEq : ranges.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : machine_footprint.has_membership_loop ranges index = (do
        let result ← machine_footprint.has_membership_loop.body ranges index
        match result with
        | .cont next => machine_footprint.has_membership_loop ranges next
        | .done output => .ok output) := by
      unfold machine_footprint.has_membership_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result <;> rfl
    rw [unroll]
    by_cases inside : index.val < ranges.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq ranges
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
      have increment : (index + 1#usize : Result Usize) = .ok next := by
        simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
          UScalar.check_bounds, platformBound, next, Result.ofOption]
        rfl
      have lookup : Slice.index_usize ranges index = .ok ranges.val[index.val] := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq,
          List.getElem?_eq_getElem inside]
      have test : index < Slice.len ranges := by scalar_tac
      have body : machine_footprint.has_membership_loop.body ranges index =
          .ok (if ranges.val[index.val].1 = 18#u8 then .done true else .cont next) := by
        simp only [machine_footprint.has_membership_loop.body, test, ite_true,
          lookup, bind_ok, increment]
        rcases ranges.val[index.val] with ⟨space, owner, lo, hi⟩
        by_cases h : space = 18#u8 <;> simp [h]
      rw [body]
      simp only [bind_ok]
      rw [List.drop_eq_getElem_cons inside]
      simp only [List.any_cons]
      generalize scalarAt : ranges.val[index.val].1 = space
      by_cases found : space = 18#u8
      · simp [found]
      · simpa [found, nextVal] using
          ih (ranges.val.length - next.val) (by omega) next rfl

    · have test : ¬index < Slice.len ranges := by scalar_tac
      simp [machine_footprint.has_membership_loop.body, test,
        List.drop_eq_nil_of_le (show ranges.val.length ≤ index.val by omega)]

theorem source_has_membership (ranges : Slice Range) :
    machine_footprint.has_membership ranges =
      .ok (ranges.val.any (fun x => decide (x.1 = 18#u8))) := by
  simpa [machine_footprint.has_membership] using source_membership_loop ranges 0#usize

theorem source_finish_context (fp : Option machine_footprint.Footprint) :
    machine_footprint.finish_context fp = .ok {
      footprint := fp.map (fun x => {
        mode := recordMode x.sync_only,
        has_membership := (x.r.val ++ x.w.val).any (fun range => decide (range.1 = 18#u8)) }) } := by
  cases fp with
  | none => simp [machine_footprint.finish_context, core.option.Option.map]
  | some fp =>
    simp only [machine_footprint.finish_context, core.option.Option.map,
      machine_footprint.finish_context.closure.Insts.CoreOpsFunctionFnOnceTupleShared0FootprintFinishFootprint.call_once,
      source_mode, bind_ok, source_has_membership, deref_value]
    cases h : fp.r.val.any (fun range => decide (range.1 = 18#u8)) <;>
        simp [h]

end MemoryWrite.MachineFootprint
