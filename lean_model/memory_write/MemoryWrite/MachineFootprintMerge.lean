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

def sequence {α : Type} (step : machine_footprint.Footprint → α → Result machine_footprint.Footprint) :
    List α → machine_footprint.Footprint → Result machine_footprint.Footprint
  | [], fp => .ok fp
  | x :: xs, fp => do
    let next ← step fp x
    sequence step xs next

def appendKey (write : Bool) (fp : machine_footprint.Footprint) (key : U64) :
    Result machine_footprint.Footprint := do
  if write then
    let keys ← alloc.vec.Vec.push fp.acq key
    .ok { fp with acq := keys }
  else
    let keys ← alloc.vec.Vec.push fp.rel key
    .ok { fp with rel := keys }

theorem source_merge_loop0 (summary : effects.Summary)
    (fp : machine_footprint.Footprint) (index : Usize) :
    machine_footprint.merge_summary_loop0 fp summary index = (do
      let next ← sequence (fun fp range => addRef fp range false) (summary.reads.val.drop index.val) fp
      .ok (next, summary.writes, summary.releases, summary.acquires, summary.all)) := by
  generalize fuelEq : summary.reads.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing fp index with
  | h fuel ih =>
    have unroll : machine_footprint.merge_summary_loop0 fp summary index = (do
        let result ← machine_footprint.merge_summary_loop0.body summary fp index
        match result with
        | .cont pair => machine_footprint.merge_summary_loop0 pair.1 summary pair.2
        | .done output => .ok output) := by
      unfold machine_footprint.merge_summary_loop0
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => rcases pair with ⟨after, next⟩; rfl
      | done output => rfl
    rw [unroll]
    by_cases inside : index.val < summary.reads.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := summary.reads.property
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
      have increment : (index + 1#usize : Result Usize) = .ok next := by
        simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
          UScalar.check_bounds, platformBound, next, Result.ofOption]
        rfl
      have lookup : alloc.vec.Vec.index_usize summary.reads index = .ok summary.reads.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      have test : index < alloc.vec.Vec.len summary.reads := by scalar_tac
      have body : machine_footprint.merge_summary_loop0.body summary fp index = (do
          let after ← (fun fp range => addRef fp range false) fp summary.reads.val[index.val]
          .ok (.cont (after, next))) := by
        simp only [machine_footprint.merge_summary_loop0.body, test, ite_true,
          alloc.vec.Vec.index_slice_index, lookup, bind_ok]
        rcases element : summary.reads.val[index.val] with ⟨space, owner, lo, hi⟩
        simp only [source_add, increment, bind_ok]
        rfl
      rw [body]
      simp only [Std.bind_assoc, bind_ok]
      rw [List.drop_eq_getElem_cons inside]
      simp only [sequence, Std.bind_assoc]
      congr 1
      funext after
      apply ih (summary.reads.val.length - next.val) (by omega) after next rfl
    · have test : ¬index < alloc.vec.Vec.len summary.reads := by scalar_tac
      simp [machine_footprint.merge_summary_loop0.body, test, sequence,
        List.drop_eq_nil_of_le (show summary.reads.val.length ≤ index.val by omega)]

theorem source_merge_loop1 (values : alloc.vec.Vec Range)
    (fp : machine_footprint.Footprint) (index : Usize) :
    machine_footprint.merge_summary_loop1 fp values index = sequence (fun fp range => addRef fp range true) (values.val.drop index.val) fp := by
  generalize fuelEq : values.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing fp index with
  | h fuel ih =>
    have unroll : machine_footprint.merge_summary_loop1 fp values index = (do
        let result ← machine_footprint.merge_summary_loop1.body values fp index
        match result with
        | .cont pair => machine_footprint.merge_summary_loop1 pair.1 values pair.2
        | .done output => .ok output) := by
      unfold machine_footprint.merge_summary_loop1
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => rcases pair with ⟨after, next⟩; rfl
      | done output => rfl
    rw [unroll]
    by_cases inside : index.val < values.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := values.property
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
      have increment : (index + 1#usize : Result Usize) = .ok next := by
        simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
          UScalar.check_bounds, platformBound, next, Result.ofOption]
        rfl
      have lookup : alloc.vec.Vec.index_usize values index = .ok values.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      have test : index < alloc.vec.Vec.len values := by scalar_tac
      have body : machine_footprint.merge_summary_loop1.body values fp index = (do
          let after ← (fun fp range => addRef fp range true) fp values.val[index.val]
          .ok (.cont (after, next))) := by
        simp only [machine_footprint.merge_summary_loop1.body, test, ite_true,
          alloc.vec.Vec.index_slice_index, lookup, bind_ok]
        rcases element : values.val[index.val] with ⟨space, owner, lo, hi⟩
        simp only [source_add, increment, bind_ok]
        rfl
      rw [body]
      simp only [Std.bind_assoc, bind_ok]
      rw [List.drop_eq_getElem_cons inside]
      simp only [sequence]
      congr 1
      funext after
      apply ih (values.val.length - next.val) (by omega) after next rfl
    · have test : ¬index < alloc.vec.Vec.len values := by scalar_tac
      simp [machine_footprint.merge_summary_loop1.body, test, sequence,
        List.drop_eq_nil_of_le (show values.val.length ≤ index.val by omega)]

theorem source_merge_loop2 (values : alloc.vec.Vec U64)
    (fp : machine_footprint.Footprint) (index : Usize) :
    machine_footprint.merge_summary_loop2 fp values index = sequence (appendKey false) (values.val.drop index.val) fp := by
  generalize fuelEq : values.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing fp index with
  | h fuel ih =>
    have unroll : machine_footprint.merge_summary_loop2 fp values index = (do
        let result ← machine_footprint.merge_summary_loop2.body values fp index
        match result with
        | .cont pair => machine_footprint.merge_summary_loop2 pair.1 values pair.2
        | .done output => .ok output) := by
      unfold machine_footprint.merge_summary_loop2
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => rcases pair with ⟨after, next⟩; rfl
      | done output => rfl
    rw [unroll]
    by_cases inside : index.val < values.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := values.property
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
      have increment : (index + 1#usize : Result Usize) = .ok next := by
        simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
          UScalar.check_bounds, platformBound, next, Result.ofOption]
        rfl
      have lookup : alloc.vec.Vec.index_usize values index = .ok values.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      have test : index < alloc.vec.Vec.len values := by scalar_tac
      have body : machine_footprint.merge_summary_loop2.body values fp index = (do
          let after ← (appendKey false) fp values.val[index.val]
          .ok (.cont (after, next))) := by
        simp only [machine_footprint.merge_summary_loop2.body, test, ite_true,
          alloc.vec.Vec.index_slice_index, lookup, bind_ok]
        simp only [appendKey, Bool.false_eq_true, ↓reduceIte, Std.bind_assoc, increment, bind_ok]
        rfl
      rw [body]
      simp only [Std.bind_assoc, bind_ok]
      rw [List.drop_eq_getElem_cons inside]
      simp only [sequence]
      congr 1
      funext after
      apply ih (values.val.length - next.val) (by omega) after next rfl
    · have test : ¬index < alloc.vec.Vec.len values := by scalar_tac
      simp [machine_footprint.merge_summary_loop2.body, test, sequence,
        List.drop_eq_nil_of_le (show values.val.length ≤ index.val by omega)]

theorem source_merge_loop3 (values : alloc.vec.Vec U64)
    (fp : machine_footprint.Footprint) (index : Usize) :
    machine_footprint.merge_summary_loop3 fp values index = sequence (appendKey true) (values.val.drop index.val) fp := by
  generalize fuelEq : values.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing fp index with
  | h fuel ih =>
    have unroll : machine_footprint.merge_summary_loop3 fp values index = (do
        let result ← machine_footprint.merge_summary_loop3.body values fp index
        match result with
        | .cont pair => machine_footprint.merge_summary_loop3 pair.1 values pair.2
        | .done output => .ok output) := by
      unfold machine_footprint.merge_summary_loop3
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => rcases pair with ⟨after, next⟩; rfl
      | done output => rfl
    rw [unroll]
    by_cases inside : index.val < values.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := values.property
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
      have increment : (index + 1#usize : Result Usize) = .ok next := by
        simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
          UScalar.check_bounds, platformBound, next, Result.ofOption]
        rfl
      have lookup : alloc.vec.Vec.index_usize values index = .ok values.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      have test : index < alloc.vec.Vec.len values := by scalar_tac
      have body : machine_footprint.merge_summary_loop3.body values fp index = (do
          let after ← (appendKey true) fp values.val[index.val]
          .ok (.cont (after, next))) := by
        simp only [machine_footprint.merge_summary_loop3.body, test, ite_true,
          alloc.vec.Vec.index_slice_index, lookup, bind_ok]
        simp only [appendKey, ↓reduceIte, Std.bind_assoc, increment, bind_ok]
        rfl
      rw [body]
      simp only [Std.bind_assoc, bind_ok]
      rw [List.drop_eq_getElem_cons inside]
      simp only [sequence]
      congr 1
      funext after
      apply ih (values.val.length - next.val) (by omega) after next rfl
    · have test : ¬index < alloc.vec.Vec.len values := by scalar_tac
      simp [machine_footprint.merge_summary_loop3.body, test, sequence,
        List.drop_eq_nil_of_le (show values.val.length ≤ index.val by omega)]

def nativeMerge (fp : machine_footprint.Footprint) (summary : effects.Summary) :
    Result machine_footprint.Footprint := do
  let afterReads ← sequence (fun fp range => addRef fp range false) summary.reads.val fp
  let afterWrites ← sequence (fun fp range => addRef fp range true) summary.writes.val afterReads
  let afterReleases ← sequence (appendKey false) summary.releases.val afterWrites
  let afterAcquires ← sequence (appendKey true) summary.acquires.val afterReleases
  .ok { afterAcquires with all := afterAcquires.all || summary.all }

/-- Exact Result equality also preserves any checked-capacity failure of the native update. -/
theorem source_merge_summary (fp : machine_footprint.Footprint) (summary : effects.Summary) :
    machine_footprint.merge_summary fp summary = nativeMerge fp summary := by
  simp [machine_footprint.merge_summary, source_merge_loop0, source_merge_loop1,
    source_merge_loop2, source_merge_loop3, nativeMerge, Std.bind_assoc]

theorem native_append_length (ranges : List Range) (range : Range) :
    (nativeAppend ranges range).length ≤ ranges.length + 1 := by
  unfold nativeAppend
  split
  · simp
  · split
    · simp
    · simp

theorem sequence_ranges_frame (ranges : List Range) (fp : machine_footprint.Footprint)
    (write : Bool)
    (readRoom : fp.r.val.length + ranges.length < Usize.max)
    (writeRoom : fp.w.val.length + ranges.length < Usize.max) :
    sequence (fun fp range => addRef fp range write) ranges fp ⦃ updated =>
      updated.sync_only = fp.sync_only ∧ updated.all = fp.all ∧
      updated.rel = fp.rel ∧ updated.acq = fp.acq ∧
      updated.r.val.length ≤ fp.r.val.length + ranges.length ∧
      updated.w.val.length ≤ fp.w.val.length + ranges.length ⦄ := by
  induction ranges generalizing fp with
  | nil => simp [sequence]
  | cons range rest ih =>
    simp only [sequence]
    step with source_add_update fp range write (by simp at readRoom; omega)
      (by simp at writeRoom; omega) as ⟨first, mode, all, rel, acq, reads, writes⟩
    have rlen : first.r.val.length ≤ fp.r.val.length + 1 := by
      rw [reads]
      split
      · omega
      · exact native_append_length _ _
    have wlen : first.w.val.length ≤ fp.w.val.length + 1 := by
      rw [writes]
      split
      · omega
      · exact native_append_length _ _
    step with ih first (by simp at readRoom; omega) (by simp at writeRoom; omega)
      as ⟨updated, mode', all', rel', acq', reads', writes'⟩
    refine ⟨mode'.trans mode, all'.trans all, rel'.trans rel, acq'.trans acq, ?_, ?_⟩ <;>
      simp at readRoom writeRoom ⊢ <;> omega

theorem sequence_keys (keys : List U64) (fp : machine_footprint.Footprint) (write : Bool)
    (relRoom : fp.rel.val.length + keys.length < Usize.max)
    (acqRoom : fp.acq.val.length + keys.length < Usize.max) :
    sequence (appendKey write) keys fp ⦃ updated =>
      updated.sync_only = fp.sync_only ∧ updated.all = fp.all ∧
      updated.r = fp.r ∧ updated.w = fp.w ∧
      updated.rel.val = (if write then fp.rel.val else fp.rel.val ++ keys) ∧
      updated.acq.val = (if write then fp.acq.val ++ keys else fp.acq.val) ⦄ := by
  induction keys generalizing fp with
  | nil => cases write <;> simp [sequence]
  | cons key rest ih =>
    cases write <;> simp only [sequence, appendKey, Bool.false_eq_true, ↓reduceIte, Std.bind_assoc]
    · step as ⟨next, value⟩
      step with ih { fp with rel := next } (by simp_all; omega) (by simp_all; omega)
        as ⟨updated, mode, all, reads, writes, rel, acq⟩
      simp [mode, all, reads, writes, rel, acq, value, List.append_assoc]
    · step as ⟨next, value⟩
      step with ih { fp with acq := next } (by simp_all; omega) (by simp_all; omega)
        as ⟨updated, mode, all, reads, writes, rel, acq⟩
      simp [mode, all, reads, writes, rel, acq, value, List.append_assoc]

theorem source_merge_summary_keys (fp : machine_footprint.Footprint) (summary : effects.Summary)
    (readRoom : fp.r.val.length + summary.reads.val.length + summary.writes.val.length < Usize.max)
    (writeRoom : fp.w.val.length + summary.reads.val.length + summary.writes.val.length < Usize.max)
    (relRoom : fp.rel.val.length + summary.releases.val.length + summary.acquires.val.length < Usize.max)
    (acqRoom : fp.acq.val.length + summary.releases.val.length + summary.acquires.val.length < Usize.max) :
    machine_footprint.merge_summary fp summary ⦃ updated =>
      updated.sync_only = fp.sync_only ∧ updated.all = (fp.all || summary.all) ∧
      updated.rel.val = fp.rel.val ++ summary.releases.val ∧
      updated.acq.val = fp.acq.val ++ summary.acquires.val ⦄ := by
  rw [source_merge_summary]
  unfold nativeMerge
  step with sequence_ranges_frame summary.reads.val fp false (by omega) (by omega)
    as ⟨afterReads, mode0, all0, rel0, acq0, rlen0, wlen0⟩
  step with sequence_ranges_frame summary.writes.val afterReads true (by omega) (by omega)
    as ⟨afterWrites, mode1, all1, rel1, acq1, rlen1, wlen1⟩
  step with sequence_keys summary.releases.val afterWrites false (by rw [rel1, rel0]; omega)
    (by rw [acq1, acq0]; omega) as ⟨afterReleases, mode2, all2, r2, w2, rel2, acq2⟩
  simp only [Bool.false_eq_true, ite_false] at rel2 acq2
  step with sequence_keys summary.acquires.val afterReleases true
    (by rw [rel2, rel1, rel0]; simp; omega)
    (by rw [acq2, acq1, acq0]; omega) as ⟨afterAcquires, mode3, all3, r3, w3, rel3, acq3⟩
  simp only [ite_true] at rel3 acq3
  simp [mode3, mode2, mode1, mode0, all3, all2, all1, all0,
    rel3, rel2, rel1, rel0, acq3, acq2, acq1, acq0]

end MemoryWrite.MachineFootprint
