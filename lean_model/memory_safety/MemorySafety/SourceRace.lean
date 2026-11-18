/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceCertificate
import MemorySafety.SourcePrimitives

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety.Source

theorem source_overlaps_exact (a aw b bw : U64) :
    memory_safety.overlaps a aw b bw = .ok (decide (Overlap a.val aw.val b.val bw.val)) := by
  obtain ⟨output, computed, meaning⟩ := MemoryWrite.spec_returns _ _ (source_overlaps a aw b bw)
  have equal : output = decide (Overlap a.val aw.val b.val bw.val) := by
    apply Bool.eq_iff_iff.mpr
    simpa only [decide_eq_true_eq] using meaning
  simpa only [equal] using computed

def checkRace (previous : memory_safety.AccessRecord) (thread : Usize)
    (address width : U64) (isWrite valid : Bool) : Result Bool := do
  if previous.thread != thread then
    if previous.is_write then
      let overlaps ← memory_safety.overlaps previous.address previous.width address width
      if overlaps then .ok false else .ok valid
    else if isWrite then
      let overlaps ← memory_safety.overlaps previous.address previous.width address width
      if overlaps then .ok false else .ok valid
    else .ok valid
  else .ok valid

theorem check_race_exact (previous : memory_safety.AccessRecord) (thread : Usize)
    (address width : U64) (isWrite valid : Bool) :
    checkRace previous thread address width isWrite valid = .ok
      (valid && decide (¬Conflict (recordModel previous) ⟨thread.val, address.val, width.val, isWrite⟩)) := by
  by_cases same : previous.thread = thread <;>
    by_cases overlap : Overlap previous.address.val previous.width.val address.val width.val <;>
    cases oldWrite : previous.is_write <;> cases isWrite <;> cases valid <;>
    simp [checkRace, source_overlaps_exact, same, overlap, oldWrite, Conflict, recordModel,
      scalar_val_injective]

theorem source_race_loop (history : alloc.vec.Vec memory_safety.AccessRecord)
    (thread : Usize) (address width : U64) (isWrite valid : Bool) (index : Usize) :
    memory_safety.access_valid_loop1 history thread address width isWrite valid index = .ok
      (valid && (history.val.drop index.val).all (fun previous =>
        decide (¬Conflict (recordModel previous) ⟨thread.val, address.val, width.val, isWrite⟩))) := by
  generalize fuelEq : history.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing valid index with
  | h fuel ih =>
    have unroll : memory_safety.access_valid_loop1 history thread address width isWrite valid index = (do
        let result ← memory_safety.access_valid_loop1.body history thread address width isWrite valid index
        match result with
        | .cont pair => memory_safety.access_valid_loop1 history thread address width isWrite pair.1 pair.2
        | .done output => .ok output) := by
      unfold memory_safety.access_valid_loop1
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll]
    by_cases inside : index.val < history.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq history
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have increment := usize_increment index bound
      have lookup : alloc.vec.Vec.index_usize history index = .ok history.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      have test : index < alloc.vec.Vec.len history := by scalar_tac
      cases valid with
      | false => simp [memory_safety.access_valid_loop1.body, test]
      | true =>
        have body : memory_safety.access_valid_loop1.body history thread address width isWrite true index = (do
            let safe ← checkRace history.val[index.val] thread address width isWrite true
            .ok (.cont (safe, next))) := by
          simp only [memory_safety.access_valid_loop1.body, test, ite_true,
            alloc.vec.Vec.index_slice_index, lookup, bind_ok]
          change Aeneas.Std.bind (checkRace history.val[index.val] thread address width isWrite true) _ = _
          congr 1
          funext checked
          rw [increment]
          simp only [bind_ok]
          rfl
        rw [body, check_race_exact]
        simp only [bind_ok, Bool.true_and]
        rw [ih (history.val.length - next.val) (by omega) _ next rfl]
        rw [List.drop_eq_getElem_cons inside]
        simp only [List.all_cons, nextVal]
        rfl
    · have test : ¬index < alloc.vec.Vec.len history := by scalar_tac
      simp [memory_safety.access_valid_loop1.body, test,
        List.drop_eq_nil_of_le (show history.val.length ≤ index.val by omega)]

@[step]
theorem source_access_races (history : alloc.vec.Vec memory_safety.AccessRecord)
    (thread : Usize) (address width : U64) (isWrite valid : Bool) :
    memory_safety.access_valid_loop1 history thread address width isWrite valid 0#usize
      ⦃ result => result = true → valid = true ∧
        ∀ previous ∈ history.val,
          ¬Conflict (recordModel previous) ⟨thread.val, address.val, width.val, isWrite⟩ ⦄ := by
  rw [source_race_loop]
  simp only [WP.spec_ok]
  change valid && history.val.all (fun previous =>
    decide (¬Conflict (recordModel previous) ⟨thread.val, address.val, width.val, isWrite⟩)) = true → _
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq]
  exact id

end MemorySafety.Source
