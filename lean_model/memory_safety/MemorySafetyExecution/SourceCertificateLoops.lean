/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceCertificate

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

theorem source_closed_loop (certificate : Slice checked_execution.State)
    (next : alloc.vec.Vec checked_execution.State) (index : Usize) :
    checked_execution.state_closed_loop certificate next index =
      checkAll (checked_execution.state_present certificate) (next.val.drop index.val) := by
  generalize fuelEq : next.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : checked_execution.state_closed_loop certificate next index = (do
        let result ← checked_execution.state_closed_loop.body certificate next index
        match result with
        | .cont after => checked_execution.state_closed_loop certificate next after
        | .done output => .ok output) := by
      unfold checked_execution.state_closed_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result <;> rfl
    rw [unroll]
    by_cases inside : index.val < next.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq next
        scalar_tac
      let after : Usize := Usize.ofNatCore (index.val + 1) bound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := usize_increment index bound
      have lookup : alloc.vec.Vec.index_usize next index = .ok next.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      have test : index < alloc.vec.Vec.len next := by scalar_tac
      simp only [checked_execution.state_closed_loop.body, test, ite_true,
        alloc.vec.Vec.index_slice_index, lookup, bind_ok, Std.bind_assoc,
        List.drop_eq_getElem_cons inside, checkAll]
      congr 1
      funext checked
      cases checked with
      | false => simp only [Bool.false_eq_true, ite_false, bind_ok]
      | true =>
        simp only [ite_true, increment, bind_ok]
        rw [ih (next.val.length - after.val) (by omega) after rfl]
        simp only [afterVal]
    · have test : ¬index < alloc.vec.Vec.len next := by scalar_tac
      simp [checked_execution.state_closed_loop.body, test, checkAll,
        List.drop_eq_nil_of_le (show next.val.length ≤ index.val by omega)]

theorem source_verify_loop (program : execution_core.Program)
    (certificate : Slice checked_execution.State) (index : Usize) :
    checked_execution.verify_loop program certificate index =
      checkAll (checked_execution.state_closed program certificate)
        (certificate.val.drop index.val) := by
  generalize fuelEq : certificate.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : checked_execution.verify_loop program certificate index = (do
        let result ← checked_execution.verify_loop.body program certificate index
        match result with
        | .cont after => checked_execution.verify_loop program certificate after
        | .done output => .ok output) := by
      unfold checked_execution.verify_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result <;> rfl
    rw [unroll]
    by_cases inside : index.val < certificate.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq certificate
        scalar_tac
      let after : Usize := Usize.ofNatCore (index.val + 1) bound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := usize_increment index bound
      have lookup : Slice.index_usize certificate index = .ok certificate.val[index.val] := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside]
      have test : index < Slice.len certificate := by scalar_tac
      simp only [checked_execution.verify_loop.body, test, ite_true,
        lookup, bind_ok, Std.bind_assoc, List.drop_eq_getElem_cons inside, checkAll]
      congr 1
      funext checked
      cases checked with
      | false => simp only [Bool.false_eq_true, ite_false, bind_ok]
      | true =>
        simp only [ite_true, increment, bind_ok]
        rw [ih (certificate.val.length - after.val) (by omega) after rfl]
        simp only [afterVal]
    · have test : ¬index < Slice.len certificate := by scalar_tac
      simp [checked_execution.verify_loop.body, test, checkAll,
        List.drop_eq_nil_of_le (show certificate.val.length ≤ index.val by omega)]

end MemorySafetyExecution.Source
