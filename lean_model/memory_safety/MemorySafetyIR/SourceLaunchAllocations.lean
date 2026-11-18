/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchModel
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source

set_option autoImplicit false
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

@[step]
theorem source_project_allocation (allocation : machine.Alloc) (index : Usize) :
    strict_memory_launch.project_allocation allocation index
      ⦃ result => ∀ output, result = .Ok output →
        allocationModel output = allocationMemoryModel allocation ∧
        allocation.base.val + allocation.init.val.length ≤ U64.max ⦄ := by
  unfold strict_memory_launch.project_allocation
  step with UScalar.cast_inBounds_spec .U64 (alloc.vec.Vec.len allocation.init)
    (by have := alloc.vec.Vec.len_ineq allocation.init; rcases Usize.bounds_eq with bound | bound <;> scalar_tac) as ⟨bytes, bytesVal⟩
  have checked := U64.checked_add_bv_spec allocation.base bytes
  cases computed : U64.checked_add allocation.base bytes with
  | none => simp [lift]
  | some sum =>
    simp only [computed] at checked
    simp only [lift, bind_ok, Option.isNone, Bool.false_eq_true,
      ite_false, WP.spec_ok]
    intro output equal
    cases equal
    refine ⟨?_, ?_⟩
    · simp [allocationModel, allocationMemoryModel, bytesVal]
    · simpa only [bytesVal, alloc.vec.Vec.len_val] using checked.1

theorem source_allocations_loop (launch : machine.Launch)
    (allocations output : alloc.vec.Vec memory_safety.Allocation)
    (failure : Option strict_memory_admission.Refusal) (index : Usize)
    (computed : strict_memory_launch.project_allocations_loop launch allocations failure index =
      .ok (output, none)) :
    failure = none ∧ output.val.map allocationModel = allocations.val.map allocationModel ++
      (launch.allocs.val.drop index.val).map allocationMemoryModel := by
  generalize fuelEq : launch.allocs.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing allocations failure index with
  | h fuel ih =>
    have unroll : strict_memory_launch.project_allocations_loop launch allocations failure index = (do
        let result ← strict_memory_launch.project_allocations_loop.body launch allocations failure index
        match result with
        | .cont (next, failed, position) =>
          strict_memory_launch.project_allocations_loop launch next failed position
        | .done result => .ok result) := by
      unfold strict_memory_launch.project_allocations_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont triple => rcases triple with ⟨a, b, c⟩; rfl
      | done result => rfl
    rw [unroll] at computed
    by_cases inside : index.val < launch.allocs.val.length
    · have test : index < alloc.vec.Vec.len launch.allocs := by scalar_tac
      cases failure with
      | some reason =>
        simp [strict_memory_launch.project_allocations_loop.body, test] at computed
      | none =>
        have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
          have := alloc.vec.Vec.len_ineq launch.allocs
          scalar_tac
        let after : Usize := Usize.ofNatCore (index.val + 1) bound
        have afterVal : after.val = index.val + 1 := by simp [after]
        have increment := source_usize_increment index bound
        let allocation := launch.allocs.val[index.val]
        have allocationRead : alloc.vec.Vec.index_usize launch.allocs index = .ok allocation := by
          simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside, allocation]
        simp only [strict_memory_launch.project_allocations_loop.body, test, ite_true,
          core.option.Option.is_none, Option.isNone, alloc.vec.Vec.index_slice_index, allocationRead, bind_ok,
          Std.bind_assoc] at computed
        obtain ⟨result, projected, finished⟩ := source_bind_success _ _ _ computed
        cases result with
        | Err reason =>
          simp only [bind_ok, increment] at finished
          change Std.bind (.ok (ControlFlow.cont (allocations, some reason, after))) _ = .ok (output, none) at finished
          simp only [bind_ok] at finished
          have tail := ih (launch.allocs.val.length - after.val) (by omega)
            allocations (some reason) after finished rfl
          cases tail.1
        | Ok value =>
          simp only [Std.bind_assoc, bind_ok] at finished
          obtain ⟨next, pushed, finished⟩ := source_bind_success _ _ _ finished
          simp only [bind_ok, increment] at finished
          change Std.bind (.ok (ControlFlow.cont (next, none, after))) _ = .ok (output, none) at finished
          simp only [bind_ok] at finished
          have tail := ih (launch.allocs.val.length - after.val) (by omega) next none after finished rfl
          have contract := source_project_allocation allocation index
          rw [projected] at contract
          simp only [WP.spec_ok] at contract
          have mapped := (contract value rfl).1
          refine ⟨rfl, ?_⟩
          rw [tail.2, vec_push_success allocations next value pushed, List.map_append,
            List.map_cons, List.map_nil, mapped]
          rw [List.drop_eq_getElem_cons inside]
          simp only [List.map_cons, List.append_assoc, List.singleton_append, afterVal]
          rfl
    · have test : ¬index < alloc.vec.Vec.len launch.allocs := by scalar_tac
      simp only [strict_memory_launch.project_allocations_loop.body, test, ite_false, bind_ok,
        ok.injEq, Prod.mk.injEq] at computed
      rcases computed with ⟨rfl, failed⟩
      refine ⟨failed, ?_⟩
      simp [List.drop_eq_nil_of_le (show launch.allocs.val.length ≤ index.val by omega)]

theorem source_project_allocations (launch : machine.Launch)
    (output : alloc.vec.Vec memory_safety.Allocation)
    (computed : strict_memory_launch.project_allocations launch = .ok (.Ok output)) :
    output.val.map allocationModel = launchAllocations launch := by
  unfold strict_memory_launch.project_allocations at computed
  obtain ⟨⟨allocations, failure⟩, ran, finished⟩ := source_bind_success _ _ _ computed
  cases failure with
  | some reason => simp at finished
  | none =>
    have equal : allocations = output := by simpa using finished
    subst allocations
    have same := (source_allocations_loop launch (alloc.vec.Vec.new _) output none 0#usize ran).2
    simpa [alloc.vec.Vec.new, launchAllocations] using same

end MemorySafetyIR
